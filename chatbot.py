"""
chatbot.py
==========
Implements the AI Chatbot (Feature #6). This is the orchestration
layer that ties together:

  - rag_engine.py   -> retrieves relevant chunks from the user's own reports
  - ai_engine.py    -> generates the actual natural-language answer
  - models.py       -> persists every message (both user & assistant) so
                       the chatbot "remembers" past conversations across
                       sessions, not just within one browser tab

FLOW for a single chat turn:
  1. Save the user's message to the database immediately.
  2. Pull recent chat history (for conversational continuity).
  3. Use rag_engine.search_index() to retrieve relevant report chunks.
  4. Build the grounded prompt (prompts.py) and call ai_engine.
  5. Save the assistant's reply to the database.
  6. Return the reply + which chunks were used (for UI transparency).
"""

import logging
from typing import List, Optional

from sqlalchemy.orm import Session

from ai_engine import generate_ai_response, AIEngineError
from config import settings
from models import ChatMessage
from prompts import build_rag_chat_prompt
from rag_engine import search_index

logger = logging.getLogger(__name__)


def _get_recent_history(db: Session, user_id: int, limit: int = 6) -> str:
    """
    Pulls the last `limit` messages for this user and formats them as
    plain text, so the LLM has short-term conversational context
    (e.g. so "what about the second one?" makes sense as a follow-up).

    We deliberately keep this SHORT (6 messages = ~3 exchanges) rather
    than the full history, since old chat turns add cost and can
    distract the model from the current question.
    """
    messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.user_id == user_id)
        .order_by(ChatMessage.created_at.desc())
        .limit(limit)
        .all()
    )
    messages.reverse()  # oldest first, for natural reading order

    lines = [f"{m.role.capitalize()}: {m.content}" for m in messages]
    return "\n".join(lines)


def handle_chat_message(
    db: Session,
    user_id: int,
    user_message: str,
    report_id: Optional[int] = None,
) -> dict:
    """
    Main entry point called by routes.py's /chat endpoint.

    Returns a dict with keys: "reply", "sources_used", "disclaimer" -
    matching schemas.ChatResponse.
    """
    # 1. Persist the user's message immediately, so it's never lost
    #    even if the AI call below fails.
    user_entry = ChatMessage(user_id=user_id, report_id=report_id, role="user", content=user_message)
    db.add(user_entry)
    db.commit()

    # 2. Recent conversational context
    history = _get_recent_history(db, user_id)

    # 3. Retrieve relevant report chunks via RAG
    retrieved_chunks: List[str] = []
    try:
        retrieved_chunks = search_index(user_id, user_message, report_id=report_id)
    except Exception as exc:
        # If the user has no reports yet (no index file exists), that's
        # expected and not an error - search_index already returns []
        # in that case. This except is for genuinely unexpected failures
        # (e.g. a corrupted index file), which we log but don't crash on.
        logger.warning("RAG retrieval failed for user %d: %s", user_id, exc)

    retrieved_context = "\n---\n".join(retrieved_chunks)

    # 4. Build the grounded prompt and call the LLM
    prompt = build_rag_chat_prompt(user_message, retrieved_context, history)

    try:
        reply_text = generate_ai_response(prompt)
    except AIEngineError as exc:
        # Even if the AI call fails, we give the user a clear, honest
        # message rather than a raw stack trace or a silent failure.
        reply_text = (
            "I'm sorry, I couldn't generate a response right now "
            f"({exc}). Please check your AI provider configuration, "
            "or try again in a moment."
        )

    # 5. Persist the assistant's reply
    assistant_entry = ChatMessage(user_id=user_id, report_id=report_id, role="assistant", content=reply_text)
    db.add(assistant_entry)
    db.commit()

    return {
        "reply": reply_text,
        "sources_used": retrieved_chunks,
        "disclaimer": settings.MEDICAL_DISCLAIMER,
    }


def get_chat_history(db: Session, user_id: int, report_id: Optional[int] = None) -> List[ChatMessage]:
    """
    Returns the full chat history for a user (optionally scoped to one
    report), oldest first - used by routes.py to render the chat UI
    and by the Streamlit frontend to redraw the conversation.
    """
    query = db.query(ChatMessage).filter(ChatMessage.user_id == user_id)
    if report_id is not None:
        query = query.filter(ChatMessage.report_id == report_id)
    return query.order_by(ChatMessage.created_at.asc()).all()
