"""
ai_engine.py
============
The single place in MediLens that actually talks to an LLM.

WHY A SINGLE WRAPPER FUNCTION:
Every other module (chatbot.py, routes.py) calls ONE function -
`generate_ai_response(prompt)` - and never needs to know or care
which provider (Gemini / Claude / OpenAI / Ollama) is configured.
`config.py`'s `LLM_PROVIDER` setting decides which branch runs.
This means switching providers is a ONE-LINE change in `.env`,
with zero changes needed anywhere else in the codebase.

SAFETY GUARANTEE:
`generate_ai_response()` ALWAYS appends `settings.MEDICAL_DISCLAIMER`
to the end of every single response, in OUR code, after the LLM has
already produced its text. This means the disclaimer can never be
"forgotten" by a prompt, a model update, or a provider's own model
behavior - it is structurally guaranteed by this wrapper function
itself, not by asking the model nicely.
"""

import logging
from typing import Optional

from config import settings

logger = logging.getLogger(__name__)


class AIEngineError(Exception):
    """Raised when no LLM provider could produce a response."""
    pass


# ------------------------------------------------------------
# Per-provider call functions
# ------------------------------------------------------------
def _call_gemini(prompt: str) -> str:
    """Calls Google Gemini via the google-generativeai SDK."""
    import google.generativeai as genai

    if not settings.GEMINI_API_KEY:
        raise AIEngineError("GEMINI_API_KEY is not set in your .env file")

    genai.configure(api_key=settings.GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text


def _call_claude(prompt: str) -> str:
    """Calls Anthropic's Claude via the anthropic SDK."""
    import anthropic

    if not settings.ANTHROPIC_API_KEY:
        raise AIEngineError("ANTHROPIC_API_KEY is not set in your .env file")

    client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    # content is a list of blocks; join any text blocks together
    return "".join(block.text for block in message.content if hasattr(block, "text"))


def _call_openai(prompt: str) -> str:
    """Calls OpenAI's API via the openai SDK."""
    from openai import OpenAI

    if not settings.OPENAI_API_KEY:
        raise AIEngineError("OPENAI_API_KEY is not set in your .env file")

    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def _call_ollama(prompt: str) -> str:
    """
    Calls a locally-running Ollama server - no API key needed since
    it runs entirely on the user's own machine. Useful for developers
    who want to run MediLens fully offline/free, with the tradeoff
    of needing a reasonably powerful local machine.
    """
    import requests

    response = requests.post(
        f"{settings.OLLAMA_BASE_URL}/api/generate",
        json={"model": settings.OLLAMA_MODEL, "prompt": prompt, "stream": False},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["response"]


_PROVIDER_FUNCTIONS = {
    "gemini": _call_gemini,
    "claude": _call_claude,
    "openai": _call_openai,
    "ollama": _call_ollama,
}


# ------------------------------------------------------------
# Public entry point
# ------------------------------------------------------------
def generate_ai_response(prompt: str, provider: Optional[str] = None) -> str:
    """
    Sends `prompt` to whichever LLM provider is configured (or the
    explicit `provider` override, useful for testing a specific
    provider without changing global settings) and returns plain text.

    Always appends the medical disclaimer before returning - see the
    module docstring for why this is done here rather than in the
    prompt text itself.

    If the call fails for any reason (missing API key, network issue,
    provider outage), we raise AIEngineError with a clear message
    rather than letting a raw SDK exception bubble up to the user.
    """
    provider_name = (provider or settings.LLM_PROVIDER).lower()
    call_function = _PROVIDER_FUNCTIONS.get(provider_name)

    if call_function is None:
        raise AIEngineError(
            f"Unknown LLM_PROVIDER '{provider_name}'. "
            f"Valid options: {', '.join(_PROVIDER_FUNCTIONS)}"
        )

    try:
        logger.info("Calling LLM provider: %s", provider_name)
        raw_response = call_function(prompt)
    except AIEngineError:
        raise
    except Exception as exc:
        logger.error("LLM call failed for provider '%s': %s", provider_name, exc)
        raise AIEngineError(
            f"The AI provider '{provider_name}' could not be reached. "
            f"Please check your API key and internet connection. Details: {exc}"
        ) from exc

    if not raw_response or not raw_response.strip():
        raise AIEngineError(f"Provider '{provider_name}' returned an empty response.")

    return f"{raw_response.strip()}\n\n---\n_{settings.MEDICAL_DISCLAIMER}_"
