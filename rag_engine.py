"""
rag_engine.py
=============
Implements Retrieval-Augmented Generation (RAG) - Feature #7.

THE CORE IDEA:
Instead of stuffing a user's ENTIRE report history into every
chatbot prompt (expensive, and eventually too big to fit), we:
  1. Split each report's text into small overlapping "chunks".
  2. Convert each chunk into a numeric vector ("embedding") that
     captures its meaning.
  3. Store all chunks' vectors in a FAISS index (a fast
     nearest-neighbor search structure) - ONE INDEX PER USER, so
     one user's medical data is never searchable from another
     user's index.
  4. When the user asks a question, embed the QUESTION the same way,
     and ask FAISS for the most similar chunks.
  5. Only THOSE chunks (not the whole history) get sent to the LLM
     as context - this is what prompts.py's `build_rag_chat_prompt`
     consumes, and it's what prevents hallucination: the model
     literally cannot make things up about data it was never shown.

WHY ONE FAISS INDEX PER USER (not one global index):
Medical data is extremely sensitive. Keeping a completely separate
index file per user (vector_db/user_<id>.faiss) means there is no
code path by which one user's retrieval could ever surface another
user's report content, even by accident.
"""

import logging
import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np

from config import settings

logger = logging.getLogger(__name__)

# ------------------------------------------------------------
# Lazy-loaded embedding model (downloads weights on first use)
# ------------------------------------------------------------
_embedding_model = None


def _get_embedding_model():
    """
    Loads the sentence-transformers model once and reuses it.
    Lazy-loaded for the same reason as EasyOCR in ocr_engine.py:
    loading a neural network at import time would make every test,
    script, or FastAPI worker startup slow even when RAG isn't used.
    """
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model '%s' (first use only)...", settings.EMBEDDING_MODEL_NAME)
        _embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
    return _embedding_model


def embed_texts(texts: List[str]) -> np.ndarray:
    """Converts a list of strings into a 2D numpy array of embedding vectors."""
    model = _get_embedding_model()
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return embeddings.astype("float32")


# ------------------------------------------------------------
# Text chunking (pure logic, no ML - easy to test independently)
# ------------------------------------------------------------
def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
    """
    Splits text into overlapping chunks of roughly `chunk_size`
    characters. The `overlap` ensures a sentence that gets cut in
    half at a chunk boundary still appears in full in the NEXT
    chunk too, so we don't lose meaning at the edges.

    Splitting is done on whitespace boundaries (never mid-word) to
    keep chunks readable.
    """
    chunk_size = chunk_size or settings.RAG_CHUNK_SIZE
    overlap = overlap or settings.RAG_CHUNK_OVERLAP

    words = text.split()
    if not words:
        return []

    chunks = []
    current_words: List[str] = []
    current_len = 0

    for word in words:
        current_words.append(word)
        current_len += len(word) + 1  # +1 for the space

        if current_len >= chunk_size:
            chunks.append(" ".join(current_words))
            # Keep the last `overlap` characters worth of words for the next chunk
            overlap_words = []
            overlap_len = 0
            for w in reversed(current_words):
                overlap_len += len(w) + 1
                overlap_words.insert(0, w)
                if overlap_len >= overlap:
                    break
            current_words = overlap_words
            current_len = overlap_len

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks


# ------------------------------------------------------------
# FAISS index management (one index file per user, on disk)
# ------------------------------------------------------------
def _index_paths(user_id: int) -> Tuple[Path, Path]:
    """Returns (faiss_index_path, metadata_path) for a given user."""
    base = settings.VECTOR_DB_DIR / f"user_{user_id}"
    return base.with_suffix(".index"), base.with_suffix(".meta")


def build_or_update_index(user_id: int, report_id: int, text: str) -> int:
    """
    Chunks a report's text, embeds each chunk, and adds it to that
    user's FAISS index (creating the index if it doesn't exist yet).

    Each chunk's metadata (which report_id it came from, and the
    chunk text itself) is stored alongside the index in a pickle
    file, since FAISS itself only stores vectors, not the original text.

    Returns the number of chunks added.
    """
    import faiss

    chunks = chunk_text(text)
    if not chunks:
        logger.warning("No chunks produced for report %d (empty text)", report_id)
        return 0

    vectors = embed_texts(chunks)
    dimension = vectors.shape[1]

    index_path, meta_path = _index_paths(user_id)

    if index_path.exists():
        index = faiss.read_index(str(index_path))
        with open(meta_path, "rb") as f:
            metadata: List[dict] = pickle.load(f)
    else:
        index = faiss.IndexFlatL2(dimension)  # simple, exact L2-distance search
        metadata = []

    index.add(vectors)
    for chunk in chunks:
        metadata.append({"report_id": report_id, "text": chunk})

    faiss.write_index(index, str(index_path))
    with open(meta_path, "wb") as f:
        pickle.dump(metadata, f)

    logger.info("Added %d chunks from report %d to user %d's RAG index", len(chunks), report_id, user_id)
    return len(chunks)


def search_index(user_id: int, query: str, top_k: int = None, report_id: int = None) -> List[str]:
    """
    Embeds the user's question and retrieves the most similar chunks
    from their FAISS index.

    If `report_id` is given, results are filtered to chunks from that
    specific report only (used when the chatbot is scoped to "explain
    THIS report" rather than the user's whole history).

    Returns a list of matching chunk texts (empty list if the user
    has no index yet, e.g. they haven't uploaded any reports).
    """
    import faiss

    top_k = top_k or settings.RAG_TOP_K
    index_path, meta_path = _index_paths(user_id)

    if not index_path.exists():
        logger.info("No RAG index found for user %d yet", user_id)
        return []

    index = faiss.read_index(str(index_path))
    with open(meta_path, "rb") as f:
        metadata: List[dict] = pickle.load(f)

    query_vector = embed_texts([query])
    # Search for more than top_k when filtering by report_id, since
    # some results may be filtered out afterward.
    search_k = min(top_k * 5, index.ntotal) if report_id else min(top_k, index.ntotal)
    if search_k == 0:
        return []

    distances, indices = index.search(query_vector, search_k)

    results = []
    for idx in indices[0]:
        if idx < 0 or idx >= len(metadata):
            continue
        entry = metadata[idx]
        if report_id is not None and entry["report_id"] != report_id:
            continue
        results.append(entry["text"])
        if len(results) >= top_k:
            break

    return results
