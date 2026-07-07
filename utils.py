"""
utils.py
========
Small, reusable helper functions that don't belong to any single
module. The rule for what goes here: if two or more other files
would otherwise duplicate the same logic, it goes here instead.

Contains:
  - File validation and saving helpers (used by routes.py)
  - Text-cleaning helpers (used by ocr_engine.py and parser.py)
  - A safe-float parser (used by parser.py and ml_engine.py)
  - Logging setup (used by app.py)
"""

import logging
import re
import shutil
import uuid
from pathlib import Path
from typing import Optional

from fastapi import UploadFile, HTTPException

from config import settings


def setup_logging() -> None:
    """
    Configures logging ONCE for the whole application. Called from
    app.py at startup. Using a shared format means every log line
    from every module (auth, ocr, ai_engine, etc.) looks consistent
    and includes the module name, making debugging much easier.
    """
    logging.basicConfig(
        level=logging.INFO if not settings.DEBUG else logging.DEBUG,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ------------------------------------------------------------
# File handling
# ------------------------------------------------------------
def validate_file_extension(filename: str) -> str:
    """
    Checks that an uploaded file's extension is one MediLens supports.
    Returns the lowercase extension (e.g. ".pdf") on success, or
    raises an HTTP 400 error that FastAPI will send back to the client.
    """
    ext = Path(filename).suffix.lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{ext}'. "
                f"Allowed types: {', '.join(sorted(settings.ALLOWED_EXTENSIONS))}"
            ),
        )
    return ext


def save_upload_file(upload_file: UploadFile, user_id: int) -> Path:
    """
    Saves an uploaded file to disk under uploads/<user_id>/<uuid>_<name>
    and returns the path where it was saved.

    We use a UUID prefix so two different users (or the same user
    uploading twice) can never accidentally overwrite each other's
    files, even if they upload a file with the identical name.
    """
    ext = validate_file_extension(upload_file.filename)

    user_dir = settings.UPLOAD_DIR / str(user_id)
    user_dir.mkdir(parents=True, exist_ok=True)

    safe_name = f"{uuid.uuid4().hex}_{Path(upload_file.filename).name}"
    destination = user_dir / safe_name

    # Enforce max file size while streaming to disk, rather than
    # loading the whole file into memory first.
    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    size = 0
    with open(destination, "wb") as buffer:
        while chunk := upload_file.file.read(1024 * 1024):
            size += len(chunk)
            if size > max_bytes:
                buffer.close()
                destination.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=400,
                    detail=f"File exceeds max size of {settings.MAX_UPLOAD_SIZE_MB}MB",
                )
            buffer.write(chunk)

    return destination


# ------------------------------------------------------------
# Text cleaning
# ------------------------------------------------------------
def clean_ocr_text(text: str) -> str:
    """
    Cleans up common OCR noise before we try to parse structured data
    out of it:
      - Collapses multiple spaces/tabs into one
      - Removes stray control characters
      - Normalizes different types of dashes/bullets that OCR engines
        sometimes misread
      - Strips leading/trailing whitespace on each line
    We deliberately do NOT remove newlines - line structure matters
    a lot for table-like medical reports.
    """
    if not text:
        return ""

    # Remove non-printable control characters (keep newlines/tabs)
    text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E]", " ", text)

    # Normalize common OCR misreads of bullets/dashes
    text = text.replace("—", "-").replace("–", "-").replace("•", "-")

    # Collapse repeated spaces/tabs (but not newlines)
    text = re.sub(r"[ \t]+", " ", text)

    # Strip trailing whitespace per line, drop fully-empty lines at edges
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(lines)
    text = text.strip()

    return text


def safe_float(value: Optional[str]) -> Optional[float]:
    """
    Tries to parse a string like "13.5", "13.5 mg/dL", "< 5", or
    ">200" into a float. Returns None if no number can be found.

    Medical reports often report values like "<0.5" or "Negative" -
    we want the numeric part when it exists, and a clean None when
    it doesn't, rather than crashing.
    """
    if value is None:
        return None
    match = re.search(r"[-+]?\d*\.?\d+", str(value))
    if not match:
        return None
    try:
        return float(match.group())
    except ValueError:
        return None


def truncate(text: str, max_chars: int = 200) -> str:
    """Shortens text for logging/preview purposes, adding an ellipsis if cut."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."
