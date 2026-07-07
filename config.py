"""
config.py
=========
This is the SINGLE SOURCE OF TRUTH for all settings in MediLens.

WHY THIS FILE EXISTS:
Instead of scattering things like "the database file name" or
"how long a login token lasts" across 20 different files, we
define them ONCE here. Every other module imports `settings`
from this file.

If you ever need to change something (e.g. move to a different
database, or change the token expiry time), you change it in
ONE place, not everywhere.

We use `pydantic-settings`, which lets us:
1. Define settings as a normal Python class with type hints.
2. Automatically read values from a `.env` file if one exists.
3. Fall back to sensible defaults if no `.env` file is found.

This means a beginner can run the app immediately with defaults,
but a real deployment can override secrets via environment
variables without touching code.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


# BASE_DIR = the folder this config.py file lives in.
# We use this to build absolute paths (for uploads, database, etc.)
# so the app works no matter which folder you run it from.
BASE_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    """
    Every attribute below is a setting. The type hint (str, int, bool)
    tells pydantic what type to expect and validates it automatically.
    """

    # ------------------------------------------------------------
    # App metadata
    # ------------------------------------------------------------
    APP_NAME: str = "MediLens"
    APP_TAGLINE: str = "See Beyond the Report."
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True  # Set to False in real production deployment

    # ------------------------------------------------------------
    # Security / JWT Authentication
    # ------------------------------------------------------------
    # SECRET_KEY is used to cryptographically sign login tokens.
    # In real production, this MUST be overridden via a .env file
    # with a long random string. We give a placeholder default so
    # the app still runs for learning purposes.
    SECRET_KEY: str = "CHANGE_ME_medilens_dev_secret_key_2026"
    ALGORITHM: str = "HS256"                    # JWT signing algorithm
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24   # Tokens last 24 hours

    # ------------------------------------------------------------
    # Database
    # ------------------------------------------------------------
    # SQLite stores everything in a single file - perfect for a
    # beginner/portfolio project. No separate database server needed.
    DATABASE_URL: str = f"sqlite:///{BASE_DIR / 'medilens.db'}"

    # ------------------------------------------------------------
    # File storage
    # ------------------------------------------------------------
    UPLOAD_DIR: Path = BASE_DIR / "uploads"
    VECTOR_DB_DIR: Path = BASE_DIR / "vector_db"
    ALLOWED_EXTENSIONS: set[str] = {
        ".pdf", ".png", ".jpg", ".jpeg", ".doc", ".docx", ".txt"
    }
    MAX_UPLOAD_SIZE_MB: int = 20

    # ------------------------------------------------------------
    # LLM Provider Settings
    # ------------------------------------------------------------
    # The user can choose which AI provider powers the assistant.
    # "LLM_PROVIDER" decides which one ai_engine.py will actually call.
    # Only the API key for the chosen provider needs to be set.
    LLM_PROVIDER: str = "gemini"  # options: "gemini", "claude", "openai", "ollama"

    GEMINI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3"

    # ------------------------------------------------------------
    # Embeddings & RAG
    # ------------------------------------------------------------
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"  # small, fast, runs on CPU
    RAG_CHUNK_SIZE: int = 500       # characters per text chunk
    RAG_CHUNK_OVERLAP: int = 50     # overlap between chunks, preserves context
    RAG_TOP_K: int = 4              # how many chunks to retrieve per question

    # ------------------------------------------------------------
    # Safety / Compliance
    # ------------------------------------------------------------
    # This disclaimer is injected into every AI explanation and
    # every chatbot response. It is NOT optional and NOT editable
    # by the AI - it is appended by our own code after the AI runs.
    MEDICAL_DISCLAIMER: str = (
        "This information is for general educational purposes only. "
        "It is not a medical diagnosis and does not replace advice "
        "from a licensed healthcare professional. Please consult a "
        "doctor for interpretation of your specific results."
    )

    # This tells pydantic-settings: "if a .env file exists, read it"
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


# We create ONE instance of Settings and import THIS everywhere,
# instead of creating new Settings() objects in every file.
settings = Settings()

# Make sure the folders we need actually exist on disk the moment
# this module is imported anywhere in the app.
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
