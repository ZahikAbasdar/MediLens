"""
app.py
======
The FastAPI application entry point. This is the file you run to
start the backend server:

    uvicorn app:app --reload

WHAT HAPPENS HERE:
1. Set up logging (utils.py)
2. Create the FastAPI app instance
3. Configure CORS (so the Streamlit frontend, running on a different
   port, is allowed to call this API from the browser)
4. Register a startup event that creates database tables if they
   don't exist yet (database.py's init_db)
5. Mount every router from routes.py under the main app
6. Add a root health-check endpoint
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from database import init_db
from utils import setup_logging
from routes import (
    auth_router, reports_router, chat_router,
    dashboard_router, explain_router, export_router,
)

setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.APP_NAME,
    description=f"{settings.APP_TAGLINE} - An AI Healthcare Screening Assistant. "
                f"Educational use only - not a substitute for professional medical advice.",
    version=settings.APP_VERSION,
)

# CORS: allows the Streamlit frontend (typically http://localhost:8501)
# to make requests to this API (typically http://localhost:8000).
# In a real production deployment, replace "*" with your actual
# frontend domain for tighter security.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    """Runs once when the server starts. Creates DB tables if needed."""
    logger.info("Starting %s v%s...", settings.APP_NAME, settings.APP_VERSION)
    init_db()


# Mount all feature routers
app.include_router(auth_router)
app.include_router(reports_router)
app.include_router(chat_router)
app.include_router(dashboard_router)
app.include_router(explain_router)
app.include_router(export_router)


@app.get("/")
def root():
    """Simple health-check / welcome endpoint."""
    return {
        "app": settings.APP_NAME,
        "tagline": settings.APP_TAGLINE,
        "status": "running",
        "docs": "/docs",
        "disclaimer": settings.MEDICAL_DISCLAIMER,
    }


@app.get("/health")
def health_check():
    """Used by deployment platforms / uptime monitors."""
    return {"status": "healthy"}
