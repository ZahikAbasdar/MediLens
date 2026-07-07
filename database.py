"""
database.py
===========
This file sets up the connection between our Python code and the
SQLite database file.

WHY THIS FILE EXISTS:
SQLAlchemy needs three things to work:
1. An "engine" - the actual connection to the database file.
2. A "session factory" - creates temporary conversations with the
   database (called a "session") each time we need to read/write data.
3. A "Base" class - all our table definitions (in models.py) will
   inherit from this, so SQLAlchemy knows about them.

We centralize this here so every other file just does:
    from database import get_db, Base
instead of re-configuring the database connection repeatedly.
"""

import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

from config import settings

logger = logging.getLogger(__name__)

# ------------------------------------------------------------
# The Engine
# ------------------------------------------------------------
# `create_engine` doesn't connect immediately - it just prepares
# HOW to connect. The actual connection happens per-request.
#
# `connect_args={"check_same_thread": False}` is SQLite-specific.
# By default, SQLite only allows the thread that created a connection
# to use it. FastAPI handles requests using multiple threads, so we
# need to relax this restriction. This is safe because SQLAlchemy's
# session management still prevents data corruption.
engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=False,  # Set True temporarily if you want to see every SQL query printed
)

# ------------------------------------------------------------
# The Session Factory
# ------------------------------------------------------------
# A "session" is like a temporary workspace for talking to the DB:
# you add/query objects, then `commit()` to save them permanently.
#
# autocommit=False -> we must explicitly call commit(), which
#                      prevents accidental half-finished writes.
# autoflush=False  -> SQLAlchemy won't auto-sync every tiny change,
#                      giving us more predictable control.
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# ------------------------------------------------------------
# The Base class
# ------------------------------------------------------------
# Every table model (User, Report, ChatMessage, etc. in models.py)
# will inherit from this Base. SQLAlchemy uses it to keep track
# of all table definitions so it can create them in the database.
Base = declarative_base()


def get_db():
    """
    This is a FastAPI "dependency". Any route that needs database
    access will declare a parameter like:

        def some_route(db: Session = Depends(get_db)):

    FastAPI will call this function, get a session, hand it to the
    route, and — critically — the `finally` block guarantees the
    session is closed afterward, EVEN IF the route raises an error.
    This prevents leaking open database connections over time.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """
    Creates all tables in the database based on the models defined
    in models.py, but ONLY if they don't already exist. Safe to call
    every time the app starts.

    This must be called AFTER models.py has been imported at least
    once (so Base knows what tables to create) - app.py handles this
    ordering for us.
    """
    logger.info("Initializing database and creating tables if needed...")
    Base.metadata.create_all(bind=engine)
    logger.info("Database ready at %s", settings.DATABASE_URL)
