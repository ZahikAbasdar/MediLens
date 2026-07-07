"""
models.py
=========
This file defines every DATABASE TABLE in MediLens as a Python class.
This pattern is called an "ORM" (Object-Relational Mapper) - it lets
us work with database rows as if they were normal Python objects,
instead of writing raw SQL everywhere.

WHY THESE 5 TABLES (and not more):
We keep the schema minimal but complete:

1. User          -> who is using the app
2. Report        -> an uploaded medical report + its extracted metadata
3. LabValue      -> one row per test result inside a report
                     (e.g. "Hemoglobin: 10.2 g/dL") - this is what
                     powers trend analysis, alerts, and explanations
4. ChatMessage    -> every message in every chatbot conversation
5. HealthSnapshot -> one row per health-score calculation, used to
                     draw the dashboard timeline and trend charts

Every table inherits from `Base` (imported from database.py), which
is how SQLAlchemy knows to create these as real tables.
"""

from datetime import datetime, timezone
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, Text, ForeignKey, DateTime
)
from sqlalchemy.orm import relationship

from database import Base


def utc_now() -> datetime:
    """
    Small helper so every timestamp in the app is stored in UTC,
    not the server's local time. This avoids subtle bugs when the
    app is deployed to a server in a different timezone than the
    developer's machine.
    """
    return datetime.now(timezone.utc)


class User(Base):
    """
    One row per registered user.

    Passwords are NEVER stored in plain text - only `hashed_password`
    (produced by auth.py using bcrypt) is saved. Even if the database
    were leaked, actual passwords could not be recovered from it.
    """
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String, nullable=True)
    age = Column(Integer, nullable=True)
    gender = Column(String, nullable=True)
    preferred_language = Column(String, default="English")
    created_at = Column(DateTime, default=utc_now)

    # --- Relationships ---
    # `relationship()` doesn't create a database column - it lets
    # Python code do `user.reports` to get all reports belonging to
    # this user, without writing a manual JOIN query.
    # `cascade="all, delete-orphan"` means: if a User is deleted,
    # automatically delete all their reports/chats/snapshots too,
    # so we never end up with orphaned data nobody owns.
    reports = relationship("Report", back_populates="owner", cascade="all, delete-orphan")
    chat_messages = relationship("ChatMessage", back_populates="owner", cascade="all, delete-orphan")
    health_snapshots = relationship("HealthSnapshot", back_populates="owner", cascade="all, delete-orphan")


class Report(Base):
    """
    One row per uploaded medical report (a PDF, image, or document).

    This stores BOTH:
    - Raw data (file path, OCR text) -> ground truth from the document
    - Extracted metadata (patient name, doctor, hospital, date) -> parsed
      by parser.py from the OCR text

    Keeping raw_text here means we never need to re-run OCR once a
    report has been processed once - it also becomes the source
    document for RAG (Module 7).
    """
    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    # --- File info ---
    original_filename = Column(String, nullable=False)
    stored_file_path = Column(String, nullable=False)
    file_type = Column(String, nullable=False)  # pdf, png, jpg, docx, txt...

    # --- OCR output ---
    raw_text = Column(Text, nullable=True)
    ocr_confidence = Column(Float, nullable=True)  # 0.0 - 1.0

    # --- Extracted metadata (filled in by parser.py, may be partially empty) ---
    report_type = Column(String, nullable=True)     # e.g. "Blood Test", "MRI", "Prescription"
    patient_name = Column(String, nullable=True)
    patient_age = Column(String, nullable=True)
    patient_gender = Column(String, nullable=True)
    hospital_name = Column(String, nullable=True)
    doctor_name = Column(String, nullable=True)
    laboratory_name = Column(String, nullable=True)
    report_date = Column(String, nullable=True)  # kept as string: source dates are inconsistently formatted

    uploaded_at = Column(DateTime, default=utc_now)

    # --- Relationships ---
    owner = relationship("User", back_populates="reports")
    lab_values = relationship("LabValue", back_populates="report", cascade="all, delete-orphan")
    health_snapshots = relationship("HealthSnapshot", back_populates="report", cascade="all, delete-orphan")


class LabValue(Base):
    """
    One row per individual test result within a report.

    Example: a single blood test report might produce 15 LabValue
    rows: Hemoglobin, WBC Count, Platelet Count, etc.

    This granular structure is what makes trend analysis possible:
    we can later query "show me every Creatinine LabValue for this
    user, ordered by date" to build a trend chart.
    """
    __tablename__ = "lab_values"

    id = Column(Integer, primary_key=True, index=True)
    report_id = Column(Integer, ForeignKey("reports.id"), nullable=False)

    test_name = Column(String, nullable=False, index=True)  # e.g. "Hemoglobin"
    value = Column(String, nullable=True)                    # kept as string: some results are text (e.g. "Negative")
    numeric_value = Column(Float, nullable=True)             # parsed float, when possible - used for math/charts
    unit = Column(String, nullable=True)                     # e.g. "g/dL"
    reference_range = Column(String, nullable=True)          # e.g. "13.0 - 17.0"

    is_abnormal = Column(Boolean, default=False)
    is_critical = Column(Boolean, default=False)

    related_organ = Column(String, nullable=True)  # e.g. "kidney" - used by visual_engine.py (Module 9)

    report = relationship("Report", back_populates="lab_values")


class ChatMessage(Base):
    """
    One row per message in the AI chatbot conversation.

    Storing both the user's messages AND the assistant's replies in
    the SAME table (distinguished by `role`) makes it trivial to
    reconstruct a full conversation in order, just by querying
    "all messages for this user, sorted by created_at".
    """
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    report_id = Column(Integer, ForeignKey("reports.id"), nullable=True)  # optional: message may relate to a specific report

    role = Column(String, nullable=False)   # "user" or "assistant"
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=utc_now)

    owner = relationship("User", back_populates="chat_messages")


class HealthSnapshot(Base):
    """
    One row per health-score calculation (Module 5, ml_engine.py
    produces these). This is the backbone of the Health Dashboard's
    timeline and trend charts - instead of recalculating the health
    score from scratch every time the dashboard loads, we store each
    calculated snapshot with its date.
    """
    __tablename__ = "health_snapshots"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    report_id = Column(Integer, ForeignKey("reports.id"), nullable=True)

    health_score = Column(Float, nullable=False)   # 0-100
    risk_level = Column(String, nullable=False)    # "Low", "Moderate", "High"
    abnormal_count = Column(Integer, default=0)
    critical_count = Column(Integer, default=0)

    created_at = Column(DateTime, default=utc_now)

    owner = relationship("User", back_populates="health_snapshots")
    report = relationship("Report", back_populates="health_snapshots")
