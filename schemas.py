"""
schemas.py
==========
Pydantic "schemas" define the SHAPE of data going IN and OUT of our
API. They are different from models.py (which defines database
tables). A schema validates what a client is allowed to send us,
and controls exactly what we send back (so we never accidentally
leak a password hash, for example).

Naming convention used throughout this file:
  - <Thing>Create  -> what the client sends to create something
  - <Thing>Out     -> what we send back to the client
  - <Thing>Update  -> what the client sends to update something
"""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, ConfigDict, EmailStr, Field


# ------------------------------------------------------------
# Auth / User schemas
# ------------------------------------------------------------
class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6, description="Minimum 6 characters")
    full_name: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserOut(BaseModel):
    id: int
    email: EmailStr
    full_name: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    preferred_language: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class UserUpdate(BaseModel):
    full_name: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    preferred_language: Optional[str] = None


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class ResetPasswordRequest(BaseModel):
    email: EmailStr
    reset_code: str
    new_password: str = Field(min_length=6)


# ------------------------------------------------------------
# Lab value schemas
# ------------------------------------------------------------
class LabValueOut(BaseModel):
    id: int
    test_name: str
    value: Optional[str]
    numeric_value: Optional[float]
    unit: Optional[str]
    reference_range: Optional[str]
    is_abnormal: bool
    is_critical: bool
    related_organ: Optional[str]

    model_config = ConfigDict(from_attributes=True)


# ------------------------------------------------------------
# Report schemas
# ------------------------------------------------------------
class ReportOut(BaseModel):
    id: int
    original_filename: str
    file_type: str
    report_type: Optional[str]
    patient_name: Optional[str]
    patient_age: Optional[str]
    patient_gender: Optional[str]
    hospital_name: Optional[str]
    doctor_name: Optional[str]
    laboratory_name: Optional[str]
    report_date: Optional[str]
    ocr_confidence: Optional[float]
    uploaded_at: datetime
    lab_values: List[LabValueOut] = []

    model_config = ConfigDict(from_attributes=True)


class ReportSummary(BaseModel):
    """Lightweight version used in lists, avoids sending full lab_values every time."""
    id: int
    original_filename: str
    report_type: Optional[str]
    uploaded_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ------------------------------------------------------------
# Chat schemas
# ------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str
    report_id: Optional[int] = None  # if set, chat is scoped to one report


class ChatMessageOut(BaseModel):
    id: int
    role: str
    content: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ChatResponse(BaseModel):
    reply: str
    sources_used: List[str] = []  # which report chunks were used (RAG transparency)
    disclaimer: str


# ------------------------------------------------------------
# Dashboard / ML schemas
# ------------------------------------------------------------
class HealthSnapshotOut(BaseModel):
    id: int
    health_score: float
    risk_level: str
    abnormal_count: int
    critical_count: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class DashboardOut(BaseModel):
    latest_health_score: Optional[float]
    latest_risk_level: Optional[str]
    total_reports: int
    total_abnormal_values: int
    total_critical_values: int
    recent_reports: List[ReportSummary]
    score_history: List[HealthSnapshotOut]


class TrendPoint(BaseModel):
    date: datetime
    value: Optional[float]
    is_abnormal: bool


class TrendAnalysisOut(BaseModel):
    test_name: str
    unit: Optional[str]
    reference_range: Optional[str]
    points: List[TrendPoint]
    direction: str  # "Improving", "Worsening", "Stable", "Insufficient data"


# ------------------------------------------------------------
# Explainer / visual schemas
# ------------------------------------------------------------
class ExplainTermRequest(BaseModel):
    term: str


class OrganExplanationOut(BaseModel):
    organ: str
    svg: str
    function: str
    why_it_matters: str
    lifestyle_tips: List[str]
