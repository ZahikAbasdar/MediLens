"""
routes.py
=========
Every HTTP endpoint in MediLens lives here, grouped into FastAPI
"routers" by feature area. app.py imports these routers and mounts
them onto the main app.

WHY ONE FILE FOR ALL ROUTES (not one file per feature):
With ~25 endpoints total, splitting into 8+ separate route files
would violate the project's "minimum meaningful files" rule for
marginal benefit. Instead, this file is organized into clearly
commented sections, one per feature area, which gives the same
readability without the file-count overhead.

Router prefixes:
  /auth      - signup, login, profile
  /reports   - upload, list, get, delete
  /chat      - chatbot messaging + history
  /dashboard - health score, trends, alerts
  /explain   - term/organ/medicine explanations
  /export    - PDF/CSV/JSON export
"""

import logging
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from sqlalchemy.orm import Session

import ml_engine
import rag_engine
import visual_engine
from ai_engine import generate_ai_response, AIEngineError
from auth import (
    authenticate_user, create_access_token, get_current_user, hash_password,
    generate_reset_code, verify_reset_code, clear_reset_code,
)
from chatbot import handle_chat_message, get_chat_history
from config import settings
from database import get_db
from knowledge_base import search_dictionary
from models import User, Report, LabValue, HealthSnapshot
from ocr_engine import extract_text_from_file
from parser import parse_report
from prompts import (
    build_lab_value_explanation_prompt, build_term_explanation_prompt,
    build_medicine_explanation_prompt, build_summary_prompt,
)
from schemas import (
    UserCreate, UserOut, UserUpdate, Token, ForgotPasswordRequest, ResetPasswordRequest,
    ReportOut, ReportSummary, ChatRequest, ChatResponse, ChatMessageOut,
    DashboardOut, HealthSnapshotOut, TrendAnalysisOut, TrendPoint,
    ExplainTermRequest, OrganExplanationOut,
)
from utils import save_upload_file, validate_file_extension

logger = logging.getLogger(__name__)

# ------------------------------------------------------------
# Routers
# ------------------------------------------------------------
auth_router = APIRouter(prefix="/auth", tags=["Authentication"])
reports_router = APIRouter(prefix="/reports", tags=["Reports"])
chat_router = APIRouter(prefix="/chat", tags=["Chatbot"])
dashboard_router = APIRouter(prefix="/dashboard", tags=["Dashboard"])
explain_router = APIRouter(prefix="/explain", tags=["Explainer"])
export_router = APIRouter(prefix="/export", tags=["Export"])


# ============================================================
# AUTH ROUTES
# ============================================================
@auth_router.post("/signup", response_model=UserOut, status_code=status.HTTP_201_CREATED)
def signup(payload: UserCreate, db: Session = Depends(get_db)):
    """Registers a new user. Rejects duplicate emails."""
    existing = db.query(User).filter(User.email == payload.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="An account with this email already exists")

    user = User(
        email=payload.email,
        hashed_password=hash_password(payload.password),
        full_name=payload.full_name,
        age=payload.age,
        gender=payload.gender,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    logger.info("New user registered: %s", user.email)
    return user


@auth_router.post("/login", response_model=Token)
def login(email: str, password: str, db: Session = Depends(get_db)):
    """
    Verifies credentials and returns a JWT access token.
    NOTE: Uses simple query params (not OAuth2PasswordRequestForm) to
    keep the Streamlit frontend integration simple - it still works
    perfectly with the Bearer token flow afterward.
    """
    user = authenticate_user(db, email, password)
    if not user:
        raise HTTPException(status_code=401, detail="Incorrect email or password")

    access_token = create_access_token(data={"sub": user.email})
    return {"access_token": access_token, "token_type": "bearer"}


@auth_router.post("/logout")
def logout():
    """
    JWTs are stateless, so 'logout' is enforced client-side (the
    client simply discards the token). We still expose this endpoint
    for a consistent API surface and so the frontend has a clear
    action to call.
    """
    return {"message": "Logged out successfully. Please discard your access token."}


@auth_router.post("/forgot-password")
def forgot_password(payload: ForgotPasswordRequest, db: Session = Depends(get_db)):
    """
    Generates a password reset code. See auth.py's docstring for why
    this returns the code directly in DEBUG mode instead of emailing
    it (no SMTP server required for this portfolio project).
    """
    user = db.query(User).filter(User.email == payload.email).first()
    if not user:
        # Deliberately vague message - never reveal whether an email is registered
        return {"message": "If this email is registered, a reset code has been generated."}

    code = generate_reset_code(payload.email)
    response = {"message": "If this email is registered, a reset code has been generated."}
    if settings.DEBUG:
        response["debug_reset_code"] = code  # only exposed in DEBUG/dev mode
    return response


@auth_router.post("/reset-password")
def reset_password(payload: ResetPasswordRequest, db: Session = Depends(get_db)):
    """Verifies the reset code and updates the user's password."""
    if not verify_reset_code(payload.email, payload.reset_code):
        raise HTTPException(status_code=400, detail="Invalid or expired reset code")

    user = db.query(User).filter(User.email == payload.email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    user.hashed_password = hash_password(payload.new_password)
    db.commit()
    clear_reset_code(payload.email)
    return {"message": "Password reset successfully"}


@auth_router.get("/profile", response_model=UserOut)
def get_profile(current_user: User = Depends(get_current_user)):
    return current_user


@auth_router.put("/profile", response_model=UserOut)
def update_profile(
    payload: UserUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(current_user, field, value)
    db.commit()
    db.refresh(current_user)
    return current_user


# ============================================================
# REPORT UPLOAD & MANAGEMENT ROUTES
# ============================================================
@reports_router.post("/upload", response_model=ReportOut, status_code=status.HTTP_201_CREATED)
def upload_report(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    The full upload pipeline, in order:
      1. Validate + save the file to disk (utils.py)
      2. Extract text via OCR (ocr_engine.py)
      3. Parse structured fields + lab values (parser.py)
      4. Persist Report + LabValue rows (models.py)
      5. Add the report's text to the user's RAG index (rag_engine.py)
      6. Compute and store a HealthSnapshot (ml_engine.py)
    """
    file_ext = validate_file_extension(file.filename)
    saved_path = save_upload_file(file, current_user.id)

    try:
        raw_text, confidence, _tables = extract_text_from_file(saved_path, file_ext)
    except Exception as exc:
        logger.error("OCR failed for %s: %s", saved_path, exc)
        raise HTTPException(status_code=422, detail=f"Could not extract text from file: {exc}")

    parsed = parse_report(raw_text)

    report = Report(
        user_id=current_user.id,
        original_filename=file.filename,
        stored_file_path=str(saved_path),
        file_type=file_ext,
        raw_text=raw_text,
        ocr_confidence=confidence,
        report_type=parsed.report_type,
        patient_name=parsed.patient_name,
        patient_age=parsed.patient_age,
        patient_gender=parsed.patient_gender,
        hospital_name=parsed.hospital_name,
        doctor_name=parsed.doctor_name,
        laboratory_name=parsed.laboratory_name,
        report_date=parsed.report_date,
    )
    db.add(report)
    db.commit()
    db.refresh(report)

    for lv in parsed.lab_values:
        db.add(LabValue(
            report_id=report.id,
            test_name=lv.test_name,
            value=lv.value,
            numeric_value=lv.numeric_value,
            unit=lv.unit,
            reference_range=lv.reference_range,
            is_abnormal=lv.is_abnormal,
            is_critical=lv.is_critical,
            related_organ=lv.related_organ,
        ))
    db.commit()
    db.refresh(report)

    # Add to RAG index (best-effort - a failure here shouldn't fail the whole upload)
    try:
        if raw_text.strip():
            rag_engine.build_or_update_index(current_user.id, report.id, raw_text)
    except Exception as exc:
        logger.warning("Could not add report %d to RAG index: %s", report.id, exc)

    # Compute and store health snapshot for this report
    score, risk_level = ml_engine.calculate_health_score(report.lab_values)
    db.add(HealthSnapshot(
        user_id=current_user.id,
        report_id=report.id,
        health_score=score,
        risk_level=risk_level,
        abnormal_count=sum(1 for lv in report.lab_values if lv.is_abnormal),
        critical_count=sum(1 for lv in report.lab_values if lv.is_critical),
    ))
    db.commit()

    logger.info("Report %d uploaded and processed for user %d", report.id, current_user.id)
    return report


@reports_router.get("", response_model=List[ReportSummary])
def list_reports(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return (
        db.query(Report)
        .filter(Report.user_id == current_user.id)
        .order_by(Report.uploaded_at.desc())
        .all()
    )


@reports_router.get("/{report_id}", response_model=ReportOut)
def get_report(report_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    report = _get_owned_report(db, report_id, current_user.id)
    return report


@reports_router.delete("/{report_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_report(report_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    report = _get_owned_report(db, report_id, current_user.id)
    db.delete(report)
    db.commit()
    return None


def _get_owned_report(db: Session, report_id: int, user_id: int) -> Report:
    """Shared helper: fetch a report and verify it belongs to the requesting user."""
    report = db.query(Report).filter(Report.id == report_id).first()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    if report.user_id != user_id:
        raise HTTPException(status_code=403, detail="You do not have access to this report")
    return report


# ============================================================
# CHATBOT ROUTES
# ============================================================
@chat_router.post("", response_model=ChatResponse)
def chat(payload: ChatRequest, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if payload.report_id is not None:
        _get_owned_report(db, payload.report_id, current_user.id)  # ownership check
    return handle_chat_message(db, current_user.id, payload.message, payload.report_id)


@chat_router.get("/history", response_model=List[ChatMessageOut])
def chat_history(
    report_id: Optional[int] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return get_chat_history(db, current_user.id, report_id)


# ============================================================
# DASHBOARD ROUTES
# ============================================================
@dashboard_router.get("", response_model=DashboardOut)
def get_dashboard(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """
    Assembles Feature #10 - the Health Dashboard: latest score/risk,
    report counts, abnormal/critical totals, recent reports, and
    score history (for the trend chart).
    """
    reports = db.query(Report).filter(Report.user_id == current_user.id).all()
    snapshots = (
        db.query(HealthSnapshot)
        .filter(HealthSnapshot.user_id == current_user.id)
        .order_by(HealthSnapshot.created_at.asc())
        .all()
    )

    total_abnormal = sum(s.abnormal_count for s in snapshots)
    total_critical = sum(s.critical_count for s in snapshots)
    latest = snapshots[-1] if snapshots else None

    recent_reports = (
        db.query(Report)
        .filter(Report.user_id == current_user.id)
        .order_by(Report.uploaded_at.desc())
        .limit(5)
        .all()
    )

    return DashboardOut(
        latest_health_score=latest.health_score if latest else None,
        latest_risk_level=latest.risk_level if latest else None,
        total_reports=len(reports),
        total_abnormal_values=total_abnormal,
        total_critical_values=total_critical,
        recent_reports=recent_reports,
        score_history=snapshots,
    )


@dashboard_router.get("/trend/{test_name}", response_model=TrendAnalysisOut)
def get_trend(test_name: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """
    Feature #11 - Trend Analysis. Pulls every historical LabValue
    matching `test_name` across ALL of this user's reports, ordered
    by upload date, and computes the direction using ml_engine.
    """
    rows = (
        db.query(LabValue, Report.uploaded_at)
        .join(Report, LabValue.report_id == Report.id)
        .filter(Report.user_id == current_user.id)
        .filter(LabValue.test_name.ilike(f"%{test_name}%"))
        .order_by(Report.uploaded_at.asc())
        .all()
    )

    if not rows:
        raise HTTPException(status_code=404, detail=f"No historical data found for '{test_name}'")

    points = [
        TrendPoint(date=uploaded_at, value=lv.numeric_value, is_abnormal=lv.is_abnormal)
        for lv, uploaded_at in rows
    ]
    direction = ml_engine.analyze_trend(points)

    first_lab_value = rows[0][0]
    return TrendAnalysisOut(
        test_name=first_lab_value.test_name,
        unit=first_lab_value.unit,
        reference_range=first_lab_value.reference_range,
        points=points,
        direction=direction,
    )


@dashboard_router.get("/alerts")
def get_smart_alerts(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """
    Feature #15 - Smart Alerts. Returns every critical lab value
    across the user's reports, most recent first, so the frontend
    can prominently surface anything requiring urgent attention.
    """
    rows = (
        db.query(LabValue, Report.uploaded_at, Report.original_filename)
        .join(Report, LabValue.report_id == Report.id)
        .filter(Report.user_id == current_user.id)
        .filter(LabValue.is_critical == True)  # noqa: E712 (SQLAlchemy requires == not `is`)
        .order_by(Report.uploaded_at.desc())
        .all()
    )
    return [
        {
            "test_name": lv.test_name,
            "value": lv.value,
            "unit": lv.unit,
            "reference_range": lv.reference_range,
            "report_filename": filename,
            "date": uploaded_at,
            "message": (
                f"{lv.test_name} was reported at a critical level ({lv.value} {lv.unit or ''}). "
                "Please seek prompt medical evaluation."
            ),
        }
        for lv, uploaded_at, filename in rows
    ]


# ============================================================
# EXPLAINER ROUTES (lab values, terms, organs, medicines)
# ============================================================
@explain_router.get("/lab-value/{lab_value_id}")
def explain_lab_value(
    lab_value_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Feature #5 - full AI explanation of one specific lab value."""
    lab_value = db.query(LabValue).filter(LabValue.id == lab_value_id).first()
    if not lab_value:
        raise HTTPException(status_code=404, detail="Lab value not found")
    _get_owned_report(db, lab_value.report_id, current_user.id)  # ownership check

    prompt = build_lab_value_explanation_prompt(
        test_name=lab_value.test_name,
        value=lab_value.value or "N/A",
        unit=lab_value.unit or "",
        reference_range=lab_value.reference_range or "Not specified on report",
        is_abnormal=lab_value.is_abnormal,
        is_critical=lab_value.is_critical,
    )
    try:
        explanation = generate_ai_response(prompt)
    except AIEngineError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    organ = visual_engine.get_organ_for_test(lab_value.test_name, lab_value.related_organ)
    visual = visual_engine.build_organ_explanation(organ)

    return {"explanation": explanation, "visual": visual}


@explain_router.post("/term", response_model=None)
def explain_term(payload: ExplainTermRequest):
    """
    Feature #16/#20 - Medical Dictionary lookup. Checks the static
    knowledge_base.py first (instant, free); falls back to the LLM
    for terms not in our curated dictionary.
    """
    cached = search_dictionary(payload.term)
    if cached:
        return {"source": "dictionary", **cached}

    try:
        explanation = generate_ai_response(build_term_explanation_prompt(payload.term))
    except AIEngineError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return {"source": "ai", "term": payload.term, "explanation": explanation}


@explain_router.get("/organ/{organ_name}", response_model=OrganExplanationOut)
def explain_organ(organ_name: str):
    """Feature #22 - Visual Medical Explainer for a specific organ."""
    return visual_engine.build_organ_explanation(organ_name)


@explain_router.get("/organs")
def list_organs():
    return {"organs": visual_engine.list_available_organs()}


@explain_router.get("/medicine/{medicine_name}")
def explain_medicine(medicine_name: str):
    """Feature #9 - Medicine Reader term lookup."""
    cached = search_dictionary(medicine_name)
    if cached and cached.get("category") == "Medicine":
        return {"source": "dictionary", **cached}

    try:
        explanation = generate_ai_response(build_medicine_explanation_prompt(medicine_name))
    except AIEngineError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return {"source": "ai", "medicine": medicine_name, "explanation": explanation}


# ============================================================
# EXPORT ROUTES
# ============================================================
@export_router.get("/report/{report_id}/json")
def export_report_json(report_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Feature #21 - JSON export of a single report + its lab values."""
    report = _get_owned_report(db, report_id, current_user.id)
    return {
        "report": ReportOut.model_validate(report).model_dump(mode="json"),
        "exported_at": datetime.utcnow().isoformat(),
        "disclaimer": settings.MEDICAL_DISCLAIMER,
    }


@export_router.get("/report/{report_id}/csv")
def export_report_csv(report_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Feature #21 - CSV export of a report's lab values, as raw CSV text."""
    import csv
    import io

    report = _get_owned_report(db, report_id, current_user.id)

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["Test Name", "Value", "Unit", "Reference Range", "Abnormal", "Critical"])
    for lv in report.lab_values:
        writer.writerow([lv.test_name, lv.value, lv.unit, lv.reference_range, lv.is_abnormal, lv.is_critical])

    from fastapi.responses import StreamingResponse
    buffer.seek(0)
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=report_{report_id}.csv"},
    )


@export_router.get("/summary/{report_id}")
def export_summary(
    report_id: int,
    summary_type: str = "simple",
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Feature #12 - AI Summary (simple / detailed / doctor_visit)."""
    report = _get_owned_report(db, report_id, current_user.id)
    if not report.raw_text:
        raise HTTPException(status_code=400, detail="This report has no extracted text to summarize")

    try:
        summary = generate_ai_response(build_summary_prompt(report.raw_text, summary_type))
    except AIEngineError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    return {"report_id": report_id, "summary_type": summary_type, "summary": summary}
