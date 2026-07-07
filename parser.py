"""
parser.py
=========
Takes the raw text (from ocr_engine.py) and turns it into STRUCTURED
data: patient info, hospital/doctor, and a list of individual lab
test results (name, value, unit, reference range, abnormal flags).

WHY REGEX AND NOT AN LLM FOR THIS STEP:
It's tempting to just ask an LLM "extract the structured fields from
this text". But that is slower, costs API calls on every upload, and
-- critically -- is non-deterministic, which is dangerous for medical
NUMBERS. A regex-based parser is fast, free, runs offline, and always
gives the exact same answer for the same input. We reserve the LLM
(ai_engine.py) for the part it's actually good at: EXPLAINING what
the numbers mean in plain English, not extracting the numbers
themselves.

This parser is intentionally built to be "good enough, always safe"
rather than "perfect for every possible report layout" - any field
it can't confidently find is simply left as None instead of guessing.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

from utils import safe_float

logger = logging.getLogger(__name__)


# ------------------------------------------------------------
# Data structures returned by this module
# ------------------------------------------------------------
@dataclass
class ParsedLabValue:
    test_name: str
    value: Optional[str]
    numeric_value: Optional[float]
    unit: Optional[str]
    reference_range: Optional[str]
    is_abnormal: bool = False
    is_critical: bool = False
    related_organ: Optional[str] = None


@dataclass
class ParsedReport:
    report_type: Optional[str] = None
    patient_name: Optional[str] = None
    patient_age: Optional[str] = None
    patient_gender: Optional[str] = None
    hospital_name: Optional[str] = None
    doctor_name: Optional[str] = None
    laboratory_name: Optional[str] = None
    report_date: Optional[str] = None
    lab_values: List[ParsedLabValue] = field(default_factory=list)


# ------------------------------------------------------------
# Reference data: known tests, their normal ranges, and which
# organ/body-system each one relates to (used by visual_engine.py)
# ------------------------------------------------------------
# NOTE: These reference ranges are widely-cited general adult ranges
# for educational purposes ONLY. Real reference ranges vary by lab,
# method, age, and sex - the report's OWN printed reference range
# (if found) always takes priority over this table.
KNOWN_TESTS: dict[str, dict] = {
    "hemoglobin":     {"unit": "g/dL",   "low": 13.0, "high": 17.0, "organ": "blood"},
    "hba1c":          {"unit": "%",      "low": 4.0,  "high": 5.6,  "organ": "pancreas"},
    "glucose":        {"unit": "mg/dL",  "low": 70,   "high": 100,  "organ": "pancreas"},
    "creatinine":     {"unit": "mg/dL",  "low": 0.6,  "high": 1.3,  "organ": "kidney"},
    "urea":           {"unit": "mg/dL",  "low": 7,    "high": 20,   "organ": "kidney"},
    "sgpt":           {"unit": "U/L",    "low": 7,    "high": 56,   "organ": "liver"},
    "alt":            {"unit": "U/L",    "low": 7,    "high": 56,   "organ": "liver"},
    "sgot":           {"unit": "U/L",    "low": 8,    "high": 48,   "organ": "liver"},
    "ast":            {"unit": "U/L",    "low": 8,    "high": 48,   "organ": "liver"},
    "cholesterol":    {"unit": "mg/dL",  "low": 0,    "high": 200,  "organ": "heart"},
    "ldl":            {"unit": "mg/dL",  "low": 0,    "high": 100,  "organ": "heart"},
    "hdl":            {"unit": "mg/dL",  "low": 40,   "high": 60,   "organ": "heart"},
    "triglycerides":  {"unit": "mg/dL",  "low": 0,    "high": 150,  "organ": "heart"},
    "tsh":            {"unit": "uIU/mL", "low": 0.4,  "high": 4.0,  "organ": "thyroid"},
    "wbc":            {"unit": "cells/uL", "low": 4000, "high": 11000, "organ": "blood"},
    "platelet":       {"unit": "cells/uL", "low": 150000, "high": 450000, "organ": "blood"},
    "vitamin d":      {"unit": "ng/mL",  "low": 30,   "high": 100,  "organ": "bones"},
    "vitamin b12":    {"unit": "pg/mL",  "low": 200,  "high": 900,  "organ": "blood"},
}

# Maps a test name substring to the "critical" threshold - values
# beyond these should trigger a strong smart-alert (Feature #15).
CRITICAL_THRESHOLDS: dict[str, dict] = {
    "creatinine":  {"critical_high": 3.0},
    "glucose":     {"critical_high": 400, "critical_low": 54},
    "hemoglobin":  {"critical_low": 7.0},
    "potassium":   {"critical_high": 6.5, "critical_low": 2.5},
    "sodium":      {"critical_high": 160, "critical_low": 120},
    "platelet":    {"critical_low": 50000},
}


# ------------------------------------------------------------
# Metadata extraction (patient info, hospital, doctor, date)
# ------------------------------------------------------------
_METADATA_PATTERNS = {
    "patient_name": r"(?:patient\s*name|name\s*of\s*patient|name)\s*[:\-]\s*([A-Za-z .]{2,40})",
    "patient_age": r"age\s*[:\-]?\s*(\d{1,3})\s*(?:yrs?|years?)?",
    "patient_gender": r"(?:sex|gender)\s*[:\-]\s*(male|female|m|f|other)",
    "hospital_name": r"(?:hospital|clinic|medical\s*center)\s*[:\-]\s*([A-Za-z0-9 .,&\-]{3,60})",
    "doctor_name": r"(?:dr\.?|doctor|referred\s*by|physician)\s*[:\-]?\s*([A-Za-z .]{3,40})",
    "laboratory_name": r"(?:laboratory|lab)\s*[:\-]\s*([A-Za-z0-9 .,&\-]{3,60})",
    "report_date": r"(?:date|collected\s*on|report\s*date)\s*[:\-]\s*(\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4})",
}


def _extract_metadata(text: str) -> dict:
    """Runs each metadata regex against the text and returns whatever it finds."""
    lowered = text.lower()
    results: dict[str, Optional[str]] = {}

    for field_name, pattern in _METADATA_PATTERNS.items():
        match = re.search(pattern, lowered, re.IGNORECASE)
        if match:
            value = match.group(1).strip().title() if field_name != "report_date" else match.group(1).strip()
            # Guard against the regex accidentally swallowing the next
            # label (e.g. "Name: John Age: 45" -> don't let name = "John Age")
            value = re.split(r"\s{2,}", value)[0].strip(" .,-")
            if value:
                results[field_name] = value
    return results


def _guess_report_type(text: str) -> Optional[str]:
    """
    Simple keyword-based classifier for what KIND of report this is.
    (ml_engine.py later builds a proper ML classifier on top of this
    as a fallback/confirmation - see Module 5.)
    """
    lowered = text.lower()
    type_keywords = {
        "Blood Test / CBC": ["hemoglobin", "wbc", "rbc", "platelet", "complete blood count"],
        "Diabetes Panel": ["hba1c", "glucose", "blood sugar"],
        "Kidney Function Test": ["creatinine", "urea", "egfr"],
        "Liver Function Test": ["sgpt", "sgot", "bilirubin", "alt", "ast"],
        "Lipid Profile": ["cholesterol", "triglycerides", "ldl", "hdl"],
        "Thyroid Panel": ["tsh", "t3", "t4"],
        "MRI Report": ["mri", "magnetic resonance"],
        "ECG Report": ["ecg", "electrocardiogram"],
        "X-Ray Report": ["x-ray", "radiograph"],
        "Prescription": ["rx", "tablet", "capsule", "dosage", "prescribed"],
    }
    best_match, best_score = None, 0
    for report_type, keywords in type_keywords.items():
        score = sum(1 for kw in keywords if kw in lowered)
        if score > best_score:
            best_match, best_score = report_type, score
    return best_match


# ------------------------------------------------------------
# Lab value extraction
# ------------------------------------------------------------
# Matches lines like:
#   "Hemoglobin      13.5   g/dL      13.0 - 17.0"
#   "Creatinine: 1.8 mg/dL (Normal: 0.6-1.3)"
#   "Cholesterol - 240 mg/dL   High"
_LAB_LINE_PATTERN = re.compile(
    r"^(?P<name>[A-Za-z][A-Za-z0-9()/ .\-]{2,40}?)\s*[:\-]?\s+"
    r"(?P<value>[<>]?\d+\.?\d*)\s*"
    r"(?P<unit>[A-Za-z%/µ]{1,15})?\s*"
    r"(?:\(?(?:ref(?:erence)?\.?|normal)?[:\- ]*\)?\s*)?"
    r"(?P<range>\d+\.?\d*\s*-\s*\d+\.?\d*)?",
    re.IGNORECASE,
)


def _match_known_test(test_name: str) -> Optional[dict]:
    """Finds a KNOWN_TESTS entry whose key is a substring of test_name (or vice versa)."""
    lowered = test_name.lower().strip()
    for known_name, info in KNOWN_TESTS.items():
        if known_name in lowered or lowered in known_name:
            return {"matched_name": known_name, **info}
    return None


def _check_abnormal(numeric_value: Optional[float], reference_range: Optional[str],
                     known_info: Optional[dict]) -> bool:
    """
    Decides if a value is abnormal. Priority:
    1. The report's OWN printed reference range (most authoritative -
       it reflects the actual lab's method and population).
    2. Our KNOWN_TESTS fallback table, only if no range was printed.
    """
    if numeric_value is None:
        return False

    if reference_range:
        numbers = re.findall(r"\d+\.?\d*", reference_range)
        if len(numbers) >= 2:
            low, high = float(numbers[0]), float(numbers[1])
            return numeric_value < low or numeric_value > high

    if known_info:
        return numeric_value < known_info["low"] or numeric_value > known_info["high"]

    return False


def _check_critical(test_name: str, numeric_value: Optional[float]) -> bool:
    """Checks a value against hardcoded critical/emergency thresholds."""
    if numeric_value is None:
        return False
    lowered = test_name.lower()
    for key, thresholds in CRITICAL_THRESHOLDS.items():
        if key in lowered:
            if "critical_high" in thresholds and numeric_value >= thresholds["critical_high"]:
                return True
            if "critical_low" in thresholds and numeric_value <= thresholds["critical_low"]:
                return True
    return False


# Lines that are clearly patient/report METADATA, not lab results.
# We exclude these before pattern matching so fields like
# "Age: 52 yrs" or "Date: 15/03/2026" never get mistaken for a
# test named "Age" with value 52 and unit "yrs".
_METADATA_LINE_PREFIXES = (
    "age", "date", "sex", "gender", "name", "patient", "patient name",
    "doctor", "dr.", "dr ", "referred by", "physician", "hospital",
    "clinic", "laboratory", "lab no", "lab id", "reg no", "registration",
    "sample", "specimen", "collected", "reported", "phone", "address",
    "page",
)


def _is_metadata_line(line: str) -> bool:
    lowered = line.strip().lower()
    return any(lowered.startswith(prefix) for prefix in _METADATA_LINE_PREFIXES)


def _extract_lab_values(text: str) -> List[ParsedLabValue]:
    """
    Scans the text line-by-line looking for lines that look like
    "TestName  Value  Unit  ReferenceRange". Lines that don't match
    the pattern (headers, disclaimers, addresses) are simply skipped.
    """
    lab_values: List[ParsedLabValue] = []

    for line in text.splitlines():
        line = line.strip()
        if len(line) < 4:
            continue

        if _is_metadata_line(line):
            continue

        match = _LAB_LINE_PATTERN.match(line)
        if not match:
            continue

        test_name = match.group("name").strip(" :-")
        raw_value = match.group("value")
        unit = match.group("unit")
        reference_range = match.group("range")

        # Skip obvious false positives: lines that are just a page
        # number, phone number, or something with no letters at all.
        if not re.search(r"[A-Za-z]{3,}", test_name):
            continue

        numeric_value = safe_float(raw_value)
        known_info = _match_known_test(test_name)

        if not unit and known_info:
            unit = known_info["unit"]
        if not reference_range and known_info:
            reference_range = f"{known_info['low']} - {known_info['high']}"

        is_abnormal = _check_abnormal(numeric_value, reference_range, known_info)
        is_critical = _check_critical(test_name, numeric_value)

        lab_values.append(ParsedLabValue(
            test_name=test_name.title(),
            value=raw_value,
            numeric_value=numeric_value,
            unit=unit,
            reference_range=reference_range,
            is_abnormal=is_abnormal,
            is_critical=is_critical,
            related_organ=known_info["organ"] if known_info else None,
        ))

    return lab_values


# ------------------------------------------------------------
# Public entry point
# ------------------------------------------------------------
def parse_report(raw_text: str) -> ParsedReport:
    """
    Main function called by routes.py right after OCR. Combines
    metadata extraction + report-type guessing + lab value extraction
    into one ParsedReport object.
    """
    if not raw_text or not raw_text.strip():
        logger.warning("parse_report called with empty text")
        return ParsedReport()

    metadata = _extract_metadata(raw_text)
    report_type = _guess_report_type(raw_text)
    lab_values = _extract_lab_values(raw_text)

    logger.info(
        "Parsed report: type=%s, %d lab values found, %d abnormal, %d critical",
        report_type, len(lab_values),
        sum(1 for v in lab_values if v.is_abnormal),
        sum(1 for v in lab_values if v.is_critical),
    )

    return ParsedReport(
        report_type=report_type,
        patient_name=metadata.get("patient_name"),
        patient_age=metadata.get("patient_age"),
        patient_gender=metadata.get("patient_gender"),
        hospital_name=metadata.get("hospital_name"),
        doctor_name=metadata.get("doctor_name"),
        laboratory_name=metadata.get("laboratory_name"),
        report_date=metadata.get("report_date"),
        lab_values=lab_values,
    )
