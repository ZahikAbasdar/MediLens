"""
tests/test_basic.py
====================
Automated tests for MediLens's core, network-independent logic:
auth, parsing, ML scoring, RAG chunking, and the visual explainer.

WHY THESE MODULES AND NOT THE LLM CALLS:
Tests should be fast, free, and runnable without API keys or
internet access (so they work in any CI pipeline). The functions
tested here are exactly the ones that DON'T require an external
service - they're also the functions where a silent bug would be
most dangerous (e.g. a parsing bug that mislabels a critical value
as normal). LLM-dependent code (ai_engine.py's actual API calls) is
integration-tested manually against a real provider instead, since
mocking an LLM's exact response teaches us little about correctness.

RUN WITH:
    pytest tests/ -v
(from the project root, so imports resolve correctly)
"""

import sys
from pathlib import Path

# Allow running `pytest tests/` from the project root without
# needing to install the project as a package first.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from auth import hash_password, verify_password, create_access_token, decode_access_token
from parser import parse_report
from ml_engine import calculate_health_score, analyze_trend, ocr_confidence_bucket
from rag_engine import chunk_text
from schemas import TrendPoint
from visual_engine import get_organ_for_test, build_organ_explanation
from utils import clean_ocr_text, safe_float
from knowledge_base import search_dictionary


# ------------------------------------------------------------
# auth.py
# ------------------------------------------------------------
class TestAuth:
    def test_password_hash_and_verify(self):
        hashed = hash_password("mySecurePass123")
        assert hashed != "mySecurePass123"  # never store plain text
        assert verify_password("mySecurePass123", hashed) is True
        assert verify_password("wrongPassword", hashed) is False

    def test_jwt_roundtrip(self):
        token = create_access_token({"sub": "user@example.com"})
        payload = decode_access_token(token)
        assert payload["sub"] == "user@example.com"

    def test_jwt_tampering_fails(self):
        from jose import JWTError
        token = create_access_token({"sub": "user@example.com"})
        tampered = token[:-4] + "abcd"  # corrupt the signature
        with pytest.raises(JWTError):
            decode_access_token(tampered)


# ------------------------------------------------------------
# parser.py
# ------------------------------------------------------------
class TestParser:
    SAMPLE_REPORT = """
    Patient Name: Test Patient
    Age: 45 yrs
    Sex: Female
    Date: 01/01/2026
    Dr. Test Doctor

    Hemoglobin 11.5 g/dL 13.0-17.0
    Creatinine 4.0 mg/dL 0.6-1.3
    Glucose 90 mg/dL 70-100
    """

    def test_extracts_patient_metadata(self):
        result = parse_report(self.SAMPLE_REPORT)
        assert result.patient_name == "Test Patient"
        assert result.patient_age == "45"
        assert result.patient_gender == "Female"

    def test_extracts_correct_number_of_lab_values(self):
        result = parse_report(self.SAMPLE_REPORT)
        assert len(result.lab_values) == 3

    def test_flags_abnormal_and_critical_correctly(self):
        result = parse_report(self.SAMPLE_REPORT)
        by_name = {lv.test_name: lv for lv in result.lab_values}
        assert by_name["Hemoglobin"].is_abnormal is True
        assert by_name["Creatinine"].is_critical is True  # 4.0 exceeds critical threshold of 3.0
        assert by_name["Glucose"].is_abnormal is False

    def test_metadata_lines_never_become_lab_values(self):
        """Regression test: 'Age: 45 yrs' must never be parsed as a lab value."""
        result = parse_report(self.SAMPLE_REPORT)
        test_names = [lv.test_name.lower() for lv in result.lab_values]
        assert "age" not in test_names
        assert "date" not in test_names

    def test_empty_text_does_not_crash(self):
        result = parse_report("")
        assert result.lab_values == []


# ------------------------------------------------------------
# ml_engine.py
# ------------------------------------------------------------
class FakeLabValue:
    def __init__(self, is_abnormal, is_critical):
        self.is_abnormal = is_abnormal
        self.is_critical = is_critical


class TestMLEngine:
    def test_all_normal_gives_perfect_score(self):
        score, risk = calculate_health_score([FakeLabValue(False, False) for _ in range(4)])
        assert score == 100.0
        assert risk == "Low"

    def test_critical_value_forces_high_risk(self):
        """
        Regression test: even if the numeric score alone would land in
        the 'Moderate' band, a single critical value must force 'High'
        risk - this is a deliberate safety rule, not just a scoring detail.
        """
        score, risk = calculate_health_score([FakeLabValue(True, True), FakeLabValue(False, False)])
        assert risk == "High"

    def test_score_never_goes_below_zero(self):
        score, _ = calculate_health_score([FakeLabValue(True, True) for _ in range(20)])
        assert score >= 0.0

    def test_trend_detects_increasing_direction(self):
        from datetime import datetime
        points = [TrendPoint(date=datetime.now(), value=v, is_abnormal=False) for v in [1.0, 1.5, 2.0, 2.5]]
        assert analyze_trend(points) == "Increasing"

    def test_trend_insufficient_data(self):
        from datetime import datetime
        assert analyze_trend([TrendPoint(date=datetime.now(), value=1.0, is_abnormal=False)]) == "Insufficient data"

    def test_ocr_confidence_bucket_labels(self):
        assert ocr_confidence_bucket(0.95) == "High confidence"
        assert ocr_confidence_bucket(None) == "Unknown"


# ------------------------------------------------------------
# rag_engine.py
# ------------------------------------------------------------
class TestRagEngine:
    def test_chunking_produces_overlapping_chunks(self):
        text = " ".join(f"word{i}" for i in range(100))
        chunks = chunk_text(text, chunk_size=100, overlap=20)
        assert len(chunks) > 1
        # Every chunk should be non-empty
        assert all(len(c) > 0 for c in chunks)

    def test_empty_text_produces_no_chunks(self):
        assert chunk_text("") == []


# ------------------------------------------------------------
# visual_engine.py
# ------------------------------------------------------------
class TestVisualEngine:
    def test_known_test_maps_to_correct_organ(self):
        assert get_organ_for_test("Creatinine") == "kidney"
        assert get_organ_for_test("SGPT") == "liver"
        assert get_organ_for_test("Cholesterol") == "heart"

    def test_organ_explanation_returns_valid_svg(self):
        result = build_organ_explanation("kidney")
        assert "<svg" in result["svg"]
        assert "</svg>" in result["svg"]
        assert result["function"]  # non-empty

    def test_unknown_organ_does_not_crash(self):
        result = build_organ_explanation("not_a_real_organ")
        assert result["svg"]  # still returns a valid (unhighlighted) SVG


# ------------------------------------------------------------
# utils.py
# ------------------------------------------------------------
class TestUtils:
    def test_safe_float_parses_numbers_with_units(self):
        assert safe_float("13.5 mg/dL") == 13.5
        assert safe_float("<0.5") == 0.5

    def test_safe_float_returns_none_for_non_numeric(self):
        assert safe_float("Negative") is None
        assert safe_float(None) is None

    def test_clean_ocr_text_collapses_whitespace(self):
        assert clean_ocr_text("Hello    World") == "Hello World"


# ------------------------------------------------------------
# knowledge_base.py
# ------------------------------------------------------------
class TestKnowledgeBase:
    def test_finds_known_lab_test(self):
        result = search_dictionary("creatinine")
        assert result is not None
        assert result["category"] == "Lab Test"

    def test_unknown_term_returns_none(self):
        assert search_dictionary("xyzabc123notreal") is None
