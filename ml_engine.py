"""
ml_engine.py
============
This is the Machine Learning core of MediLens. It covers every ML
feature from the spec using a mix of:
  - Deterministic scoring rules (transparent, explainable, always
    give a defensible answer for a health-related number)
  - Actual scikit-learn models (IsolationForest for anomaly
    detection, a small classifier for report-type confirmation)

DESIGN DECISION - why not one giant black-box model for everything:
Health scoring needs to be EXPLAINABLE ("your score is 72 because 2
values are abnormal and 1 is critical" - not "the neural net said
so"). So the health score and risk level are computed with clear,
auditable rules built on the parsed lab values. We reserve trained
scikit-learn models (IsolationForest) specifically for anomaly
detection, where "does this pattern of numbers look unusual as a
whole" genuinely benefits from a statistical model rather than
simple range checks parser.py already does per-value.

Functions in this file:
  - calculate_health_score()   -> Feature #10 (Health Dashboard score)
  - detect_anomalies()         -> Feature #19 (Anomaly Detection, IsolationForest)
  - classify_report_type()     -> Feature #19 (Report Classification, ML-backed)
  - analyze_trend()            -> Feature #11 (Trend Analysis across reports)
  - ocr_confidence_bucket()    -> Feature #19 (OCR Confidence interpretation)
"""

import logging
from typing import List, Optional, Sequence

import numpy as np
from sklearn.ensemble import IsolationForest

from models import LabValue
from schemas import TrendPoint

logger = logging.getLogger(__name__)


# ------------------------------------------------------------
# Health Score & Risk Level
# ------------------------------------------------------------
def calculate_health_score(lab_values: Sequence[LabValue]) -> tuple[float, str]:
    """
    Produces a 0-100 "health score" and a risk label from a set of
    lab values belonging to ONE report.

    RULE (fully transparent, on purpose):
      - Start at 100.
      - Each abnormal value costs 8 points.
      - Each CRITICAL value costs an additional 15 points on top
        of the abnormal penalty (so a critical value costs 23 total).
      - Score is clamped to [0, 100].

    Risk level thresholds:
      - score >= 80              -> "Low"
      - 50 <= score < 80         -> "Moderate"
      - score < 50                -> "High"
      - ANY critical value present -> risk is forced to at least "High"
        regardless of the numeric score, because a single critical
        value (e.g. dangerously high creatinine) should never be
        diluted into "Moderate" just because other tests were normal.
    """
    if not lab_values:
        return 100.0, "Low"

    score = 100.0
    critical_count = 0

    for lv in lab_values:
        if lv.is_abnormal:
            score -= 8
        if lv.is_critical:
            score -= 15
            critical_count += 1

    score = max(0.0, min(100.0, score))

    if critical_count > 0:
        risk_level = "High"
    elif score >= 80:
        risk_level = "Low"
    elif score >= 50:
        risk_level = "Moderate"
    else:
        risk_level = "High"

    return round(score, 1), risk_level


# ------------------------------------------------------------
# Anomaly Detection (IsolationForest)
# ------------------------------------------------------------
def detect_anomalies(numeric_values: List[float]) -> List[bool]:
    """
    Given a list of numeric lab values (all from the SAME report,
    mixed different tests), uses IsolationForest to flag values that
    are statistically unusual RELATIVE TO EACH OTHER.

    This is deliberately a SECONDARY signal on top of parser.py's
    reference-range check, not a replacement for it: reference
    ranges tell you "is this number outside the medically normal
    range", while IsolationForest tells you "does this number look
    like a statistical outlier compared to the rest of this specific
    report" - useful for catching OCR errors (e.g. a decimal point
    misread turning 10.2 into 102).

    Returns a list of booleans, same length and order as the input,
    where True means "flagged as an outlier".

    NOTE: IsolationForest needs a reasonable number of samples to be
    meaningful. With fewer than 5 values we skip it entirely (not
    enough data for a statistical model to say anything reliable)
    and just return all-False.
    """
    if len(numeric_values) < 5:
        return [False] * len(numeric_values)

    X = np.array(numeric_values).reshape(-1, 1)

    # contamination=0.1 means "assume roughly 10% of values could be
    # outliers" - a conservative default so we don't over-flag.
    model = IsolationForest(contamination=0.1, random_state=42)
    predictions = model.fit_predict(X)  # -1 = outlier, 1 = normal

    return [pred == -1 for pred in predictions]


# ------------------------------------------------------------
# OCR Confidence interpretation
# ------------------------------------------------------------
def ocr_confidence_bucket(confidence: Optional[float]) -> str:
    """
    Translates a raw 0.0-1.0 OCR confidence float into a human label,
    used by the frontend to warn users when text extraction quality
    was poor (e.g. blurry photo of a report) and results might be
    less reliable.
    """
    if confidence is None:
        return "Unknown"
    if confidence >= 0.85:
        return "High confidence"
    if confidence >= 0.6:
        return "Moderate confidence - please verify critical values"
    return "Low confidence - image quality may have affected accuracy"


# ------------------------------------------------------------
# Trend Analysis
# ------------------------------------------------------------
def analyze_trend(points: List[TrendPoint]) -> str:
    """
    Given a time-ordered list of the SAME test from multiple reports,
    determines whether the trend is Improving, Worsening, or Stable.

    METHOD: simple linear regression slope (via numpy.polyfit) on
    (time_index, value) pairs. This is intentionally simple and
    explainable rather than a complex forecasting model - for a
    handful of data points (most patients have 2-6 historical
    reports), a straight-line trend is both accurate enough and
    easy to justify to a non-technical user.

    NOTE: "Improving" vs "Worsening" direction depends on whether
    higher-is-better or lower-is-better for the given test. Since we
    don't always know that generically, we report direction as
    "Increasing" / "Decreasing" / "Stable" and let the caller (which
    knows the test's reference range) label clinical meaning if needed.
    """
    valid_points = [p for p in points if p.value is not None]
    if len(valid_points) < 2:
        return "Insufficient data"

    x = np.arange(len(valid_points))
    y = np.array([p.value for p in valid_points])

    slope = float(np.polyfit(x, y, 1)[0])

    # Threshold as a percentage of the mean value, so it scales
    # correctly whether the test's numbers are in the 0-10 range or
    # the thousands (e.g. WBC counts).
    mean_value = float(np.mean(y)) or 1.0
    relative_slope = abs(slope) / mean_value

    if relative_slope < 0.02:
        return "Stable"
    return "Increasing" if slope > 0 else "Decreasing"


# ------------------------------------------------------------
# Report type classification (ML-backed confirmation)
# ------------------------------------------------------------
# parser.py already does keyword-based report-type guessing (fast,
# zero dependencies). Here we add a tiny, genuinely-trained
# scikit-learn text classifier as an ML-backed CONFIRMATION step,
# fulfilling the "Report Classification" ML requirement with a real
# trained model rather than just re-using the keyword heuristic.
_classifier = None
_vectorizer = None

_TRAINING_DATA = [
    ("hemoglobin wbc rbc platelet count complete blood count", "Blood Test / CBC"),
    ("hba1c glucose blood sugar fasting random diabetes", "Diabetes Panel"),
    ("creatinine urea egfr kidney function bun", "Kidney Function Test"),
    ("sgpt sgot alt ast bilirubin liver function", "Liver Function Test"),
    ("cholesterol triglycerides ldl hdl lipid profile", "Lipid Profile"),
    ("tsh t3 t4 thyroid stimulating hormone", "Thyroid Panel"),
    ("mri magnetic resonance imaging brain scan contrast", "MRI Report"),
    ("ecg electrocardiogram heart rhythm rate pr interval", "ECG Report"),
    ("x-ray radiograph chest bone fracture", "X-Ray Report"),
    ("tablet capsule mg dosage prescribed twice daily rx", "Prescription"),
]


def _get_classifier():
    """
    Lazily trains a tiny TF-IDF + Naive Bayes classifier on a small
    built-in training set the first time it's needed. This is
    intentionally small and fast (<1 second to train) - the goal is
    a genuine trained ML model backing report classification, not a
    state-of-the-art model requiring gigabytes of training data.
    """
    global _classifier, _vectorizer
    if _classifier is None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.naive_bayes import MultinomialNB

        texts, labels = zip(*_TRAINING_DATA)
        _vectorizer = TfidfVectorizer()
        X = _vectorizer.fit_transform(texts)
        _classifier = MultinomialNB()
        _classifier.fit(X, labels)
        logger.info("Trained lightweight report-type classifier on %d examples", len(texts))
    return _classifier, _vectorizer


def classify_report_type(text: str) -> tuple[str, float]:
    """
    Returns (predicted_label, confidence) using the trained classifier.
    Used as a confirmation/fallback alongside parser.py's keyword guess.
    """
    classifier, vectorizer = _get_classifier()
    X = vectorizer.transform([text.lower()])
    probabilities = classifier.predict_proba(X)[0]
    best_index = int(np.argmax(probabilities))
    return classifier.classes_[best_index], round(float(probabilities[best_index]), 3)
