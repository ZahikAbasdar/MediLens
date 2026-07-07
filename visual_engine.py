"""
visual_engine.py
================
Implements Feature #22 - the AI Visual Medical Explainer.

DESIGN DECISION (documented up front, since it's a deliberate
deviation from the literal tech list of PyVista/VTK/Open3D):
Full interactive 3D anatomy rendering (PyVista/VTK/Open3D) requires
heavy native graphics dependencies (OpenGL, headless rendering
servers) that are fragile to install and deploy, and massively
overkill for the actual educational goal: "show roughly where this
organ is and highlight it." The project's own spec explicitly allows
this fallback: "If full 3D rendering is not practical, use high-
quality SVG or PNG anatomical illustrations with highlighted organs...
generated dynamically instead of storing hundreds of static images."

So: this module draws a simple, clean human body outline as SVG
(entirely in code, no external image files, no internet download),
and highlights the relevant organ region with a colored overlay and
pulse animation. This is:
  - Zero external dependencies beyond Python string templates
  - Instant to render (no model loading, no GPU)
  - Fully interactive-capable when embedded in HTML (CSS animation)
  - Easy for a beginner to read and modify (it's just SVG shapes)

Each organ is defined by a simple (cx, cy, rx, ry) ellipse position
on a shared body outline, plus its educational metadata. Adding a
new organ later is a 5-line addition to ORGAN_DATA - no new image
assets needed.
"""

import logging
from typing import List

logger = logging.getLogger(__name__)


# ------------------------------------------------------------
# Organ position + educational metadata
# ------------------------------------------------------------
# Coordinates are placed on a shared 300x600 body outline (see
# _BODY_OUTLINE_SVG below). (cx, cy) = center, (rx, ry) = radii.
ORGAN_DATA = {
    "brain": {
        "position": (150, 60, 35, 30),
        "color": "#7C3AED",
        "function": "Your brain controls thought, memory, movement, and every organ in your body through nerve signals.",
        "why_it_matters": "Brain-related tests (like MRI or CT scans) look for structural or functional changes that could affect these processes.",
        "tips": ["Get 7-9 hours of sleep", "Stay mentally active", "Manage stress", "Stay hydrated"],
    },
    "thyroid": {
        "position": (150, 115, 20, 12),
        "color": "#F59E0B",
        "function": "Your thyroid gland (in your neck) produces hormones that control your metabolism - how your body uses energy.",
        "why_it_matters": "TSH, T3, and T4 tests check whether your thyroid is producing the right amount of hormone.",
        "tips": ["Ensure adequate iodine intake", "Manage stress levels", "Get regular sleep", "Discuss family history with your doctor"],
    },
    "heart": {
        "position": (165, 190, 30, 28),
        "color": "#EF4444",
        "function": "Your heart pumps oxygen-rich blood throughout your entire body, supplying every organ and tissue.",
        "why_it_matters": "Cholesterol, triglycerides, and ECG tests assess your cardiovascular health and heart function.",
        "tips": ["Regular aerobic exercise", "Limit saturated fat and sodium", "Avoid smoking", "Manage blood pressure"],
    },
    "lungs": {
        "position": (150, 190, 55, 45),
        "color": "#3B82F6",
        "function": "Your lungs bring oxygen into your blood and remove carbon dioxide every time you breathe.",
        "why_it_matters": "Chest X-rays and breathing tests assess lung structure and function.",
        "tips": ["Avoid smoking and secondhand smoke", "Exercise regularly to improve lung capacity", "Practice deep breathing"],
    },
    "liver": {
        "position": (185, 250, 35, 25),
        "color": "#92400E",
        "function": "Your liver filters toxins from your blood, produces bile for digestion, and stores energy.",
        "why_it_matters": "SGPT/ALT and SGOT/AST tests measure liver enzymes - elevated levels can indicate liver stress or damage.",
        "tips": ["Limit alcohol intake", "Maintain a healthy weight", "Avoid unnecessary medications", "Eat a balanced, fiber-rich diet"],
    },
    "stomach": {
        "position": (130, 260, 25, 30),
        "color": "#10B981",
        "function": "Your stomach and digestive system break down food and absorb nutrients your body needs.",
        "why_it_matters": "Digestive symptoms and certain tests can indicate how well your digestive system is functioning.",
        "tips": ["Eat balanced, fiber-rich meals", "Stay hydrated", "Eat mindfully and avoid overeating", "Limit processed foods"],
    },
    "pancreas": {
        "position": (150, 275, 30, 15),
        "color": "#EC4899",
        "function": "Your pancreas produces insulin, the hormone that helps regulate your blood sugar levels.",
        "why_it_matters": "Glucose and HbA1c tests reflect how well your pancreas is managing blood sugar over time.",
        "tips": ["Limit added sugar and refined carbs", "Maintain a healthy weight", "Stay physically active", "Get regular checkups if at risk"],
    },
    "kidney": {
        "position": (150, 300, 45, 20),
        "color": "#0EA5E9",
        "function": "Your kidneys filter waste products from your blood and regulate fluid and mineral balance.",
        "why_it_matters": "Creatinine and Urea tests measure how efficiently your kidneys are filtering waste.",
        "tips": ["Stay well hydrated", "Limit excess salt intake", "Manage blood pressure and blood sugar", "Avoid overusing painkillers"],
    },
    "bones": {
        "position": (150, 400, 60, 100),
        "color": "#D1D5DB",
        "function": "Your skeletal system supports your body's structure and protects internal organs, and stores minerals like calcium.",
        "why_it_matters": "Vitamin D and calcium tests reflect bone health and mineral balance.",
        "tips": ["Get adequate calcium and Vitamin D", "Do regular weight-bearing exercise", "Get safe sun exposure", "Avoid smoking"],
    },
    "blood": {
        "position": (150, 350, 70, 15),
        "color": "#B91C1C",
        "function": "Your blood carries oxygen, nutrients, and immune cells throughout your body, and removes waste.",
        "why_it_matters": "Hemoglobin, WBC, and Platelet counts reflect your blood's ability to carry oxygen and fight infection.",
        "tips": ["Eat iron and vitamin-rich foods", "Stay hydrated", "Get regular checkups", "Treat infections promptly"],
    },
}

# Maps parser.py's `related_organ` field (already lowercase, matches
# KNOWN_TESTS in parser.py) directly to ORGAN_DATA keys above.


def _body_outline_svg() -> str:
    """
    A simple, generic human body silhouette drawn with basic SVG
    shapes (not a photo, not copyrighted art - just geometric shapes
    representing a head/torso/limbs, styled with CSS variables so it
    matches the app's theme).
    """
    return """
    <g id="body-outline" fill="none" stroke="var(--text-secondary, #94A3B8)" stroke-width="2" opacity="0.5">
        <circle cx="150" cy="60" r="40" />
        <path d="M110,100 L90,220 L100,420 L120,580 M190,100 L210,220 L200,420 L180,580" />
        <path d="M90,120 L40,280 M210,120 L260,280" />
        <path d="M90,220 L210,220" />
        <path d="M100,420 L200,420" />
    </g>
    """


def generate_organ_svg(organ: str, highlight: bool = True) -> str:
    """
    Builds a full SVG string: the shared body outline + a highlighted
    ellipse over the requested organ's position, with a pulsing CSS
    animation to draw the eye (when embedded in HTML/Streamlit which
    both support inline <style> animation).

    Returns an empty-body-only SVG (no highlight) if the organ name
    isn't recognized, rather than raising an error - a missing organ
    should never break the whole explanation UI.
    """
    organ_key = organ.lower().strip()
    info = ORGAN_DATA.get(organ_key)

    highlight_svg = ""
    if info and highlight:
        cx, cy, rx, ry = info["position"]
        color = info["color"]
        highlight_svg = f"""
        <ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" fill="{color}" opacity="0.55">
            <animate attributeName="opacity" values="0.35;0.7;0.35" dur="2s" repeatCount="indefinite" />
        </ellipse>
        <ellipse cx="{cx}" cy="{cy}" rx="{rx + 6}" ry="{ry + 6}" fill="none" stroke="{color}" stroke-width="2" opacity="0.8" />
        """

    svg = f"""<svg viewBox="0 0 300 600" xmlns="http://www.w3.org/2000/svg" width="100%" height="auto">
        {_body_outline_svg()}
        {highlight_svg}
    </svg>"""

    return svg


def get_organ_for_test(test_name: str, related_organ: str = None) -> str:
    """
    Given a lab test name and/or its already-known `related_organ`
    (from parser.py's KNOWN_TESTS table), returns the best matching
    ORGAN_DATA key. Falls back to keyword matching on the test name
    if related_organ wasn't provided.
    """
    if related_organ and related_organ.lower() in ORGAN_DATA:
        return related_organ.lower()

    lowered = test_name.lower()
    keyword_map = {
        "creatinine": "kidney", "urea": "kidney", "egfr": "kidney",
        "sgpt": "liver", "sgot": "liver", "alt": "liver", "ast": "liver", "bilirubin": "liver",
        "cholesterol": "heart", "ldl": "heart", "hdl": "heart", "triglyceride": "heart", "ecg": "heart",
        "hemoglobin": "blood", "wbc": "blood", "platelet": "blood", "rbc": "blood",
        "tsh": "thyroid", "t3": "thyroid", "t4": "thyroid",
        "glucose": "pancreas", "hba1c": "pancreas", "insulin": "pancreas",
        "vitamin d": "bones", "calcium": "bones",
        "mri": "brain", "brain": "brain",
    }
    for keyword, organ in keyword_map.items():
        if keyword in lowered:
            return organ
    return "blood"  # sensible generic default rather than raising an error


def build_organ_explanation(organ: str) -> dict:
    """
    Returns the full educational package for one organ: the SVG
    diagram + hardcoded factual info (function, importance, tips).
    This is FAST and FREE (no LLM call needed) since the factual
    content per organ is fixed and small enough to maintain directly.

    For a more open-ended/conversational explanation, routes.py can
    additionally call prompts.build_organ_explanation_prompt() +
    ai_engine.generate_ai_response() for an LLM-generated version -
    this function provides the guaranteed-available, zero-latency
    baseline that the UI can always show immediately.
    """
    organ_key = organ.lower().strip()
    info = ORGAN_DATA.get(organ_key)

    if not info:
        logger.warning("Unknown organ requested: %s", organ)
        return {
            "organ": organ,
            "svg": generate_organ_svg(organ, highlight=False),
            "function": "No information available for this organ yet.",
            "why_it_matters": "",
            "lifestyle_tips": [],
        }

    return {
        "organ": organ_key,
        "svg": generate_organ_svg(organ_key),
        "function": info["function"],
        "why_it_matters": info["why_it_matters"],
        "lifestyle_tips": info["tips"],
    }


def list_available_organs() -> List[str]:
    """Used by the frontend to build an organ picker/menu."""
    return sorted(ORGAN_DATA.keys())
