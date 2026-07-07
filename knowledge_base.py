"""
knowledge_base.py
=================
Backing data for two related features:
  - Feature #8: Medical Knowledge Base (diseases, tests, medicines, terms)
  - Feature #16: Medical Dictionary (searchable lookup)

WHY STATIC DATA INSTEAD OF ALWAYS CALLING THE LLM:
For common, well-established terms, a hardcoded, doctor-reviewable
dictionary is:
  - Instant (no API latency or cost)
  - Consistent (the same term always gets the same safe explanation)
  - Auditable (a real clinician could review this exact file)

For a term NOT found here, routes.py falls back to
prompts.build_term_explanation_prompt() + ai_engine, so the dictionary
is a fast-path cache in front of the LLM, not a hard limit on what
users can look up.
"""

from typing import Optional

# ------------------------------------------------------------
# Lab tests
# ------------------------------------------------------------
LAB_TEST_DICTIONARY: dict[str, dict] = {
    "hemoglobin": {
        "category": "Lab Test",
        "definition": "A protein in red blood cells that carries oxygen from your lungs to the rest of your body.",
        "normal_range": "13.0 - 17.0 g/dL (adult male), 12.0 - 15.5 g/dL (adult female)",
        "importance": "Low levels may indicate anemia; high levels can relate to dehydration or other conditions.",
    },
    "creatinine": {
        "category": "Lab Test",
        "definition": "A waste product from muscle activity that your kidneys filter out of your blood.",
        "normal_range": "0.6 - 1.3 mg/dL",
        "importance": "Elevated levels can indicate reduced kidney function.",
    },
    "hba1c": {
        "category": "Lab Test",
        "definition": "Reflects your average blood sugar level over the past 2-3 months.",
        "normal_range": "Below 5.7% (non-diabetic range)",
        "importance": "Used to monitor and diagnose diabetes and prediabetes.",
    },
    "cholesterol": {
        "category": "Lab Test",
        "definition": "A fat-like substance in your blood, used to build cells, but harmful in excess.",
        "normal_range": "Below 200 mg/dL (total cholesterol)",
        "importance": "High levels are linked to increased cardiovascular risk over time.",
    },
    "tsh": {
        "category": "Lab Test",
        "definition": "Thyroid Stimulating Hormone - signals your thyroid gland to produce thyroid hormone.",
        "normal_range": "0.4 - 4.0 uIU/mL",
        "importance": "Abnormal levels can indicate an overactive or underactive thyroid.",
    },
}

# ------------------------------------------------------------
# Diseases / conditions (general educational descriptions only)
# ------------------------------------------------------------
DISEASE_DICTIONARY: dict[str, dict] = {
    "diabetes": {
        "category": "Disease",
        "definition": "A chronic condition where the body has trouble regulating blood sugar levels.",
        "common_causes": "Insulin resistance, insufficient insulin production, genetics, lifestyle factors.",
        "wellness_tips": "Balanced diet, regular exercise, routine blood sugar monitoring, regular checkups.",
    },
    "hypertension": {
        "category": "Disease",
        "definition": "Persistently elevated blood pressure in the arteries, often with no obvious symptoms.",
        "common_causes": "Genetics, high sodium intake, stress, obesity, lack of physical activity.",
        "wellness_tips": "Reduce sodium intake, exercise regularly, manage stress, monitor blood pressure.",
    },
    "anemia": {
        "category": "Disease",
        "definition": "A condition where you lack enough healthy red blood cells to carry adequate oxygen.",
        "common_causes": "Iron deficiency, vitamin B12/folate deficiency, chronic disease, blood loss.",
        "wellness_tips": "Iron and vitamin-rich diet, follow up with a doctor for the underlying cause.",
    },
    "hypothyroidism": {
        "category": "Disease",
        "definition": "A condition where the thyroid gland doesn't produce enough thyroid hormone.",
        "common_causes": "Autoimmune conditions, iodine deficiency, certain medications.",
        "wellness_tips": "Consistent medication timing (as prescribed), adequate iodine, regular monitoring.",
    },
}

# ------------------------------------------------------------
# Common medicines (general educational info only, no dosing)
# ------------------------------------------------------------
MEDICINE_DICTIONARY: dict[str, dict] = {
    "metformin": {
        "category": "Medicine",
        "drug_class": "Biguanide (oral diabetes medication)",
        "common_uses": "Managing type 2 diabetes by improving insulin sensitivity.",
        "common_side_effects": "Nausea, stomach upset, diarrhea (often improves over time).",
        "precautions": "Should not be used with certain kidney conditions - always discuss with your doctor.",
    },
    "atorvastatin": {
        "category": "Medicine",
        "drug_class": "Statin (cholesterol-lowering medication)",
        "common_uses": "Lowering LDL cholesterol to reduce cardiovascular risk.",
        "common_side_effects": "Muscle aches, mild digestive upset.",
        "precautions": "Regular liver function monitoring is often recommended by doctors.",
    },
    "levothyroxine": {
        "category": "Medicine",
        "drug_class": "Thyroid hormone replacement",
        "common_uses": "Treating hypothyroidism (underactive thyroid).",
        "common_side_effects": "Usually well-tolerated at the correct dose; too high a dose can cause palpitations.",
        "precautions": "Typically taken on an empty stomach; timing matters for absorption.",
    },
    "amlodipine": {
        "category": "Medicine",
        "drug_class": "Calcium channel blocker (blood pressure medication)",
        "common_uses": "Treating high blood pressure and certain types of chest pain.",
        "common_side_effects": "Swelling in ankles/feet, flushing, headache.",
        "precautions": "Discuss with your doctor before combining with other blood pressure medications.",
    },
}


def search_dictionary(term: str) -> Optional[dict]:
    """
    Searches all three dictionaries (tests, diseases, medicines) for a
    matching term. Returns the FIRST match found, or None if the term
    isn't in our static dictionary (caller should fall back to the LLM).

    Matching is substring-based and case-insensitive, so searching
    "sugar" won't match "hba1c" but "cholesterol" will match a search
    for "chol".
    """
    lowered = term.lower().strip()

    for dictionary in (LAB_TEST_DICTIONARY, DISEASE_DICTIONARY, MEDICINE_DICTIONARY):
        for key, info in dictionary.items():
            if lowered in key or key in lowered:
                return {"term": key, **info}

    return None


def list_all_terms() -> dict[str, list[str]]:
    """Used by the frontend to build browsable category lists."""
    return {
        "Lab Tests": sorted(LAB_TEST_DICTIONARY.keys()),
        "Diseases": sorted(DISEASE_DICTIONARY.keys()),
        "Medicines": sorted(MEDICINE_DICTIONARY.keys()),
    }
