"""
prompts.py
==========
Every prompt template sent to an LLM lives HERE, and only here.

WHY CENTRALIZE PROMPTS:
1. Consistency - every feature that explains a lab value uses the
   SAME safety framing, so we never accidentally ship one feature
   that forgets to say "this isn't a diagnosis".
2. Easy auditing - a reviewer (or you, the developer) can read this
   ONE file to see exactly what MediLens is allowed to say to the AI
   and how it's instructed to respond, without hunting through
   ai_engine.py's request-handling code.
3. Easy tuning - improving how explanations read means editing text
   here, not touching any logic in ai_engine.py.

EVERY prompt in this file includes the same non-negotiable safety
instructions. This is intentional redundancy: even if a future
developer adds a new prompt and forgets the full safety framing,
the SAFETY_RULES block is designed to be trivially copy-pasted in.
"""

# ------------------------------------------------------------
# Non-negotiable safety instructions, included in every prompt
# ------------------------------------------------------------
SAFETY_RULES = """
CRITICAL RULES YOU MUST ALWAYS FOLLOW:
- You are an EDUCATIONAL assistant, NOT a doctor. NEVER diagnose a disease or condition.
- NEVER tell the user what medication to take, start, stop, or change.
- NEVER claim certainty about what a value "means" for this specific person - always
  frame explanations in terms of "this test generally indicates..." or "values in this
  range are commonly associated with...", not "you have X".
- ALWAYS distinguish between FACTS extracted from the report (exact numbers, names) and
  your OWN educational explanation of what those facts commonly mean.
- If a value is in a CRITICAL/dangerous range, clearly state that the user should seek
  prompt medical attention, without being alarmist or causing panic.
- Keep language SIMPLE. Assume the reader has no medical background.
- Do not use overly technical jargon without explaining it in plain words.
"""


# ------------------------------------------------------------
# Feature: AI Report Analyzer (Module 4 in the feature list)
# ------------------------------------------------------------
def build_lab_value_explanation_prompt(
    test_name: str,
    value: str,
    unit: str,
    reference_range: str,
    is_abnormal: bool,
    is_critical: bool,
) -> str:
    """
    Builds the prompt used to explain ONE lab test result in plain
    English, covering everything Feature #5 in the spec asks for:
    what it means, why it's tested, possible reasons if high/low,
    lifestyle/nutrition tips, questions for the doctor, warning signs.
    """
    status = "CRITICAL - requires prompt medical attention" if is_critical else (
        "Abnormal" if is_abnormal else "Within normal range"
    )

    return f"""{SAFETY_RULES}

A patient's lab report shows the following result:

Test: {test_name}
Reported Value: {value} {unit}
Reference Range: {reference_range}
Status: {status}

Write a clear, structured, plain-English explanation covering EXACTLY these sections
(use these as markdown headers):

### What This Test Measures
### What Your Result Means
### Why Doctors Order This Test
### Possible Reasons If High
### Possible Reasons If Low
### Lifestyle Suggestions
### Nutrition Suggestions
### Questions To Ask Your Doctor
### Warning Signs That Need Urgent Care

Keep each section to 2-4 short sentences or bullet points. Be reassuring but honest.
"""


# ------------------------------------------------------------
# Feature: AI Chatbot + RAG (Modules 6-7)
# ------------------------------------------------------------
def build_rag_chat_prompt(user_question: str, retrieved_context: str, chat_history: str = "") -> str:
    """
    Builds the RAG-grounded chatbot prompt. The retrieved_context
    comes from rag_engine.py's FAISS similarity search over the
    user's OWN uploaded reports - this is what prevents hallucination:
    the model is instructed to answer ONLY from this context.
    """
    history_block = f"\nPrevious conversation (for context only):\n{chat_history}\n" if chat_history else ""

    return f"""{SAFETY_RULES}

You are answering a question using ONLY the context below, which was retrieved from the
user's own uploaded medical reports. If the context does not contain enough information
to answer confidently, say so honestly instead of guessing or using outside knowledge
about the user specifically. You MAY use general medical education knowledge to explain
what a term means, but any specific facts about THIS patient must come only from the
context provided.

--- RETRIEVED CONTEXT FROM USER'S REPORTS ---
{retrieved_context if retrieved_context.strip() else "(No relevant report content was found for this question.)"}
--- END CONTEXT ---
{history_block}
User's question: {user_question}

Answer clearly and simply. If the context doesn't cover the question, say what
information would be needed and suggest the user consult their doctor or upload
the relevant report.
"""


# ------------------------------------------------------------
# Feature: Medical Dictionary / Knowledge Base term lookups
# ------------------------------------------------------------
def build_term_explanation_prompt(term: str) -> str:
    """Used when a user clicks/searches a medical term with no report context needed."""
    return f"""{SAFETY_RULES}

Explain the medical term "{term}" for a patient with no medical background. Include:

### Definition
### Normal Range (if applicable)
### Why It Matters
### Common Causes If Abnormal
### General Wellness Tips

Keep it concise - 2-3 sentences per section.
"""


# ------------------------------------------------------------
# Feature: Medicine Reader (Module 9)
# ------------------------------------------------------------
def build_medicine_explanation_prompt(medicine_name: str) -> str:
    """
    Explains a medicine found in a prescription. NEVER recommends
    dosage changes or prescribes - purely educational identification.
    """
    return f"""{SAFETY_RULES}

A prescription mentions the medicine "{medicine_name}". Provide general educational
information ONLY:

### General Purpose / Drug Class
### Common Uses
### Commonly Reported Side Effects
### General Precautions

Do NOT suggest a dosage. Do NOT tell the user whether they should keep taking it,
stop it, or change it - always defer that decision explicitly to their prescribing
doctor or pharmacist.
"""


# ------------------------------------------------------------
# Feature: AI Summary (Module 12)
# ------------------------------------------------------------
def build_summary_prompt(report_text: str, summary_type: str) -> str:
    """
    summary_type is one of: "simple", "detailed", "doctor_visit"
    Each produces a differently-scoped summary of the SAME report.
    """
    instructions = {
        "simple": (
            "Write a short, friendly 4-6 sentence summary in very simple language, "
            "as if explaining to a family member with no medical background."
        ),
        "detailed": (
            "Write a comprehensive summary covering every abnormal or notable value, "
            "organized by category (e.g. blood, kidney, liver), with brief explanations."
        ),
        "doctor_visit": (
            "Write a concise, structured summary formatted for the PATIENT to bring to "
            "a doctor's visit: list of abnormal values, any critical values flagged "
            "clearly at the top, and 3-5 specific questions the patient could ask."
        ),
    }
    instruction = instructions.get(summary_type, instructions["simple"])

    return f"""{SAFETY_RULES}

Below is the extracted text of a patient's medical report:

--- REPORT TEXT ---
{report_text}
--- END REPORT TEXT ---

{instruction}
"""


# ------------------------------------------------------------
# Feature: Organ / Visual Explainer (Module 22 in the follow-up spec)
# ------------------------------------------------------------
def build_organ_explanation_prompt(organ: str, related_test: str = "", value_context: str = "") -> str:
    """
    Builds the short educational text that accompanies the dynamically
    generated SVG body diagram in visual_engine.py.
    """
    context_line = (
        f"This is being shown because the test '{related_test}' was abnormal ({value_context})."
        if related_test else ""
    )
    return f"""{SAFETY_RULES}

Explain the {organ} for a patient with no medical background, in 3 short sections:

### What This Organ Does
### Why It Matters For Your Health
### General Wellness Tips For This Organ

{context_line}
Keep the entire response under 120 words total.
"""
