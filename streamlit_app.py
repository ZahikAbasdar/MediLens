"""
streamlit_app.py
================
The complete MediLens frontend, built with Streamlit.

WHY ONE FILE FOR THE WHOLE FRONTEND:
Streamlit apps are naturally suited to a single-file-with-functions
structure - `st.session_state` (Streamlit's built-in state store)
holds the JWT token and current page, and each "page" is just a
Python function. Splitting this into 8 separate files for 8 pages
would add import overhead and navigation complexity for no real
benefit at this project's size.

HOW THIS TALKS TO THE BACKEND:
This frontend NEVER imports backend modules directly (no `from
ai_engine import ...`). It only makes HTTP requests to the FastAPI
server (app.py), exactly like a real separate frontend application
would. This is the correct architecture: frontend and backend are
genuinely decoupled, and the backend could be swapped for a mobile
app or a different frontend without any backend code changes.

RUN WITH:
    streamlit run streamlit_app.py
(after starting the backend separately with `uvicorn app:app --reload`)
"""

import requests
import streamlit.components.v1 as components
import streamlit as st
import plotly.graph_objects as go

from voice import is_voice_available, text_to_speech, VoiceError

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
API_BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title="MediLens - See Beyond the Report",
    page_icon="🩺",
    layout="wide",
)

# ------------------------------------------------------------
# Theme (Blue/White, light custom CSS - Dark Mode follows Streamlit's
# native theme toggle in the settings menu, which this CSS respects
# by using Streamlit's own CSS variables rather than hardcoded colors)
# ------------------------------------------------------------
st.markdown("""
<style>
    .medilens-header { font-size: 2.2rem; font-weight: 700; color: #1E5EFF; margin-bottom: 0; }
    .medilens-tagline { font-size: 1rem; color: #64748B; margin-top: 0; font-style: italic; }
    .disclaimer-box {
        background-color: rgba(30, 94, 255, 0.08);
        border-left: 4px solid #1E5EFF;
        padding: 0.75rem 1rem;
        border-radius: 6px;
        font-size: 0.85rem;
        margin: 1rem 0;
    }
    .alert-critical {
        background-color: rgba(220, 38, 38, 0.1);
        border-left: 4px solid #DC2626;
        padding: 0.75rem 1rem;
        border-radius: 6px;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------
# Session state initialization
# ------------------------------------------------------------
def init_session_state():
    defaults = {
        "token": None,
        "user_email": None,
        "user_name": None,
        "page": "Login",
        "selected_report_id": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# ------------------------------------------------------------
# API helper functions
# ------------------------------------------------------------
def auth_headers() -> dict:
    return {"Authorization": f"Bearer {st.session_state.token}"} if st.session_state.token else {}


def api_get(path: str, **kwargs):
    try:
        response = requests.get(f"{API_BASE_URL}{path}", headers=auth_headers(), timeout=30, **kwargs)
        return response
    except requests.exceptions.ConnectionError:
        st.error(
            "⚠️ Could not connect to the MediLens backend. "
            "Make sure it's running: `uvicorn app:app --reload`"
        )
        st.stop()


def api_post(path: str, **kwargs):
    try:
        response = requests.post(f"{API_BASE_URL}{path}", headers=auth_headers(), timeout=60, **kwargs)
        return response
    except requests.exceptions.ConnectionError:
        st.error(
            "⚠️ Could not connect to the MediLens backend. "
            "Make sure it's running: `uvicorn app:app --reload`"
        )
        st.stop()


def api_delete(path: str, **kwargs):
    try:
        return requests.delete(f"{API_BASE_URL}{path}", headers=auth_headers(), timeout=30, **kwargs)
    except requests.exceptions.ConnectionError:
        st.error("⚠️ Could not connect to the MediLens backend.")
        st.stop()


# ------------------------------------------------------------
# Header (shown on every page)
# ------------------------------------------------------------
def render_header():
    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown(
            '<p class="medilens-header">🩺 MediLens</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="medilens-tagline">"See Beyond the Report."</p>',
            unsafe_allow_html=True,
        )

    with col2:
        if st.session_state.token:
            st.write(f"👤 {st.session_state.user_name or st.session_state.user_email}")

            if st.button("Logout", use_container_width=True):
                for key in ("token", "user_email", "user_name"):
                    st.session_state[key] = None

                st.session_state.page = "Login"
                st.rerun()


# ------------------------------------------------------------
# PAGE: Login / Signup
# ------------------------------------------------------------
def page_login():
    st.subheader("Welcome to MediLens")
    tab_login, tab_signup, tab_forgot = st.tabs(["Login", "Sign Up", "Forgot Password"])

    with tab_login:
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login", use_container_width=True)
            if submitted:
                response = api_post("/auth/login", params={"email": email, "password": password})
                if response.status_code == 200:
                    st.session_state.token = response.json()["access_token"]
                    st.session_state.user_email = email
                    profile = api_get("/auth/profile").json()
                    st.session_state.user_name = profile.get("full_name")
                    st.session_state.page = "Dashboard"
                    st.success("Logged in successfully!")
                    st.rerun()
                else:
                    st.error(response.json().get("detail", "Login failed"))

    with tab_signup:
        with st.form("signup_form"):
            full_name = st.text_input("Full Name")
            email = st.text_input("Email", key="signup_email")
            age = st.number_input("Age", min_value=0, max_value=120, value=30)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            password = st.text_input("Password (min 6 characters)", type="password", key="signup_password")
            submitted = st.form_submit_button("Create Account", use_container_width=True)
            if submitted:
                response = api_post("/auth/signup", json={
                    "email": email, "password": password, "full_name": full_name,
                    "age": int(age), "gender": gender,
                })
                if response.status_code == 201:
                    st.success("Account created! Please log in from the Login tab.")
                else:
                    st.error(response.json().get("detail", "Signup failed"))

    with tab_forgot:
        with st.form("forgot_form"):
            email = st.text_input("Your account email")
            submitted = st.form_submit_button("Send Reset Code", use_container_width=True)
            if submitted:
                response = api_post("/auth/forgot-password", json={"email": email})
                data = response.json()
                st.info(data.get("message"))
                if "debug_reset_code" in data:
                    st.warning(f"DEV MODE - your reset code is: {data['debug_reset_code']}")

        with st.form("reset_form"):
            st.write("Have a reset code? Enter it below:")
            reset_email = st.text_input("Email", key="reset_email")
            reset_code = st.text_input("Reset Code")
            new_password = st.text_input("New Password", type="password")
            submitted = st.form_submit_button("Reset Password", use_container_width=True)
            if submitted:
                response = api_post("/auth/reset-password", json={
                    "email": reset_email, "reset_code": reset_code, "new_password": new_password,
                })
                if response.status_code == 200:
                    st.success("Password reset! Please log in.")
                else:
                    st.error(response.json().get("detail", "Reset failed"))


# ------------------------------------------------------------
# PAGE: Upload
# ------------------------------------------------------------
def page_upload():
    st.subheader("📤 Upload a Medical Report")
    st.caption("Supported formats: PDF, PNG, JPG, JPEG, DOC, DOCX, TXT")

    uploaded_files = st.file_uploader(
        "Drag & drop your report(s) here", accept_multiple_files=True,
        type=["pdf", "png", "jpg", "jpeg", "doc", "docx", "txt"],
    )

    if uploaded_files and st.button("Process Reports", type="primary"):
        for uploaded_file in uploaded_files:
            with st.spinner(f"Processing {uploaded_file.name}... (OCR + parsing + AI indexing)"):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                response = api_post("/reports/upload", files=files)

            if response.status_code == 201:
                report = response.json()
                st.success(f"✅ {uploaded_file.name} processed successfully!")
                with st.expander(f"View extracted data for {uploaded_file.name}", expanded=True):
                    col1, col2 = st.columns(2)
                    col1.metric("Report Type", report.get("report_type") or "Unknown")
                    col2.metric("OCR Confidence", f"{(report.get('ocr_confidence') or 0) * 100:.0f}%")
                    st.write(f"**Patient:** {report.get('patient_name') or 'Not detected'}")
                    st.write(f"**Doctor:** {report.get('doctor_name') or 'Not detected'}")
                    st.write(f"**Date:** {report.get('report_date') or 'Not detected'}")

                    if report["lab_values"]:
                        st.write("**Lab Values Found:**")
                        for lv in report["lab_values"]:
                            flag = "🔴" if lv["is_critical"] else ("🟡" if lv["is_abnormal"] else "🟢")
                            st.write(f"{flag} {lv['test_name']}: {lv['value']} {lv['unit'] or ''} (Ref: {lv['reference_range'] or 'N/A'})")
                    else:
                        st.info("No structured lab values were detected in this report.")
            else:
                st.error(f"❌ Failed to process {uploaded_file.name}: {response.json().get('detail', 'Unknown error')}")


# ------------------------------------------------------------
# PAGE: Dashboard
# ------------------------------------------------------------
def page_dashboard():
    st.subheader("📊 Health Dashboard")

    response = api_get("/dashboard")
    if response.status_code != 200:
        st.error("Could not load dashboard data.")
        return
    data = response.json()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Health Score", f"{data['latest_health_score']:.0f}/100" if data["latest_health_score"] is not None else "N/A")
    col2.metric("Risk Level", data["latest_risk_level"] or "N/A")
    col3.metric("Total Reports", data["total_reports"])
    col4.metric("Critical Values", data["total_critical_values"])

    # Score history chart
    if data["score_history"]:
        st.write("### Health Score Timeline")
        dates = [s["created_at"] for s in data["score_history"]]
        scores = [s["health_score"] for s in data["score_history"]]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=scores, mode="lines+markers", line=dict(color="#1E5EFF", width=3)))
        fig.update_layout(yaxis_range=[0, 100], height=350, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    # Smart alerts
    alerts_response = api_get("/dashboard/alerts")
    if alerts_response.status_code == 200 and alerts_response.json():
        st.write("### 🚨 Smart Alerts - Critical Values")
        for alert in alerts_response.json():
            st.markdown(f'<div class="alert-critical">⚠️ {alert["message"]}</div>', unsafe_allow_html=True)

    # Recent reports
    st.write("### Recent Reports")
    for report in data["recent_reports"]:
        col1, col2, col3 = st.columns([3, 2, 1])
        col1.write(f"📄 {report['original_filename']}")
        col2.write(report.get("report_type") or "Unknown type")
        if col3.button("View", key=f"view_{report['id']}"):
            st.session_state.selected_report_id = report["id"]
            st.session_state.page = "Report Detail"
            st.rerun()


# ------------------------------------------------------------
# PAGE: Report Detail (lab values + trend + organ explainer)
# ------------------------------------------------------------
def page_report_detail():
    report_id = st.session_state.selected_report_id
    if not report_id:
        st.warning("No report selected.")
        return

    response = api_get(f"/reports/{report_id}")
    if response.status_code != 200:
        st.error("Could not load report.")
        return
    report = response.json()

    st.subheader(f"📄 {report['original_filename']}")
    st.caption(f"Uploaded {report['uploaded_at']} | Type: {report.get('report_type') or 'Unknown'}")

    for lv in report["lab_values"]:
        flag = "🔴 CRITICAL" if lv["is_critical"] else ("🟡 Abnormal" if lv["is_abnormal"] else "🟢 Normal")
        with st.expander(f"{lv['test_name']}: {lv['value']} {lv['unit'] or ''} — {flag}"):
            col1, col2 = st.columns([1, 2])
            with col1:
                organ_resp = api_get(f"/explain/organ/{lv['related_organ'] or 'blood'}")
                if organ_resp.status_code == 200:
                    organ_data = organ_resp.json()
                    components.html(
    f"""
    <!DOCTYPE html>
    <html>
    <body style="margin:0;padding:0;background:white;">
        {organ_data["svg"]}
    </body>
    </html>
    """,
    height=650,
    scrolling=False,
)
            with col2:
                if st.button("Get AI Explanation", key=f"explain_{lv['id']}"):
                    with st.spinner("Generating explanation..."):
                        explain_resp = api_get(f"/explain/lab-value/{lv['id']}")
                    if explain_resp.status_code == 200:
                        result = explain_resp.json()
                        st.markdown(result["explanation"])
                        st.write("**About this organ:**", result["visual"]["function"])
                    else:
                        st.error(explain_resp.json().get("detail", "Could not generate explanation"))

    # Trend analysis for each test
    st.write("### 📈 Trend Analysis")
    test_names = list({lv["test_name"] for lv in report["lab_values"]})
    if test_names:
        selected_test = st.selectbox("Select a test to see its trend over time", test_names)
        if st.button("Show Trend"):
            trend_resp = api_get(f"/dashboard/trend/{selected_test}")
            if trend_resp.status_code == 200:
                trend = trend_resp.json()
                st.write(f"**Direction:** {trend['direction']}")
                dates = [p["date"] for p in trend["points"]]
                values = [p["value"] for p in trend["points"]]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=dates, y=values, mode="lines+markers"))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(trend_resp.json().get("detail", "Not enough historical data yet."))

    # Export buttons
    st.write("### 📥 Export")
    col1, col2, col3 = st.columns(3)
    if col1.button("Export as JSON"):
        r = api_get(f"/export/report/{report_id}/json")
        st.download_button("Download JSON", r.text, file_name=f"report_{report_id}.json", mime="application/json")
    if col2.button("Export as CSV"):
        r = api_get(f"/export/report/{report_id}/csv")
        st.download_button("Download CSV", r.content, file_name=f"report_{report_id}.csv", mime="text/csv")
    if col3.button("Generate Doctor Visit Summary"):
        with st.spinner("Generating summary..."):
            r = api_get(f"/export/summary/{report_id}", params={"summary_type": "doctor_visit"})
        if r.status_code == 200:
            st.markdown(r.json()["summary"])


# ------------------------------------------------------------
# PAGE: Chatbot
# ------------------------------------------------------------
def page_chat():
    st.subheader("💬 AI Chatbot")
    st.caption("Ask about your uploaded reports - e.g. \"Explain my hemoglobin\" or \"What does my latest report show?\"")

    history_resp = api_get("/chat/history")
    if history_resp.status_code == 200:
        for msg in history_resp.json():
            with st.chat_message("user" if msg["role"] == "user" else "assistant"):
                st.write(msg["content"])

    voice_status = is_voice_available()

    if prompt := st.chat_input("Type your question..."):
        with st.chat_message("user"):
            st.write(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = api_post("/chat", json={"message": prompt})
            if response.status_code == 200:
                reply = response.json()["reply"]
                st.write(reply)
                if voice_status["text_to_speech"]:
                    if st.button("🔊 Read aloud", key=f"tts_{len(prompt)}"):
                        try:
                            audio = text_to_speech(reply)
                            st.audio(audio, format="audio/mp3")
                        except VoiceError as e:
                            st.warning(str(e))
            else:
                st.error("Could not get a response. Please try again.")
        st.rerun()


# ------------------------------------------------------------
# PAGE: Medical Dictionary
# ------------------------------------------------------------
def page_dictionary():
    st.subheader("📖 Medical Dictionary")
    term = st.text_input("Search for a disease, medicine, lab test, or medical term")
    if term and st.button("Search"):
        with st.spinner("Looking up..."):
            response = api_post("/explain/term", json={"term": term})
        if response.status_code == 200:
            result = response.json()
            if result["source"] == "dictionary":
                st.success(f"Found in curated medical dictionary: **{result['term'].title()}**")
                for key, value in result.items():
                    if key not in ("source", "term", "category"):
                        st.write(f"**{key.replace('_', ' ').title()}:** {value}")
            else:
                st.info("Generated by AI (not in curated dictionary)")
                st.markdown(result["explanation"])


# ------------------------------------------------------------
# PAGE: Organ Explorer
# ------------------------------------------------------------
def page_organ_explorer():
    st.subheader("🫀 Visual Medical Explainer")
    st.caption("Explore how your body works, organ by organ.")

    organs_resp = api_get("/explain/organs")

    if organs_resp.status_code != 200:
        st.error("Could not load organ list.")
        return

    organs = organs_resp.json().get("organs", [])

    if not organs:
        st.warning("No organs available.")
        return

    selected_organ = st.selectbox("Choose an organ", organs)

    if not selected_organ:
        return

    response = api_get(f"/explain/organ/{selected_organ}")

    if response.status_code != 200:
        st.error("Could not load organ information.")
        return

    data = response.json()

    col1, col2 = st.columns([1, 2])

    with col1:
        components.html(
            f"""
            <!DOCTYPE html>
            <html>
            <body style="margin:0;padding:0;background:white;">
                {data["svg"]}
            </body>
            </html>
            """,
            height=650,
            scrolling=False,
        )

    with col2:
        st.subheader(selected_organ.title())

        st.markdown("### 🩺 What it does")
        st.write(data.get("function", "N/A"))

        st.markdown("### ❓ Why it matters")
        st.write(data.get("why_it_matters", "N/A"))

        st.markdown("### 💚 Lifestyle Tips")

        for tip in data.get("lifestyle_tips", []):
            st.markdown(f"✅ {tip}")

# ------------------------------------------------------------
# Main app router
# ------------------------------------------------------------
def main():
    render_header()

    st.markdown(
        '<div class="disclaimer-box">⚕️ MediLens provides general educational information only. '
        'It does not diagnose conditions or replace professional medical advice. '
        'Always consult a licensed healthcare provider.</div>',
        unsafe_allow_html=True,
    )

    if not st.session_state.token:
        page_login()
        return

    with st.sidebar:
        st.write("### Navigation")
        pages = ["Dashboard", "Upload", "Chat", "Dictionary", "Organ Explorer"]
        for p in pages:
            if st.button(p, use_container_width=True):
                st.session_state.page = p
                st.rerun()

    page = st.session_state.page
    if page == "Dashboard":
        page_dashboard()
    elif page == "Upload":
        page_upload()
    elif page == "Chat":
        page_chat()
    elif page == "Dictionary":
        page_dictionary()
    elif page == "Organ Explorer":
        page_organ_explorer()
    elif page == "Report Detail":
        page_report_detail()
    else:
        page_dashboard()


if __name__ == "__main__":
    main()
