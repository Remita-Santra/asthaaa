import os
import json
import uuid
import tempfile
import datetime
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

# Safe environment configuration block
load_dotenv()


def _clean_api_key(raw_key):
    """Strip whitespace and accidental wrapping quotes copied into .env / secrets."""
    if not raw_key:
        return raw_key
    cleaned = raw_key.strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in ("'", '"'):
        cleaned = cleaned[1:-1].strip()
    return cleaned


api_key = _clean_api_key(os.environ.get("GEMINI_API_KEY"))
if not api_key:
    try:
        api_key = _clean_api_key(st.secrets.get("GEMINI_API_KEY"))
    except Exception:
        api_key = None

if not api_key:
    st.error("🛑 Environment Variable 'GEMINI_API_KEY' is missing. Please check your system settings or configuration secrets panel.")
    st.stop()

# Import the Google GenAI module safely
from google import genai

client = genai.Client(api_key=api_key, vertexai=False)


@st.cache_resource(show_spinner=False)
def _verify_gemini_credentials(key_fingerprint: str):
    """
    Make one cheap live call at startup to confirm the API key actually
    authenticates, instead of letting every node fail identically with a
    401 the first time a real case is submitted. Cached per key value
    (key_fingerprint) so it only runs once per key per process, not on
    every Streamlit rerun.
    """
    try:
        client.models.generate_content(model="gemini-3.5-flash", contents="ping")
        return True, None
    except Exception as e:
        return False, str(e)


_verified_ok, _verify_error = _verify_gemini_credentials(api_key[-8:] if len(api_key) >= 8 else api_key)
if not _verified_ok:
    st.error(
        "🛑 Could not authenticate with the Gemini API using the configured "
        "GEMINI_API_KEY. This app will not be able to transcribe, translate, "
        "or extract anything until this is fixed.\n\n"
        "Common causes:\n"
        "- The key is invalid, expired, or was revoked — generate a fresh one at "
        "https://aistudio.google.com/apikey\n"
        "- `GOOGLE_GENAI_USE_VERTEXAI` is set to true somewhere in this environment, "
        "forcing Vertex AI (OAuth) auth instead of a plain API key\n"
        "- The key has API restrictions that exclude the Generative Language API\n\n"
        f"Raw error: {_verify_error}"
    )
    st.stop()

from graph import asha_agent_graph
import db

st.set_page_config(page_title="ASHTHA FOR ASHA", page_icon="🩺", layout="centered")

# --- CUSTOM CSS: LIGHT PINK / MAROON / WHITE THEME ---
# Forced with !important on stable data-testid selectors so the palette
# stays identical whether the user's Streamlit client is set to Light or
# Dark mode — the app never switches to Streamlit's own dark palette.
PALETTE = {
    "bg": "#FFFDF9",          # warm white / cream app background
    "card_bg": "#FFF0F2",     # light pink card background
    "card_border": "#C48B95", # dusty rose border
    "sidebar_bg": "#FFF5F6",  # very light pink sidebar
    "sidebar_border": "#E8C1C7",
    "maroon": "#5A0E1A",      # primary text / primary button
    "accent": "#D34E65",      # rose accent / hover / focus
    "input_bg": "#FFFFFF",
    "placeholder": "#B98A93",
    "disabled": "#E8C1C7",
}

st.markdown(f"""
    <style>
        /* ---- Base app surface (overrides both light & dark Streamlit themes) ---- */
        html, body, .stApp,
        [data-testid="stAppViewContainer"], [data-testid="stMain"],
        [data-testid="stHeader"], [data-testid="stBottomBlockContainer"],
        [data-testid="stToolbar"] {{
            background-color: {PALETTE['bg']} !important;
            color: {PALETTE['maroon']} !important;
        }}

        /* ---- Typography ---- */
        h1, h2, h3, h4, h5, h6, label, p, span,
        .stMarkdown, .stMarkdown p, .stCaption,
        [data-testid="stCaptionContainer"], [data-testid="stMarkdownContainer"] {{
            color: {PALETTE['maroon']} !important;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }}

        /* ---- Bordered cards / containers & expanders ---- */
        div[data-testid="stContainerBorder"], div[data-testid="stExpander"] {{
            background-color: {PALETTE['card_bg']} !important;
            border: 1px solid {PALETTE['card_border']} !important;
            border-radius: 10px;
        }}
        div[data-testid="stContainerBorder"] {{ padding: 1.5rem; }}
        [data-testid="stExpander"] summary,
        [data-testid="stExpanderDetails"] {{
            background-color: {PALETTE['card_bg']} !important;
            color: {PALETTE['maroon']} !important;
        }}

        /* ---- Buttons ---- */
        button[data-testid="baseButton-primary"] {{
            background-color: {PALETTE['maroon']} !important;
            color: {PALETTE['bg']} !important;
            border: 2px solid {PALETTE['maroon']} !important;
            border-radius: 8px !important;
            transition: all 0.3s ease;
        }}
        button[data-testid="baseButton-primary"]:hover {{
            background-color: {PALETTE['accent']} !important;
            color: {PALETTE['bg']} !important;
            border-color: {PALETTE['accent']} !important;
        }}
        button[data-testid="baseButton-primary"]:disabled {{
            background-color: {PALETTE['disabled']} !important;
            color: {PALETTE['bg']} !important;
            border-color: {PALETTE['disabled']} !important;
            opacity: 1 !important;
        }}
        button[data-testid="baseButton-secondary"] {{
            background-color: {PALETTE['input_bg']} !important;
            color: {PALETTE['maroon']} !important;
            border: 1px solid {PALETTE['card_border']} !important;
        }}
        button[data-testid="baseButton-secondary"]:hover {{
            background-color: {PALETTE['card_bg']} !important;
            color: {PALETTE['maroon']} !important;
            border-color: {PALETTE['accent']} !important;
        }}
        [data-testid="stDownloadButton"] button {{
            background-color: {PALETTE['input_bg']} !important;
            color: {PALETTE['maroon']} !important;
            border: 1px solid {PALETTE['card_border']} !important;
        }}
        [data-testid="stDownloadButton"] button:hover {{
            background-color: {PALETTE['card_bg']} !important;
            border-color: {PALETTE['accent']} !important;
        }}

        /* ---- Sidebar ---- */
        section[data-testid="stSidebar"], [data-testid="stSidebarContent"] {{
            background-color: {PALETTE['sidebar_bg']} !important;
            border-right: 1px solid {PALETTE['sidebar_border']};
        }}
        section[data-testid="stSidebar"] * {{ color: {PALETTE['maroon']} !important; }}

        /* ---- Text / number inputs & text areas ---- */
        .stTextInput input, .stTextArea textarea, .stNumberInput input {{
            background-color: {PALETTE['input_bg']} !important;
            color: {PALETTE['maroon']} !important;
            border: 1px solid {PALETTE['card_border']} !important;
            border-radius: 6px !important;
        }}
        .stTextInput input::placeholder, .stTextArea textarea::placeholder {{
            color: {PALETTE['placeholder']} !important;
        }}
        .stTextInput input:focus, .stTextArea textarea:focus, .stNumberInput input:focus {{
            border-color: {PALETTE['accent']} !important;
            box-shadow: 0 0 0 1px {PALETTE['accent']} !important;
        }}

        /* ---- Radio buttons & checkboxes ---- */
        div[role="radiogroup"] label p, .stCheckbox label p {{
            color: {PALETTE['maroon']} !important;
        }}

        /* ---- Select boxes ---- */
        div[data-baseweb="select"] > div {{
            background-color: {PALETTE['input_bg']} !important;
            color: {PALETTE['maroon']} !important;
            border: 1px solid {PALETTE['card_border']} !important;
        }}
        ul[data-testid="stSelectboxVirtualDropdown"] {{
            background-color: {PALETTE['input_bg']} !important;
        }}
        ul[data-testid="stSelectboxVirtualDropdown"] li {{
            color: {PALETTE['maroon']} !important;
        }}

        /* ---- Tabs ---- */
        button[data-baseweb="tab"] p {{ color: {PALETTE['maroon']} !important; }}
        button[data-baseweb="tab"][aria-selected="true"] p {{ color: {PALETTE['accent']} !important; }}
        div[data-baseweb="tab-highlight"] {{ background-color: {PALETTE['accent']} !important; }}
        div[data-baseweb="tab-border"] {{ background-color: {PALETTE['sidebar_border']} !important; }}

        /* ---- Alerts: info / success / warning / error ---- */
        .stAlert, [data-testid="stNotification"] {{
            background-color: {PALETTE['card_bg']} !important;
            color: {PALETTE['maroon']} !important;
            border-left: 5px solid {PALETTE['accent']} !important;
        }}
        .stAlert p, [data-testid="stNotification"] p {{ color: {PALETTE['maroon']} !important; }}

        /* ---- JSON viewer / code blocks ---- */
        [data-testid="stJson"], pre, code {{
            background-color: {PALETTE['sidebar_bg']} !important;
            color: {PALETTE['maroon']} !important;
        }}

        /* ---- Divider & spinner ---- */
        hr {{ border-color: {PALETTE['sidebar_border']} !important; }}
        [data-testid="stSpinner"] div {{ color: {PALETTE['maroon']} !important; }}
    </style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# PAGE ROUTER
# ---------------------------------------------------------------------------
# Simple session-state page router: login -> dashboard -> new_case -> report.
# Login is required before anything else is reachable.
if "page" not in st.session_state:
    st.session_state.page = "login"


def go_to(page_name: str):
    st.session_state.page = page_name
    st.rerun()


PATIENT_CATEGORY_OPTIONS = {
    "🤰 Pregnant Woman": "PREGNANT_WOMAN",
    "🧒 Child (under 5)": "CHILD",
    "🧑 Other / General Patient": "OTHER",
}
PATIENT_CATEGORY_LABELS = {v: k for k, v in PATIENT_CATEGORY_OPTIONS.items()}


def _risk_level_of(raw_risk):
    """Normalize a stored risk_assessment (dict / JSON string) to a risk_level string."""
    if isinstance(raw_risk, dict):
        return raw_risk.get("risk_level", "—")
    if isinstance(raw_risk, str) and raw_risk.strip():
        try:
            parsed = json.loads(raw_risk)
            if isinstance(parsed, dict):
                return parsed.get("risk_level", "—")
        except json.JSONDecodeError:
            return "—"
    return "—"


def _reasons_of(raw_risk):
    if isinstance(raw_risk, dict):
        return raw_risk.get("reasons", [])
    if isinstance(raw_risk, str) and raw_risk.strip():
        try:
            parsed = json.loads(raw_risk)
            if isinstance(parsed, dict):
                return parsed.get("reasons", [])
        except json.JSONDecodeError:
            return []
    return []


def print_button(label="🖨️ Print This Report"):
    """
    Streamlit has no native print action, so this renders a small HTML
    button that triggers the browser's own print dialog (Ctrl+P) for the
    page via window.parent.print(). The ⬇️ Download button next to it is
    the more reliable option if the browser blocks the print call.
    """
    components.html(
        f"""
        <button onclick="window.parent.print()"
            style="background-color:#5A0E1A;color:#FFFDF9;border:2px solid #5A0E1A;
            border-radius:8px;padding:0.55rem 1rem;font-size:1rem;cursor:pointer;
            width:100%;font-family:'Segoe UI',Tahoma,Geneva,Verdana,sans-serif;">
            {label}
        </button>
        """,
        height=55,
    )


# ---------------------------------------------------------------------------
# PAGE: LOGIN
# ---------------------------------------------------------------------------
def render_login_page():
    st.title("ASHTHA FOR ASHA")
    st.caption(
        "Turn every ASHA home visit into a fully automated health record — "
        "voice-first capture, auto-filled forms, vaccine/ANC schedule checks, "
        "and an auto-scheduled follow-up. No typing, no paper forms, no delays."
    )

    st.markdown("### 📋 ASHA Worker Log In")
    with st.container(border=True):
        worker_id = st.text_input(
            "ASHA Worker ID",
            value=st.session_state.get("worker_id", ""),
            placeholder="e.g. ASHA-WEST-4092",
            key="login_worker_id_input",
        )
        village = st.text_input(
            "Assigned Village / Settlement",
            value=st.session_state.get("village", ""),
            placeholder="e.g. Rampur Sub-Center",
            key="login_village_input",
        )

        can_login = bool(worker_id.strip()) and bool(village.strip())
        if st.button("Log In", type="primary", use_container_width=True, disabled=not can_login):
            st.session_state.worker_id = worker_id.strip()
            st.session_state.village = village.strip()
            go_to("dashboard")

        if not can_login:
            st.caption("Enter both your Worker ID and Village to continue.")


# ---------------------------------------------------------------------------
# PAGE: DASHBOARD — records of previous reports
# ---------------------------------------------------------------------------
def render_dashboard_page():
    col_title, col_logout = st.columns([4, 1])
    with col_title:
        st.title("🏠 Dashboard")
        st.caption(f"Logged in as **{st.session_state.worker_id}** · {st.session_state.village}")
    with col_logout:
        st.write("")
        if st.button("Log Out", use_container_width=True):
            for key in ("worker_id", "village", "last_result"):
                st.session_state.pop(key, None)
            go_to("login")

    if st.button("➕ New Patient Case", type="primary", use_container_width=True):
        go_to("new_case")

    st.markdown("### 🗂️ Previous Visit Records")
    if st.button("🔄 Refresh Records", type="secondary"):
        st.session_state["records"] = db.fetch_all_records()

    if "records" not in st.session_state:
        st.session_state["records"] = db.fetch_all_records()

    records = st.session_state.get("records", [])
    if not records:
        st.caption("No records yet — start a new patient case to create the first one.")
        return

    for rec in records:
        risk_level = _risk_level_of(rec.get("risk_assessment"))
        reasons = _reasons_of(rec.get("risk_assessment"))
        category_label = PATIENT_CATEGORY_LABELS.get(rec.get("patient_category"), rec.get("patient_type", "Unknown"))
        color = {"LOW": "green", "MODERATE": "orange", "HIGH": "red", "URGENT_REFERRAL": "red"}.get(risk_level, "grey")

        with st.container(border=True):
            st.markdown(
                f"**{rec.get('patient_name', 'Unknown')}** · {category_label} · "
                f":{color}[{risk_level}]  \n"
                f"`{rec.get('abha_ref', 'N/A')}` · recorded {rec.get('recorded_at', '—')} "
                f"by {rec.get('recorded_by_worker') or 'Unknown worker'}"
            )
            if rec.get("follow_up_date"):
                st.caption(f"📅 Follow-up scheduled: {rec['follow_up_date']}")
            if reasons:
                st.caption(" · ".join(reasons[:2]))

            if rec.get("detailed_report"):
                with st.expander("View Full Report"):
                    st.text_area(
                        "Report",
                        rec["detailed_report"],
                        height=240,
                        key=f"report_view_{rec.get('abha_ref')}",
                        label_visibility="collapsed",
                    )
                    st.download_button(
                        "⬇️ Download This Report (.txt)",
                        data=rec["detailed_report"],
                        file_name=f"ASHA_report_{rec.get('abha_ref', 'record')}.txt",
                        mime="text/plain",
                        key=f"download_{rec.get('abha_ref')}",
                    )
            else:
                st.caption("Full report text not available for this record.")


# ---------------------------------------------------------------------------
# PAGE: NEW CASE — patient details / conversation capture / MUAC (separate blocks)
# ---------------------------------------------------------------------------
def render_new_case_page():
    col_title, col_back = st.columns([4, 1])
    with col_title:
        st.title("📝 New Patient Case")
        st.caption(f"Worker **{st.session_state.worker_id}** · {st.session_state.village}")
    with col_back:
        st.write("")
        if st.button("← Dashboard", use_container_width=True):
            go_to("dashboard")

    # --- BLOCK 1: Patient Details ---
    st.markdown("### 🧑‍🤝‍🧑 Patient Details")
    with st.container(border=True):
        patient_name = st.text_input(
            "Patient Name",
            value=st.session_state.get("patient_name", ""),
            placeholder="e.g. Sunita Devi",
            key="patient_name_input",
        )
        patient_name = patient_name.strip()

        category_display = st.radio(
            "Who is this visit for?",
            list(PATIENT_CATEGORY_OPTIONS.keys()),
            horizontal=True,
            key="patient_category_input",
        )
        patient_category = PATIENT_CATEGORY_OPTIONS[category_display]

        worker_gestational_weeks = None
        worker_child_age_months = None

        if patient_category == "PREGNANT_WOMAN":
            worker_gestational_weeks = st.number_input(
                "Weeks of pregnancy (if known — leave at 0 if unknown)",
                min_value=0, max_value=45, value=0, step=1,
            )
            worker_gestational_weeks = worker_gestational_weeks or None
        elif patient_category == "CHILD":
            worker_child_age_months = st.number_input(
                "Child's age in months (if known — leave at 0 if unknown)",
                min_value=0, max_value=71, value=0, step=1,
            )
            worker_child_age_months = worker_child_age_months or None

    # --- BLOCK 2: Patient Case Capture (the ASHA-patient conversation) ---
    st.markdown("### 🎙️ Patient Case Capture — Conversation")
    with st.container(border=True):
        mode = st.radio(
            "Input method — records the ASHA–patient conversation",
            ["Type note manually", "Record live voice note"],
            horizontal=True,
        )

        raw_text = ""
        audio_path = None

        if mode == "Type note manually":
            raw_text = st.text_area(
                "Observation Notes",
                placeholder="e.g. Pregnant woman with high fever, missing ANC checkup, took 2 ORS kits from the drug log.",
                height=100,
            )
        else:
            audio_file = st.audio_input("Tap microphone to record the ASHA–patient conversation")
            if audio_file is not None:
                st.session_state["cached_audio_bytes"] = audio_file.read()

            if "cached_audio_bytes" in st.session_state:
                tmp_dir = tempfile.gettempdir()
                stable_audio_path = os.path.join(tmp_dir, "asha_live_speech.wav")
                with open(stable_audio_path, "wb") as f:
                    f.write(st.session_state["cached_audio_bytes"])
                audio_path = stable_audio_path
                st.audio(st.session_state["cached_audio_bytes"], format="audio/wav")
                st.caption("This recording is converted to text automatically once you run the workflow below.")

    # --- BLOCK 3: MUAC Band Analysis (separate block, mainly for children) ---
    st.markdown("### 📸 MUAC Band Analysis (Child Malnutrition Check)")
    muac_image_path = None
    with st.container(border=True):
        if patient_category != "CHILD":
            st.caption("Optional — mainly used for child patients. Switch Patient Details to 'Child' to prioritize this.")
        enable_camera = st.checkbox("Enable MUAC Scanner Camera")
        if enable_camera:
            img_file = st.camera_input("Position the MUAC band clearly in frame")
            if img_file is not None:
                tmp_dir = tempfile.gettempdir()
                stable_img_path = os.path.join(tmp_dir, "muac_snapshot.jpg")
                with open(stable_img_path, "wb") as f:
                    f.write(img_file.getbuffer())
                muac_image_path = stable_img_path

    # --- RUN WORKFLOW ---
    st.markdown("### 🚀 Run Agent Workflow")
    button_disabled = not patient_name
    if not patient_name:
        st.info("ℹ️ Enter the patient's name above to enable the workflow.")

    raw_text_stripped = raw_text.strip() if raw_text else ""

    if st.button("Analyze & Run Agent Workflow", type="primary", use_container_width=True, disabled=button_disabled):
        if not raw_text_stripped and not audio_path and not muac_image_path:
            st.error("Cannot submit blank data. Please type text notes, record voice input, or supply an alignment image first.")
        else:
            initial_state = {
                "session_id": str(uuid.uuid4()),
                "asha_worker_id": st.session_state.worker_id,
                "village": st.session_state.village,
                "patient_name": patient_name,
                "patient_category": patient_category,
                "worker_child_age_months": worker_child_age_months,
                "worker_gestational_age_weeks": worker_gestational_weeks,
                "input_mode": "audio" if (mode == "Record live voice note" and audio_path) else "text",
                "raw_text": raw_text_stripped,
                "raw_audio_path": audio_path,
                "muac_image_path": muac_image_path,
                "errors": [],
            }

            with st.spinner("Processing speech, textual details, and vision metrics with Gemini AI..."):
                try:
                    final_state = asha_agent_graph.invoke(initial_state)
                except Exception as e:
                    st.error(f"Agent workflow failed to complete: {str(e)}")
                    final_state = None

                if final_state and "risk_assessment" in final_state and "patient_type" in final_state:
                    generated_abha_ref = f"ABHA-{str(uuid.uuid4())[:8].upper()}"
                    try:
                        db.save_record(
                            generated_abha_ref,
                            final_state.get("patient_name"),
                            final_state["patient_type"],
                            final_state["risk_assessment"],
                            (final_state.get("follow_up_plan") or {}).get("follow_up_date"),
                            patient_category=final_state.get("patient_category"),
                            recorded_by_worker=final_state.get("asha_worker_id"),
                            village=final_state.get("village"),
                            guidance_text_en=final_state.get("guidance_text_en"),
                            guidance_text_local=final_state.get("guidance_text_local"),
                            detailed_report=final_state.get("detailed_report"),
                        )
                        final_state["abha_ref"] = generated_abha_ref
                    except Exception as e:
                        st.warning(f"Record processed but could not be saved to history: {str(e)}")

            if final_state:
                st.session_state["last_result"] = final_state
                st.session_state.pop("cached_audio_bytes", None)
                st.session_state["records"] = db.fetch_all_records()
                go_to("report")


# ---------------------------------------------------------------------------
# PAGE: REPORT — conversation as text, risk/guidance, required fill-ups, print
# ---------------------------------------------------------------------------
def render_report_page():
    result = st.session_state.get("last_result")

    col_title, col_nav = st.columns([4, 1])
    with col_title:
        st.title("📄 Visit Report")
    with col_nav:
        st.write("")
        if st.button("← Dashboard", use_container_width=True):
            go_to("dashboard")

    if not result:
        st.info("No report to show yet. Start a new patient case first.")
        if st.button("➕ New Patient Case"):
            go_to("new_case")
        return

    display_name = result.get("patient_name") or "the patient"
    category_label = PATIENT_CATEGORY_LABELS.get(result.get("patient_category"), "")
    st.success(f"Triage Evaluation Complete for {display_name}" + (f" ({category_label})" if category_label else "") + ".")
    if result.get("abha_ref"):
        st.caption(f"ABHA Reference: `{result['abha_ref']}`")

    # --- Conversation recorded, shown as text ---
    st.subheader("🗣️ Conversation Recorded (Converted to Text)")
    st.write(result.get("translated_text_en") or "No conversation text was captured for this visit.")
    st.caption(f"Detected input: {result.get('detected_language', 'Unknown')}")

    # --- Risk assessment ---
    risk = result.get("risk_assessment", {})
    level = risk.get("risk_level", "LOW")
    color = {"LOW": "green", "MODERATE": "orange", "HIGH": "red", "URGENT_REFERRAL": "red"}.get(level, "red")
    st.markdown(f"### Assessment Outcome: :{color}[{level}]")
    for reason in risk.get("reasons", []):
        st.write(f"- {reason}")

    # --- Guidance ---
    st.subheader("✅ Action Steps for ASHA Worker")
    tab_en, tab_local = st.tabs(["English Guidance Instructions", "Hindi / Local Dialect Translation"])
    with tab_en:
        st.info(result.get("guidance_text_en", "No guidance text received."))
    with tab_local:
        st.info(result.get("guidance_text_local", "कोई निर्देश उपलब्ध नहीं है।"))

    # --- Required fill-ups / review ---
    fill_ups = result.get("required_fill_ups") or []
    if fill_ups:
        st.subheader("📝 Required Fill-Ups & Review")
        for item in fill_ups:
            st.warning(item)
    else:
        st.caption("✔️ No missing fields flagged for this visit.")

    # --- Vaccine / ANC schedule check ---
    schedule_result = result.get("schedule_check_result")
    if schedule_result:
        st.subheader("💉 Vaccine / ANC Schedule Check")
        if schedule_result.get("items_due"):
            st.warning("**Due / Overdue:** " + ", ".join(schedule_result["items_due"]))
        if schedule_result.get("items_completed"):
            st.caption("Completed: " + ", ".join(schedule_result["items_completed"]))
        for note in schedule_result.get("notes", []):
            st.caption(f"ℹ️ {note}")

    # --- Follow-up + SMS reminder ---
    plan = result.get("follow_up_plan")
    sms = result.get("sms_reminder_status")
    if plan or sms:
        st.subheader("📅 Follow-Up & Reminder")
        if plan:
            st.write(f"**Next visit auto-scheduled:** {plan.get('follow_up_date')}  ({plan.get('reason')})")
        if sms:
            sms_status = sms.get("status")
            if sms_status == "SIMULATED_SENT":
                st.caption(f"📱 SMS reminder prepared (simulated — no gateway configured): {sms.get('message')}")
            else:
                st.caption(f"📱 SMS reminder status: {sms_status}")

    if result.get("muac_result"):
        with st.expander("🔍 Diagnostics: MUAC Image Verification Metrics"):
            st.json(result["muac_result"])

    if result.get("maternal_risk_result"):
        with st.expander("🔍 Diagnostics: Extracted Text Risk Markers"):
            st.json(result["maternal_risk_result"])

    with st.expander("🗂️ Unified Patient Metadata (ABHA System Output Payload)"):
        metadata_payload = {
            "administrative_metadata": {
                "abha_ref": result.get("abha_ref"),
                "patient_name": result.get("patient_name"),
                "patient_category": result.get("patient_category"),
                "managed_by_worker": result.get("asha_worker_id"),
                "registered_village": result.get("village"),
                "detected_language": result.get("detected_language"),
                "translated_text_en": result.get("translated_text_en"),
                "abha_sync_status": result.get("abha_sync_status"),
            },
            "extracted_multi_domain_records": result.get("unified_metadata", {}),
        }
        st.json(metadata_payload)

    # --- Printable / downloadable detailed report ---
    if result.get("detailed_report"):
        st.subheader("🖨️ Detailed Visit Report")
        st.text_area("Report Preview", result["detailed_report"], height=280, key="report_preview")
        safe_name = (result.get("patient_name") or "patient").replace(" ", "_")
        col_print, col_download = st.columns(2)
        with col_print:
            print_button()
        with col_download:
            st.download_button(
                label="⬇️ Download Report (.txt)",
                data=result["detailed_report"],
                file_name=f"ASHA_report_{safe_name}_{result.get('abha_ref', 'record')}.txt",
                mime="text/plain",
                use_container_width=True,
            )

    if result.get("errors"):
        with st.expander("⚠️ Backend Execution Warning Logs"):
            for error_log in result["errors"]:
                st.write("-", error_log)

    st.divider()
    if st.button("➕ Start Another Patient Case", type="primary", use_container_width=True):
        go_to("new_case")


# ---------------------------------------------------------------------------
# ROUTE
# ---------------------------------------------------------------------------
if st.session_state.page == "login" or "worker_id" not in st.session_state:
    render_login_page()
elif st.session_state.page == "dashboard":
    render_dashboard_page()
elif st.session_state.page == "new_case":
    render_new_case_page()
elif st.session_state.page == "report":
    render_report_page()
else:
    render_dashboard_page()
