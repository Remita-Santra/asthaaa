import os
import json
import uuid
import tempfile
import streamlit as st
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
        client.models.generate_content(
            model="gemini-3.5-flash",
            contents="ping",
        )
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

# --- CUSTOM CSS FOR PINK, MAROON, AND CREAM THEME ---
st.markdown("""
    <style>
        .stApp {
            background-color:#FFFDF9;
            color: #5A0E1A;
        }
        h1, h2, h3, h4, h5, h6, label, .stMarkdown p {
            color: #5A0E1A !important;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        div[data-testid="stContainerBorder"] {
            background-color: #FFF0F2 !important;
            border: 1px solid #C48B95 !important;
            border-radius: 10px;
            padding: 1.5rem;
        }
        button[data-testid="baseButton-primary"] {
            background-color: #5A0E1A !important;
            color: #FFFDF9 !important;
            border: 2px solid #5A0E1A !important;
            border-radius: 8px !important;
            transition: all 0.3s ease;
        }
        button[data-testid="baseButton-primary"]:hover {
            background-color: #D34E65 !important;
            color: #FFFDF9 !important;
            border-color: #D34E65 !important;
        }
        button[data-testid="baseButton-secondary"] {
            background-color: #FFFDF9 !important;
            color: #5A0E1A !important;
            border: 1px solid #C48B95 !important;
        }
        section[data-testid="stSidebar"] {
            background-color: #FFF5F6 !important;
            border-right: 1px solid #E8C1C7;
        }
        .stAlert {
            background-color: #FFF0F2 !important;
            color: #5A0E1A !important;
            border-left: 5px solid #D34E65 !important;
        }
    </style>
""", unsafe_allow_html=True)


st.title("ASHTHA FOR ASHA")
st.caption(
    "Turn every ASHA home visit into a fully automated health record — "
    "voice-first capture, auto-filled forms, vaccine/ANC schedule checks, "
    "and an auto-scheduled follow-up. No typing, no paper forms, no delays."
)

# --- SECTION 1: AUTHENTICATION ---
st.markdown("### 📋 ASHA Worker Authentication")
with st.container(border=True):
    col1, col2 = st.columns(2)
    with col1:
        worker_id = st.text_input(
            "Enter ASHA Worker ID",
            value=st.session_state.get("worker_id", ""),
            placeholder="e.g. ASHA-WEST-4092",
            key="worker_id_input"
        )
    with col2:
        village = st.text_input(
            "Assigned Village / Settlement",
            value=st.session_state.get("village", ""),
            placeholder="e.g. Rampur Sub-Center",
            key="village_input"
        )

    worker_id = worker_id.strip()
    village = village.strip()

    if not worker_id or not village:
        st.warning("⚠️ Please provide both your Worker ID and Village details to activate the agent workflow below.")
    else:
        st.success(f"Verified Session: Active log session for Worker {worker_id} at {village}.")

with st.sidebar:
    st.header("Sync History Log")
    if st.button("Refresh Historical Records", type="secondary"):
        st.session_state["records"] = db.fetch_all_records()

    records = st.session_state.get("records", [])
    if records:
        for rec in records:
            raw_risk = rec.get('risk_assessment')
            risk_level = "—"

            # 1. Handle if it's already a native dict/list
            if isinstance(raw_risk, dict):
                risk_level = raw_risk.get('risk_level', '—')
            elif isinstance(raw_risk, list) and raw_risk:
                # If it's a list, look inside the first element if it's a dict
                first_item = raw_risk[0]
                risk_level = first_item.get('risk_level', '—') if isinstance(first_item, dict) else '—'

            # 2. Handle if it's stored as a JSON string
            elif isinstance(raw_risk, str) and raw_risk.strip():
                try:
                    parsed_risk = json.loads(raw_risk)

                    if isinstance(parsed_risk, dict):
                        risk_level = parsed_risk.get('risk_level', '—')
                    elif isinstance(parsed_risk, list) and parsed_risk:
                        # Safely unpack the first element out of the parsed list
                        first_item = parsed_risk[0]
                        risk_level = first_item.get('risk_level', '—') if isinstance(first_item, dict) else '—'
                except json.JSONDecodeError:
                    risk_level = "—"

            # 3. Render output cleanly
            patient_name = rec.get('patient_name', 'Unknown')
            follow_up = rec.get('follow_up_date')
            follow_up_str = f" · next visit {follow_up}" if follow_up else ""
            st.write(
                f"**{rec.get('abha_ref', 'N/A')}** · {patient_name} · "
                f"{rec.get('patient_type', 'Unknown')} · {risk_level}{follow_up_str}"
            )
    else:
        st.caption("No historical records fetched for this session.")

# --- SECTION 2: PATIENT CASE ENCOUNTER DATA ENTRY ---
st.markdown("### 📝 Patient Case Capture")

patient_name = st.text_input(
    "Patient Name",
    value=st.session_state.get("patient_name", ""),
    placeholder="e.g. Sunita Devi",
    key="patient_name_input",
    help="Whose visit is this? If left blank, the agent will try to pick a name out of the note/recording.",
)
patient_name = patient_name.strip()

mode = st.radio(
    "Input method — records the ASHA–patient conversation",
    ["Type note manually", "Record live voice note"],
    horizontal=True,
)

raw_text = ""
audio_path = None
muac_image_path = None

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

st.markdown("### 📸 MUAC Band Image Capture (Optional)")
enable_camera = st.checkbox("Toggle Child Malnutrition Scanner")

if enable_camera:
    img_file = st.camera_input("Position the MUAC band clearly in frame")
    if img_file is not None:
        tmp_dir = tempfile.gettempdir()
        stable_img_path = os.path.join(tmp_dir, "muac_snapshot.jpg")
        with open(stable_img_path, "wb") as f:
            f.write(img_file.getbuffer())
        muac_image_path = stable_img_path

# --- SECTION 3: PIPELINE INVOCATION AND RESPONSE VISUALIZATION ---
# Patient name is required — every saved record needs to be attributable
# to a specific patient, not just an ASHA worker/village pair.
button_disabled = not worker_id or not village or not patient_name

if not patient_name and worker_id and village:
    st.info("ℹ️ Enter the patient's name above to enable the workflow.")

# Trim whitespace-only notes so they don't slip past the "blank data" check below.
raw_text_stripped = raw_text.strip() if raw_text else ""

if st.button("Analyze & Run Agent Workflow", type="primary", use_container_width=True, disabled=button_disabled):
    if not raw_text_stripped and not audio_path and not muac_image_path:
        st.error("Cannot submit blank data. Please type text notes, record voice input, or supply an alignment image first.")
    else:
        initial_state = {
            "session_id": str(uuid.uuid4()),
            "asha_worker_id": worker_id,
            "village": village,
            "patient_name": patient_name,
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

            generated_abha_ref = None
            if final_state and "risk_assessment" in final_state and "patient_type" in final_state:
                generated_abha_ref = f"ABHA-{str(uuid.uuid4())[:8].upper()}"
                try:
                    db.save_record(
                        generated_abha_ref,
                        final_state.get("patient_name"),
                        final_state["patient_type"],
                        final_state["risk_assessment"],
                        (final_state.get("follow_up_plan") or {}).get("follow_up_date"),
                    )
                    final_state["abha_ref"] = generated_abha_ref
                except Exception as e:
                    st.warning(f"Record processed but could not be saved to history: {str(e)}")

        if final_state:
            st.session_state["last_result"] = final_state

# Render output cards dynamically
result = st.session_state.get("last_result")
if result:
    display_name = result.get("patient_name") or "the patient"
    st.success(f"Triage Evaluation Complete for {display_name}.")

    risk = result.get("risk_assessment", {})
    level = risk.get("risk_level", "LOW")
    color = {"LOW": "green", "MODERATE": "orange", "HIGH": "red", "URGENT_REFERRAL": "red"}.get(level, "red")

    st.markdown(f"### Assessment Outcome: :{color}[{level}]")
    for reason in risk.get("reasons", []):
        st.write(f"- {reason}")

    st.subheader("Action Steps for ASHA Worker")
    tab_en, tab_local = st.tabs(["English Guidance Instructions", "Hindi / Local Dialect Translation"])
    with tab_en:
        st.info(result.get("guidance_text_en", "No guidance text received."))
    with tab_local:
        st.info(result.get("guidance_text_local", "कोई निर्देश उपलब्ध नहीं है।"))

    # --- Vaccine / ANC Schedule Check ---
    schedule_result = result.get("schedule_check_result")
    if schedule_result:
        st.subheader("💉 Vaccine / ANC Schedule Check")
        if schedule_result.get("items_due"):
            st.warning("**Due / Overdue:** " + ", ".join(schedule_result["items_due"]))
        if schedule_result.get("items_completed"):
            st.caption("Completed: " + ", ".join(schedule_result["items_completed"]))
        for note in schedule_result.get("notes", []):
            st.caption(f"ℹ️ {note}")

    # --- Follow-Up + SMS Reminder ---
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
                "managed_by_worker": result.get("asha_worker_id"),
                "registered_village": result.get("village"),
                "detected_language": result.get("detected_language"),
                "translated_text_en": result.get("translated_text_en"),
                "abha_sync_status": result.get("abha_sync_status")
            },
            "extracted_multi_domain_records": result.get("unified_metadata", {})
        }
        st.json(metadata_payload)

    # --- Detailed, printable/downloadable report ---
    if result.get("detailed_report"):
        st.subheader("🖨️ Detailed Visit Report")
        st.text_area("Report Preview", result["detailed_report"], height=280, key="report_preview")
        safe_name = (result.get("patient_name") or "patient").replace(" ", "_")
        st.download_button(
            label="⬇️ Download Detailed Report (.txt)",
            data=result["detailed_report"],
            file_name=f"ASHA_report_{safe_name}_{result.get('abha_ref', 'record')}.txt",
            mime="text/plain",
            use_container_width=True,
        )

    if result.get("errors"):
        with st.expander("⚠️ Backend Execution Warning Logs"):
            for error_log in result["errors"]:
                st.write("-", error_log)
