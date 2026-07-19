import os
import streamlit as st
from dotenv import load_dotenv
from google import genai

load_dotenv()

# Try loading from standard environment, fallback to Streamlit secrets
api_key = os.environ.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")

if not api_key:
    raise ValueError("Missing key! Please set GEMINI_API_KEY in your cloud secrets dashboard.")

client = genai.Client(api_key=api_key)


import json
import uuid
import tempfile
import streamlit as st

from graph import asha_agent_graph
import db

st.set_page_config(page_title="ASHTHA FOR ASHA", page_icon="🩺", layout="centered")

# --- CUSTOM CSS FOR PINK, MAROON, AND CREAM THEME ---
st.markdown("""
    <style>
        /* Base application background (Cream) and main text (Maroon) */
        .stApp {
            background-color:#FFFDF9;
            color: #5A0E1A;
        }
        
        /* Headers and subheaders (Deep Maroon) */
        h1, h2, h3, h4, h5, h6, label, .stMarkdown p {
            color: #5A0E1A !important;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        /* Container boxes styling (Creamy Pink background with Maroon border) */
        div[data-testid="stContainerBorder"] {
            background-color: #FFF0F2 !important;
            border: 1px solid #C48B95 !important;
            border-radius: 10px;
            padding: 1.5rem;
        }
        
        /* Primary Run Action Button styling (Maroon back, Pink accent on text) */
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

        /* Secondary actions button styling (Pink outline) */
        button[data-testid="baseButton-secondary"] {
            background-color: #FFFDF9 !important;
            color: #5A0E1A !important;
            border: 1px solid #C48B95 !important;
        }
        button[data-testid="baseButton-secondary"]:hover {
            background-color: #FFF0F2 !important;
            border-color: #5A0E1A !important;
        }

        /* Sidebar container custom overrides */
        section[data-testid="stSidebar"] {
            background-color: #FFF5F6 !important;
            border-right: 1px solid #E8C1C7;
        }

        /* Info boxes (Subtle pinkish accent backdrops) */
        .stAlert {
            background-color: #FFF0F2 !important;
            color: #5A0E1A !important;
            border-left: 5px solid #D34E65 !important;
        }
    </style>
""", unsafe_allow_html=True)


st.title("ASHTHA FOR ASHA")
st.caption("Live multimodal capture using unified voice processing, text, and image analysis.")

# --- SECTION 1: USER INPUT FOR WORKER DETAILS ---
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
    
    # Simple validation visual indicator
    if not worker_id or not village:
        st.warning("⚠️ Please provide both your Worker ID and Village details to activate the agent workflow below.")
    else:
        st.success(f"Verified Session: Active log session for Worker {worker_id} at {village}.")

# Keep track of records in a sidebar ledger panel
with st.sidebar:
    st.header("Sync History Log")
    if st.button("Refresh Historical Records"):
        st.session_state["records"] = db.fetch_all_records()
    
    records = st.session_state.get("records", [])
    if records:
        for rec in records:
            st.write(f"**{rec['abha_ref']}** · {rec['patient_type']} · "
                     f"{json.loads(rec['risk_assessment'] or '{}').get('risk_level', '—')}")
    else:
        st.caption("No historical records fetched for this session.")

# --- SECTION 2: PATIENT CASE ENCOUNTER DATA ENTRY ---

st.markdown("### 📝 Patient Case Capture")
mode = st.radio("Input method for case description", ["Type note manually", "Record live voice note"], horizontal=True)

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
    audio_file = st.audio_input("Tap microphone to record patient vocal symptoms")
    
    if audio_file is not None:
        st.session_state["cached_audio_bytes"] = audio_file.read()
        
    if "cached_audio_bytes" in st.session_state:
        tmp_dir = tempfile.gettempdir()
        stable_audio_path = os.path.join(tmp_dir, "asha_live_speech.wav")
        with open(stable_audio_path, "wb") as f:
            f.write(st.session_state["cached_audio_bytes"])
        audio_path = stable_audio_path
        st.audio(st.session_state["cached_audio_bytes"], format="audio/wav")

# --- CAMERA SCANNING COMPONENT FOR MUAC ---
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
button_disabled = not worker_id or not village

if st.button("Analyze & Run Agent Workflow", type="primary", use_container_width=True, disabled=button_disabled):
    if not raw_text and not audio_path and not muac_image_path:
        st.error("Cannot submit blank data. Please type text notes, record voice input, or supply an alignment image first.")
    else:
        initial_state = {
            "session_id": str(uuid.uuid4()),
            "asha_worker_id": worker_id,
            "village": village,
            "input_mode": "audio" if audio_path else "text",
            "raw_text": raw_text,
            "raw_audio_path": audio_path,
            "muac_image_path": muac_image_path,
            "errors": [],
        }
        
        with st.spinner("Processing speech, textual details, and vision metrics with Gemini AI..."):
            final_state = asha_agent_graph.invoke(initial_state)
            
            if "risk_assessment" in final_state and "patient_type" in final_state:
                # Format to persist triage record safely
                db.save_record(
                    f"ABHA-{str(uuid.uuid4())[:8].upper()}", 
                    final_state["patient_type"], 
                    json.dumps(final_state["risk_assessment"]) if isinstance(final_state["risk_assessment"], dict) else final_state["risk_assessment"]
                )
                
        st.session_state["last_result"] = final_state

# Render output cards dynamically based on state outcomes
result = st.session_state.get("last_result")
if result:
    st.success("Triage Evaluation Complete.")

    risk = result.get("risk_assessment", {})
    level = risk.get("risk_level", "LOW")
    color = {"LOW": "green", "MODERATE": "orange", "HIGH": "red", "URGENT_REFERRAL": "red"}.get(level, "gray")
    
    st.markdown(f"### Assessment Outcome: :{color}[{level}]")
    for reason in risk.get("reasons", []):
        st.write(f"- {reason}")

    st.subheader("Action Steps for ASHA Worker")
    tab_en, tab_local = st.tabs(["English Guidance Instructions", "Hindi / Local Dialect Translation"])
    with tab_en:
        st.info(result.get("guidance_text_en", "No guidance text received."))
    with tab_local:
        st.info(result.get("guidance_text_local", "कोई निर्देश उपलब्ध नहीं है।"))

    if result.get("muac_result"):
        with st.expander("🔍 Diagnostics: MUAC Image Verification Metrics"):
            st.json(result["muac_result"])

    if result.get("maternal_risk_result"):
        with st.expander("🔍 Diagnostics: Extracted Text Risk Markers"):
            st.json(result["maternal_risk_result"])

    # --- UPDATED: Render the full expanded 7-domain data payload payload structure ---
    with st.expander("🗂️ Unified Patient Metadata (ABHA System Output Payload)"):
        metadata_payload = {
            "administrative_metadata": {
                "managed_by_worker": result.get("asha_worker_id"),
                "registered_village": result.get("village"),
                "detected_language": result.get("detected_language"),
                "translated_text_en": result.get("translated_text_en"),
                "abha_sync_status": result.get("abha_sync_status")
            },
            "extracted_multi_domain_records": result.get("unified_metadata", {})
        }
        st.json(metadata_payload)

    if result.get("errors"):
        with st.expander("⚠️ Backend Execution Warning Logs"):
            for error_log in result["errors"]:
                st.write("-", error_log)
