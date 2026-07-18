# app.py
import os
from dotenv import load_dotenv

# 1. Force explicit environment variable mapping from your local config file
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, '.env')
load_dotenv(dotenv_path=env_path)

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
            background-color: #FFFDF9;
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


st.title("🩺 ASHA Digital Assistant")
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

if mode == "Type note manually":
    raw_text = st.text_area(
        "Observation Notes (Mixing regional words/Hindi with English is perfectly supported)",
        placeholder="e.g. Garbhwati mahila, age 28, severe headache, BP 145/95, hemoglobin 9.5 hai",
        height=100,
    )
else:
    audio_file = st.audio_input("Tap microphone to record patient vocal symptoms")
    if audio_file is not None:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp.write(audio_file.read())
        tmp.close()
        audio_path = tmp.name

st.markdown("### 📸 Clinical Image Capture (Optional)")
muac_file = st.camera_input("Line up the child's MUAC tape boundary markers in the viewfinder")
muac_image_path = None

if muac_file is not None:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    tmp.write(muac_file.read())
    tmp.close()
    muac_image_path = tmp.name

st.divider()

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
                db.save_record(
                    f"ABHA-{str(uuid.uuid4())[:8].upper()}", 
                    final_state["patient_type"], 
                    final_state["risk_assessment"]
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

    with st.expander("🗂️ Unified Patient Metadata (ABHA System Output Payload)"):
        st.json({
            "managed_by_worker": result.get("asha_worker_id"),
            "registered_village": result.get("village"),
            "detected_language": result.get("detected_language"),
            "translated_text_en": result.get("translated_text_en"),
            "extracted_vitals": result.get("extracted_vitals"),
            "abha_sync_status": result.get("abha_sync_status"),
        })

    if result.get("errors"):
        with st.expander("⚠️ Backend Execution Warning Logs"):
            for error_log in result["errors"]:
                st.write("-", error_log)
