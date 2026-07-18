import json
import uuid
import tempfile
import os
import streamlit as st

# app_streamlit.py
import os
from dotenv import load_dotenv
load_dotenv()  # 👈 ADDS THE API KEY TO ENVIRONMENT MEMORY BEFORE THE APP RUNS

import json
import uuid
import tempfile
import streamlit as st

from graph import asha_agent_graph
import db

# ... Rest of your app_streamlit.py code remains exactly the same ...

from graph import asha_agent_graph
import db

st.set_page_config(page_title="ASHA Digital Assistant", page_icon="🩺", layout="centered")

st.title("🩺 ASHTHA FOR ASHA ")
st.caption("Your very owm Digital Assistant!")

with st.sidebar:
    st.header("Worker details")
    worker_id = st.text_input("ASHA Worker ID", value="ASHA-DEMO-001")
    village = st.text_input("Village", value="Demo Village")
    st.divider()
    st.header("Past synced records")
    if st.button("Refresh records"):
        st.session_state["records"] = db.fetch_all_records()
    for rec in st.session_state.get("records", []):
        st.write(f"**{rec['abha_ref']}** · {rec['patient_type']} · "
                 f"{json.loads(rec['risk_assessment'] or '{}').get('risk_level', '—')}")

st.subheader("1. Field note capture")
mode = st.radio("Input method", ["Type note", "Record live voice note"], horizontal=True)

raw_text = ""
audio_path = None

if mode == "Type note":
    raw_text = st.text_area(
        "Note (any Indian language is fine)",
        placeholder="e.g. Garbhwati mahila, 32 saptah, BP 150/95, hemoglobin 9.2 hai",
        height=100,
    )
else:
    audio_file = st.st.audio_input("Tap microphone to record patient notes")
    if audio_file is not None:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp.write(audio_file.read())
        tmp.close()
        audio_path = tmp.name

st.subheader("2. MUAC tape snapshot (only needed for children under 5)")
muac_file = st.camera_input("Line up the tape measure in the camera viewfinder")
muac_image_path = None

if muac_file is not None:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    tmp.write(muac_file.read())
    tmp.close()
    muac_image_path = tmp.name

st.divider()

if st.button("Run agent", type="primary", use_container_width=True):
    if not raw_text and not audio_path and not muac_image_path:
        st.error("Please record audio, write notes, or snap a photo before invoking the pipeline.")
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
        with st.spinner("Analyzing live multi-modal assets with Gemini backend..."):
            final_state = asha_agent_graph.invoke(initial_state)
            
            # Persist it into local mock ledger historical logging
            if "risk_assessment" in final_state and "patient_type" in final_state:
                db.save_record(
                    f"ABHA-{str(uuid.uuid4())[:8].upper()}", 
                    final_state["patient_type"], 
                    final_state["risk_assessment"]
                )
                
        st.session_state["last_result"] = final_state

result = st.session_state.get("last_result")
if result:
    st.success("Done — Multi-modal data evaluated successfully.")

    risk = result.get("risk_assessment", {})
    level = risk.get("risk_level", "—")
    color = {"LOW": "green", "MODERATE": "orange", "HIGH": "red", "URGENT_REFERRAL": "red"}.get(level, "gray")
    st.markdown(f"### Risk level: :{color}[{level}]")
    for reason in risk.get("reasons", []):
        st.write(f"- {reason}")

    st.subheader("Guidance to read aloud")
    tab_en, tab_local = st.tabs(["English", "Hindi Translation"])
    with tab_en:
        st.info(result.get("guidance_text_en", "No text compiled."))
    with tab_local:
        st.info(result.get("guidance_text_local", "कोई निर्देश उपलब्ध नहीं है।"))

    if result.get("muac_result"):
        st.subheader("Image Analysis: MUAC nutrition screening")
        st.json(result["muac_result"])

    if result.get("maternal_risk_result"):
        st.subheader("Text/Voice Analysis: Maternal risk flags")
        st.json(result["maternal_risk_result"])

    with st.expander("Full structured record (for ABHA / supervisor review)"):
        st.json({
            "detected_language": result.get("detected_language"),
            "translated_text_en": result.get("translated_text_en"),
            "patient_record": result.get("patient_record"),
            "extracted_vitals": result.get("extracted_vitals"),
            "abha_sync_status": result.get("abha_sync_status"),
        })

    if result.get("errors"):
        with st.expander("⚠️ Pipeline Traces"):
            for e in result["errors"]:
                st.write("-", e)
