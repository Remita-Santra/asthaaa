import os
import json
import time
from typing import Dict, Any
from google import genai
from google.genai import types
from state import ASHAAgentState

# nodes.py
import os
import json
import time
from typing import Dict, Any
from dotenv import load_dotenv

# Force load_dotenv to look in the exact directory of this specific file
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, '.env')
load_dotenv(dotenv_path=env_path) 

from google import genai
from google.genai import types
from state import ASHAAgentState

# Fallback check: If environment lookup still fails, read it or pass it directly
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    # ⚠️ TEMPORARY SAFEGUARD: If your .env file is completely broken or unreadable,
    # you can paste your key directly here to get unblocked right away:
    api_key = "AIzaSy..." 

# Initialize the client explicitly passing your key variable
client = genai.Client(api_key=api_key)

# nodes.py
import os
import json
import time
from typing import Dict, Any
from dotenv import load_dotenv

# Initialize environment config variables
load_dotenv()  # 👈 ENSURES THE KEY IS LOADED DIRECTLY FOR THE GENAI ENGINE

from google import genai
from google.genai import types
from state import ASHAAgentState

# The unified client can now read the injected variable seamlessly
client = genai.Client()

# ... Rest of your nodes.py code remains exactly the same ...

# Initialize the unified GenAI Client (Picks up GEMINI_API_KEY from environment)
client = genai.Client()

def ingest_node(state: ASHAAgentState) -> Dict[str, Any]:
    """
    Natively uploads voice notes or relies on typed notes. 
    Uses Gemini to handle transcription and translation cleanly in one step.
    """
    errors = list(state.get("errors", []))
    raw_text = state.get("raw_text", "")
    audio_path = state.get("raw_audio_path")
    
    translated_en = ""
    detected_language = "English"

    # If audio is present, use Gemini to transcribe it natively
    # --- nodes.py ingest_node UPDATE ---
# --- nodes.py ingest_node UPDATE ---
if state.get("input_mode") == "audio" and audio_path and os.path.exists(audio_path):
    try:
        print(f"[Nodes] Uploading stabilized voice notes track: {audio_path}")
        
        # Explicitly define the audio standard MIME payload parameter
        uploaded_audio = client.files.upload(
            file=audio_path,
            config=types.UploadFileConfig(mime_type="audio/wav")
        )
        
        # Pool status explicitly before executing content extraction layers
        while uploaded_audio.state.name == "PROCESSING":
            time.sleep(1)
            uploaded_audio = client.files.get(name=uploaded_audio.name)
            
        if uploaded_audio.state.name == "FAILED":
            raise Exception("Google API processing layer failed to decode file structure.")

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                "Transcribe this voice note spoken by a community health worker. "
                "Provide the output translated completely into standard English.", 
                uploaded_audio
            ]
        )
        translated_en = response.text.strip()
        detected_language = "Detected from Audio"
        
        # Remove deletion or introduce short buffer sleep to avoid server side clipping
        time.sleep(0.5) 
        client.files.delete(name=uploaded_audio.name)
        
    except Exception as e:
        errors.append(f"Audio processing error: {str(e)}")
        translated_en = f"Audio processing failed runtime tracking: {str(e)}"
    else:
        # If text is typed, use a quick LLM call to translate it to English if needed
        if raw_text:
            try:
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=f"Translate this Indian healthcare field note precisely into English: {raw_text}"
                )
                translated_en = response.text.strip()
                detected_language = "Typed Input"
            except Exception as e:
                errors.append(f"Translation error: {str(e)}")
                translated_en = raw_text

    return {
        "translated_text_en": translated_en, 
        "detected_language": detected_language, 
        "errors": errors
    }


def language_and_translate_node(state: ASHAAgentState) -> Dict[str, Any]:
    """Pass-through node since translation is safely bundled with ingest."""
    return {}


def extract_vitals_node(state: ASHAAgentState) -> Dict[str, Any]:
    """
    Uses structured schema parsing to extract medical vitals from text fields.
    """
    translated_text = state.get("translated_text_en", "")
    errors = list(state.get("errors", []))
    
    # Force Structured JSON Extraction from text
    prompt = f"""
    Analyze this English medical note and extract vitals or records into JSON format.
    
    Note: "{translated_text}"
    
    Respond STRICTLY with a valid JSON object matching this structure:
    {{
        "patient_type": "MATERNAL" or "CHILD" or "GENERAL",
        "extracted_vitals": {{
            "systolic": integer or null,
            "diastolic": integer or null,
            "hemoglobin": float or null,
            "gestational_age_weeks": integer or null
        }},
        "patient_record": {{
            "conditions": ["list", "of", "conditions"]
        }}
    }}
    """
    
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )
        extracted_data = json.loads(response.text.strip())
    except Exception as e:
        errors.append(f"Structured vitals extraction failed: {str(e)}")
        extracted_data = {
            "patient_type": "GENERAL",
            "extracted_vitals": {"systolic": None, "diastolic": None, "hemoglobin": None, "gestational_age_weeks": None},
            "patient_record": {"conditions": []}
        }
        
    # Contextual type override if a MUAC photo was taken
    if state.get("muac_image_path"):
        extracted_data["patient_type"] = "CHILD"

    return {
        "patient_type": extracted_data.get("patient_type", "GENERAL"),
        "extracted_vitals": extracted_data.get("extracted_vitals"),
        "patient_record": extracted_data.get("patient_record"),
        "errors": errors
    }


def route_by_patient_type(state: ASHAAgentState) -> str:
    p_type = state.get("patient_type", "GENERAL")
    if p_type == "CHILD":
        return "muac_analysis"
    elif p_type == "MATERNAL":
        return "maternal_risk"
    return "triage"


def muac_analysis_node(state: ASHAAgentState) -> Dict[str, Any]:
    """
    Natively processes a photo of a MUAC band using computer vision to diagnose malnutrition.
    """
    image_path = state.get("muac_image_path")
    errors = list(state.get("errors", []))
    
    if not image_path or not os.path.exists(image_path):
        return {"muac_result": {"classification": "UNKNOWN", "reasons": ["No MUAC image provided."]}}
        
    try:
        print(f"[Nodes] Uploading and processing camera snapshot: {image_path}")
        uploaded_img = client.files.upload(file=image_path)
        
        prompt = """
        Analyze this image of a Mid-Upper Arm Circumference (MUAC) tape measurement on a child's arm.
        Determine the visible centimeter measurement or the band color (Green, Yellow, Red).
        Return a valid JSON response with this structure:
        {
            "measured_circumference_cm": float or null,
            "classification": "SAM (Severe Acute Malnutrition)" or "MAM (Moderate Acute Malnutrition)" or "NORMAL"
        }
        """
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt, uploaded_img],
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        muac_json = json.loads(response.text.strip())
        client.files.delete(name=uploaded_img.name)
        return {"muac_result": muac_json}
        
    except Exception as e:
        errors.append(f"Image analysis failed: {str(e)}")
        return {"muac_result": {"classification": "UNKNOWN"}, "errors": errors}


def maternal_risk_node(state: ASHAAgentState) -> Dict[str, Any]:
    """Algorithmic parsing node evaluating vital trends."""
    vitals = state.get("extracted_vitals", {}) or {}
    systolic = vitals.get("systolic")
    diastolic = vitals.get("diastolic")
    hb = vitals.get("hemoglobin")
    
    flags = []
    if systolic and systolic >= 140 or diastolic and diastolic >= 90:
        flags.append(f"Gestational Hypertension Risk (Blood Pressure: {systolic}/{diastolic})")
    if hb and hb < 11.0:
        flags.append(f"Anemia Detected (Hb level measured low at {hb} g/dL)")
        
    return {
        "maternal_risk_result": {
            "risk_flags": flags,
            "monitored_parameters": len(flags)
        }
    }


def triage_node(state: ASHAAgentState) -> Dict[str, Any]:
    """Consolidates individual risk flags into a clean classification."""
    maternal = state.get("maternal_risk_result") or {}
    muac = state.get("muac_result") or {}
    
    risk_level = "LOW"
    reasons = []
    
    if maternal.get("risk_flags"):
        risk_level = "HIGH"
        reasons.extend(maternal["risk_flags"])
        
    muac_class = muac.get("classification", "NORMAL")
    if "SAM" in muac_class:
        risk_level = "URGENT_REFERRAL"
        reasons.append("Severe Acute Malnutrition (SAM) confirmed by image processing.")
    elif "MAM" in muac_class:
        if risk_level != "HIGH":
            risk_level = "MODERATE"
        reasons.append("Moderate Acute Malnutrition (MAM) detected via image processing.")
        
    if not reasons:
        reasons.append("All inputs parsed successfully and fall within regular parameters.")
        
    return {"risk_assessment": {"risk_level": risk_level, "reasons": reasons}}


def guidance_generation_node(state: ASHAAgentState) -> Dict[str, Any]:
    """Generates localized audio-ready field text depending on the risk state."""
    triage = state.get("risk_assessment") or {}
    vitals = state.get("extracted_vitals") or {}
    level = triage.get("risk_level", "LOW")
    reasons_str = ", ".join(triage.get("reasons", []))
    
    prompt = f"""
    You are an AI assistant helping a community health worker (ASHA worker) in rural India.
    Based on this clinical synthesis, generate action guidance for the worker to read aloud.
    
    Risk Level: {level}
    Medical Flags: {reasons_str}
    
    Generate instructions in two variations:
    1. A clear English medical instruction string.
    2. A matching local language guidance string translated entirely into clear, simple Hindi.
    
    Respond with a JSON object format matching:
    {{
        "guidance_text_en": "string",
        "guidance_text_local": "string"
    }}
    """
    
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        guidance_data = json.loads(response.text.strip())
    except Exception:
        guidance_data = {
            "guidance_text_en": "Follow standard regional operating protocols.",
            "guidance_text_local": "मानक क्षेत्रीय नियमों का पालन करें।"
        }
        
    return {
        "guidance_text_en": guidance_data.get("guidance_text_en"),
        "guidance_text_local": guidance_data.get("guidance_text_local"),
        "abha_sync_status": {"synced": True, "transaction_id": "MOCK-TXN-SUCCESS-902"}
    }
