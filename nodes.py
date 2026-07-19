# nodes.py
import os
import json
import time
from typing import Dict, Any
from dotenv import load_dotenv

# 1. Force explicit environment variable loading from the project directory
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, '.env')
load_dotenv(dotenv_path=env_path) 

from google import genai
from google.genai import types
from state import ASHAAgentState

# 2. Fallback check: Read key from environment variables
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    # ⚠️ SAFEGUARD: If your .env file is completely missing or unreadable,
    # you can paste your key string directly below:
    api_key = "AIzaSy..." 

# 3. Initialize the unified GenAI Client precisely once
client = genai.Client(api_key=api_key)


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

    if state.get("input_mode") == "audio" and audio_path and os.path.exists(audio_path):
        try:
            print(f"[Nodes] Uploading stabilized voice notes track: {audio_path}")
            
            uploaded_audio = client.files.upload(
                file=audio_path,
                config=types.UploadFileConfig(mime_type="audio/wav")
            )
            
            while uploaded_audio.state.name == "PROCESSING":
                time.sleep(1)
                uploaded_audio = client.files.get(name=uploaded_audio.name)
                
            if uploaded_audio.state.name == "FAILED":
                raise Exception("Google API processing layer failed to decode file structure.")

            transcription_prompt = (
                "You are an expert medical transcriptionist and translator working with rural community health systems. "
                "Examine this voice note spoken by an ASHA worker carefully. "
                "1. Transcribe the raw text accurately, paying close attention to vital numerical indicators, patient fields, and data counts. "
                "2. The recording contains mixed Hindi and English clinical phrases (Hinglish). "
                "3. Provide the final output translated completely and seamlessly into clear, standard clinical English. "
                "Do not add conversational commentary or summaries; output only the high-accuracy translation."
            )

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[transcription_prompt, uploaded_audio]
            )
            translated_en = response.text.strip()
            detected_language = "Detected from Audio"
            
            time.sleep(0.5) 
            client.files.delete(name=uploaded_audio.name)
            
        except Exception as e:
            errors.append(f"Audio processing error: {str(e)}")
            translated_en = f"Audio processing failed runtime tracking: {str(e)}"
   
    else:
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
    Uses structured schema parsing to extract medical records, metrics, 
    and administrative fields into an expanded unified metadata schema.
    """
    translated_text = state.get("translated_text_en", "")
    errors = list(state.get("errors", []))
    
    prompt = f"""
    Analyze this English medical/community note and extract patient metadata, variables, and logs into a clean JSON format.
    
    Note: "{translated_text}"
    
    Respond STRICTLY with a valid JSON object matching this structure exactly. Populate unmentioned fields as null or empty arrays:
    {{
        "primary_domain": "MATERNAL_HEALTH" or "CHILD_HEALTH" or "VITAL_EVENTS" or "DISEASE_SCREENING" or "COMMUNITY_DEMOGRAPHICS" or "DRUG_SUPPLIES" or "WORK_LOGS",
        "maternal_health": {{
            "is_pregnant": boolean or null,
            "anc_checkup_count": integer or null,
            "institutional_delivery": boolean or null,
            "postpartum_care_received": boolean or null,
            "gestational_age_weeks": integer or null,
            "systolic_bp": integer or null,
            "diastolic_bp": integer or null,
            "hemoglobin": float or null
        }},
        "child_health_immunization": {{
            "has_birth_record": boolean or null,
            "immunizations_given": ["list", "of", "vaccines"],
            "birth_weight_kg": float or null,
            "breastfeeding_progress_status": "EXCLUSIVE" or "PARTIAL" or "ISSUES" or null
        }},
        "vital_events": {{
            "is_birth_event": boolean or null,
            "is_death_event": boolean or null,
            "infant_child_mortality_flag": boolean or null,
            "demographic_notes": string or null
        }},
        "disease_screening": {{
            "communicable_symptoms": ["malaria", "leprosy", "tuberculosis", "etc"],
            "ncd_screening_results": string or null,
            "requires_immediate_isolation": boolean or null
        }},
        "community_demographics": {{
            "eligible_family_planning_couple": boolean or null,
            "malnourished_child_flag": boolean or null,
            "targeted_nutritional_support_required": boolean or null
        }},
        "drug_supplies_services": {{
            "items_consumed": [
                {{"item_name": "ORS", "quantity": 2}},
                {{"item_name": "IFA_tablets", "quantity": 30}},
                {{"item_name": "contraceptives", "quantity": 5}}
            ]
        }},
        "work_logs": {{
            "activity_description": string or null,
            "performance_incentive_eligible": boolean or null,
            "children_mobilized_count": integer or null
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
        errors.append(f"Structured domain extraction failed: {str(e)}")
        # Fallback empty structure matching schema
        extracted_data = {
            "primary_domain": "WORK_LOGS",
            "maternal_health": {}, "child_health_immunization": {}, "vital_events": {},
            "disease_screening": {}, "community_demographics": {}, "drug_supplies_services": {}, "work_logs": {}
        }
        
    # Contextual type adjustments based on edge case context
    if state.get("muac_image_path"):
        extracted_data["primary_domain"] = "CHILD_HEALTH"
        if not extracted_data.get("community_demographics"):
            extracted_data["community_demographics"] = {}
        extracted_data["community_demographics"]["malnourished_child_flag"] = True

    return {
        "patient_type": extracted_data.get("primary_domain", "WORK_LOGS"),
        "unified_metadata": extracted_data,
        "errors": errors
    }


def route_by_patient_type(state: ASHAAgentState) -> str:
    """Routes state based on the multi-domain classification engine."""
    p_type = state.get("patient_type", "WORK_LOGS")
    if p_type == "CHILD_HEALTH":
        return "muac_analysis"
    elif p_type == "MATERNAL_HEALTH":
        return "maternal_risk"
    return "triage"


def muac_analysis_node(state: ASHAAgentState) -> Dict[str, Any]:
    """Natively processes a photo of a MUAC band using computer vision."""
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
    """Algorithmic evaluation layer examining unified maternal records."""
    metadata = state.get("unified_metadata", {})
    maternal = metadata.get("maternal_health", {}) or {}
    
    systolic = maternal.get("systolic_bp")
    diastolic = maternal.get("diastolic_bp")
    hb = maternal.get("hemoglobin")
    
    flags = []
    if (systolic and systolic >= 140) or (diastolic and diastolic >= 90):
        flags.append(f"Gestational Hypertension Risk (Blood Pressure: {systolic}/{diastolic})")
    if hb and hb < 11.0:
        flags.append(f"Anemia Detected (Hb level measured low at {hb} g/dL)")
    if maternal.get("postpartum_care_received") is False:
        flags.append("Missing required immediate postpartum follow-up care.")
        
    return {
        "maternal_risk_result": {
            "risk_flags": flags,
            "monitored_parameters": len(flags)
        }
    }


def triage_node(state: ASHAAgentState) -> Dict[str, Any]:
    """Consolidates cross-domain fields and clinical observations into a high-level triage ranking."""
    metadata = state.get("unified_metadata", {}) or {}
    maternal = state.get("maternal_risk_result") or {}
    muac = state.get("muac_result") or {}
    
    risk_level = "LOW"
    reasons = []
    
    # 1. Evaluate Maternal System Checks
    if maternal.get("risk_flags"):
        risk_level = "HIGH"
        reasons.extend(maternal["risk_flags"])
        
    # 2. Evaluate Child Health / Nutrition Visual Checks
    muac_class = muac.get("classification", "NORMAL")
    if "SAM" in muac_class:
        risk_level = "URGENT_REFERRAL"
        reasons.append("Severe Acute Malnutrition (SAM) confirmed by image processing.")
    elif "MAM" in muac_class:
        if risk_level != "HIGH" and risk_level != "URGENT_REFERRAL":
            risk_level = "MODERATE"
        reasons.append("Moderate Acute Malnutrition (MAM) detected via image processing.")
        
    # 3. Evaluate Disease and Screening Signals
    screening = metadata.get("disease_screening", {}) or {}
    if screening.get("communicable_symptoms"):
        risk_level = "HIGH"
        syms = ", ".join(screening.get("communicable_symptoms", []))
        reasons.append(f"Communicable disease indicators flagged: {syms}")
        
    # 4. Evaluate Vital Incident Signals
    vitals = metadata.get("vital_events", {}) or {}
    if vitals.get("infant_child_mortality_flag"):
        risk_level = "URGENT_REFERRAL"
        reasons.append("Critical vital incident record: Child mortality tracking event reported.")

    if not reasons:
        reasons.append("All inputs parsed successfully and fall within standard parameters.")
        
    return {"risk_assessment": {"risk_level": risk_level, "reasons": reasons}}


def guidance_generation_node(state: ASHAAgentState) -> Dict[str, Any]:
    """Generates localized action directions mapped directly to the active clinical domain profile."""
    triage = state.get("risk_assessment") or {}
    metadata = state.get("unified_metadata", {}) or {}
    
    level = triage.get("risk_level", "LOW")
    reasons_str = ", ".join(triage.get("reasons", []))
    domain = metadata.get("primary_domain", "GENERAL")
    
    prompt = f"""
    You are an AI assistant helping a community health worker (ASHA worker) in rural India.
    Based on this clinical synthesis across domain '{domain}', generate contextual field guidance.
    
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
            "guidance_text_en": "Follow standard regional operating protocols for metadata filing.",
            "guidance_text_local": "मानक क्षेत्रीय नियमों का पालन करें।"
        }
        
    return {
        "guidance_text_en": guidance_data.get("guidance_text_en"),
        "guidance_text_local": guidance_data.get("guidance_text_local"),
        "abha_sync_status": {"synced": True, "transaction_id": "MOCK-TXN-SUCCESS-902"}
    }
