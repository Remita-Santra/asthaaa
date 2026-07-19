# nodes.py
import os
import json
import time
from typing import Dict, Any
from dotenv import load_dotenv

from google import genai
from google.genai import types
from state import ASHAAgentState

# 2. Fallback check: Read key from environment variables
api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    raise ValueError("Missing key!")
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

            # OPTIMIZATION 3: Few-shot audio stabilization mapping template
            transcription_prompt = (
                "You are an elite clinical transcriptionist tracking rural public health data.\n"
                "Convert mixed audio descriptions into explicit, flat, translated clinical facts.\n\n"
                "EXAMPLES:\n"
                "- Spoken: 'Maternal health report, sister ka hemoglobin level 9.5 hai aur bp bahut high hai, 150 over 95'\n"
                "  Output: 'Maternal profile case indicator: Patient has a low hemoglobin level tracking at 9.5 g/dL and high blood pressure registering at 150/95 mmHg.'\n"
                "- Spoken: 'Child vaccination camp done, teen bache mobilized and we distributed 4 ORS packets'\n"
                "  Output: 'Activity log reporting execution: 3 children mobilized for immunization services, 4 ORS replacement kit pieces consumed.'\n\n"
                "Maintain raw numerical figures with precise decimals. Convert everything into clinical English. "
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
    """
    #OPTIMIZATION 1: Native Structural JSON Validation Schema Layer
    
    response_schema = {
        "type": "OBJECT",
        "properties": {
            "primary_domain": {
                "type": "STRING",
                "enum": ["MATERNAL_HEALTH", "CHILD_HEALTH", "VITAL_EVENTS", "DISEASE_SCREENING", "COMMUNITY_DEMOGRAPHICS", "DRUG_SUPPLIES", "WORK_LOGS"]
            },
            "maternal_health": {
                "type": "OBJECT",
                "properties": {
                    "is_pregnant": {"type": "BOOLEAN"},
                    "anc_checkup_count": {"type": "INTEGER"},
                    "institutional_delivery": {"type": "BOOLEAN"},
                    "postpartum_care_received": {"type": "BOOLEAN"},
                    "gestational_age_weeks": {"type": "INTEGER"},
                    "systolic_bp": {"type": "INTEGER"},
                    "diastolic_bp": {"type": "INTEGER"},
                    "hemoglobin": {"type": "NUMBER"}
                }
            },
            "child_health_immunization": {
                "type": "OBJECT",
                "properties": {
                    "has_birth_record": {"type": "BOOLEAN"},
                    "immunizations_given": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "birth_weight_kg": {"type": "NUMBER"},
                    "breastfeeding_progress_status": {"type": "STRING", "enum": ["EXCLUSIVE", "PARTIAL", "ISSUES"]}
                }
            },
            "vital_events": {
                "type": "OBJECT",
                "properties": {
                    "is_birth_event": {"type": "BOOLEAN"},
                    "is_death_event": {"type": "BOOLEAN"},
                    "infant_child_mortality_flag": {"type": "BOOLEAN"},
                    "demographic_notes": {"type": "STRING"}
                }
            },
            "disease_screening": {
                "type": "OBJECT",
                "properties": {
                    "communicable_symptoms": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "ncd_screening_results": {"type": "STRING"},
                    "requires_immediate_isolation": {"type": "BOOLEAN"}
                }
            },
            "community_demographics": {
                "type": "OBJECT",
                "properties": {
                    "eligible_family_planning_couple": {"type": "BOOLEAN"},
                    "malnourished_child_flag": {"type": "BOOLEAN"},
                    "targeted_nutritional_support_required": {"type": "BOOLEAN"}
                }
            },
            "drug_supplies_services": {
                "type": "OBJECT",
                "properties": {
                    "items_consumed": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "item_name": {"type": "STRING"},
                                "quantity": {"type": "INTEGER"}
                            }
                        }
                    }
                }
            },
            "work_logs": {
                "type": "OBJECT",
                "properties": {
                    "activity_description": {"type": "STRING"},
                    "performance_incentive_eligible": {"type": "BOOLEAN"},
                    "children_mobilized_count": {"type": "INTEGER"}
                }
            }
        },
        "required": [
            "primary_domain", "maternal_health", "child_health_immunization", 
            "vital_events", "disease_screening", "community_demographics", 
            "drug_supplies_services", "work_logs"
        ]
    }
    
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=response_schema
            )
        )
        extracted_data = json.loads(response.text.strip())
    except Exception as e:
        errors.append(f"Structured domain extraction failed: {str(e)}")
        extracted_data = {
            "primary_domain": "WORK_LOGS",
            "maternal_health": {}, "child_health_immunization": {}, "vital_events": {},
            "disease_screening": {}, "community_demographics": {}, "drug_supplies_services": {}, "work_logs": {}
        }
        
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
    """
    OPTIMIZATION 2: Advanced Multi-Domain Triage Matrix
    Consolidates cross-domain fields and clinical observations into a high-level triage ranking.
    """
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
        reasons.append("Severe Acute Malnutrition (SAM) confirmed via visual measurement tool.")
    elif "MAM" in muac_class:
        if risk_level not in ["HIGH", "URGENT_REFERRAL"]:
            risk_level = "MODERATE"
        reasons.append("Moderate Acute Malnutrition (MAM) detected.")
        
    # 3. Evaluate Critical Vital Incidents
    vitals = metadata.get("vital_events", {}) or {}
    if vitals.get("infant_child_mortality_flag") is True:
        risk_level = "URGENT_REFERRAL"
        reasons.append("CRITICAL INCIDENT: Local infant/child mortality tracking event reported.")
        
    # 4. Evaluate Disease and Screening Signals
    screening = metadata.get("disease_screening", {}) or {}
    if screening.get("requires_immediate_isolation") is True:
        risk_level = "URGENT_REFERRAL"
        reasons.append("DISEASE RISK: Symptom footprint flags immediate isolation warning.")
    elif screening.get("communicable_symptoms"):
        if risk_level != "URGENT_REFERRAL":
            risk_level = "HIGH"
        syms = ", ".join(screening.get("communicable_symptoms", []))
        reasons.append(f"Communicable disease indicators flagged: {syms}")
        
    # 5. Evaluate Community Demographics (Cohort Fallbacks)
    demographics = metadata.get("community_demographics", {}) or {}
    if demographics.get("targeted_nutritional_support_required") is True:
        if risk_level == "LOW":
            risk_level = "MODERATE"
        reasons.append("High-risk community cohort requires targeted nutrition tracking.")

    # 6. Evaluate Drug Supplies Consumption Logs
    drugs = metadata.get("drug_supplies_services", {}) or {}
    for item in drugs.get("items_consumed", []):
        if item.get("item_name") == "ORS" and item.get("quantity", 0) > 5:
            if risk_level == "LOW":
                risk_level = "MODERATE"
            reasons.append("High volume diarrhea mitigation items distributed.")

    if not reasons:
        reasons.append("Record successfully processed. Indicators fall within normal parameters.")
        
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
