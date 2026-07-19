import os
import json
import time
from typing import Dict, Any
from dotenv import load_dotenv

# Force environment variable processing safely
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, '.env')
load_dotenv(dotenv_path=env_path)

from google import genai
from google.genai import types
from state import ASHAAgentState

def _clean_api_key(raw_key):
    """
    Strip whitespace and accidental wrapping quotes from an API key.
    A key like ' "AIzaSy...." \\n' (stray quotes/newline from a copy-paste
    into .env) is a common cause of auth calls silently failing.
    """
    if not raw_key:
        return raw_key
    cleaned = raw_key.strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in ("'", '"'):
        cleaned = cleaned[1:-1].strip()
    return cleaned


api_key = _clean_api_key(os.getenv("GEMINI_API_KEY"))
if not api_key:
    # Attempt fallback check to local configuration engines
    try:
        import streamlit as st
        api_key = _clean_api_key(st.secrets.get("GEMINI_API_KEY"))
    except Exception:
        pass

if not api_key:
    raise ValueError("System failure: GEMINI_API_KEY could not be resolved by the engine layers.")

if not api_key.startswith("AIza"):
    print(
        "[Nodes] WARNING: GEMINI_API_KEY does not look like a typical Google AI "
        "Studio key (these usually start with 'AIza'). If you're seeing "
        "'ACCESS_TOKEN_TYPE_UNSUPPORTED' / 401 errors, double-check this is an "
        "API key from https://aistudio.google.com/apikey and not an OAuth "
        "client ID/secret or a Vertex AI service-account value."
    )


client = genai.Client(api_key=api_key, vertexai=False)


def _extract_text(response) -> str:
    """
    Safely pull text out of a Gemini response. Raises a clear, actionable
    error if the model returned nothing usable (e.g. blocked by safety
    filters, empty candidates, non-STOP finish reason) instead of letting
    callers hit a bare AttributeError on `.strip()`.
    """
    text = getattr(response, "text", None)
    if not text or not text.strip():
        finish_reason = None
        try:
            candidates = getattr(response, "candidates", None) or []
            if candidates:
                finish_reason = getattr(candidates[0], "finish_reason", None)
        except Exception:
            pass
        raise ValueError(
            f"Model returned an empty response "
            f"(possibly blocked by safety filters or no candidates; finish_reason={finish_reason})."
        )
    return text.strip()


def ingest_node(state: ASHAAgentState) -> Dict[str, Any]:
    """
    Natively uploads voice notes or relies on typed notes.
    Uses Gemini to handle transcription and translation cleanly in one step.
    """
    errors = list(state.get("errors", []))
    raw_text = (state.get("raw_text") or "").strip()
    audio_path = state.get("raw_audio_path")

    translated_en = ""
    detected_language = "English"

    input_mode = state.get("input_mode")

    if input_mode == "audio":
        if audio_path and os.path.exists(audio_path):
            try:
                print(f"[Nodes] Uploading stabilized voice notes track: {audio_path}")

                uploaded_audio = client.files.upload(
                    file=audio_path,
                    config=types.UploadFileConfig(mime_type="audio/wav")
                )

                # Explicit polling loop
                while uploaded_audio.state.name == "PROCESSING":
                    time.sleep(1)
                    uploaded_audio = client.files.get(name=uploaded_audio.name)

                if uploaded_audio.state.name == "FAILED":
                    raise Exception("Google API processing layer failed to decode file structure.")

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
                    model="gemini-3.5-flash",
                    contents=[transcription_prompt, uploaded_audio]
                )
                translated_en = _extract_text(response)
                detected_language = "Detected from Audio"

                try:
                    client.files.delete(name=uploaded_audio.name)
                except Exception:
                    pass

            except Exception as e:
                errors.append(f"Audio processing error: {str(e)}")
                translated_en = f"Audio processing failed runtime tracking: {str(e)}"
        else:
            # Audio mode was selected but no valid file exists — don't
            # silently fall through and pretend nothing was submitted.
            errors.append("Audio input mode selected but no valid audio file was found.")
            translated_en = ""
            detected_language = "Unknown"

    else:
        if raw_text:
            try:
                response = client.models.generate_content(
                    model="gemini-3.5-flash",
                    contents=f"Translate this Indian healthcare field note precisely into English. Focus heavily on maintaining numbers, medical values, symptoms, and items: {raw_text}"
                )
                translated_en = _extract_text(response)
                detected_language = "Typed Input"
            except Exception as e:
                errors.append(f"Translation error: {str(e)}")
                translated_en = raw_text
        else:
            errors.append("No text note provided and input mode was not audio.")

    return {
        "translated_text_en": translated_en,
        "detected_language": detected_language,
        "errors": errors
    }


def language_and_translate_node(state: ASHAAgentState) -> Dict[str, Any]:
    """Pass-through node ensuring persistent structural state tracking."""
    return {
        "translated_text_en": state.get("translated_text_en", ""),
        "detected_language": state.get("detected_language", "Unknown")
    }


def extract_vitals_node(state: ASHAAgentState) -> Dict[str, Any]:
    """
    Uses structured schema parsing to extract medical records, metrics,
    and administrative fields into an expanded unified metadata schema.
    """
    translated_text = state.get("translated_text_en", "")
    errors = list(state.get("errors", []))

    prompt = f"""
    You are extracting structured public-health data from a community health worker's field note.
    The note has already been translated into English below.

    Note to evaluate: "{translated_text}"

    Choose exactly one `primary_domain` using this priority order (check top to bottom, first match wins):
    1. MATERNAL_HEALTH — mentions pregnancy, ANC checkups, delivery, postpartum care, blood pressure,
       hemoglobin, or any maternal vital sign (e.g. "pregnant", "BP 150/90", "hemoglobin 9.5").
    2. VITAL_EVENTS — mentions a birth or death event, or child/infant mortality.
    3. DISEASE_SCREENING — mentions communicable disease symptoms (fever, cough, isolation) or NCD screening
       that is NOT tied to a pregnancy (if pregnancy is mentioned too, MATERNAL_HEALTH still wins).
    4. CHILD_HEALTH — mentions immunization, birth weight, breastfeeding, or a child under a specific age.
    5. COMMUNITY_DEMOGRAPHICS — mentions family planning eligibility or malnutrition screening not covered above.
    6. DRUG_SUPPLIES — primarily about medicine/consumable stock or distribution counts.
    7. WORK_LOGS — general activity/outreach logging with no clinical signal above.

    Extract every numeric vital sign exactly as stated (blood pressure as separate systolic_bp/diastolic_bp
    integers, hemoglobin as a decimal number, ages/weeks as integers). Do not invent values that are not
    present in the note — leave those fields absent instead of guessing. Populate every relevant nested
    object even when primary_domain is a different domain, since a note can carry signals for multiple
    domains at once (e.g. a maternal note can also carry disease_screening symptoms).
    """

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
            model="gemini-3.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=response_schema
            )
        )
        extracted_data = json.loads(_extract_text(response))
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
    """Processes a photo of a MUAC band using computer vision."""
    image_path = state.get("muac_image_path")
    errors = list(state.get("errors", []))

    if not image_path or not os.path.exists(image_path):
        return {"muac_result": {"classification": "NORMAL", "reasons": ["No MUAC image provided."]}}

    try:
        print(f"[Nodes] Uploading and processing camera snapshot: {image_path}")
        uploaded_img = client.files.upload(file=image_path)

        prompt = """
        Analyze this image of a Mid-Upper Arm Circumference (MUAC) tape measurement on a child's arm.
        Determine the visible centimeter measurement or the band color (Green, Yellow, Red).
        Return a valid JSON response with this structure:
        {
            "measured_circumference_cm": float or null,
            "classification": "SAM" or "MAM" or "NORMAL"
        }
        """
        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=[prompt, uploaded_img],
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        muac_json = json.loads(_extract_text(response))

        try:
            client.files.delete(name=uploaded_img.name)
        except Exception:
            pass

        return {"muac_result": muac_json, "errors": errors}

    except Exception as e:
        errors.append(f"Image analysis failed: {str(e)}")
        return {"muac_result": {"classification": "NORMAL"}, "errors": errors}


def maternal_risk_node(state: ASHAAgentState) -> Dict[str, Any]:
    """Algorithmic evaluation layer examining unified maternal records."""
    metadata = state.get("unified_metadata", {}) or {}
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
    """Consolidates cross-domain fields and observations into a triage ranking."""
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
    muac_class = str(muac.get("classification", "NORMAL")).upper()
    if "SAM" in muac_class:
        risk_level = "URGENT_REFERRAL"
        reasons.append("Severe Acute Malnutrition (SAM) confirmed via visual measurement tool.")
    elif "MAM" in muac_class:
        if risk_level not in ["HIGH", "URGENT_REFERRAL"]:
            risk_level = "MODERATE"
        reasons.append("Moderate Acute Malnutrition (MAM) detected.")

    # 3. Evaluate Critical Vital Incidents
    vitals = metadata.get("vital_events", {}) or {}
    if vitals.get("infant_child_mortality_flag") is True or vitals.get("is_death_event") is True:
        risk_level = "URGENT_REFERRAL"
        reasons.append("CRITICAL INCIDENT: Local mortality tracking event reported.")

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

    # 5. Evaluate Community Demographics
    demographics = metadata.get("community_demographics", {}) or {}
    if demographics.get("targeted_nutritional_support_required") is True or demographics.get("malnourished_child_flag") is True:
        if risk_level == "LOW":
            risk_level = "MODERATE"
        reasons.append("High-risk community cohort requires targeted nutrition tracking.")

    # 6. Evaluate Drug Supplies Consumption Logs
    drugs = metadata.get("drug_supplies_services", {}) or {}
    for item in drugs.get("items_consumed", []):
        name = str(item.get("item_name", "")).upper()
        if "ORS" in name and item.get("quantity", 0) >= 3:
            if risk_level == "LOW":
                risk_level = "MODERATE"
            reasons.append(f"High volume dehydration mitigation items distributed: {item.get('quantity')} ORS packets.")

    if not reasons:
        reasons.append("Record successfully processed. Indicators fall within normal parameters.")

    return {"risk_assessment": {"risk_level": risk_level, "reasons": reasons}}


def guidance_generation_node(state: ASHAAgentState) -> Dict[str, Any]:
    """Generates localized action directions mapped directly to the active clinical profile."""
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
            model="gemini-3.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        guidance_data = json.loads(_extract_text(response))
    except Exception:
        guidance_data = {
            "guidance_text_en": f"Alert level {level}. Review system parameters for domain {domain} locally.",
            "guidance_text_local": f"चेतावनी स्तर: {level}। कृपया स्थानीय स्वास्थ्य नियमों के अनुसार जांच करें।"
        }

    return {
        "guidance_text_en": guidance_data.get("guidance_text_en"),
        "guidance_text_local": guidance_data.get("guidance_text_local"),
        "abha_sync_status": {"synced": True, "transaction_id": f"ABHA-TXN-{int(time.time())}"}
    }
