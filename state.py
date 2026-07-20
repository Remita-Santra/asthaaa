from typing import Dict, Any, List, Optional
from typing_extensions import TypedDict


class ASHAAgentState(TypedDict):
    # Core Metadata & Context
    session_id: str
    asha_worker_id: str
    village: str
    patient_name: Optional[str]

    # Worker-selected patient category — set directly on the "Patient
    # Details" form (PREGNANT_WOMAN / CHILD / OTHER). More reliable than
    # inferring it purely from the note, so it's used to steer routing.
    patient_category: Optional[str]
    # Optional manually-entered values from the same form, used to
    # fill in / override whatever the LLM extracts from the conversation.
    worker_child_age_months: Optional[int]
    worker_gestational_age_weeks: Optional[int]

    input_mode: str  # "text" or "audio"

    # Raw Inputs
    raw_text: Optional[str]
    raw_audio_path: Optional[str]
    muac_image_path: Optional[str]

    # Processed Text Streams
    detected_language: Optional[str]
    translated_text_en: Optional[str]

    # Structured Medical Extractions
    # NOTE: "unified_metadata" is the exact key extract_vitals_node returns
    # (see nodes.py) and the exact key muac_analysis_node, maternal_risk_node,
    # triage_node, and app.py all read back via state.get("unified_metadata").
    # Every key here MUST match a key some node actually returns — LangGraph
    # only persists keys declared in this TypedDict, so a state key a node
    # writes but this file doesn't declare is silently dropped on merge.
    patient_type: Optional[str]  # one of the primary_domain enum values from nodes.py,
                                  # e.g. "MATERNAL_HEALTH", "CHILD_HEALTH", "WORK_LOGS", etc.
    unified_metadata: Optional[Dict[str, Any]]

    # Evaluation Modules Outputs
    muac_result: Optional[Dict[str, Any]]
    maternal_risk_result: Optional[Dict[str, Any]]
    risk_assessment: Optional[Dict[str, Any]]  # Combined triage matrix

    # Localizable Care Guidances
    guidance_text_en: Optional[str]
    guidance_text_local: Optional[str]

    # Vaccine / ANC schedule compliance check (rule-based, runs after triage)
    schedule_check_result: Optional[Dict[str, Any]]

    # Follow-up automation
    follow_up_plan: Optional[Dict[str, Any]]
    sms_reminder_status: Optional[Dict[str, Any]]

    # Consolidated human-readable report (printed + downloadable)
    detailed_report: Optional[str]

    # Checklist of fields/measurements the ASHA worker still needs to
    # follow up on for this domain (missing vitals, missing age, etc.)
    required_fill_ups: Optional[List[str]]

    # Synchronization tracking
    abha_sync_status: Optional[Dict[str, Any]]

    errors: List[str]
