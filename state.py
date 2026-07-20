from typing import Dict, Any, List, Optional
from typing_extensions import TypedDict


class ASHAAgentState(TypedDict):
    # Core Metadata & Context
    session_id: str
    asha_worker_id: str
    village: str
    patient_name: Optional[str]
    input_mode: str  # "text" or "audio"

    # Raw Inputs
    raw_text: Optional[str]
    raw_audio_path: Optional[str]
    muac_image_path: Optional[str]

    # Processed Text Streams
    detected_language: Optional[str]
    translated_text_en: Optional[str]


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

    # Synchronization tracking
    abha_sync_status: Optional[Dict[str, Any]]

    errors: List[str]
