from typing import Dict, Any, List, Optional
from typing_extensions import TypedDict


class ASHAAgentState(TypedDict):
    # Core Metadata & Context
    session_id: str
    asha_worker_id: str
    village: str
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
    # This schema previously declared "patient_record"/"extracted_vitals"
    # instead, which no node ever wrote to — so the real data extract_vitals_node
    # produced was silently dropped by LangGraph on every state merge (LangGraph
    # only persists keys declared in this TypedDict). That was the root cause of
    # "extracted_multi_domain_records" always rendering as an empty {}.
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

    # Synchronization tracking
    abha_sync_status: Optional[Dict[str, Any]]
    errors: List[str]
