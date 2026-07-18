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
    patient_type: Optional[str]  # "CHILD", "MATERNAL", "GENERAL"
    patient_record: Optional[Dict[str, Any]]
    extracted_vitals: Optional[Dict[str, Any]]
    
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