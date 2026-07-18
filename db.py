import json
from typing import List, Dict, Any

# Simple In-Memory Datastore Engine
_MOCK_DATABASE_LEDGER: List[Dict[str, Any]] = [
    {
        "abha_ref": "ABHA-9921-2201",
        "patient_type": "MATERNAL",
        "risk_assessment": json.dumps({
            "risk_level": "HIGH",
            "reasons": ["Severe Hypertension Flagged (155/100)"]
        })
    },
    {
        "abha_ref": "ABHA-4410-8839",
        "patient_type": "CHILD",
        "risk_assessment": json.dumps({
            "risk_level": "MODERATE",
            "reasons": ["Moderate Acute Malnutrition (MAM)"]
        })
    }
]

def fetch_all_records() -> List[Dict[str, Any]]:
    """Retrieves synced administrative patient data objects."""
    return _MOCK_DATABASE_LEDGER

def save_record(abha_ref: str, patient_type: str, risk_assessment_dict: Dict[str, Any]) -> None:
    """Appends validated graph logs to the persistent local data layer."""
    _MOCK_DATABASE_LEDGER.append({
        "abha_ref": abha_ref,
        "patient_type": patient_type,
        "risk_assessment": json.dumps(risk_assessment_dict)
    })