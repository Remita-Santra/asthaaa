import json
import datetime
from typing import List, Dict, Any, Optional


_MOCK_DATABASE_LEDGER: List[Dict[str, Any]] = [
    {
        "abha_ref": "ABHA-9921-2201",
        "patient_name": "Sunita Devi",
        "patient_type": "MATERNAL_HEALTH",
        "risk_assessment": json.dumps({
            "risk_level": "HIGH",
            "reasons": ["Severe Hypertension Flagged (155/100)"]
        }),
        "follow_up_date": (datetime.date.today() + datetime.timedelta(days=3)).isoformat(),
        "recorded_at": datetime.datetime.now().isoformat(timespec="seconds"),
    },
    {
        "abha_ref": "ABHA-4410-8839",
        "patient_name": "Baby Rohan",
        "patient_type": "CHILD_HEALTH",
        "risk_assessment": json.dumps({
            "risk_level": "MODERATE",
            "reasons": ["Moderate Acute Malnutrition (MAM)"]
        }),
        "follow_up_date": (datetime.date.today() + datetime.timedelta(days=7)).isoformat(),
        "recorded_at": datetime.datetime.now().isoformat(timespec="seconds"),
    },
]


def fetch_all_records() -> List[Dict[str, Any]]:
    """
    Retrieves synced administrative patient data objects.
    Returns a shallow copy so callers (e.g. Streamlit session_state) can't
    accidentally mutate the underlying in-memory ledger by modifying the
    returned list in place.
    """
    return list(_MOCK_DATABASE_LEDGER)


def save_record(
    abha_ref: str,
    patient_name: Optional[str],
    patient_type: str,
    risk_assessment: Any,
    follow_up_date: Optional[str] = None,
) -> None:
    """
    Appends a validated visit record to the persistent local data layer.
    `risk_assessment` may be passed as a dict (it will be JSON-serialized
    here) or as an already-serialized JSON string.
    """
    _MOCK_DATABASE_LEDGER.append({
        "abha_ref": abha_ref,
        "patient_name": (patient_name or "Unknown").strip() or "Unknown",
        "patient_type": patient_type,
        "risk_assessment": json.dumps(risk_assessment) if isinstance(risk_assessment, (dict, list)) else risk_assessment,
        "follow_up_date": follow_up_date,
        "recorded_at": datetime.datetime.now().isoformat(timespec="seconds"),
    })
