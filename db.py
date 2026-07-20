import json
import datetime
from typing import List, Dict, Any, Optional

# Simple In-Memory Datastore Engine
# NOTE: This is a mock, process-local ledger for demo purposes — it resets
# every time the process restarts. In a real deployment this would be
# swapped for a persistent store (a proper database, or the ABHA-linked
# backend), but fetch_all_records() / save_record() are written so that
# swap can happen without touching app.py or nodes.py.
_MOCK_DATABASE_LEDGER: List[Dict[str, Any]] = [
    {
        "abha_ref": "ABHA-9921-2201",
        "patient_name": "Sunita Devi",
        "patient_category": "PREGNANT_WOMAN",
        "patient_type": "MATERNAL_HEALTH",
        "risk_assessment": json.dumps({
            "risk_level": "HIGH",
            "reasons": ["Severe Hypertension Flagged (155/100)"]
        }),
        "follow_up_date": (datetime.date.today() + datetime.timedelta(days=3)).isoformat(),
        "recorded_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "recorded_by_worker": "ASHA-DEMO-0001",
        "village": "Rampur Sub-Center",
        "guidance_text_en": "Refer for urgent blood pressure monitoring and advise immediate facility visit.",
        "guidance_text_local": "तुरंत रक्तचाप जांच के लिए केंद्र पर भेजें।",
        "detailed_report": None,
    },
    {
        "abha_ref": "ABHA-4410-8839",
        "patient_name": "Baby Rohan",
        "patient_category": "CHILD",
        "patient_type": "CHILD_HEALTH",
        "risk_assessment": json.dumps({
            "risk_level": "MODERATE",
            "reasons": ["Moderate Acute Malnutrition (MAM)"]
        }),
        "follow_up_date": (datetime.date.today() + datetime.timedelta(days=7)).isoformat(),
        "recorded_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "recorded_by_worker": "ASHA-DEMO-0001",
        "village": "Rampur Sub-Center",
        "guidance_text_en": "Enroll in supplementary nutrition program and recheck MUAC in 2 weeks.",
        "guidance_text_local": "पूरक पोषण कार्यक्रम में दाखिला दें और 2 हफ्ते में फिर से जांच करें।",
        "detailed_report": None,
    },
]


def fetch_all_records() -> List[Dict[str, Any]]:
    """
    Retrieves synced administrative patient data objects, most recent first.
    Returns a shallow copy so callers (e.g. Streamlit session_state) can't
    accidentally mutate the underlying in-memory ledger by modifying the
    returned list in place.
    """
    return list(reversed(_MOCK_DATABASE_LEDGER))


def save_record(
    abha_ref: str,
    patient_name: Optional[str],
    patient_type: str,
    risk_assessment: Any,
    follow_up_date: Optional[str] = None,
    patient_category: Optional[str] = None,
    recorded_by_worker: Optional[str] = None,
    village: Optional[str] = None,
    guidance_text_en: Optional[str] = None,
    guidance_text_local: Optional[str] = None,
    detailed_report: Optional[str] = None,
) -> None:
    """
    Appends a validated visit record to the persistent local data layer.
    `risk_assessment` may be passed as a dict (it will be JSON-serialized
    here) or as an already-serialized JSON string. The full report text is
    stored so a past visit can be reopened and reviewed/printed in full
    from the dashboard, not just its summary line.
    """
    _MOCK_DATABASE_LEDGER.append({
        "abha_ref": abha_ref,
        "patient_name": (patient_name or "Unknown").strip() or "Unknown",
        "patient_category": patient_category,
        "patient_type": patient_type,
        "risk_assessment": json.dumps(risk_assessment) if isinstance(risk_assessment, (dict, list)) else risk_assessment,
        "follow_up_date": follow_up_date,
        "recorded_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "recorded_by_worker": recorded_by_worker,
        "village": village,
        "guidance_text_en": guidance_text_en,
        "guidance_text_local": guidance_text_local,
        "detailed_report": detailed_report,
    })
