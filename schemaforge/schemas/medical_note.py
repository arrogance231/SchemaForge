"""Medical note schema: clinical-note extraction targets.

The ``ontology`` expands common clinical abbreviations to canonical condition
names (e.g. ``HTN`` -> ``Hypertension``, ``T2DM`` -> ``Type 2 Diabetes Mellitus``).
"""

from datetime import date

from pydantic import BaseModel

from schemaforge.registry import SchemaSpec, register


class Medication(BaseModel):
    name: str | None = None
    dosage: str | None = None


class MedicalNote(BaseModel):
    patient_name: str | None = None
    patient_age: int | None = None
    visit_date: date | None = None
    diagnosis: str | None = None
    physician_name: str | None = None
    medications: list[Medication] | None = None
    follow_up_date: date | None = None


ONTOLOGY = {
    "HTN": "Hypertension",
    "htn": "Hypertension",
    "DM2": "Type 2 Diabetes Mellitus",
    "T2DM": "Type 2 Diabetes Mellitus",
    "MI": "Myocardial Infarction",
    "CHF": "Congestive Heart Failure",
    "COPD": "Chronic Obstructive Pulmonary Disease",
    "CKD": "Chronic Kidney Disease",
    "PUD": "Peptic Ulcer Disease",
}

SPEC = SchemaSpec(
    name="medical_note",
    model=MedicalNote,
    deterministic_fields=frozenset({"visit_date", "follow_up_date"}),
    semantic_fields=frozenset(
        {
            "patient_name",
            "patient_age",
            "diagnosis",
            "physician_name",
            "medications[].name",
            "medications[].dosage",
        }
    ),
    ontology=ONTOLOGY,
)

register(SPEC)
