"""Insurance claim schema (held out): relational claim-extraction targets.

Designated held-out so the model never trains on this shape; it exercises
nested entities (``involved_parties[]``) plus a peril-normalization ontology.
"""

from datetime import date
from decimal import Decimal

from pydantic import BaseModel

from schemaforge.registry import SchemaSpec, register


class InvolvedParty(BaseModel):
    name: str | None = None
    role: str | None = None


class InsuranceClaim(BaseModel):
    claim_number: str | None = None
    policy_number: str | None = None
    claimant_name: str | None = None
    claim_type: str | None = None
    incident_date: date | None = None
    amount_requested: Decimal | None = None
    status: str | None = None
    involved_parties: list[InvolvedParty] | None = None


ONTOLOGY = {
    "car accident": "Auto Collision",
    "auto collision": "Auto Collision",
    "fender bender": "Auto Collision",
    "house fire": "Fire",
    "kitchen fire": "Fire",
    "burst pipe": "Water Damage",
    "flood": "Water Damage",
    "slip and fall": "Liability",
    "theft": "Theft",
    "stolen": "Theft",
}

SPEC = SchemaSpec(
    name="insurance_claim",
    model=InsuranceClaim,
    deterministic_fields=frozenset(
        {"claim_number", "policy_number", "incident_date", "amount_requested"}
    ),
    semantic_fields=frozenset(
        {
            "claimant_name",
            "claim_type",
            "status",
            "involved_parties[].name",
            "involved_parties[].role",
        }
    ),
    ontology=ONTOLOGY,
    held_out=True,
)

register(SPEC)
