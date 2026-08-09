"""CRM record schema: sales-opportunity extraction targets."""

from datetime import date
from decimal import Decimal

from pydantic import BaseModel

from schemaforge.registry import SchemaSpec, register


class CrmRecord(BaseModel):
    contact_name: str | None = None
    company_name: str | None = None
    email: str | None = None
    phone: str | None = None
    deal_stage: str | None = None
    deal_value: Decimal | None = None
    probability: float | None = None
    owner: str | None = None
    last_contact_date: date | None = None


ONTOLOGY = {
    "Corp.": "Corporation",
    "Inc.": "Incorporated",
    "Intl.": "International",
    "Tech.": "Technologies",
}

SPEC = SchemaSpec(
    name="crm_record",
    model=CrmRecord,
    deterministic_fields=frozenset({"email", "phone", "deal_value", "last_contact_date"}),
    semantic_fields=frozenset(
        {"contact_name", "company_name", "deal_stage", "probability", "owner"}
    ),
    ontology=ONTOLOGY,
)

register(SPEC)
