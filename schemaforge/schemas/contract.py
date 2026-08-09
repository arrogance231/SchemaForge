"""Contract schema: agreement-terms extraction targets."""

from datetime import date
from decimal import Decimal

from pydantic import BaseModel

from schemaforge.registry import SchemaSpec, register


class Contract(BaseModel):
    contract_type: str | None = None
    party_a: str | None = None
    party_b: str | None = None
    start_date: date | None = None
    end_date: date | None = None
    auto_renewal: bool | None = None
    liability_cap: Decimal | None = None
    jurisdiction: str | None = None


SPEC = SchemaSpec(
    name="contract",
    model=Contract,
    deterministic_fields=frozenset({"start_date", "end_date", "liability_cap"}),
    semantic_fields=frozenset(
        {"contract_type", "party_a", "party_b", "auto_renewal", "jurisdiction"}
    ),
    ontology={},
)

register(SPEC)
