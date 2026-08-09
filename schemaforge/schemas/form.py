"""Form schema: web-form submission extraction targets."""

from datetime import date

from pydantic import BaseModel

from schemaforge.registry import SchemaSpec, register


class Form(BaseModel):
    form_id: str | None = None
    submitted_date: date | None = None
    submitter_name: str | None = None
    submitter_email: str | None = None
    request_type: str | None = None
    details: str | None = None
    status: str | None = None
    priority: str | None = None


SPEC = SchemaSpec(
    name="form",
    model=Form,
    deterministic_fields=frozenset({"form_id", "submitted_date", "submitter_email"}),
    semantic_fields=frozenset(
        {"submitter_name", "request_type", "details", "status", "priority"}
    ),
    ontology={},
)

register(SPEC)
