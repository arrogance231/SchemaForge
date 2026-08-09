"""Email schema: message-metadata extraction targets."""

from datetime import date

from pydantic import BaseModel

from schemaforge.registry import SchemaSpec, register


class Email(BaseModel):
    sender_email: str | None = None
    recipient_email: str | None = None
    subject: str | None = None
    body: str | None = None
    sent_date: date | None = None
    importance: str | None = None
    attachments_present: bool | None = None
    reply_needed: bool | None = None


SPEC = SchemaSpec(
    name="email",
    model=Email,
    deterministic_fields=frozenset({"sender_email", "recipient_email", "sent_date"}),
    semantic_fields=frozenset(
        {"subject", "body", "importance", "attachments_present", "reply_needed"}
    ),
    ontology={},
)

register(SPEC)
