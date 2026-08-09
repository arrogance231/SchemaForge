"""Support ticket schema: customer-support issue extraction targets.

The ``ontology`` normalizes free-form intent / issue phrasing (e.g.
``"money back"`` -> ``Refund``, ``"battery drain"`` -> ``Battery``).
"""

from datetime import date

from pydantic import BaseModel

from schemaforge.registry import SchemaSpec, register


class SupportTicket(BaseModel):
    ticket_id: str | None = None
    created_date: date | None = None
    intent: str | None = None
    issue_category: str | None = None
    product: str | None = None
    urgency: str | None = None
    sentiment: str | None = None
    resolution_status: str | None = None
    customer_email: str | None = None


ONTOLOGY = {
    "money back": "Refund",
    "refund": "Refund",
    "refund me": "Refund",
    "cancel": "Cancellation",
    "cancellation": "Cancellation",
    "cancel my order": "Cancellation",
    "won't charge": "Battery",
    "wont charge": "Battery",
    "battery drain": "Battery",
    "screen cracked": "Display",
    "cracked screen": "Display",
    "broken screen": "Display",
}

SPEC = SchemaSpec(
    name="support_ticket",
    model=SupportTicket,
    deterministic_fields=frozenset({"ticket_id", "created_date", "customer_email"}),
    semantic_fields=frozenset(
        {
            "intent",
            "issue_category",
            "product",
            "urgency",
            "sentiment",
            "resolution_status",
        }
    ),
    ontology=ONTOLOGY,
)

register(SPEC)
