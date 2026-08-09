"""Invoice schema: structured purchase-invoice extraction targets."""

from datetime import date
from decimal import Decimal

from pydantic import BaseModel

from schemaforge.registry import SchemaSpec, register


class LineItem(BaseModel):
    description: str | None = None
    quantity: Decimal | None = None
    unit_price: Decimal | None = None
    line_total: Decimal | None = None


class Invoice(BaseModel):
    invoice_number: str | None = None
    invoice_date: date | None = None
    due_date: date | None = None
    vendor_name: str | None = None
    vendor_email: str | None = None
    vendor_phone: str | None = None
    total_amount: Decimal | None = None
    tax_amount: Decimal | None = None
    line_items: list[LineItem] | None = None


SPEC = SchemaSpec(
    name="invoice",
    model=Invoice,
    deterministic_fields=frozenset(
        {
            "invoice_number",
            "invoice_date",
            "due_date",
            "vendor_email",
            "vendor_phone",
            "total_amount",
            "tax_amount",
            "line_items[].quantity",
            "line_items[].unit_price",
            "line_items[].line_total",
        }
    ),
    semantic_fields=frozenset({"vendor_name", "line_items[].description"}),
    ontology={},
)

register(SPEC)
