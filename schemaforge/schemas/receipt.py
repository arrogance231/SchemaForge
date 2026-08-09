"""Receipt schema: point-of-sale receipt extraction targets."""

from datetime import date
from decimal import Decimal

from pydantic import BaseModel

from schemaforge.registry import SchemaSpec, register


class LineItem(BaseModel):
    description: str | None = None
    quantity: Decimal | None = None
    unit_price: Decimal | None = None
    line_total: Decimal | None = None


class Receipt(BaseModel):
    receipt_number: str | None = None
    receipt_date: date | None = None
    store_name: str | None = None
    cashier_name: str | None = None
    total_amount: Decimal | None = None
    tax_amount: Decimal | None = None
    payment_method: str | None = None
    refunded: bool | None = None
    line_items: list[LineItem] | None = None


SPEC = SchemaSpec(
    name="receipt",
    model=Receipt,
    deterministic_fields=frozenset(
        {
            "receipt_number",
            "receipt_date",
            "total_amount",
            "tax_amount",
            "line_items[].quantity",
            "line_items[].unit_price",
            "line_items[].line_total",
        }
    ),
    semantic_fields=frozenset(
        {"store_name", "cashier_name", "payment_method", "refunded", "line_items[].description"}
    ),
    ontology={},
)

register(SPEC)
