"""Tests for the hybrid merge pipeline (research direction §6, "hybrid (rules → SchemaForge)").

Focus: the pre-pass winning conflicts on its own fields, semantic fields filled
from the model, deterministic leaves falling through to the model when the
pre-pass missed them, absence (not null) as the "neither system produced a
value" signal, the rules-only baseline, and the ``[]`` unflatten inversion of
``flatten``.
"""

from datetime import date
from decimal import Decimal

import pytest

import schemaforge  # noqa: F401  (importing the package populates the registry)
from schemaforge.evaluation.json_utils import flatten
from schemaforge.hybrid.pipeline import _unflatten, merge_prediction, rules_only_prediction
from schemaforge.registry import get_schema

INVOICE_RESOLVED = {
    "invoice_number": "INV-1001",
    "invoice_date": date(2026, 4, 10),
    "due_date": date(2026, 5, 10),
    "vendor_email": "billing@acme.com",
    "vendor_phone": "+1 555 0100",
    "total_amount": Decimal("518.40"),
    "tax_amount": Decimal("38.40"),
    "line_items[].quantity": [Decimal("4")],
    "line_items[].unit_price": [Decimal("120.00")],
    "line_items[].line_total": [Decimal("480.00")],
}


def test_prepass_resolved_deterministic_field_wins_over_model():
    spec = get_schema("invoice")
    model = {
        "invoice_number": "INV-9999",  # the pre-pass already resolved this
        "vendor_name": "Acme Corp",
    }
    merged = merge_prediction(INVOICE_RESOLVED, model, spec)
    assert merged["invoice_number"] == "INV-1001"
    assert merged["vendor_name"] == "Acme Corp"


def test_semantic_field_filled_from_model():
    spec = get_schema("invoice")
    merged = merge_prediction(INVOICE_RESOLVED, {"vendor_name": "Acme Corp"}, spec)
    assert merged["vendor_name"] == "Acme Corp"


def test_unresolved_deterministic_field_falls_through_to_model():
    spec = get_schema("invoice")
    prepass = {path: value for path, value in INVOICE_RESOLVED.items() if path != "due_date"}
    assert "due_date" not in prepass  # simulate a failed regex
    merged = merge_prediction(prepass, {"due_date": "2026-05-10"}, spec)
    assert merged["due_date"] == "2026-05-10"


def test_field_neither_system_produced_is_absent_not_none():
    spec = get_schema("invoice")
    merged = merge_prediction(INVOICE_RESOLVED, {"vendor_name": None}, spec)
    flat = flatten(merged)
    assert "vendor_name" not in flat


def test_none_model_prediction_equals_rules_only_prediction():
    spec = get_schema("invoice")
    assert merge_prediction(INVOICE_RESOLVED, None, spec) == rules_only_prediction(INVOICE_RESOLVED)
    assert merge_prediction(INVOICE_RESOLVED, None, spec) == {
        "invoice_number": "INV-1001",
        "invoice_date": date(2026, 4, 10),
        "due_date": date(2026, 5, 10),
        "vendor_email": "billing@acme.com",
        "vendor_phone": "+1 555 0100",
        "total_amount": Decimal("518.40"),
        "tax_amount": Decimal("38.40"),
        "line_items": [
            {
                "quantity": Decimal("4"),
                "unit_price": Decimal("120.00"),
                "line_total": Decimal("480.00"),
            }
        ],
    }


def test_line_items_zip_deterministic_and_semantic_values_per_item():
    spec = get_schema("invoice")
    prepass = {
        "invoice_number": "INV-1001",
        "total_amount": Decimal("2246.40"),
        "line_items[].quantity": [Decimal("4"), Decimal("2")],
        "line_items[].unit_price": [Decimal("120.00"), Decimal("800.00")],
        "line_items[].line_total": [Decimal("480.00"), Decimal("1600.00")],
    }
    model = {"line_items": [{"description": "Office Chairs"}, {"description": "Server Rack"}]}
    merged = merge_prediction(prepass, model, spec)
    assert merged["line_items"] == [
        {
            "quantity": Decimal("4"),
            "unit_price": Decimal("120.00"),
            "line_total": Decimal("480.00"),
            "description": "Office Chairs",
        },
        {
            "quantity": Decimal("2"),
            "unit_price": Decimal("800.00"),
            "line_total": Decimal("1600.00"),
            "description": "Server Rack",
        },
    ]
    flat = flatten(merged)
    assert flat["line_items[].quantity"] == [Decimal("4"), Decimal("2")]
    assert flat["line_items[].description"] == ["Office Chairs", "Server Rack"]


@pytest.mark.parametrize(
    "obj",
    [
        {"invoice_number": "INV-1001", "total_amount": Decimal("518.40"), "due_date": "2026-05-10"},
        {
            "line_items": [
                {"quantity": Decimal("4"), "description": "Widget"},
                {"quantity": Decimal("2"), "description": "Gadget"},
            ]
        },
    ],
)
def test_unflatten_round_trips_flatten_exactly(obj):
    assert _unflatten(flatten(obj)) == obj


def test_unflatten_empty_list_round_trip_matches_flatten_drop():
    # flatten drops empty dict/list values by contract ("Empty dict/list values
    # produce no leaf"), so the inverse cannot recover {"tags": []}: the round
    # trip is exact precisely on the non-empty leaves flatten keeps.
    assert flatten({"tags": []}) == {}
    assert _unflatten(flatten({"tags": []})) == {}
