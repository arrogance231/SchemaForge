"""Tests for the SchemaForge V2 deterministic pre-pass.

Focus: the field-ownership split (never resolve a semantic field), the
``nearest_label`` selection rule on a real invoice where subtotal / tax / total
are all money, residual ``unresolved`` handling, and safety across every
registered schema.
"""

from datetime import date
from decimal import Decimal

import schemaforge  # noqa: F401  (importing the package populates the registry)
from schemaforge.deterministic.extractors import find_money, resolve_overlaps
from schemaforge.deterministic.prepass import (
    PrepassResult,
    _select_nearest_label,
    run_prepass,
    unresolved_prompt_fields,
)
from schemaforge.registry import all_schemas, get_schema, leaf_paths

INVOICE_DOC = (
    "INVOICE #INV-1001. Vendor: Acme Supply Co. Date: 2026-04-10. "
    "Item: Office Chairs x 4 @ $120.00 = $480.00. Subtotal: $480.00. "
    "Tax (8%): $38.40. Total: $518.40."
)

TWO_LINE_ITEMS_DOC = (
    "INVOICE #INV-2002. Vendor: Acme Supply Co. Date: 2026-04-10. "
    "Item: Office Chairs x 4 @ $120.00 = $480.00. "
    "Item: Server Rack x 2 @ $800.00 = $1600.00. "
    "Subtotal: $2080.00. Tax: $166.40. Total: $2246.40."
)

GENERIC_DOC = (
    "Date: 2026-04-10. Email a@b.com. Phone (415) 555-2671. "
    "Amount: $100.00. Ticket TKT-12345."
)


def test_invoice_deterministic_fields_resolve_correctly():
    result = run_prepass(INVOICE_DOC, get_schema("invoice"))
    assert result.resolved["invoice_number"] == "INV-1001"
    assert result.resolved["invoice_date"] == date(2026, 4, 10)
    assert result.resolved["total_amount"] == Decimal("518.40")
    assert result.resolved["tax_amount"] == Decimal("38.40")
    # NOT the first money match in the document (the $120.00 unit price).
    assert result.resolved["total_amount"] != Decimal("120.00")
    assert result.provenance["total_amount"].extractor == "money"
    assert result.provenance["total_amount"].span[0] > 0


def test_nearest_label_picks_subtotal_and_total_not_the_first_money_match():
    matches = resolve_overlaps(find_money(INVOICE_DOC))
    money_values = [m.value for m in matches]
    assert Decimal("120.00") in money_values  # the first money match exists
    subtotal = _select_nearest_label(INVOICE_DOC, matches, r"(?i)(?<![a-z0-9])subtotal\s*[:.]?")
    total = _select_nearest_label(INVOICE_DOC, matches, r"(?i)(?<![a-z0-9])total\s*[:.]?")
    assert subtotal is not None and subtotal.value == Decimal("480.00")
    assert total is not None and total.value == Decimal("518.40")


def test_line_items_resolve_quantity_unit_price_and_line_total():
    result = run_prepass(INVOICE_DOC, get_schema("invoice"))
    assert result.resolved["line_items[].quantity"] == [Decimal("4")]
    assert result.resolved["line_items[].unit_price"] == [Decimal("120.00")]
    assert result.resolved["line_items[].line_total"] == [Decimal("480.00")]
    assert result.resolved["total_amount"] == Decimal("518.40")
    assert result.resolved["total_amount"] != Decimal("480.00")
    assert "line_items[].description" not in result.resolved
    assert "line_items[].description" in result.unresolved
    assert result.provenance["line_items[].quantity"].value == Decimal("4")
    assert result.provenance["line_items[].unit_price"].extractor == "money"


def test_two_line_items_resolve_in_document_order():
    result = run_prepass(TWO_LINE_ITEMS_DOC, get_schema("invoice"))
    assert result.resolved["line_items[].quantity"] == [Decimal("4"), Decimal("2")]
    assert result.resolved["line_items[].unit_price"] == [Decimal("120.00"), Decimal("800.00")]
    assert result.resolved["line_items[].line_total"] == [Decimal("480.00"), Decimal("1600.00")]
    assert result.resolved["total_amount"] == Decimal("2246.40")
    assert "line_items[].description" in result.unresolved


def test_no_line_item_segments_leaves_line_item_paths_unresolved():
    doc = "Invoice #INV-1001. Vendor: Acme Supply Co. Total: $518.40."
    result = run_prepass(doc, get_schema("invoice"))
    assert "line_items[].quantity" in result.unresolved
    assert "line_items[].unit_price" in result.unresolved
    assert "line_items[].line_total" in result.unresolved
    assert "line_items[].description" in result.unresolved


def test_vendor_name_is_semantic_and_never_resolved():
    result = run_prepass(INVOICE_DOC, get_schema("invoice"))
    assert "vendor_name" not in result.resolved
    assert "vendor_name" in result.unresolved
    assert "line_items[].description" not in result.resolved
    assert "line_items[].description" in result.unresolved


def test_unresolved_is_every_model_leaf_minus_resolved():
    spec = get_schema("invoice")
    result = run_prepass(INVOICE_DOC, spec)
    assert result.unresolved == sorted(set(leaf_paths(spec.model)) - set(result.resolved))


def test_failed_deterministic_field_appears_in_unresolved():
    spec = get_schema("invoice")
    doc = "Invoice #INV-1001. Vendor: Acme Supply Co. Total: $518.40."
    result = run_prepass(doc, spec)
    assert "invoice_date" in result.unresolved  # no date present
    assert "vendor_email" in result.unresolved  # no email present
    assert "total_amount" not in result.unresolved


def test_every_registered_schema_runs_without_error_and_never_populates_semantic_fields():
    for spec in all_schemas():
        result = run_prepass(GENERIC_DOC, spec)
        assert isinstance(result, PrepassResult)
        assert set(result.resolved).isdisjoint(spec.semantic_fields), spec.name
        assert set(result.resolved) <= set(spec.deterministic_fields), spec.name
        assert result.unresolved == sorted(set(leaf_paths(spec.model)) - set(result.resolved))


def test_generic_document_resolves_something_for_common_schemas():
    spec = get_schema("invoice")
    result = run_prepass(GENERIC_DOC, spec)
    # The generic doc's "Date:" label drives invoice_date, and $100.00 is the
    # first money after "Amount:" -- but total needs a "Total" label, so it stays
    # unresolved. Only assert the stable, label-driven resolutions.
    assert result.resolved.get("invoice_date") == date(2026, 4, 10)


def test_unresolved_prompt_fields_returns_the_residual_list():
    spec = get_schema("invoice")
    result = run_prepass(INVOICE_DOC, spec)
    residual = unresolved_prompt_fields(spec, result)
    assert residual == result.unresolved
    assert "vendor_name" in residual
    assert "invoice_number" not in residual


def test_dayfirst_and_region_parameters_flow_through():
    doc = "Date: 10/04/2026. Call (415) 555-2671."
    spec = get_schema("invoice")
    mdy = run_prepass(doc, spec, dayfirst=False)
    dmy = run_prepass(doc, spec, dayfirst=True)
    assert mdy.resolved.get("invoice_date") == date(2026, 10, 4)
    assert dmy.resolved.get("invoice_date") == date(2026, 4, 10)
