"""Tests for the teacher-output validation gate (research direction §4.2)."""

import json

import pytest

import schemaforge  # noqa: F401  (importing the package populates the registry)
from schemaforge.validation.gate import (
    GateResult,
    _fuzzy_supported,
    rejection_rate,
    validate_teacher_output,
)

INVOICE_SOURCE = (
    "Invoice INV-2024-001 from ACME Supplies, billing@acme.example, 555-0100. "
    "Date: 2024-06-01. Due: 2024-07-01. 2x widgets at 50.00 each, 1x gadgets at 25.50. "
    "Total 125.50 including tax 10.50."
)


def _invoice_output(**overrides) -> str:
    doc = {
        "invoice_number": "INV-2024-001",
        "invoice_date": "2024-06-01",
        "due_date": "2024-07-01",
        "vendor_name": "ACME Supplies",
        "vendor_email": "billing@acme.example",
        "vendor_phone": "555-0100",
        "total_amount": 125.50,
        "tax_amount": 10.50,
        "line_items": [
            {"description": "widgets", "quantity": 2, "unit_price": 50.00, "line_total": 100.00},
            {"description": "gadgets", "quantity": 1, "unit_price": 25.50, "line_total": 25.50},
        ],
    }
    doc.update(overrides)
    return json.dumps(doc)


def test_valid_invoice_output_accepted():
    result = validate_teacher_output("invoice", INVOICE_SOURCE, _invoice_output())
    assert result.accepted is True
    assert result.reasons == []
    assert result.unsupported_fields == []
    assert isinstance(result.parsed, dict)
    assert result.parsed["vendor_name"] == "ACME Supplies"


def test_malformed_json_rejected_at_step1():
    result = validate_teacher_output("invoice", INVOICE_SOURCE, "the model rambles {unclosed")
    assert result.accepted is False
    assert result.parsed is None
    assert result.unsupported_fields == []
    assert result.reasons[0].startswith("step1_json_parse")


def test_schema_violation_rejected_at_step2():
    result = validate_teacher_output("invoice", INVOICE_SOURCE, '{"invoice_date": "not-a-date"}')
    assert result.accepted is False
    assert result.reasons[0].startswith("step2_schema_validation")
    # parsed stays as the raw (unvalidated) dict for inspection on a step2 rejection.
    assert result.parsed == {"invoice_date": "not-a-date"}
    assert result.unsupported_fields == []


def test_schema_violation_wrong_type_rejected_at_step2():
    result = validate_teacher_output("invoice", INVOICE_SOURCE, '{"line_items": "not-a-list"}')
    assert result.accepted is False
    assert result.reasons[0].startswith("step2_schema_validation")


def test_unsupported_semantic_value_rejected_at_step3():
    result = validate_teacher_output(
        "invoice",
        INVOICE_SOURCE,
        _invoice_output(vendor_name="Fantasy Corp"),
    )
    assert result.accepted is False
    assert result.reasons[0].startswith("step3_source_support")
    assert result.unsupported_fields == ["vendor_name"]


def test_unsupported_value_in_repeated_list_item_rejected_at_step3():
    result = validate_teacher_output(
        "invoice",
        INVOICE_SOURCE,
        '{"vendor_name": "ACME Supplies", "line_items": [{"description": "gadgets"}, {"description": "frobnicators"}]}',
    )
    assert result.accepted is False
    assert result.reasons[0].startswith("step3_source_support")
    # The []-collapsed path is reported once, not per element.
    assert result.unsupported_fields == ["line_items[].description"]


def test_semantic_value_present_in_source_accepted():
    result = validate_teacher_output(
        "invoice",
        "Shipped by Acme Corp on request.",
        '{"vendor_name": "Acme Corp"}',
    )
    assert result.accepted is True
    assert result.reasons == []
    assert result.parsed["vendor_name"] == "Acme Corp"


def test_semantic_value_case_insensitive_substring_accepted():
    result = validate_teacher_output(
        "invoice",
        "SHIPPED BY ACME CORP.",
        '{"vendor_name": "acme corp"}',
    )
    assert result.accepted is True


def test_semantic_value_via_ontology_surface_accepted():
    # Source spells the condition "HTN"; the teacher outputs the canonical
    # "Hypertension", which is derivable via the medical_note ontology.
    result = validate_teacher_output(
        "medical_note",
        "Patient Jane Doe presented with HTN and COPD.",
        '{"diagnosis": "Hypertension", "patient_name": "Jane Doe"}',
    )
    assert result.accepted is True
    assert result.reasons == []


def test_extra_top_level_field_rejected_at_step4():
    # These models use pydantic's default extra="ignore", so an invented
    # top-level key survives step2 and is only caught by the step4 check.
    result = validate_teacher_output(
        "invoice",
        INVOICE_SOURCE,
        '{"invoice_number": "INV-2024-001", "made_up_field": "invented"}',
    )
    assert result.accepted is False
    assert result.reasons[0].startswith("step4_no_overassertion")
    assert "made_up_field" in result.reasons[0]


def test_extra_key_inside_list_item_rejected_at_step4():
    # pydantic silently ignores an extra key inside a list[SubModel] item; step4
    # compares the raw pre-validation flatten against leaf_paths to reject it.
    result = validate_teacher_output(
        "invoice",
        INVOICE_SOURCE,
        '{"invoice_number": "INV-2024-001", "line_items": [{"description": "widgets", "quantity": 2, "extra_in_item": "y"}]}',
    )
    assert result.accepted is False
    assert result.reasons[0].startswith("step4_no_overassertion")
    assert "line_items[].extra_in_item" in result.reasons[0]


def test_rejection_rate_mixed():
    ok = GateResult(accepted=True, parsed={}, reasons=[], unsupported_fields=[])
    bad = GateResult(accepted=False, parsed=None, reasons=["x"], unsupported_fields=[])
    assert rejection_rate([ok, bad]) == 0.5
    assert rejection_rate([ok]) == 0.0
    assert rejection_rate([bad]) == 1.0


def test_rejection_rate_empty_raises():
    with pytest.raises(ValueError):
        rejection_rate([])


def test_typo_denoised_value_rejected_by_default_accepted_with_fuzzy():
    source = "Shipped by Acme Crporation on request."
    typo_fixed = '{"vendor_name": "Acme Corporation"}'
    assert validate_teacher_output("invoice", source, typo_fixed).accepted is False
    assert (
        validate_teacher_output(
            "invoice", source, typo_fixed, fuzzy_support=True
        ).accepted
        is True
    )


def test_genuine_hallucination_rejected_even_with_fuzzy():
    source = "Invoice from ACME Supplies for the quarterly audit."
    result = validate_teacher_output(
        "invoice",
        source,
        '{"vendor_name": "Quantum Dynamics"}',
        fuzzy_support=True,
    )
    assert result.accepted is False
    assert result.reasons[0].startswith("step3_source_support")
    assert result.unsupported_fields == ["vendor_name"]


def test_fuzzy_word_count_mismatch_rejected():
    source = "From Acme."
    result = validate_teacher_output(
        "invoice",
        source,
        '{"vendor_name": "Acme Corporation"}',
        fuzzy_support=True,
    )
    assert result.accepted is False
    assert result.reasons[0].startswith("step3_source_support")


def test_fuzzy_supported_near_identical_window_true():
    assert _fuzzy_supported("Acme Corporation", "shipped by acme crporation inc") is True
    assert _fuzzy_supported("ACME Supplies", "billing from acme supplies on file") is True


def test_fuzzy_supported_common_short_words_false():
    assert _fuzzy_supported("blue sky over the hills", "the end of the road") is False


def test_fuzzy_supported_empty_value_false():
    assert _fuzzy_supported("   ", "anything here") is False
    assert _fuzzy_supported("", "anything here") is False


def test_fuzzy_supported_source_too_short_false():
    assert _fuzzy_supported("two words here", "one") is False
