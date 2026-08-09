"""Tests for field-level value comparison and micro-averaged metrics (part 1)."""

import pytest
from datetime import date
from decimal import Decimal
from pydantic import BaseModel

import schemaforge  # noqa: F401  (importing the package populates the registry)
from schemaforge.evaluation.metrics import (
    _try_decimal,
    aggregate,
    evaluate_record,
    values_equal,
)
from schemaforge.registry import SchemaSpec, get_schema

MED_REFERENCE = {
    "patient_name": "Jane Doe",
    "patient_age": 42,
    "visit_date": "2026-08-03",
    "diagnosis": "Hypertension",
    "physician_name": "Dr. Smith",
    "medications": [{"name": "lisinopril", "dosage": "10 mg"}],
    "follow_up_date": "2026-09-03",
}

MED_SOURCE = (
    "Jane Doe, age 42, seen 2026-08-03 for Hypertension. "
    "Medications: lisinopril 10 mg. Attending: Dr. Smith. Follow-up 2026-09-03."
)


def test_perfect_prediction():
    spec = get_schema("medical_note")
    res = evaluate_record(dict(MED_REFERENCE), MED_REFERENCE, spec, source_text=MED_SOURCE)
    assert res.exact_match is True
    assert res.schema_valid is True
    assert res.n_correct == res.n_predicted == res.n_reference == 8
    assert res.missing == []
    assert res.hallucinated == []
    assert res.errors == {}
    agg = aggregate([res])
    assert agg["exact_match"] == 1.0
    assert agg["schema_validity"] == 1.0
    assert agg["field_precision"] == pytest.approx(1.0)
    assert agg["field_recall"] == pytest.approx(1.0)
    assert agg["field_f1"] == pytest.approx(1.0)
    assert agg["n_records"] == 1.0


def test_prediction_none_is_missing_everything_without_raising():
    spec = get_schema("medical_note")
    res = evaluate_record(None, MED_REFERENCE, spec, source_text=MED_SOURCE)
    assert res.schema_valid is False
    assert res.exact_match is False
    assert res.n_predicted == 0
    assert res.n_reference == 8
    assert len(res.missing) == 8
    agg = aggregate([res])
    assert agg["field_precision"] == 0.0
    assert agg["field_recall"] == 0.0
    assert agg["field_f1"] == 0.0


def test_ontology_normalization_counts_as_correct():
    spec = get_schema("medical_note")
    assert values_equal("HTN", "Hypertension", spec=spec) is True
    pred = dict(MED_REFERENCE)
    pred["diagnosis"] = "HTN"
    res = evaluate_record(pred, MED_REFERENCE, spec, source_text=MED_SOURCE)
    assert "diagnosis" not in res.errors
    assert res.n_correct == res.n_reference == 8
    assert res.exact_match is True


def test_money_string_and_numeric_forms_equal():
    assert values_equal("$1,234.50", 1234.5) is True
    assert values_equal("1,234.50", "1234.5") is True
    assert values_equal(1234.50, "1234.5") is True
    assert values_equal("$1,234.50", "1235.50") is False


def test_date_forms_equal_when_parseable():
    assert values_equal(date(2026, 8, 3), "2026-08-03") is True
    assert values_equal("Aug 3, 2026", "2026-08-03") is True
    assert values_equal("08/03/2026", date(2026, 8, 3)) is True
    assert values_equal("2026-08-03", "2026-08-04") is False


def test_lists_compare_as_multisets_order_independent():
    assert values_equal(["aspirin", "metformin"], ["metformin", "aspirin"]) is True
    assert values_equal(["aspirin"], ["metformin"]) is False
    assert values_equal(["aspirin", "aspirin"], ["aspirin"]) is False


def test_none_equals_only_none():
    assert values_equal(None, None) is True
    assert values_equal(None, "x") is False
    assert values_equal("x", None) is False


def test_hallucination_flags_strings_but_not_numbers():
    spec = get_schema("invoice")
    reference = {
        "invoice_number": "INV-1001",
        "invoice_date": "2026-08-03",
        "due_date": "2026-09-03",
        "vendor_name": "Acme Corp",
        "vendor_email": "billing@acme.com",
        "vendor_phone": "+1 555 0100",
        "total_amount": 1234.50,
        "tax_amount": 100.00,
    }
    source_text = (
        "Invoice INV-1001 dated 2026-08-03, due 2026-09-03, from Acme Corp "
        "(billing@acme.com, +1 555 0100). Total 1,234.50 and tax 100.00."
    )
    prediction = {
        "invoice_number": "INV-1001",
        "invoice_date": "2026-08-03",
        "due_date": "2026-09-03",
        "vendor_name": "Ghost Corporation",  # absent from the source -> hallucinated
        "vendor_email": "billing@acme.com",
        "vendor_phone": "+1 555 0100",
        "total_amount": "$1,234.50",  # numeric, formatting differs -> NOT flagged
        "tax_amount": 100.00,
    }
    res = evaluate_record(prediction, reference, spec, source_text=source_text)
    assert "vendor_name" in res.hallucinated
    assert "total_amount" not in res.hallucinated


def test_micro_precision_pools_over_leaf_counts():
    class Mini(BaseModel):
        a: str | None = None
        b: str | None = None

    spec = SchemaSpec(
        name="mini",
        model=Mini,
        deterministic_fields=frozenset(),
        semantic_fields=frozenset({"a", "b"}),
        ontology={},
    )
    # Record 1: 1 correct of 2 predicted leaves (a correct, b wrong).
    # Record 2: 1 correct of 1 predicted leaf (a correct).
    # Pooled micro precision = sum(n_correct) / sum(n_predicted) = (1+1)/(2+1) = 2/3.
    # A macro mean would instead give (1/2 + 1/1)/2 = 3/4 — we assert the micro one.
    r1 = evaluate_record({"a": "x", "b": "zzz"}, {"a": "x", "b": "y"}, spec)
    r2 = evaluate_record({"a": "x"}, {"a": "x"}, spec)
    agg = aggregate([r1, r2])
    assert agg["field_precision"] == pytest.approx(2 / 3)
    assert agg["field_recall"] == pytest.approx(2 / 3)
    assert agg["field_f1"] == pytest.approx(2 / 3)
    assert agg["n_records"] == 2.0


def test_missing_field_rate_counts_leaf_units_not_paths():
    spec = get_schema("medical_note")
    ref = {
        "patient_name": "Jane",
        "medications": [
            {"name": "aspirin", "dosage": "81mg"},
            {"name": "lisinopril", "dosage": "10mg"},
        ],
    }
    pred = {"patient_name": "Jane"}
    res = evaluate_record(pred, ref, spec)
    # n_reference = 5 leaf units (patient_name + 2 medications x 2 leaves).
    # Both medication paths are missing entirely -> 4/5 = 0.8, not 2/5 = 0.4.
    assert res.n_reference == 5
    assert res.missing == ["medications[].dosage", "medications[].name"]
    assert res.n_missing == 4
    agg = aggregate([res])
    assert agg["missing_field_rate"] == pytest.approx(4 / 5)


def test_missing_field_rate_counts_shorter_list_shortfall():
    class Tags(BaseModel):
        note: str | None = None
        tags: list[str] | None = None

    spec = SchemaSpec(
        name="tags",
        model=Tags,
        deterministic_fields=frozenset(),
        semantic_fields=frozenset({"note", "tags"}),
        ontology={},
    )
    ref = {"note": "x", "tags": ["a", "b", "c"]}
    pred = {"note": "x", "tags": ["a"]}
    res = evaluate_record(pred, ref, spec)
    # Reference list has 3 elements, prediction supplies 1: the 2-element
    # shortfall is counted (the old path-count code counted 0 because the path
    # was present). n_reference = 1 + 3 = 4 leaf units, n_missing = 2.
    assert res.missing == []
    assert res.n_reference == 4
    assert res.n_missing == 2
    agg = aggregate([res])
    assert agg["missing_field_rate"] == pytest.approx(2 / 4)


def test_hallucination_rate_counts_list_elements():
    spec = get_schema("invoice")
    reference = {
        "invoice_number": "INV-1001",
        "invoice_date": "2026-08-03",
        "due_date": "2026-09-03",
        "vendor_name": "Acme Corp",
        "vendor_email": "billing@acme.com",
        "vendor_phone": "+1 555 0100",
        "total_amount": 1234.50,
        "tax_amount": 100.00,
    }
    source_text = (
        "Invoice INV-1001 dated 2026-08-03, due 2026-09-03, from Acme Corp "
        "(billing@acme.com, +1 555 0100). Total 1,234.50 and tax 100.00."
    )
    prediction = {
        "invoice_number": "INV-1001",
        "invoice_date": "2026-08-03",
        "due_date": "2026-09-03",
        "vendor_name": "Acme Corp",
        "vendor_email": "billing@acme.com",
        "vendor_phone": "+1 555 0100",
        "total_amount": 1234.50,
        "tax_amount": 100.00,
        "line_items": [{"description": "Widget"}, {"description": "Gadget"}],
    }
    res = evaluate_record(prediction, reference, spec, source_text=source_text)
    # One path, but 2 unsupported list elements -> 2 leaf units, not 1.
    assert res.hallucinated == ["line_items[].description"]
    assert res.n_hallucinated == 2
    agg = aggregate([res])
    # n_predicted = 8 scalar leaves + 2 list elements = 10.
    assert agg["hallucination_rate"] == pytest.approx(2 / 10)


def test_partial_credit_for_shorter_list():
    spec = get_schema("medical_note")
    ref = {"medications": [{"name": "aspirin"}, {"name": "lisinopril"}]}
    pred = {"medications": [{"name": "aspirin"}]}  # 1 of 2 correct
    res = evaluate_record(pred, ref, spec)
    assert res.n_correct == 1
    assert res.n_predicted == 1
    assert res.n_reference == 2
    assert res.errors == {"medications[].name": (["aspirin"], ["aspirin", "lisinopril"])}


def test_partial_credit_two_of_three_correct():
    spec = get_schema("medical_note")
    ref = {
        "medications": [
            {"name": "aspirin"},
            {"name": "lisinopril"},
            {"name": "metformin"},
        ]
    }
    pred = {"medications": [{"name": "aspirin"}, {"name": "metformin"}]}  # 2 of 3 correct
    res = evaluate_record(pred, ref, spec)
    assert res.n_correct == 2
    assert res.n_predicted == 2
    assert res.n_reference == 3


def test_try_decimal_european_and_us_forms():
    # Both-separator rule (identical to extractors._normalize_money): the LAST
    # separator is the decimal separator, every other is thousands grouping.
    assert _try_decimal("1.234,50") == Decimal("1234.50")
    assert _try_decimal("$1,234.50") == Decimal("1234.50")
    assert _try_decimal("1,234.50") == Decimal("1234.50")
    assert _try_decimal("1,23.50") is None  # invalid thousands grouping
    assert values_equal("1.234,50", 1234.50) is True


def test_try_decimal_single_separator_group_size_rule():
    # Single-separator rule: a 3-digit trailing group after a 1-3 digit leading
    # group reads as THOUSANDS, so "1.234" and "1,234" both read as 1234.
    assert _try_decimal("1.234") == Decimal("1234")
    assert _try_decimal("1,234") == Decimal("1234")
    assert _try_decimal("1234.50") == Decimal("1234.50")
    assert _try_decimal("1234,50") == Decimal("1234.50")
    assert _try_decimal("1.5") == Decimal("1.5")
