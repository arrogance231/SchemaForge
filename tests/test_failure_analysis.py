"""Tests for the automatic failure classifier (research direction §7).

Integration-style: every ``classify_record`` fixture is driven through the REAL
``evaluate_record`` from ``schemaforge.evaluation.metrics`` so that
``result.missing``/``result.hallucinated``/``result.errors`` are genuinely
produced, not hand-constructed.
"""

import schemaforge  # noqa: F401  (importing the package populates the registry)
from schemaforge.evaluation.harness import EvalRecord
from schemaforge.evaluation.metrics import evaluate_record
from schemaforge.failure_analysis.analyze import _CATEGORIES, build_report, classify_record
from schemaforge.failure_analysis import FailureInstance
from schemaforge.registry import get_schema

MED_SOURCE = (
    "Jane Doe, age 42, seen 2026-08-03 for Hypertension. "
    "Medications: lisinopril 10 mg. Attending: Dr. Smith. Follow-up 2026-09-03."
)
MED_REFERENCE = {
    "patient_name": "Jane Doe",
    "patient_age": 42,
    "visit_date": "2026-08-03",
    "diagnosis": "Hypertension",
    "physician_name": "Dr. Smith",
    "medications": [{"name": "lisinopril", "dosage": "10 mg"}],
    "follow_up_date": "2026-09-03",
}

INV_SOURCE = (
    "Invoice INV-1001 dated 2026-08-03, due 2026-09-03, from Acme Corp "
    "(billing@acme.com, +1 555 0100). Total 1,234.50 and tax 100.00."
)
INV_REFERENCE = {
    "invoice_number": "INV-1001",
    "invoice_date": "2026-08-03",
    "due_date": "2026-09-03",
    "vendor_name": "Acme Corp",
    "vendor_email": "billing@acme.com",
    "vendor_phone": "+1 555 0100",
    "total_amount": 1234.50,
    "tax_amount": 100.00,
}


def classify(schema_name, prediction, reference, source_text, tags=()):
    """Run the real evaluator then the classifier over one record."""
    record = EvalRecord(
        schema=schema_name,
        source_text=source_text,
        reference=reference,
        tags=list(tags),
    )
    spec = get_schema(schema_name)
    result = evaluate_record(prediction, reference, spec, source_text=source_text)
    return classify_record(record, prediction, result, spec)


def _by_category(instances):
    return {inst.category for inst in instances}


def test_missing_field():
    prediction = {
        "patient_name": "Jane Doe",
        "patient_age": 42,
        "visit_date": "2026-08-03",
        "physician_name": "Dr. Smith",
        "medications": [{"name": "lisinopril", "dosage": "10 mg"}],
        "follow_up_date": "2026-09-03",
    }
    instances = classify("medical_note", prediction, MED_REFERENCE, MED_SOURCE)
    assert len(instances) == 1
    inst = instances[0]
    assert inst.category == "missing_field"
    assert inst.path == "diagnosis"
    assert inst.predicted is None
    assert inst.reference == "Hypertension"
    assert inst.schema == "medical_note"


def test_hallucinated_field():
    prediction = {**INV_REFERENCE, "notes": "Made Up"}  # not a model leaf path
    instances = classify("invoice", prediction, INV_REFERENCE, INV_SOURCE)
    assert len(instances) == 1
    inst = instances[0]
    assert inst.category == "hallucinated_field"
    assert inst.path == "notes"
    assert inst.predicted == "Made Up"
    assert inst.reference is None


def test_schema_violation_does_not_short_circuit_other_failures():
    prediction = {
        "patient_name": "Jane Doe",
        "patient_age": 42,
        "visit_date": "2026-08-03",
        "physician_name": "Dr. Smith",
        "medications": [{"name": "lisinopril", "dosage": "10 mg"}],
        "follow_up_date": "not-a-date",  # wrong type -> schema-invalid
        # "diagnosis" omitted -> still reported as missing
    }
    instances = classify("medical_note", prediction, MED_REFERENCE, MED_SOURCE)
    categories = _by_category(instances)
    assert "schema_violation" in categories
    assert "missing_field" in categories
    assert "unclassified_mismatch" in categories  # the follow_up_date error
    schema_violation = next(inst for inst in instances if inst.category == "schema_violation")
    assert schema_violation.path is None
    assert schema_violation.predicted == prediction
    missing = next(inst for inst in instances if inst.category == "missing_field")
    assert missing.path == "diagnosis"


def test_ambiguate_emits_exactly_one_instance():
    prediction = {"diagnosis": "Totally Wrong", "patient_name": "Nobody"}
    instances = classify(
        "medical_note", prediction, MED_REFERENCE, MED_SOURCE, tags=["ambiguate"]
    )
    assert len(instances) == 1
    inst = instances[0]
    assert inst.category == "ambiguous_input"
    assert inst.path is None
    assert inst.predicted == prediction
    assert inst.reference == MED_REFERENCE
    assert inst.tags == ["ambiguate"]


def test_incorrect_normalization_wrong_ontology_entry():
    prediction = {**MED_REFERENCE, "diagnosis": "T2DM"}
    instances = classify("medical_note", prediction, MED_REFERENCE, MED_SOURCE)
    inst = next(i for i in instances if i.category == "incorrect_normalization")
    assert inst.path == "diagnosis"
    assert inst.predicted == "T2DM"
    assert inst.reference == "Hypertension"


def test_incorrect_nesting_correct_value_at_wrong_path():
    prediction = {
        "patient_name": "Dr. Smith",  # swapped with physician_name
        "patient_age": 42,
        "visit_date": "2026-08-03",
        "diagnosis": "Hypertension",
        "physician_name": "Jane Doe",
        "medications": [{"name": "lisinopril", "dosage": "10 mg"}],
        "follow_up_date": "2026-09-03",
    }
    instances = classify("medical_note", prediction, MED_REFERENCE, MED_SOURCE)
    nesting_paths = {i.path for i in instances if i.category == "incorrect_nesting"}
    assert nesting_paths == {"patient_name", "physician_name"}
    inst = next(i for i in instances if i.path == "physician_name")
    assert inst.category == "incorrect_nesting"
    assert inst.predicted == "Jane Doe"
    assert inst.reference == "Dr. Smith"


def test_wrong_entity_boundary_substring_span():
    source = (
        "Jane Doe, age 42, seen 2026-08-03 for Hypertension. "
        "Attending: Dr. Smith MD. Follow-up 2026-09-03."
    )
    reference = {**MED_REFERENCE, "physician_name": "Dr. Smith"}
    prediction = {**reference, "physician_name": "Dr. Smith MD"}
    instances = classify("medical_note", prediction, reference, source)
    inst = next(i for i in instances if i.category == "wrong_entity_boundary")
    assert inst.path == "physician_name"
    assert inst.predicted == "Dr. Smith MD"
    assert inst.reference == "Dr. Smith"


def test_wrong_inferred_value_neither_side_supported_by_source():
    source = (
        "Jane Doe, age 42, seen 2026-08-03. Medications: lisinopril 10 mg. "
        "Attending: Dr. Smith. Follow-up 2026-09-03."
    )
    reference = {**MED_REFERENCE, "diagnosis": "Migraine"}
    prediction = {**reference, "diagnosis": "Tension Headache"}
    instances = classify("medical_note", prediction, reference, source)
    inst = next(i for i in instances if i.category == "wrong_inferred_value")
    assert inst.path == "diagnosis"
    assert inst.predicted == "Tension Headache"
    assert inst.reference == "Migraine"


def test_explicit_null_field_classified_only_once_as_missing():
    prediction = {**INV_REFERENCE, "vendor_name": None}
    instances = classify("invoice", prediction, INV_REFERENCE, INV_SOURCE)
    by_path = [inst for inst in instances if inst.path == "vendor_name"]
    assert len(by_path) == 1
    inst = by_path[0]
    assert inst.category == "missing_field"
    assert inst.predicted is None
    assert inst.reference == "Acme Corp"
    assert not any(inst.category == "unclassified_mismatch" for inst in instances)


def test_unclassified_mismatch_numeric_disagreement():
    prediction = {**INV_REFERENCE, "total_amount": 2000.00}
    instances = classify("invoice", prediction, INV_REFERENCE, INV_SOURCE)
    inst = next(i for i in instances if i.category == "unclassified_mismatch")
    assert inst.path == "total_amount"
    assert inst.predicted == 2000.00
    assert inst.reference == 1234.50


def _instance(category, schema, tags):
    return FailureInstance(
        category=category,
        schema=schema,
        path="some.path",
        predicted=None,
        reference=None,
        tags=list(tags),
        source_text="",
    )


def test_build_report_counts_and_slices():
    instances = [
        _instance("missing_field", "invoice", ["ocr_noise"]),
        _instance("missing_field", "invoice", ["ocr_noise"]),
        _instance("missing_field", "contract", ["severity=0.6"]),
        _instance("hallucinated_field", "invoice", ["ocr_noise"]),
        _instance("hallucinated_field", "contract", ["severity=0.6", "implicit"]),
    ]
    report = build_report(instances, n_worst=2)

    assert report.by_category["missing_field"] == 3
    assert report.by_category["hallucinated_field"] == 2
    assert set(report.by_category) == set(_CATEGORIES)
    assert report.by_category["incorrect_normalization"] == 0

    assert report.by_category_and_schema["missing_field"] == {"invoice": 2, "contract": 1}
    assert report.by_category_and_schema["hallucinated_field"] == {"invoice": 1, "contract": 1}

    assert report.by_category_and_operator["missing_field"] == {"ocr_noise": 2}
    assert report.by_category_and_operator["hallucinated_field"] == {"ocr_noise": 1, "implicit": 1}
    assert all("severity=0.6" not in ops for ops in report.by_category_and_operator.values())

    # "first N encountered", not ranked-worst: order preserved, capped at n_worst.
    assert report.worst_examples["missing_field"] == instances[:2]
    assert report.worst_examples["hallucinated_field"] == instances[3:]
    assert report.worst_examples["schema_violation"] == []


def test_build_report_empty_input_is_all_zero():
    report = build_report([])
    assert all(count == 0 for count in report.by_category.values())
    assert set(report.by_category) == set(_CATEGORIES)
    assert report.by_category_and_schema == {}
    assert report.by_category_and_operator == {}
    assert set(report.worst_examples) == set(_CATEGORIES)
    assert all(examples == [] for examples in report.worst_examples.values())
