"""Tests for the evaluation harness: tag-based exclusion from the accuracy pool.

Per SCHEMAFORGE_V2_RESEARCH_DIRECTION.md §3, ``ambiguate`` items are
deliberately ambiguous and belong in the confidence evaluation, not in the
accuracy numerator.  ``evaluate`` must therefore exclude them from ``overall``
and ``by_schema`` while keeping them visible under ``excluded`` and in
``by_tag``.
"""

import schemaforge  # noqa: F401  (importing the package populates the registry)
from schemaforge.evaluation.harness import EvalRecord, evaluate, format_table

MEDICAL_REFERENCE = {
    "patient_name": "Jane Doe",
    "patient_age": 42,
    "visit_date": "2026-08-03",
    "diagnosis": "Hypertension",
    "physician_name": "Dr. Smith",
    "medications": [{"name": "lisinopril", "dosage": "10 mg"}],
    "follow_up_date": "2026-09-03",
}

MEDICAL_SOURCE = (
    "Jane Doe, age 42, seen 2026-08-03 for Hypertension. "
    "Medications: lisinopril 10 mg. Attending: Dr. Smith. Follow-up 2026-09-03."
)


def _clean_record() -> EvalRecord:
    return EvalRecord(
        schema="medical_note",
        source_text=MEDICAL_SOURCE,
        reference=MEDICAL_REFERENCE,
        tags=["ocr_noise"],
    )


def _ambiguate_record() -> EvalRecord:
    return EvalRecord(
        schema="medical_note",
        source_text=MEDICAL_SOURCE,
        reference=MEDICAL_REFERENCE,
        tags=["ambiguate"],
    )


def test_ambiguate_record_is_excluded_from_overall_and_by_schema():
    clean = _clean_record()
    ambiguous = _ambiguate_record()

    def predict(record):
        return record.reference

    results = evaluate([clean, ambiguous], predict)
    assert results["overall"]["n_records"] == 1.0
    assert results["by_schema"]["medical_note"]["n_records"] == 1.0
    assert "ambiguate" in results["excluded"]
    assert results["excluded"]["ambiguate"]["n_records"] == 1.0
    assert results["by_tag"]["ambiguate"]["n_records"] == 1.0


def test_ambiguate_record_leaves_do_not_contribute_to_overall_metrics():
    clean = _clean_record()
    ambiguous = _ambiguate_record()

    def predict(record):
        return record.reference if "ambiguate" not in record.tags else {}

    results = evaluate([clean, ambiguous], predict)
    assert results["overall"]["n_records"] == 1.0
    assert results["overall"]["field_recall"] == 1.0
    assert results["excluded"]["ambiguate"]["field_recall"] == 0.0
    assert results["by_tag"]["ambiguate"]["n_records"] == 1.0


def test_exclude_tags_empty_restores_pool_everything_behaviour():
    clean = _clean_record()
    ambiguous = _ambiguate_record()

    def predict(record):
        return record.reference if "ambiguate" not in record.tags else {}

    pooled = evaluate([clean, ambiguous], predict, exclude_tags=())
    assert pooled["overall"]["n_records"] == 2.0
    assert pooled["excluded"] == {}
    assert pooled["overall"]["field_recall"] < 1.0


def test_format_table_labelled_excluded_columns():
    results = evaluate([_clean_record(), _ambiguate_record()], lambda record: record.reference)
    table = format_table(results)
    assert "excluded from overall:ambiguate" in table
    assert "overall" in table.splitlines()[0]
