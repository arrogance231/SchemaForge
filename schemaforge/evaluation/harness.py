"""Field-level evaluation harness.

Runs a predictor over a JSONL dataset of records and slices the results per
schema and per corruption tag, per SCHEMAFORGE_V2_RESEARCH_DIRECTION.md §6
("slicing is the point").  No model loading, no torch import, no network
access anywhere in this module.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from schemaforge.evaluation.json_utils import parse_json
from schemaforge.evaluation.metrics import RecordResult, aggregate, evaluate_record
from schemaforge.registry import get_schema

_METRIC_ORDER = (
    "exact_match",
    "schema_validity",
    "field_precision",
    "field_recall",
    "field_f1",
    "hallucination_rate",
    "missing_field_rate",
    "n_records",
)


@dataclass
class EvalRecord:
    """One evaluation example: a schema name, source text, gold reference and tags."""

    schema: str
    source_text: str
    reference: dict
    tags: list[str]


def load_records(path: str) -> list[EvalRecord]:
    """Read a JSONL file of evaluation records.

    Each non-empty line is a JSON object with ``schema`` (str), ``source_text``
    (str), ``reference`` (object) and optionally ``tags`` (list of str).
    A malformed line raises ``ValueError`` naming the line number.
    """
    records: list[EvalRecord] = []
    with open(path, "r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {lineno} of {path}: {exc}") from exc
            if not isinstance(data, dict):
                raise ValueError(f"line {lineno} of {path}: record must be a JSON object")
            schema = data.get("schema")
            if not isinstance(schema, str) or not schema:
                raise ValueError(f"line {lineno} of {path}: missing non-empty 'schema'")
            reference = data.get("reference")
            if not isinstance(reference, dict):
                raise ValueError(f"line {lineno} of {path}: 'reference' must be a JSON object")
            tags = data.get("tags") or []
            if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
                raise ValueError(f"line {lineno} of {path}: 'tags' must be a list of strings")
            records.append(
                EvalRecord(
                    schema=schema,
                    source_text=data.get("source_text") or "",
                    reference=reference,
                    tags=list(tags),
                )
            )
    return records


def _resolve_prediction(predict_fn: Callable[[EvalRecord], str | dict | None], record: EvalRecord) -> dict | None:
    """Invoke ``predict_fn`` and normalize its return to a dict or ``None``."""
    output = predict_fn(record)
    if output is None or isinstance(output, dict):
        return output
    if isinstance(output, str):
        return parse_json(output)
    raise TypeError(
        f"predict_fn must return str, dict, or None; got {type(output).__name__}"
    )


def evaluate(
    records: list[EvalRecord],
    predict_fn: Callable[[EvalRecord], str | dict | None],
    *,
    slice_by_tag: bool = True,
    exclude_tags: Sequence[str] = ("ambiguate",),
) -> dict:
    """Evaluate ``predict_fn`` over ``records`` and slice the aggregate metrics.

    ``predict_fn(record)`` returns a raw model string (parsed with
    ``parse_json``), a dict (used directly), or ``None`` (system declined).
    The result is ``{"overall": {...}, "by_schema": {name: {...}},
    "by_tag": {tag: {...}}, "excluded": {tag: {...}}}`` where every value is
    the output of :func:`aggregate`.  Per research direction §6 slicing is
    mandatory: the aggregate alone is not a usable result.  Slicing by tag is
    controlled by ``slice_by_tag``; records with no tags appear only in
    ``overall``.

    Records carrying any tag in ``exclude_tags`` are excluded from ``overall``
    and from ``by_schema``: their gold reference is contestable by design (an
    ``ambiguate`` item belongs in the confidence evaluation, research direction
    §3, not in the accuracy numerator).  They remain visible under the new
    ``excluded`` key as ``{tag: aggregate(group)}`` so the numbers are not
    discarded, and they are still sliced in ``by_tag`` -- their own tag slice
    is exactly where they are meaningful.  ``exclude_tags=()`` restores the
    pool-everything behaviour.
    """
    excluded = set(exclude_tags) if exclude_tags else set()

    evaluated: list[tuple[EvalRecord, RecordResult]] = []
    for record in records:
        prediction = _resolve_prediction(predict_fn, record)
        result = evaluate_record(
            prediction,
            record.reference,
            get_schema(record.schema),
            source_text=record.source_text,
        )
        evaluated.append((record, result))

    by_schema: dict[str, list[RecordResult]] = {}
    by_tag: dict[str, list[RecordResult]] = {}
    for record, result in evaluated:
        if excluded.isdisjoint(record.tags):
            by_schema.setdefault(record.schema, []).append(result)
        if slice_by_tag:
            for tag in record.tags:
                by_tag.setdefault(tag, []).append(result)

    return {
        "overall": aggregate(
            [result for record, result in evaluated if excluded.isdisjoint(record.tags)]
        ),
        "by_schema": {
            name: aggregate(group) for name, group in sorted(by_schema.items())
        },
        "by_tag": {
            tag: aggregate(group) for tag, group in sorted(by_tag.items())
        },
        "excluded": {
            tag: aggregate(
                [result for record, result in evaluated if tag in record.tags]
            )
            for tag in sorted(excluded)
        },
    }


def _format_value(metric: str, value: Any) -> str:
    """Format one aggregate value: counts as integers, everything else to 4 decimals."""
    if metric == "n_records":
        return str(int(value)) if isinstance(value, float) and value.is_integer() else f"{value:.4f}"
    return f"{value:.4f}"


def format_table(results: dict) -> str:
    """Render evaluation ``results`` as a dependency-free plain-text table.

    Columns are the ``overall`` slice followed by each ``by_schema:name``,
    ``by_tag:tag`` and ``excluded from overall:tag`` slice; rows are the
    metrics in ``_METRIC_ORDER``.  The ``excluded`` columns are the records
    that carry an ``exclude_tags`` tag and are therefore NOT counted in
    ``overall``.
    """
    columns: list[tuple[str, dict]] = [("overall", results.get("overall", {}))]
    for label, group in (
        ("by_schema", results.get("by_schema", {})),
        ("by_tag", results.get("by_tag", {})),
        ("excluded from overall", results.get("excluded", {})),
    ):
        for name, agg in group.items():
            columns.append((f"{label}:{name}", agg))

    widths = [len("metric")] + [len(name) for name, _ in columns]
    lines = [
        "  ".join(
            ["metric".ljust(widths[0])]
            + [name.ljust(width) for (name, _), width in zip(columns, widths[1:])]
        )
    ]
    for metric in _METRIC_ORDER:
        row = [metric.ljust(widths[0])]
        for (_, agg), width in zip(columns, widths[1:]):
            row.append(_format_value(metric, agg.get(metric, 0.0)).ljust(width))
        lines.append("  ".join(row))
    return "\n".join(lines)


if __name__ == "__main__":
    import schemaforge  # noqa: F401  # populate the registry on import

    demo_record = EvalRecord(
        schema="medical_note",
        source_text=(
            "Jane Doe, age 42, seen 2026-08-03 for Hypertension. "
            "Medications: lisinopril 10 mg. Attending: Dr. Smith. Follow-up 2026-09-03."
        ),
        reference={
            "patient_name": "Jane Doe",
            "patient_age": 42,
            "visit_date": "2026-08-03",
            "diagnosis": "Hypertension",
            "physician_name": "Dr. Smith",
            "medications": [{"name": "lisinopril", "dosage": "10 mg"}],
            "follow_up_date": "2026-09-03",
        },
        tags=["abbreviate"],
    )

    def demo_predict(record: EvalRecord) -> dict:
        return {
            "patient_name": "Jane Doe",
            "patient_age": 42,
            "visit_date": "2026-08-03",
            "diagnosis": "HTN",
            "physician_name": "Dr. Smith",
            "medications": [{"name": "lisinopril", "dosage": "10 mg"}],
            "follow_up_date": "2026-09-03",
        }

    print(format_table(evaluate([demo_record], demo_predict)))
