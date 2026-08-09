"""Automatic error classifier for research direction §7 (Failure Analysis).

Runs AFTER ``schemaforge.evaluation.metrics.evaluate_record`` and buckets every
field-level discrepancy into one of the 8 categories of §7, plus a 9th
``unclassified_mismatch`` catch-all: not every value mismatch cleanly fits one
of the named buckets, and silently mislabelling one is worse than an honest
catch-all.

Pure module: no network, file I/O, or torch anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from schemaforge.evaluation.harness import EvalRecord
from schemaforge.evaluation.json_utils import flatten
from schemaforge.evaluation.metrics import RecordResult
from schemaforge.registry import SchemaSpec, normalize_via_ontology

_CATEGORIES = (
    "missing_field",
    "incorrect_normalization",
    "wrong_entity_boundary",
    "wrong_inferred_value",
    "hallucinated_field",
    "schema_violation",
    "incorrect_nesting",
    "ambiguous_input",
    "unclassified_mismatch",
)

# Tags of this form carry a corruption severity, not an operator identity; the
# per-operator breakdown slices on everything else.
_OPERATOR_TAG_EXCLUDE = "severity="


@dataclass(frozen=True)
class FailureInstance:
    """One field-level or record-level failure in an evaluated record."""

    category: str
    schema: str
    path: str | None
    predicted: Any
    reference: Any
    tags: list[str]
    source_text: str


def _is_scalar(value: Any) -> bool:
    """True for JSON-scalar-comparable values (str/int/float/bool)."""
    return isinstance(value, (str, int, float, bool))


def _value_elsewhere(flat: dict[str, Any], value: Any, exclude_path: str) -> bool:
    """True when ``value`` occurs at a flattened path other than ``exclude_path``.

    List leaves are checked element-wise, since a list-of-object path flattens
    to a list even when it holds a single element.
    """
    for other_path, other_value in flat.items():
        if other_path == exclude_path:
            continue
        if other_value == value:
            return True
        if isinstance(other_value, list) and value in other_value:
            return True
    return False


def _classify_mismatch(
    pred_value: Any,
    ref_value: Any,
    spec: SchemaSpec,
    source_text: str,
    pred_flat: dict[str, Any],
    path: str,
) -> str:
    """Classify one present-in-both, disagreeing value pair.

    The precedence order is fixed by §7: an ontology mapping that fired on
    either side outranks everything else (the model used the vocabulary, just
    the wrong or differently-formed entry), then a correctly-valued-but-
    misplaced leaf, then a span-overlap, then a pure inference disagreement.
    """
    if (
        isinstance(pred_value, str)
        and isinstance(ref_value, str)
        and spec.ontology
    ):
        pred_norm = normalize_via_ontology(spec, pred_value)
        ref_norm = normalize_via_ontology(spec, ref_value)
        if pred_norm != pred_value or ref_norm != ref_value:
            return "incorrect_normalization"
    if _is_scalar(pred_value) and _is_scalar(ref_value) and _value_elsewhere(
        pred_flat, ref_value, path
    ):
        return "incorrect_nesting"
    if (
        isinstance(pred_value, str)
        and isinstance(ref_value, str)
        and pred_value.strip()
        and ref_value.strip()
    ):
        p = pred_value.casefold().strip()
        r = ref_value.casefold().strip()
        if (p in r or r in p) and p != r:
            return "wrong_entity_boundary"
        src = source_text.casefold()
        if p not in src and r not in src:
            return "wrong_inferred_value"
    return "unclassified_mismatch"


def classify_record(
    record: "EvalRecord",
    prediction: dict | None,
    result: "RecordResult",
    spec: "SchemaSpec",
) -> list[FailureInstance]:
    """Classify every field-level and record-level failure in one evaluated record.

    ``missing``/``hallucinated`` come straight from ``result`` (evaluate_record
    already computed them); ``errors`` is where the four per-field sub-categories
    of §7 are teased apart.  Record-level categories are ``schema_violation`` and
    ``ambiguous_input``, both emitted with ``path=None``.
    """
    instances: list[FailureInstance] = []

    if "ambiguate" in record.tags:
        instances.append(
            FailureInstance(
                category="ambiguous_input",
                schema=record.schema,
                path=None,
                predicted=prediction,
                reference=record.reference,
                tags=record.tags,
                source_text=record.source_text,
            )
        )
        # Ambiguous items are scored/reported separately (research direction
        # §3/§6); counting their fields into the other buckets too would double
        # count the same disagreement.
        return instances

    if not result.schema_valid:
        instances.append(
            FailureInstance(
                category="schema_violation",
                schema=record.schema,
                path=None,
                predicted=prediction,
                reference=record.reference,
                tags=record.tags,
                source_text=record.source_text,
            )
        )
        # No early return: a schema-invalid prediction can still carry per-field
        # failures worth reporting.

    pred_flat = flatten(prediction) if isinstance(prediction, dict) else {}
    ref_flat = flatten(record.reference)

    for path in result.missing:
        instances.append(
            FailureInstance(
                category="missing_field",
                schema=record.schema,
                path=path,
                predicted=None,
                reference=ref_flat.get(path),
                tags=record.tags,
                source_text=record.source_text,
            )
        )

    for path in result.hallucinated:
        instances.append(
            FailureInstance(
                category="hallucinated_field",
                schema=record.schema,
                path=path,
                predicted=pred_flat.get(path),
                reference=None,
                tags=record.tags,
                source_text=record.source_text,
            )
        )

    for path, (pred_value, ref_value) in result.errors.items():
        if pred_value is None:
            # evaluate_record's ``missing`` loop uses ``pred_flat.get(path)``,
            # so an explicit-null prediction lands in BOTH ``missing`` AND
            # ``errors`` (the key is present in ``pred_flat`` with value
            # ``None``, so it is also in the path intersection).  The ``missing``
            # loop above already emitted this failure; skip it here to avoid
            # double-counting the same field as both missing_field and
            # unclassified_mismatch.
            continue
        instances.append(
            FailureInstance(
                category=_classify_mismatch(
                    pred_value,
                    ref_value,
                    spec,
                    record.source_text,
                    pred_flat,
                    path,
                ),
                schema=record.schema,
                path=path,
                predicted=pred_value,
                reference=ref_value,
                tags=record.tags,
                source_text=record.source_text,
            )
        )

    return instances


@dataclass
class FailureReport:
    """Aggregated failure analysis over a batch of evaluated records."""

    by_category: dict[str, int]
    by_category_and_schema: dict[str, dict[str, int]]
    by_category_and_operator: dict[str, dict[str, int]]
    worst_examples: dict[str, list[FailureInstance]]


def build_report(instances: list[FailureInstance], *, n_worst: int = 5) -> FailureReport:
    """Aggregate a flat list of :class:`FailureInstance` into a :class:`FailureReport`.

    ``worst_examples[category]`` holds the FIRST ``n_worst`` instances
    encountered for that category, in the order given in ``instances`` -- there
    is no scalar severity score available at this layer, so "first N
    encountered" is the honest choice, not a ranked-worst sample.
    ``by_category`` is keyed by every category in ``_CATEGORIES`` (zeros
    included) so downstream consumers need no ``.get(..., 0)`` guards.  An
    empty ``instances`` list yields an all-zero report.
    """
    by_category = {category: 0 for category in _CATEGORIES}
    by_category_and_schema: dict[str, dict[str, int]] = {}
    by_category_and_operator: dict[str, dict[str, int]] = {}
    worst_examples: dict[str, list[FailureInstance]] = {
        category: [] for category in _CATEGORIES
    }

    for instance in instances:
        by_category[instance.category] += 1
        by_category_and_schema.setdefault(instance.category, {})
        schema_counts = by_category_and_schema[instance.category]
        schema_counts[instance.schema] = schema_counts.get(instance.schema, 0) + 1
        for tag in instance.tags:
            if tag.startswith(_OPERATOR_TAG_EXCLUDE):
                continue
            by_category_and_operator.setdefault(instance.category, {})
            operator_counts = by_category_and_operator[instance.category]
            operator_counts[tag] = operator_counts.get(tag, 0) + 1
        if len(worst_examples[instance.category]) < n_worst:
            worst_examples[instance.category].append(instance)

    return FailureReport(
        by_category=by_category,
        by_category_and_schema=by_category_and_schema,
        by_category_and_operator=by_category_and_operator,
        worst_examples=worst_examples,
    )
