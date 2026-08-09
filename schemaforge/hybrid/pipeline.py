"""Hybrid merge: combine the pre-pass's resolved fields with a model prediction.

Research direction §6's ``hybrid (rules → SchemaForge)`` system.  The pre-pass
owns ``SchemaSpec.deterministic_fields`` (per §0/§2 a field it resolves is
never asked of the model, so a raw model output that happens to repeat one of
those paths is overridden by the rule-resolved value); the model owns every
``semantic_fields`` leaf plus any deterministic leaf the pre-pass failed to
resolve.  ``merge_prediction`` folds both into ONE nested JSON prediction ready
for ``evaluate_record``.  There is no model call, no network access and no
randomness in this module.
"""

from __future__ import annotations

from typing import Any

from schemaforge.evaluation.json_utils import flatten
from schemaforge.registry import SchemaSpec

__all__ = [
    "merge_prediction",
    "rules_only_prediction",
]


def _unflatten(flat: dict[str, Any]) -> dict:
    """Rebuild a nested JSON dict from dotted leaf paths (the inverse of ``flatten``).

    A ``path[]``-suffixed segment names a list whose per-element leaf values
    ``flatten`` collected into a list; the list node holds one dict per element
    and the leaf is assigned to each element by index, so parallel paths zip
    into per-item records.  ``flatten`` drops empty dict/list values, so an
    empty list is not recoverable and the inverse is lossy exactly there.
    """
    root: dict[str, Any] = {}
    for path, value in flat.items():
        _place(root, path, value)
    return root


def _place(node: dict[str, Any], path: str, value: Any) -> None:
    """Descend ``path`` from ``node``, assigning ``value`` at the leaf.

    The list at a repeated segment grows to the longest value list seen for any
    path under it, so a shorter parallel list simply leaves later items without
    that leaf rather than corrupting the already-built elements.
    """
    if "." in path:
        segment, rest = path.split(".", 1)
    else:
        segment, rest = path, ""
    repeated = segment.endswith("[]")
    if repeated:
        segment = segment[:-2]
    if not rest:
        node[segment] = value
        return
    if repeated:
        elements = node.setdefault(segment, [])
        items = value if isinstance(value, list) else [value]
        while len(elements) < len(items):
            elements.append({})
        for i, item in enumerate(items):
            _place(elements[i], rest, item)
    else:
        _place(node.setdefault(segment, {}), rest, value)


def merge_prediction(
    prepass_resolved: dict[str, Any],
    model_prediction: dict | None,
    spec: "SchemaSpec",
) -> dict:
    """Merge deterministic pre-pass output with a model prediction into one nested JSON dict.

    For every leaf path in ``spec.deterministic_fields`` that appears in
    ``prepass_resolved``, that value wins (the pre-pass is the trusted baseline
    for fields it owns; a model-predicted value at the same path, if
    ``model_prediction`` happens to include one, is discarded).  For every leaf
    path in ``spec.semantic_fields``, and for any ``deterministic_fields`` path
    NOT resolved by the pre-pass (a failed regex -- see ``run_prepass``'s
    docstring: it "falls through" to the model rather than vanishing), the
    model's value is used if ``model_prediction`` is a dict and has a value at
    that path; otherwise the path is simply absent from the result (do not
    insert an explicit ``None`` -- an absent key and a ``None``-valued key are
    NOT equivalent to a downstream consumer like ``evaluate_record``, and the
    correct signal here is "neither system produced a value," which is absence,
    not an explicit null assertion).

    Returns a nested JSON dict (unflattened), suitable for passing directly to
    ``schemaforge.evaluation.metrics.evaluate_record`` as the ``prediction``
    argument.  ``model_prediction`` of ``None`` is valid input (yields a result
    built purely from ``prepass_resolved``, for the deterministic-only-baseline
    case where the model wasn't even run).
    """
    merged = dict(prepass_resolved)
    if isinstance(model_prediction, dict):
        model_flat = flatten(model_prediction)
        # A None-valued model field is "no value": absence, not an explicit
        # null assertion, is the correct signal to a consumer like
        # evaluate_record.
        for path in spec.semantic_fields:
            if path not in merged and path in model_flat and model_flat[path] is not None:
                merged[path] = model_flat[path]
        for path in spec.deterministic_fields:
            if path not in merged and path in model_flat and model_flat[path] is not None:
                merged[path] = model_flat[path]
    return _unflatten(merged)


def rules_only_prediction(prepass_resolved: dict[str, Any]) -> dict:
    """Unflatten ``prepass_resolved`` alone into nested JSON, with no model contribution at all.

    This is the "deterministic pre-pass alone" system for benchmarking (research
    direction §6 lists it as one of the systems compared: "tuned regex/rule
    pipeline").  Equivalent to ``merge_prediction(prepass_resolved, None, spec)``
    for any ``spec``, but written to make that equivalence explicit at the call
    site without needing a spec object on hand, since "rules alone" has no
    notion of semantic fields to worry about defaulting.
    """
    return _unflatten(dict(prepass_resolved))
