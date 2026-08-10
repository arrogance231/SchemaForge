"""Teacher-output validation gate (research direction §4.2).

Runs the four mandatory checks -- JSON parse, pydantic schema validation,
source-support, and no-over-assertion -- in order, short-circuiting at the
first rejection and reporting a human-readable reason for it.

Step 3 optionally credits typo/OCR-denoising near-matches (off by default via
``fuzzy_support``): when enabled, a value whose same-word-count source window
clears ``fuzzy_threshold`` is accepted as supported.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass

from pydantic import ValidationError

from schemaforge.evaluation.json_utils import flatten, parse_json
from schemaforge.registry import SchemaSpec, get_schema, leaf_paths


def _norm(value: str) -> str:
    """Casefold ``value`` and collapse internal whitespace runs to one space."""
    return " ".join(value.casefold().split())


def _supported_via_ontology(spec: SchemaSpec, value: str, norm_source: str) -> bool:
    """True when ``value`` is a canonical form whose surface appears in ``norm_source``."""
    norm_value = _norm(value)
    return any(
        _norm(canonical) == norm_value and _norm(surface) in norm_source
        for surface, canonical in spec.ontology.items()
    )


def _fuzzy_supported(value: str, norm_source: str, *, threshold: float = 0.85) -> bool:
    """True when a same-word-count window of norm_source closely (typo-level) matches value.

    Slides a window of exactly `len(_norm(value).split())` words across `norm_source`'s
    whitespace-split tokens, computing `difflib.SequenceMatcher(None, norm_value, window).ratio()`
    for each window, and returns True if any window's ratio is >= `threshold`. The word-count
    constraint (only comparing equal-length spans) plus a high threshold bounds this to
    typo/OCR-level near-matches, not loose semantic similarity -- deliberately conservative.
    Returns False immediately if `value` normalizes to an empty string (avoid a degenerate
    always-true match against any window) or if `norm_source` has fewer words than `value`'s
    word count.
    """
    norm_value = _norm(value)
    tokens = norm_value.split()
    if not tokens:
        return False
    source_tokens = norm_source.split()
    if len(source_tokens) < len(tokens):
        return False
    for start in range(len(source_tokens) - len(tokens) + 1):
        window = " ".join(source_tokens[start : start + len(tokens)])
        if difflib.SequenceMatcher(None, norm_value, window).ratio() >= threshold:
            return True
    return False


@dataclass(frozen=True)
class GateResult:
    """Validation-gate outcome: acceptance flag, parsed dict, rejection reasons, unsupported field paths."""

    accepted: bool
    parsed: dict | None
    reasons: list[str]
    unsupported_fields: list[str]


def validate_teacher_output(
    schema_name: str,
    source_text: str,
    raw_teacher_output: str,
    *,
    fuzzy_support: bool = False,
    fuzzy_threshold: float = 0.85,
) -> GateResult:
    """Run the four-step §4.2 gate on a raw teacher output, short-circuiting at the first rejection.

    With ``fuzzy_support=True``, step 3 additionally credits typo/OCR-denoising near-matches
    (same-word-count window of the source scoring >= ``fuzzy_threshold``); it defaults to off,
    keeping step 3 strict literal/ontology matching only. This exists to avoid false rejections
    when the teacher corrects single-character corruption the corpus introduces, not to loosen
    matching into general fuzzy similarity."""
    parsed = parse_json(raw_teacher_output)
    if parsed is None:
        return GateResult(
            accepted=False,
            parsed=None,
            reasons=["step1_json_parse: no valid JSON object found in teacher output"],
            unsupported_fields=[],
        )

    spec = get_schema(schema_name)
    try:
        validated = spec.model.model_validate(parsed)
    except ValidationError as exc:
        return GateResult(
            accepted=False,
            parsed=parsed,
            reasons=[f"step2_schema_validation: {exc}"],
            unsupported_fields=[],
        )
    validated_dict = validated.model_dump(mode="json")

    norm_source = _norm(source_text)
    unsupported: list[str] = []
    for path, value in flatten(validated_dict).items():
        if path not in spec.semantic_fields:
            continue
        strings = [value] if isinstance(value, str) else [item for item in value if isinstance(item, str)] if isinstance(value, list) else []
        for item in strings:
            if not item:
                continue
            norm_value = _norm(item)
            if (
                norm_value in norm_source
                or _supported_via_ontology(spec, item, norm_source)
                or (
                    fuzzy_support
                    and _fuzzy_supported(item, norm_source, threshold=fuzzy_threshold)
                )
            ):
                continue
            unsupported.append(path)
            break
    if unsupported:
        return GateResult(
            accepted=False,
            parsed=validated_dict,
            reasons=["step3_source_support: unsupported field(s): " + ", ".join(unsupported)],
            unsupported_fields=unsupported,
        )

    allowed = leaf_paths(spec.model)
    extra = sorted(path for path in flatten(parsed) if path not in allowed)
    if extra:
        return GateResult(
            accepted=False,
            parsed=validated_dict,
            reasons=["step4_no_overassertion: field(s) not licensed by schema: " + ", ".join(extra)],
            unsupported_fields=[],
        )

    return GateResult(accepted=True, parsed=validated_dict, reasons=[], unsupported_fields=[])


def rejection_rate(results: list[GateResult]) -> float:
    """Fraction of results with accepted is False. Raises ValueError on empty list."""
    if not results:
        raise ValueError("rejection_rate: results must be non-empty")
    return sum(1 for r in results if not r.accepted) / len(results)
