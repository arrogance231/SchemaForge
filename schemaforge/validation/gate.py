"""Teacher-output validation gate (research direction §4.2).

Runs the four mandatory checks -- JSON parse, pydantic schema validation,
source-support, and no-over-assertion -- in order, short-circuiting at the
first rejection and reporting a human-readable reason for it.
"""

from __future__ import annotations

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


@dataclass(frozen=True)
class GateResult:
    """Validation-gate outcome: acceptance flag, parsed dict, rejection reasons, unsupported field paths."""

    accepted: bool
    parsed: dict | None
    reasons: list[str]
    unsupported_fields: list[str]


def validate_teacher_output(schema_name: str, source_text: str, raw_teacher_output: str) -> GateResult:
    """Run the four-step §4.2 gate on a raw teacher output, short-circuiting at the first rejection."""
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
            if norm_value in norm_source or _supported_via_ontology(spec, item, norm_source):
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
