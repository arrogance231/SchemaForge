"""SchemaForge V2 deterministic pre-pass (research direction §2, stage 1).

``run_prepass`` resolves every ``deterministic_fields`` leaf it can with the
regex/rule extractors and leaves the residual fields -- every ``semantic_fields``
leaf plus any deterministic leaf the extractors failed to find -- for the model.
It is BOTH the hybrid pipeline's stage 1 AND the primary benchmark baseline of
§6, so it is tuned, not a strawman.

Core invariant: the pre-pass may populate ONLY ``spec.deterministic_fields``.
A semantic field is never resolved here; ``run_prepass`` asserts this before
returning.  The repeated-element (``path[]``) NUMERIC line-item leaves
(``line_items[].quantity`` / ``[].unit_price`` / ``[].line_total`` on
``invoice`` and ``receipt``) ARE resolved, one value per line-item segment,
by :func:`_line_item_extractions`; the repeated ``line_items[].description``
leaf is semantic and is never resolved here.

There is no model call, no network access and no randomness in this module.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Any, Mapping, Sequence

import schemaforge.schemas  # noqa: F401  (populates the registry on import)
from schemaforge.deterministic.extractors import (
    Extraction,
    find_dates,
    find_emails,
    find_identifiers,
    find_money,
    find_percentages,
    find_phones,
    find_urls,
    resolve_overlaps,
)
from schemaforge.registry import SchemaSpec, all_schemas, get_schema, leaf_paths

__all__ = [
    "PrepassResult",
    "run_prepass",
    "unresolved_prompt_fields",
]


@dataclass
class PrepassResult:
    """Outcome of one deterministic pre-pass over a document.

    ``resolved`` maps each dotted leaf path that was resolved to its normalized
    value; ``provenance`` maps the same paths to the :class:`Extraction` that
    produced them (span + confidence, for hallucination checking); ``unresolved``
    is the sorted list of dotted leaf paths the model still has to fill -- every
    semantic field plus any deterministic field the extractors missed.

    For a repeated-element path (``line_items[].quantity``) ``resolved`` holds
    one value per line item, in document order.  ``provenance`` records only the
    FIRST line item's :class:`Extraction` for such a path: the mapping type
    holds one entry per path, not one per element, so the rest are not silently
    dropped but represented by the first.
    """

    resolved: dict[str, Any]
    provenance: dict[str, Extraction]
    unresolved: list[str]


# --------------------------------------------------------------------------- #
# field-to-extractor binding
# --------------------------------------------------------------------------- #

_EXTRACTOR_NAMES = frozenset({"dates", "emails", "urls", "phones", "money", "identifiers", "percentages"})
_RULE_NAMES = frozenset({"first", "last", "max", "min", "nearest_label"})


@dataclass(frozen=True)
class _Binding:
    """Bind one dotted leaf path to an extractor plus a selection rule.

    ``rule`` selects the value when the extractor returns several matches:
    ``first``/``last`` pick by document position, ``max``/``min`` by normalized
    value, and ``nearest_label`` picks the nearest match following an occurrence
    of the ``label`` regex (e.g. ``(?i)(?<![a-z0-9])total`` for ``total_amount``).
    ``patterns`` are the schema-supplied regex patterns for ``identifiers``.
    """

    extractor: str
    rule: str
    label: str | None = None
    patterns: tuple[str, ...] = ()


_BINDINGS: dict[tuple[str, str], _Binding] = {}


def _bind(
    schema: str,
    path: str,
    extractor: str,
    rule: str = "first",
    *,
    label: str | None = None,
    patterns: Sequence[str] = (),
) -> None:
    """Declare a binding for ``(schema, path)``, raising on an invalid rule."""
    if rule == "nearest_label" and not label:
        raise ValueError(f"nearest_label binding for {schema}.{path} requires a label regex")
    _BINDINGS[(schema, path)] = _Binding(extractor, rule, label, tuple(patterns))


# invoice ---------------------------------------------------------------------
_bind("invoice", "invoice_number", "identifiers", patterns=(r"\bINV-?\d+\b", r"\bINVOICE\s*#\s*\d+\b"))
_bind("invoice", "invoice_date", "dates", "nearest_label", label=r"(?i)date\s*[:.]?")
_bind("invoice", "due_date", "dates", "nearest_label", label=r"(?i)due\s*[:.]?")
_bind("invoice", "vendor_email", "emails", "nearest_label", label=r"(?i)vendor\s*[:.]?")
_bind("invoice", "vendor_phone", "phones", "nearest_label", label=r"(?i)vendor\s*[:.]?")
_bind("invoice", "total_amount", "money", "nearest_label", label=r"(?i)(?<![a-z0-9])total\s*[:.]?")
_bind("invoice", "tax_amount", "money", "nearest_label", label=r"(?i)(?<![a-z0-9])tax\s*[:.]?")

# contract --------------------------------------------------------------------
_bind("contract", "start_date", "dates", "nearest_label", label=r"(?i)start\s*[:.]?")
_bind("contract", "end_date", "dates", "nearest_label", label=r"(?i)end\s*[:.]?")
_bind("contract", "liability_cap", "money", "nearest_label", label=r"(?i)liab\w*\s*[:.]?")

# conversation (held out) -----------------------------------------------------
_bind("conversation", "started_date", "dates", "nearest_label", label=r"(?i)start(?:ed)?\s*[:.]?")

# crm_record ------------------------------------------------------------------
_bind("crm_record", "email", "emails")
_bind("crm_record", "phone", "phones")
_bind("crm_record", "deal_value", "money", "nearest_label", label=r"(?i)(?<![a-z0-9])value\s*[:.]?")
_bind("crm_record", "last_contact_date", "dates", "nearest_label", label=r"(?i)last\s+contact\s*[:.]?")

# email -----------------------------------------------------------------------
_bind("email", "sender_email", "emails", "nearest_label", label=r"(?i)(?<![a-z0-9])from\s*[:.]?")
_bind("email", "recipient_email", "emails", "nearest_label", label=r"(?i)(?<![a-z0-9])to\s*[:.]?")
_bind("email", "sent_date", "dates", "nearest_label", label=r"(?i)sent\s*[:.]?")

# form ------------------------------------------------------------------------
_bind("form", "form_id", "identifiers", patterns=(r"\b(?:FORM|FRM)-?\d+\b",))
_bind("form", "submitted_date", "dates", "nearest_label", label=r"(?i)submitted\s*[:.]?")
_bind("form", "submitter_email", "emails", "nearest_label", label=r"(?i)submitter\s*[:.]?")

# insurance_claim (held out) --------------------------------------------------
_bind("insurance_claim", "claim_number", "identifiers", patterns=(r"\b(?:CLM|CLAIM)-?\d+\b",))
_bind("insurance_claim", "policy_number", "identifiers", patterns=(r"\b(?:POLICY|POL)-?\d+\b",))
_bind("insurance_claim", "incident_date", "dates", "nearest_label", label=r"(?i)incident\s*[:.]?")
_bind("insurance_claim", "amount_requested", "money", "nearest_label", label=r"(?i)amount\s*[:.]?")

# kg_triple (held out) --------------------------------------------------------
_bind("kg_triple", "extracted_date", "dates", "nearest_label", label=r"(?i)extracted\s*[:.]?")

# medical_note ----------------------------------------------------------------
_bind("medical_note", "visit_date", "dates", "nearest_label", label=r"(?i)visit\s*[:.]?")
_bind("medical_note", "follow_up_date", "dates", "nearest_label", label=r"(?i)follow\s*-?\s*up\s*[:.]?")

# receipt ---------------------------------------------------------------------
_bind("receipt", "receipt_number", "identifiers", patterns=(r"\b(?:RECEIPT|REC)-?\d+\b",))
_bind("receipt", "receipt_date", "dates", "nearest_label", label=r"(?i)receipt\s*[:.]?")
_bind("receipt", "total_amount", "money", "nearest_label", label=r"(?i)(?<![a-z0-9])total\s*[:.]?")
_bind("receipt", "tax_amount", "money", "nearest_label", label=r"(?i)(?<![a-z0-9])tax\s*[:.]?")

# resume ----------------------------------------------------------------------
_bind("resume", "email", "emails")
_bind("resume", "phone", "phones")

# support_ticket --------------------------------------------------------------
_bind("support_ticket", "ticket_id", "identifiers", patterns=(r"\b(?:TKT|TCKT|TICKET)-?\d+\b",))
_bind("support_ticket", "created_date", "dates", "nearest_label", label=r"(?i)created\s*[:.]?")
_bind("support_ticket", "customer_email", "emails", "nearest_label", label=r"(?i)(?:customer|user)\s*[:.]?")


def _validate_bindings() -> None:
    """Every binding must name a real scalar deterministic field of a real schema."""
    for (schema_name, path), binding in _BINDINGS.items():
        spec = get_schema(schema_name)
        if path not in spec.deterministic_fields:
            raise ValueError(f"binding {schema_name}.{path} is not a deterministic field of its schema")
        if "[]" in path:
            raise ValueError(f"binding {schema_name}.{path} is a repeated-element path; the pre-pass binds scalar leaves only")
        if binding.extractor not in _EXTRACTOR_NAMES:
            raise ValueError(f"binding {schema_name}.{path} names unknown extractor {binding.extractor!r}")
        if binding.rule not in _RULE_NAMES:
            raise ValueError(f"binding {schema_name}.{path} names unknown selection rule {binding.rule!r}")


_validate_bindings()


# --------------------------------------------------------------------------- #
# selection rules
# --------------------------------------------------------------------------- #


def _value_key(extraction: Extraction) -> tuple[int, Any]:
    """Comparable sort key over an extraction's normalized value.

    Numbers and dates compare numerically (``(0, key)``); strings compare as
    strings (``(1, key)``); anything else by repr (``(2, key)``).
    """
    value = extraction.value
    if isinstance(value, bool):
        return (0, int(value))
    if isinstance(value, Decimal):
        return (0, value)
    if isinstance(value, (int, float)):
        return (0, Decimal(str(value)))
    if isinstance(value, date):
        return (0, value.toordinal())
    if isinstance(value, str):
        return (1, value)
    return (2, repr(value))


def _select_nearest_label(text: str, matches: Sequence[Extraction], label: str) -> Extraction | None:
    """Pick the nearest match FOLLOWING an occurrence of the ``label`` regex.

    For every label occurrence the earliest match at or after it is considered;
    the earliest such match overall wins (``None`` when no match follows any
    label occurrence).  This is what separates subtotal / tax / total when they
    are all money on a real invoice.
    """
    if not matches or not label:
        return None
    label_positions = [m.start() for m in re.finditer(label, text)]
    if not label_positions:
        return None
    best: Extraction | None = None
    for position in label_positions:
        for match in matches:
            if match.span[0] >= position:
                if best is None or match.span[0] < best.span[0]:
                    best = match
                break
    return best


def _select(text: str, matches: Sequence[Extraction], binding: _Binding) -> Extraction | None:
    """Apply ``binding.rule`` to ``matches``, or return ``None`` when empty."""
    if not matches:
        return None
    if binding.rule == "first":
        return matches[0]
    if binding.rule == "last":
        return matches[-1]
    if binding.rule == "max":
        return max(matches, key=_value_key)
    if binding.rule == "min":
        return min(matches, key=_value_key)
    if binding.rule == "nearest_label":
        return _select_nearest_label(text, matches, binding.label or "")
    raise ValueError(f"unknown selection rule {binding.rule!r}")


# --------------------------------------------------------------------------- #
# line-item segmentation
# --------------------------------------------------------------------------- #

# A segment is the text up to the next line break or sentence boundary (a run
# of `.`/`!`/`?` followed by whitespace or end-of-text).  Non-greedy `.+?`
# lets a decimal point inside a money value ("$120.00") be crossed so it is
# NOT mistaken for a sentence end.
_SEGMENT_RE = re.compile(r".+?(?:[.!?]+(?=\s|$)|\n|$)")
# Quantity-like tokens: "x 4" / "x4" / "x100" and "4 @".
_QTY_RE = re.compile(r"(?i)\b(?:x\s*(\d+)|(\d+)\s*@)")
_LINE_ITEM_LEAVES = ("quantity", "unit_price", "line_total")


def _line_item_paths(spec: SchemaSpec) -> dict[str, str]:
    """Map ``spec``'s repeated numeric deterministic leaf names to dotted paths.

    Only ``invoice``/``receipt`` currently declare repeated deterministic
    leaves (``line_items[].quantity`` / ``[].unit_price`` / ``[].line_total``);
    the map is empty for every other schema.
    """
    paths: dict[str, str] = {}
    for path in spec.deterministic_fields:
        if "[]" in path:
            leaf = path.rsplit(".", 1)[-1]
            if leaf in _LINE_ITEM_LEAVES:
                paths[leaf] = path
    return paths


def _money_after_marker(
    seg: str, seg_start: int, marker: str, money: Sequence[Extraction]
) -> Extraction | None:
    """Return the first money match at/after the first ``marker`` in ``seg``.

    ``money`` holds document-relative extractions known to lie inside ``seg``;
    the marker position is converted to document coordinates via ``seg_start``.
    ``None`` when the marker is absent or no money follows it inside ``seg``.
    """
    rel = seg.find(marker)
    if rel == -1:
        return None
    abs_pos = seg_start + rel
    seg_end = seg_start + len(seg)
    for extraction in money:
        if extraction.span[0] >= abs_pos and extraction.span[1] <= seg_end:
            return extraction
    return None


def _line_item_extractions(text: str, cache: dict) -> dict[str, list[Extraction]]:
    """Resolve per-line-item values from ``text``, keyed by leaf name.

    The document is segmented on line breaks and sentence boundaries; a segment
    is a line-item candidate when it contains a quantity-like token AND at
    least one money value.  Per segment ``quantity`` is the "x N" / "N @" count,
    ``unit_price`` the money following ``@`` and ``line_total`` the money
    following ``=``; a segment with exactly one money value resolves that value
    as ``line_total`` and leaves ``unit_price`` unresolved.  ``description`` is
    a semantic leaf and is never resolved here.  Returns
    ``{leaf: [Extraction, ...]}`` with one extraction per line item, in
    document order.  ``cache`` is the pre-pass extractor cache, reused for the
    document-wide ``find_money`` call.
    """
    money_key = ("money", ())
    if money_key not in cache:
        cache[money_key] = resolve_overlaps(find_money(text))
    money_all = cache[money_key]

    out: dict[str, list[Extraction]] = {}
    for match in _SEGMENT_RE.finditer(text):
        seg = match.group(0)
        if not seg.strip():
            continue
        qty = _QTY_RE.search(seg)
        if qty is None:
            continue
        seg_start, seg_end = match.start(), match.end()
        money = [e for e in money_all if seg_start <= e.span[0] and e.span[1] <= seg_end]
        if not money:
            continue
        qty_value = Decimal(qty.group(1) if qty.group(1) is not None else qty.group(2))
        out.setdefault("quantity", []).append(
            Extraction(
                value=qty_value,
                span=(seg_start + qty.start(), seg_start + qty.end()),
                raw=seg[qty.start() : qty.end()],
                extractor="line_items",
                confidence=1.0,
            )
        )
        if len(money) == 1:
            out.setdefault("line_total", []).append(money[0])
            continue
        unit_price = _money_after_marker(seg, seg_start, "@", money)
        if unit_price is not None:
            out.setdefault("unit_price", []).append(unit_price)
        line_total = _money_after_marker(seg, seg_start, "=", money)
        if line_total is not None:
            out.setdefault("line_total", []).append(line_total)
    return out


# --------------------------------------------------------------------------- #
# pre-pass
# --------------------------------------------------------------------------- #


def _extract_for(text: str, binding: _Binding, *, region: str, dayfirst: bool, cache: dict) -> Sequence[Extraction]:
    """Run the binding's extractor over ``text`` once per text, cached by call."""
    key = (binding.extractor, binding.patterns)
    if key in cache:
        return cache[key]
    if binding.extractor == "dates":
        found = find_dates(text, dayfirst=dayfirst)
    elif binding.extractor == "emails":
        found = find_emails(text)
    elif binding.extractor == "urls":
        found = find_urls(text)
    elif binding.extractor == "phones":
        found = find_phones(text, region=region)
    elif binding.extractor == "money":
        found = find_money(text)
    elif binding.extractor == "percentages":
        found = find_percentages(text)
    elif binding.extractor == "identifiers":
        found = find_identifiers(text, binding.patterns)
    else:  # pragma: no cover - guarded by _validate_bindings
        raise ValueError(f"unknown extractor {binding.extractor!r}")
    resolved = resolve_overlaps(found)
    cache[key] = resolved
    return resolved


def run_prepass(
    text: str,
    spec: SchemaSpec,
    *,
    region: str = "US",
    dayfirst: bool = False,
) -> PrepassResult:
    """Resolve ``spec.deterministic_fields`` from ``text``; leave the rest to the model.

    Only the declared scalar deterministic bindings for ``spec.name`` are
    attempted, plus the repeated numeric ``[]`` leaves
    (``line_items[].quantity`` etc.) resolved from line-item segments when the
    document contains any.  A deterministic field the extractors fail to find
    is NOT resolved and therefore falls through to ``unresolved`` (a failed
    regex must reach the model, not vanish).  Before returning, the
    semantic-field split is asserted: no ``semantic_fields`` path may appear in
    ``resolved``.
    """
    resolved: dict[str, Any] = {}
    provenance: dict[str, Extraction] = {}
    cache: dict = {}
    for (schema_name, path), binding in _BINDINGS.items():
        if schema_name != spec.name:
            continue
        matches = _extract_for(text, binding, region=region, dayfirst=dayfirst, cache=cache)
        selection = _select(text, matches, binding)
        if selection is None:
            continue
        resolved[path] = selection.value
        provenance[path] = selection

    line_item_paths = _line_item_paths(spec)
    if line_item_paths:
        for leaf, extractions in _line_item_extractions(text, cache).items():
            if not extractions:
                continue
            path = line_item_paths.get(leaf)
            if path is None:
                continue
            resolved[path] = [extraction.value for extraction in extractions]
            provenance[path] = extractions[0]

    resolved = dict(sorted(resolved.items()))
    for path in resolved:
        assert path in spec.deterministic_fields, (
            f"prepass resolved {spec.name}.{path} which is not a deterministic field"
        )
    for path in spec.semantic_fields:
        assert path not in resolved, f"prepass violated the field split: resolved semantic field {spec.name}.{path}"

    unresolved = sorted(set(leaf_paths(spec.model)) - set(resolved))
    return PrepassResult(resolved=resolved, provenance=provenance, unresolved=unresolved)


def unresolved_prompt_fields(spec: SchemaSpec, result: PrepassResult) -> list[str]:
    """Return the residual dotted leaf paths the model prompt must ask for.

    ``spec`` is accepted for interface stability (the residual set is derived
    from the model's leaves in ``run_prepass``).
    """
    return list(result.unresolved)
