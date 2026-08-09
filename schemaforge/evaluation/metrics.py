"""Field-level value comparison and micro-averaged evaluation metrics.

Implements the field-level metrics mandated by PROJECT_CHARTER.md §9 and the
research direction §6: exact match, schema validity, field precision/recall/F1,
hallucination rate and missing-field rate.  All averages are MICRO averages
(pooled over leaves), not per-record means.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

import dateparser

from schemaforge.evaluation.json_utils import flatten
from schemaforge.registry import SchemaSpec, leaf_paths, normalize_via_ontology

_CURRENCY_RE = re.compile(r"[$€£¥₹]")
_MONTH_RE = re.compile(
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)",
    re.I,
)
_NUMERIC_DATE_RE = re.compile(r"\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}")
_DATE_FORMATS = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%m/%d/%Y",
    "%d/%m/%Y",
    "%m-%d-%Y",
    "%d-%m-%Y",
    "%d %B %Y",
    "%B %d, %Y",
    "%d %b %Y",
    "%b %d, %Y",
)


def _norm_str(value: str) -> str:
    """Casefold ``value`` and collapse internal whitespace runs to one space."""
    return " ".join(value.casefold().split())


def _looks_like_date(value: str) -> bool:
    """True when ``value`` plausibly encodes a full date (month name or numeric date)."""
    return bool(_MONTH_RE.search(value)) or bool(_NUMERIC_DATE_RE.search(value))


def _to_iso_date(value: Any) -> str | None:
    """Parse ``value`` to an ISO ``YYYY-MM-DD`` string, or return ``None``.

    ``datetime.date``/``datetime.datetime`` objects are formatted directly.
    Strings are tried against the explicit formats in ``_DATE_FORMATS``, then a
    guarded ``dateparser`` fallback runs only for strings that look like dates.
    """
    if isinstance(value, (date, datetime)):
        return value.isoformat()[:10]
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    if not _looks_like_date(text):
        return None
    try:
        parsed = dateparser.parse(text, settings={"DATE_ORDER": "MDY"})
    except Exception:
        return None
    return parsed.strftime("%Y-%m-%d") if parsed is not None else None


def _valid_groups(s: str, sep: str) -> bool:
    """True when ``s`` is a valid thousands grouping: a 1-3 digit leading group
    followed by runs of exactly 3 digits (``1,234``, ``1.234.567``)."""
    parts = s.split(sep)
    if len(parts) < 2:
        return True
    if not 1 <= len(parts[0]) <= 3:
        return False
    return all(len(part) == 3 for part in parts[1:])


def _single_separator(t: str, sep: str) -> str:
    """Normalize a token with exactly one separator.

    Rule: if the trailing group is exactly 3 digits and the leading part is 1-3
    digits, the separator is a THOUSANDS separator (``1,234`` -> ``1234``,
    ``1.234`` -> ``1234``); otherwise it is the decimal separator
    (``1234.50`` -> ``1234.50``, ``1234,50`` -> ``1234.50``, ``1.5`` -> ``1.5``).
    """
    int_part, frac_part = t.split(sep, 1)
    if len(frac_part) == 3 and 1 <= len(int_part) <= 3:
        return int_part + frac_part
    return f"{int_part}.{frac_part}"


def _try_decimal(value: Any) -> Decimal | None:
    """Parse ``value`` to an exact ``Decimal``, or return ``None``.

    Accepts ``Decimal``, ``int``/``float`` (converted via ``str``) and strings
    with optional currency symbols and surrounding whitespace (e.g.
    ``"$1,234.50"``, ``"€1.234,50"``).  Anything that is not cleanly numeric
    returns ``None`` so that e.g. an ID string is never misread as a number.

    Separator rule, identical to
    ``schemaforge.deterministic.extractors._normalize_money``: when both ``,``
    and ``.`` are present the LAST separator is the decimal separator and every
    other separator must be a valid 3-digit thousands grouping (``"1,234.50"``
    and ``"1.234,50"`` both read as ``Decimal("1234.50")``; ``"1,23.50"`` is
    rejected).  When only one separator character appears, the group-size rule
    in :func:`_single_separator` decides: ``"1,234"`` -> ``Decimal("1234")``
    and ``"1.234"`` -> ``Decimal("1234")`` (thousands), while ``"1234.50"``,
    ``"1234,50"`` and ``"1.5"`` keep their decimal reading.  A string with no
    separator is read as a plain ``Decimal``.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, Decimal):
        return value
    if isinstance(value, (int, float)):
        return Decimal(str(value))
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        cleaned = _CURRENCY_RE.sub("", text)
        cleaned = cleaned.replace(" ", "").replace("\u00a0", "").replace("\u2009", "")
        if not cleaned:
            return None
        try:
            if "," not in cleaned and "." not in cleaned:
                return Decimal(cleaned)
            if "," in cleaned and "." in cleaned:
                if cleaned.rfind(",") > cleaned.rfind("."):
                    thousands, decimal = ".", ","
                else:
                    thousands, decimal = ",", "."
                int_part, frac_part = cleaned.split(decimal, 1)
                if not _valid_groups(int_part, thousands):
                    return None
                return Decimal(f"{int_part.replace(thousands, '')}.{frac_part}")
            if cleaned.count(",") >= 2:
                if not _valid_groups(cleaned, ","):
                    return None
                return Decimal(cleaned.replace(",", ""))
            if cleaned.count(".") >= 2:
                if not _valid_groups(cleaned, "."):
                    return None
                return Decimal(cleaned.replace(".", ""))
            sep = "," if "," in cleaned else "."
            return Decimal(_single_separator(cleaned, sep))
        except (InvalidOperation, ValueError):
            return None
    return None


def _norm_element(value: Any, spec: SchemaSpec | None) -> Any:
    """Return a hashable, comparable normalization of one list element."""
    if value is None:
        return None
    if isinstance(value, list):
        return tuple(_norm_element(item, spec) for item in value)
    if isinstance(value, bool):
        return ("bool", value)
    if spec is not None:
        value = normalize_via_ontology(spec, value)
    if isinstance(value, str):
        return ("str", _norm_str(value))
    number = _try_decimal(value)
    if number is not None:
        return ("num", number)
    iso = _to_iso_date(value)
    if iso is not None:
        return ("date", iso)
    if isinstance(value, (int, float)):
        return ("num", Decimal(str(value)))
    return ("other", repr(value))


def _lists_equal(a: list[Any], b: list[Any], spec: SchemaSpec | None) -> bool:
    """Compare two lists as multisets of normalized values (order independent)."""
    return Counter(_norm_element(item, spec) for item in a) == Counter(
        _norm_element(item, spec) for item in b
    )


def _multiset_match_count(
    pred: list[Any], ref: list[Any], spec: SchemaSpec | None
) -> int:
    """Number of ``pred`` elements matched to equal ``ref`` elements, as a multiset."""
    pc = Counter(_norm_element(item, spec) for item in pred)
    rc = Counter(_norm_element(item, spec) for item in ref)
    return sum(min(pc[key], rc[key]) for key in pc)


def values_equal(pred: Any, ref: Any, *, spec: SchemaSpec | None = None) -> bool:
    """Return whether ``pred`` equals ``ref`` under the value comparison rules.

    - ``None`` equals only ``None``.
    - Lists are compared as multisets of normalized values, order independent.
    - Booleans compare as booleans (never as numbers).
    - Numbers compare by exact ``Decimal`` equality after normalization; strings
      such as ``"$1,234.50"`` and ``"1234.5"`` and numeric ``1234.5`` all
      normalize to the same ``Decimal``.
    - Dates parse to ISO ``YYYY-MM-DD`` when possible and compare.
    - Strings strip, casefold and collapse internal whitespace before
      comparison; ``normalize_via_ontology`` is applied to BOTH sides first when
      ``spec`` is given.
    """
    if pred is None or ref is None:
        return pred is None and ref is None
    if isinstance(pred, list) or isinstance(ref, list):
        if isinstance(pred, list) and isinstance(ref, list):
            return _lists_equal(pred, ref, spec)
        return False
    if isinstance(pred, bool) or isinstance(ref, bool):
        return isinstance(pred, bool) and isinstance(ref, bool) and pred == ref
    number_pred = _try_decimal(pred)
    number_ref = _try_decimal(ref)
    if number_pred is not None and number_ref is not None:
        return number_pred == number_ref
    iso_pred = _to_iso_date(pred)
    iso_ref = _to_iso_date(ref)
    if iso_pred is not None and iso_ref is not None:
        return iso_pred == iso_ref
    if isinstance(pred, str) and isinstance(ref, str):
        p, r = pred, ref
        if spec is not None:
            p = normalize_via_ontology(spec, p)
            r = normalize_via_ontology(spec, r)
        return _norm_str(p) == _norm_str(r)
    return pred == ref


def _leaf_units(value: Any) -> int:
    """Number of leaf elements a flattened value represents (a repeated ``[]`` path counts once per element)."""
    return len(value) if isinstance(value, list) else 1


def _is_unsupported_string(value: Any, spec: SchemaSpec | None, norm_source: str) -> bool:
    """True when ``value`` is a non-empty string with no support in ``norm_source``.

    Numeric and date values are exempt from the support check because their
    formatting legitimately differs from the source.  Ontology normalization is
    applied before the case-insensitive, whitespace-collapsed substring check.
    """
    if not isinstance(value, str):
        return False
    text = value.strip()
    if not text:
        return False
    if _try_decimal(text) is not None or _to_iso_date(text) is not None:
        return False
    normalized = normalize_via_ontology(spec, text) if spec is not None else text
    return _norm_str(normalized) not in norm_source


@dataclass
class RecordResult:
    """Field-level result for a single prediction/reference pair.

    ``n_missing`` is the number of missing reference leaf UNITS and
    ``n_hallucinated`` the number of hallucinated predicted leaf UNITS: a
    repeated ``[]`` path contributes one unit per affected list element, not 1
    for the whole path.  ``missing``/``hallucinated`` are the corresponding
    dotted PATHS (one entry per affected path, kept for failure analysis).
    """

    exact_match: bool
    schema_valid: bool
    n_correct: int
    n_predicted: int
    n_reference: int
    missing: list[str]
    hallucinated: list[str]
    errors: dict[str, tuple[Any, Any]]
    n_missing: int = 0
    n_hallucinated: int = 0


def evaluate_record(
    prediction: dict | None,
    reference: dict,
    spec: SchemaSpec,
    *,
    source_text: str | None = None,
) -> RecordResult:
    """Evaluate one prediction against one reference under ``spec``.

    ``prediction`` of ``None`` (unparseable output) is valid input: it yields
    ``schema_valid=False``, ``n_predicted=0`` and every non-``None`` reference
    leaf listed as missing, and never raises.

    ``n_correct``/``n_predicted``/``n_reference`` are leaf-unit counts over the
    flattened objects; a repeated ``[]`` path counts once per element.  A path
    present in both where either side is a list earns PARTIAL credit: matched
    elements count toward ``n_correct`` (``_multiset_match_count``), so a
    prediction supplying ``k`` of ``m`` correct reference elements scores ``k``,
    not ``0``.  When one side is a list and the other a scalar, the scalar is
    treated as a one-element list for the comparison.
    ``n_missing`` is the number of missing reference leaf UNITS: a reference
    path holding a list of ``k`` elements that the prediction omits entirely
    contributes ``k``, and a shorter predicted list at the same path contributes
    the shortfall.  ``n_hallucinated`` is the number of hallucinated predicted
    leaf UNITS: a list path contributes one unit per unsupported or off-schema
    element.  ``missing`` lists reference paths absent or ``None`` in the
    prediction while non-``None`` in the reference.  ``hallucinated`` lists
    predicted paths that are not leaf paths of ``spec.model``, or that hold a
    non-empty string value which, after ontology normalization, does not appear
    in ``source_text`` (only checked when ``source_text`` is provided).
    ``errors`` maps each path present in both but unequal to
    ``(predicted, reference)``.
    """
    model_leaves = leaf_paths(spec.model)

    if isinstance(prediction, dict):
        try:
            spec.model.model_validate(prediction)
            schema_valid = True
        except Exception:
            schema_valid = False
    else:
        schema_valid = False

    pred_flat = flatten(prediction) if isinstance(prediction, dict) else {}
    ref_flat = flatten(reference)

    n_correct = 0
    n_predicted = sum(_leaf_units(value) for value in pred_flat.values())
    n_reference = sum(_leaf_units(value) for value in ref_flat.values())

    errors: dict[str, tuple[Any, Any]] = {}
    for path in sorted(pred_flat.keys() & ref_flat.keys()):
        pv = pred_flat[path]
        rv = ref_flat[path]
        if isinstance(pv, list) or isinstance(rv, list):
            p_list = pv if isinstance(pv, list) else [pv]
            r_list = rv if isinstance(rv, list) else [rv]
            if _lists_equal(p_list, r_list, spec):
                n_correct += len(p_list)
            else:
                errors[path] = (pv, rv)
                n_correct += _multiset_match_count(p_list, r_list, spec)
        else:
            if values_equal(pv, rv, spec=spec):
                n_correct += 1
            else:
                errors[path] = (pv, rv)

    missing: list[str] = []
    n_missing = 0
    for path in sorted(ref_flat):
        rv = ref_flat[path]
        if rv is None:
            continue
        pv = pred_flat.get(path)
        if pv is None:
            missing.append(path)
            n_missing += _leaf_units(rv)
        elif isinstance(rv, list) and isinstance(pv, list) and len(pv) < len(rv):
            n_missing += len(rv) - len(pv)

    hallucinated: list[str] = []
    n_hallucinated = 0
    if source_text is not None:
        norm_source = _norm_str(source_text)
        for path, value in pred_flat.items():
            if path not in model_leaves:
                hallucinated.append(path)
                n_hallucinated += _leaf_units(value)
                continue
            if isinstance(value, list):
                unsupported = [
                    item
                    for item in value
                    if _is_unsupported_string(item, spec, norm_source)
                ]
                if unsupported:
                    hallucinated.append(path)
                    n_hallucinated += len(unsupported)
            elif _is_unsupported_string(value, spec, norm_source):
                hallucinated.append(path)
                n_hallucinated += 1

    exact_match = set(pred_flat) == set(ref_flat) and all(
        values_equal(pred_flat[path], ref_flat[path], spec=spec)
        for path in ref_flat
    )

    return RecordResult(
        exact_match=exact_match,
        schema_valid=schema_valid,
        n_correct=n_correct,
        n_predicted=n_predicted,
        n_reference=n_reference,
        missing=missing,
        hallucinated=hallucinated,
        errors=errors,
        n_missing=n_missing,
        n_hallucinated=n_hallucinated,
    )


def aggregate(results: list[RecordResult]) -> dict[str, float]:
    """Pool a list of :class:`RecordResult` into MICRO-averaged metrics.

    These are micro averages (pooled over all records' leaves), never per-record
    means.  Formulas, identical to the code:

    - ``field_precision = sum(n_correct) / sum(n_predicted)``, 0.0 if the
      denominator is 0.
    - ``field_recall = sum(n_correct) / sum(n_reference)``, 0.0 if the
      denominator is 0.
    - ``field_f1 = 2*P*R/(P+R)`` where P/R are the above, 0.0 if ``P+R == 0``.
    - ``exact_match`` / ``schema_validity`` are the fraction of records with the
      flag true.
    - ``hallucination_rate = sum(n_hallucinated) / sum(n_predicted)``,
      0.0 if the denominator is 0.
    - ``missing_field_rate = sum(n_missing) / sum(n_reference)``, 0.0 if
      the denominator is 0.
    - ``n_records`` is the number of records.
    """
    n = len(results)
    sum_correct = sum(r.n_correct for r in results)
    sum_predicted = sum(r.n_predicted for r in results)
    sum_reference = sum(r.n_reference for r in results)

    precision = sum_correct / sum_predicted if sum_predicted else 0.0
    recall = sum_correct / sum_reference if sum_reference else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    exact_match = sum(1 for r in results if r.exact_match) / n if n else 0.0
    schema_validity = sum(1 for r in results if r.schema_valid) / n if n else 0.0

    total_hallucinated = sum(r.n_hallucinated for r in results)
    total_missing = sum(r.n_missing for r in results)

    return {
        "exact_match": exact_match,
        "schema_validity": schema_validity,
        "field_precision": precision,
        "field_recall": recall,
        "field_f1": f1,
        "hallucination_rate": total_hallucinated / sum_predicted if sum_predicted else 0.0,
        "missing_field_rate": total_missing / sum_reference if sum_reference else 0.0,
        "n_records": float(n),
    }
