"""Deterministic regex/rule extractors for the SchemaForge V2 pre-pass.

Each ``find_<name>(text)`` returns every match as an :class:`Extraction` in
document order.  Spans are exact character offsets into the ORIGINAL ``text``
and every ``raw`` is the substring ``text[span[0]:span[1]]`` -- the invariant
the hallucination check (research direction §5) depends on.

These extractors are the "tuned, not a strawman" baseline of the hybrid
architecture (research direction §2 and §6): they own the fields listed in
``SchemaSpec.deterministic_fields`` and are benchmarked head-to-head against the
model.  Only ``dateparser``, ``phonenumbers``, ``re`` and the standard library
are used; there is no model call and no randomness anywhere in this module.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any, Sequence

import dateparser
import phonenumbers

__all__ = [
    "Extraction",
    "find_dates",
    "find_emails",
    "find_urls",
    "find_phones",
    "find_money",
    "find_identifiers",
    "find_percentages",
    "resolve_overlaps",
]


@dataclass(frozen=True)
class Extraction:
    """One deterministic extraction from a source text.

    ``value`` is the normalized Python value (``str`` / ``Decimal`` /
    ``datetime.date`` / ``int``); ``span`` are character offsets into the source
    ``text`` such that ``text[span[0]:span[1]] == raw``; ``confidence`` is 1.0
    for unambiguous extractions and lower (e.g. 0.7) when the reading is
    genuinely ambiguous.
    """

    value: Any
    span: tuple[int, int]
    raw: str
    extractor: str
    confidence: float = 1.0


def _extraction(text: str, span: tuple[int, int], value: Any, confidence: float, extractor: str) -> Extraction:
    """Build an :class:`Extraction` whose ``raw`` is exactly the text slice."""
    return Extraction(
        value=value,
        span=(span[0], span[1]),
        raw=text[span[0] : span[1]],
        extractor=extractor,
        confidence=confidence,
    )


# --------------------------------------------------------------------------- #
# dates
# --------------------------------------------------------------------------- #

_MONTH_NAME = (
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
    r"aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
)

# Year-first ISO forms: 2026-04-10, 2026/04/10, 2026.04.10 (same separator each).
_ISO_DATE_RE = re.compile(
    r"\b(?P<year>\d{4})(?P<sep>[-/.])(?P<month>\d{1,2})(?P=sep)(?P<day>\d{1,2})\b"
)

# Two-component-first numeric forms: 04/10/2026, 25-04-2026, 10.04.2026.
# The (?P<year>\d{4}|\d{2}) trailing component and the \b anchors prevent this
# pattern from matching a fragment of a year-first ISO date.
_NUMERIC_DATE_RE = re.compile(
    r"\b(?P<a>\d{1,2})[-/.](?P<b>\d{1,2})[-/.](?P<year>\d{4}|\d{2})(?![\d])"
)

# Month-name forms in both orders: "April 10, 2026" / "Apr 10 2026" (month
# first) and "10 April 2026" (day first), with optional ordinal suffixes.
_MONTH_FIRST_RE = re.compile(
    rf"\b(?P<month>{_MONTH_NAME})\s+(?P<day>\d{{1,2}})(?:st|nd|rd|th)?\s*,?\s+(?P<year>\d{{4}})(?![\d])",
    re.IGNORECASE,
)
_MONTH_LAST_RE = re.compile(
    rf"\b(?P<day>\d{{1,2}})(?:st|nd|rd|th)?\s+(?P<month>{_MONTH_NAME})\s+(?P<year>\d{{4}})(?![\d])",
    re.IGNORECASE,
)


def find_dates(text: str, dayfirst: bool = False) -> list[Extraction]:
    """Return every date in ``text`` as ``datetime.date``, in document order.

    Handles at minimum: ``2026-04-10``, ``04/10/2026``, ``10/04/2026``,
    ``April 10, 2026``, ``10 April 2026``, ``Apr 10 2026``, ``2026/04/10``,
    plus ordinal month forms (``Aug 3rd, 2026``).  ``dateparser`` does the
    parsing of the regex-identified candidates; the year-first ISO forms are
    constructed directly (``dateparser`` would otherwise flip ``2026/04/10``
    under a DMY order).

    Ambiguity rule for the two-component-first numeric forms (``A/B/YYYY``,
    ``A-B-YYYY`` or ``A.B.2026``): when BOTH ``A`` and ``B`` are valid month
    numbers (1-12) the reading is genuinely ambiguous, so the form is parsed as
    MDY by default, marked ``confidence=0.7``, and ``dayfirst=True`` flips it to
    DMY.  When exactly one of ``A``/``B`` is > 12 the reading is forced (DMY or
    MDY respectively) and carries ``confidence=1.0``.  Year-first ISO and
    month-name forms are unambiguous and always ``confidence=1.0``.
    """
    results: list[Extraction] = []
    order = "DMY" if dayfirst else "MDY"
    fallback_order = "MDY" if dayfirst else "DMY"

    for m in _ISO_DATE_RE.finditer(text):
        try:
            value = date(int(m.group("year")), int(m.group("month")), int(m.group("day")))
        except ValueError:
            continue
        results.append(_extraction(text, m.span(), value, 1.0, "dates"))

    for m in _NUMERIC_DATE_RE.finditer(text):
        a, b = int(m.group("a")), int(m.group("b"))
        ambiguous = 1 <= a <= 12 and 1 <= b <= 12
        confidence = 0.7 if ambiguous else 1.0
        parsed = dateparser.parse(m.group(0), settings={"DATE_ORDER": order})
        if parsed is None:
            parsed = dateparser.parse(m.group(0), settings={"DATE_ORDER": fallback_order})
        if parsed is None:
            continue
        results.append(_extraction(text, m.span(), parsed.date(), confidence, "dates"))

    for regex in (_MONTH_FIRST_RE, _MONTH_LAST_RE):
        for m in regex.finditer(text):
            parsed = dateparser.parse(m.group(0), settings={"DATE_ORDER": order})
            if parsed is None:
                continue
            results.append(_extraction(text, m.span(), parsed.date(), 1.0, "dates"))

    results.sort(key=lambda e: e.span[0])
    return results


# --------------------------------------------------------------------------- #
# emails / urls
# --------------------------------------------------------------------------- #

_EMAIL_RE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"[A-Za-z0-9](?:[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]*[A-Za-z0-9])?"
    r"@"
    r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+"
)


def find_emails(text: str) -> list[Extraction]:
    """Return every RFC-shaped email address in ``text``, in document order.

    The local part must start and end with an alphanumeric, the domain must
    contain at least one dot-separated label, and trailing punctuation (``,``,
    ``.``, ``!`` ...) is never part of the match.
    """
    results: list[Extraction] = []
    for m in _EMAIL_RE.finditer(text):
        results.append(_extraction(text, m.span(), m.group(0), 1.0, "emails"))
    return results


_URL_RE = re.compile(r"\b(?:https?://|www\.)[^\s<>\"']+", re.IGNORECASE)


def find_urls(text: str) -> list[Extraction]:
    """Return every http/https/www URL in ``text``, in document order.

    A trailing sentence punctuation character from ``.,;:!?)`` is trimmed from
    the match so ``see http://x.io.`` yields ``http://x.io``.  Trade-off: a URL
    that genuinely ends in ``)`` has that ``)`` trimmed too.
    """
    results: list[Extraction] = []
    for m in _URL_RE.finditer(text):
        raw = m.group(0)
        stripped = raw.rstrip(".,;:!?)")
        if not stripped:
            continue
        results.append(
            Extraction(
                value=stripped,
                span=(m.start(), m.start() + len(stripped)),
                raw=stripped,
                extractor="urls",
                confidence=1.0,
            )
        )
    return results


# --------------------------------------------------------------------------- #
# phone numbers
# --------------------------------------------------------------------------- #


def find_phones(text: str, region: str = "US") -> list[Extraction]:
    """Return every phone number in ``text`` as an E.164 string, in document order.

    Uses ``phonenumbers.PhoneNumberMatcher`` with the given ``region``.
    ``region`` is validated against ``phonenumbers.SUPPORTED_REGIONS`` and an
    unsupported region raises ``ValueError`` (a caller bug).  Parsing of the
    text itself never raises: an internal error yields ``[]``.
    """
    if region not in phonenumbers.SUPPORTED_REGIONS:
        raise ValueError(f"unsupported region {region!r} for phone parsing")
    results: list[Extraction] = []
    try:
        for match in phonenumbers.PhoneNumberMatcher(text, region):
            value = phonenumbers.format_number(match.number, phonenumbers.PhoneNumberFormat.E164)
            results.append(
                Extraction(
                    value=value,
                    span=(match.start, match.end),
                    raw=text[match.start : match.end],
                    extractor="phones",
                    confidence=1.0,
                )
            )
    except Exception:
        return []
    return results


# --------------------------------------------------------------------------- #
# money / currency amounts
# --------------------------------------------------------------------------- #

_CURRENCY_SYMBOLS = re.compile(r"[$€£¥₹]")
_CURRENCY_CODES = re.compile(
    r"\b(?:USD|EUR|GBP|JPY|CNY|CHF|CAD|AUD|SEK|NOK|DKK|PLN|CZK|HUF|MXN|BRL|ZAR|INR|KRW)\b"
)
_NUMBER_CORE = r"\d+(?:[.,]\d+)*"

# Plain amount: optional currency symbol or ISO code before the number, optional
# ISO code after.  The leading lookbehind keeps the match from starting inside a
# word/number or inside a parenthesised amount; the trailing lookahead stops a
# bare number immediately before '%' (that is a percentage, not money).
_MONEY_RE = re.compile(
    rf"(?<![\w$€£¥₹])(?:{_CURRENCY_SYMBOLS.pattern}|{_CURRENCY_CODES.pattern}\s*)?"
    rf"{_NUMBER_CORE}(?:\s*{_CURRENCY_CODES.pattern})?(?![\d%])"
)
# Parenthesised (negative) amount: (1,234.50) -> Decimal("-1234.50").
_PAREN_MONEY_RE = re.compile(
    rf"\((?:{_CURRENCY_SYMBOLS.pattern}|{_CURRENCY_CODES.pattern}\s*)?"
    rf"{_NUMBER_CORE}(?:\s*{_CURRENCY_CODES.pattern})?\s*\)"
)


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


def _make_decimal(clean: str, negative: bool) -> Decimal | None:
    """Build an exact ``Decimal`` from a dot-decimal ``clean`` string."""
    try:
        value = Decimal(clean)
    except (InvalidOperation, ValueError):
        return None
    return -value if negative else value


def _normalize_money(token: str) -> Decimal | None:
    """Normalize one money token to a ``Decimal`` or ``None``.

    Steps: strip an outer pair of parentheses (negative), drop currency symbols
    and ISO codes, then read the remaining numeric core.  Separator rule: when
    both ``,`` and ``.`` are present the LAST separator is the decimal separator
    and every other separator must be a valid 3-digit thousands grouping
    (``1,234.50`` and ``1.234,50`` both read as ``1234.50``; ``1,23.50`` is
    rejected).  When only one separator character appears, the group-size rule
    in ``_single_separator`` decides.  A bare integer with no separator and no
    currency marker is NOT money (a year or count), and returns ``None``.
    """
    t = token.strip()
    negative = t.startswith("(") and t.endswith(")")
    if negative:
        t = t[1:-1].strip()
    had_currency = bool(_CURRENCY_SYMBOLS.search(t) or _CURRENCY_CODES.search(t))
    t = _CURRENCY_SYMBOLS.sub("", t)
    t = _CURRENCY_CODES.sub("", t)
    t = t.strip()
    if not t:
        return None
    if "," not in t and "." not in t:
        if not had_currency:
            return None
        return _make_decimal(t, negative)

    if "," in t and "." in t:
        if t.rfind(",") > t.rfind("."):
            thousands, decimal = ".", ","
        else:
            thousands, decimal = ",", "."
        int_part, frac_part = t.split(decimal, 1)
        if not _valid_groups(int_part, thousands):
            return None
        return _make_decimal(f"{int_part.replace(thousands, '')}.{frac_part}", negative)

    if "," in t:
        if t.count(",") >= 2:
            if not _valid_groups(t, ","):
                return None
            return _make_decimal(t.replace(",", ""), negative)
        return _make_decimal(_single_separator(t, ","), negative)

    if t.count(".") >= 2:
        if not _valid_groups(t, "."):
            return None
        return _make_decimal(t.replace(".", ""), negative)
    return _make_decimal(_single_separator(t, "."), negative)


def find_money(text: str) -> list[Extraction]:
    """Return every currency amount in ``text`` as ``Decimal``, in document order.

    Recognizes ``$1,234.50``, ``USD 1234.50``, ``1,234.50 USD``, ``€1.234,50``,
    bare ``1234.50`` and parenthesised negatives ``(1,234.50)``.  The separator
    disambiguation rule is documented in :func:`_normalize_money`.  The value is
    the exact ``Decimal`` (``1234.50``, never ``1234.5``).
    """
    results: list[Extraction] = []
    for m in _PAREN_MONEY_RE.finditer(text):
        value = _normalize_money(m.group(0))
        if value is not None:
            results.append(_extraction(text, m.span(), value, 1.0, "money"))
    for m in _MONEY_RE.finditer(text):
        if any(ps[0] <= m.start() and m.end() <= ps[1] for ps in (r.span for r in results)):
            continue
        value = _normalize_money(m.group(0))
        if value is not None:
            results.append(_extraction(text, m.span(), value, 1.0, "money"))
    results.sort(key=lambda e: e.span[0])
    return results


# --------------------------------------------------------------------------- #
# identifiers / percentages
# --------------------------------------------------------------------------- #


def find_identifiers(text: str, patterns: Sequence[str]) -> list[Extraction]:
    """Return every match of the schema-supplied ``patterns``, in document order.

    ``patterns`` are regex strings or compiled regexes; each match's full text
    is both the ``raw`` and the ``value``.  Patterns are schema configuration,
    not input: an invalid pattern string raises ``re.error``.
    """
    results: list[Extraction] = []
    for pattern in patterns:
        compiled = re.compile(pattern) if isinstance(pattern, str) else pattern
        for m in compiled.finditer(text):
            results.append(_extraction(text, m.span(), m.group(0), 1.0, "identifiers"))
    results.sort(key=lambda e: e.span[0])
    return results


_PERCENT_RE = re.compile(r"(?<![\w])(\d+(?:[.,]\d+)?)\s*%")


def find_percentages(text: str) -> list[Extraction]:
    """Return every percentage in ``text`` as a ``Decimal``, in document order.

    The value is the percentage NUMBER (``Decimal("8.5")`` for ``"8.5 %"``),
    NOT the fraction (``0.085``).  A European comma decimal is normalized to a
    dot (``8,5 %`` -> ``Decimal("8.5")``).
    """
    results: list[Extraction] = []
    for m in _PERCENT_RE.finditer(text):
        number = m.group(1).replace(",", ".")
        try:
            value = Decimal(number)
        except (InvalidOperation, ValueError):
            continue
        results.append(_extraction(text, m.span(), value, 1.0, "percentages"))
    return results


# --------------------------------------------------------------------------- #
# overlap resolution
# --------------------------------------------------------------------------- #


def resolve_overlaps(extractions: Sequence[Extraction]) -> list[Extraction]:
    """Return the maximal subset of non-overlapping ``extractions``.

    When two extractions overlap, the LONGEST is kept; ties break by higher
    confidence, then by earlier start.  The result is ordered by start.
    """
    if not extractions:
        return []
    ordered = sorted(
        extractions,
        key=lambda e: (-(e.span[1] - e.span[0]), -e.confidence, e.span[0]),
    )
    kept: list[Extraction] = []
    for e in ordered:
        if all(e.span[0] >= k.span[1] or k.span[0] >= e.span[1] for k in kept):
            kept.append(e)
    return sorted(kept, key=lambda e: e.span[0])
