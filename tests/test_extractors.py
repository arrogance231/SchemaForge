"""Tests for the deterministic extractors.

Covers the span invariant (``text[a:b] == raw``), the documented date/money
disambiguation rules, trailing-punctuation handling for emails/URLs, E.164
phone output, the no-raise guarantee on hostile input, and overlap resolution.
"""

import random
import string
from datetime import date
from decimal import Decimal

import pytest

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

SAMPLE = (
    "Invoice #INV-0011\n"
    "Date: 2026-04-10\n"
    "Vendor: Acme Supply Co. (vendor@acme.example.com, +1 (415) 555-2671)\n"
    "Total: $1,234.50\n"
    "Tax (8.25%): $101.85\n"
    "Contact a@b.com, or see http://x.io.\n"
    "Euro amount €1.234,50, and (1,234.50) pending.\n"
)


def _all_extractions(doc: str) -> list[Extraction]:
    """Run every extractor over ``doc`` (identifiers get a sample pattern)."""
    return (
        find_dates(doc)
        + find_emails(doc)
        + find_urls(doc)
        + find_phones(doc)
        + find_money(doc)
        + find_percentages(doc)
        + find_identifiers(doc, [r"\bINV-?\d+\b"])
    )


def test_span_invariant_holds_for_every_extraction():
    """Every extraction's span must slice back to exactly its raw substring."""
    extractions = _all_extractions(SAMPLE)
    assert extractions, "the sample document should trigger every extractor"
    for e in extractions:
        assert SAMPLE[e.span[0] : e.span[1]] == e.raw, (e.extractor, e.span, e.raw)


def test_extraction_is_frozen_with_default_confidence():
    e = Extraction(value="x", span=(0, 1), raw="x", extractor="test")
    assert e.confidence == 1.0
    with pytest.raises(Exception):
        e.value = "y"  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# dates
# --------------------------------------------------------------------------- #


def test_each_documented_date_format_parses_to_the_correct_date():
    cases = {
        "2026-04-10": date(2026, 4, 10),
        "04/10/2026": date(2026, 4, 10),
        "10/04/2026": date(2026, 10, 4),
        "April 10, 2026": date(2026, 4, 10),
        "10 April 2026": date(2026, 4, 10),
        "Apr 10 2026": date(2026, 4, 10),
        "2026/04/10": date(2026, 4, 10),
    }
    for text, expected in cases.items():
        results = find_dates(text)
        assert results, f"no date found for {text!r}"
        assert results[0].value == expected, f"{text!r} parsed to {results[0].value}"


def test_ambiguous_numeric_dates_get_confidence_0_7():
    assert find_dates("04/10/2026")[0].confidence == 0.7
    assert find_dates("10/04/2026")[0].confidence == 0.7


def test_unambiguous_numeric_dates_get_confidence_1_0():
    assert find_dates("04/25/2026")[0].confidence == 1.0  # month forced (25 can't be a month)
    assert find_dates("25/04/2026")[0].confidence == 1.0  # day forced (25 can't be a month)


def test_dayfirst_flips_the_ambiguous_reading():
    assert find_dates("10/04/2026")[0].value == date(2026, 10, 4)
    assert find_dates("10/04/2026", dayfirst=True)[0].value == date(2026, 4, 10)


def test_dates_do_not_swallow_trailing_text():
    results = find_dates("Deadline 2026-04-10, then shipped.")
    assert [e.raw for e in results] == ["2026-04-10"]


# --------------------------------------------------------------------------- #
# money
# --------------------------------------------------------------------------- #


def _first_money(text: str) -> Decimal:
    matches = find_money(text)
    assert matches, f"no money match in {text!r}"
    return matches[0].value


def test_money_required_forms():
    assert _first_money("$1,234.50") == Decimal("1234.50")
    assert _first_money("€1.234,50") == Decimal("1234.50")
    assert _first_money("(1,234.50)") == Decimal("-1234.50")


def test_money_extra_forms():
    assert _first_money("1234.50") == Decimal("1234.50")
    assert _first_money("USD 1234.50") == Decimal("1234.50")
    assert _first_money("1,234.50 USD") == Decimal("1234.50")
    assert _first_money("1234,50") == Decimal("1234.50")


def test_money_preserves_decimal_places():
    assert _first_money("$480.00") == Decimal("480.00")


def test_bare_integers_are_not_money():
    assert find_money("count 2026, id 1001, qty 4") == []


def test_money_rejects_invalid_thousands_grouping():
    assert find_money("1,23.50") == []  # 2-digit middle group
    assert find_money("$1,2345.50") == []  # 4-digit group


# --------------------------------------------------------------------------- #
# emails / urls
# --------------------------------------------------------------------------- #


def test_emails_and_urls_do_not_swallow_trailing_punctuation():
    text = "Contact a@b.com, or see http://x.io. Then a@b.com. and (http://x.io)"
    emails = [e.raw for e in find_emails(text)]
    urls = [e.raw for e in find_urls(text)]
    assert emails == ["a@b.com", "a@b.com"]
    assert urls == ["http://x.io", "http://x.io"]


def test_email_matches_the_whole_local_part_not_a_fragment():
    assert [e.raw for e in find_emails("xm@y.com")] == ["xm@y.com"]


def test_url_supports_www_and_paths():
    text = "www.example.com and https://example.com/a/b?x=1"
    values = [e.value for e in find_urls(text)]
    assert "www.example.com" in values
    assert "https://example.com/a/b?x=1" in values


# --------------------------------------------------------------------------- #
# phones
# --------------------------------------------------------------------------- #


def test_us_phone_yields_e164():
    results = find_phones("Call (415) 555-2671 now")
    assert any(r.value == "+14155552671" for r in results)


def test_invalid_region_raises():
    with pytest.raises(ValueError):
        find_phones("Call (415) 555-2671", region="XX")


# --------------------------------------------------------------------------- #
# percentages / identifiers
# --------------------------------------------------------------------------- #


def test_percentages_emit_the_number_not_the_fraction():
    assert find_percentages("8%")[0].value == Decimal("8")
    assert find_percentages("8.5 %")[0].value == Decimal("8.5")
    assert find_percentages("8,5%")[0].value == Decimal("8.5")


def test_identifiers_return_all_pattern_matches():
    text = "claim CLM-88 and policy POL-12 then claim CLM-90"
    values = [e.value for e in find_identifiers(text, [r"\bCLM-\d+\b", r"\bPOL-\d+\b"])]
    assert values == ["CLM-88", "POL-12", "CLM-90"]


# --------------------------------------------------------------------------- #
# robustness
# --------------------------------------------------------------------------- #


def test_no_extractor_raises_on_hostile_input():
    random.seed(42)
    noise = "".join(
        random.choice(string.ascii_letters + string.digits + string.punctuation + " ")
        for _ in range(10000)
    )
    for text in ("", "...!!!???", noise):
        find_dates(text)
        find_dates(text, dayfirst=True)
        find_emails(text)
        find_urls(text)
        find_phones(text)
        find_money(text)
        find_percentages(text)
        find_identifiers(text, [r"\bINV-?\d+\b"])


# --------------------------------------------------------------------------- #
# overlap resolution
# --------------------------------------------------------------------------- #


def test_resolve_overlaps_keeps_the_longest_match():
    a = Extraction(value=1, span=(0, 10), raw="0123456789", extractor="x")
    b = Extraction(value=2, span=(2, 5), raw="234", extractor="y")
    c = Extraction(value=3, span=(10, 12), raw="ab", extractor="z")
    out = resolve_overlaps([b, c, a])
    assert [e.value for e in out] == [1, 3]


def test_resolve_overlaps_tie_breaks_by_confidence_then_earlier_start():
    a = Extraction(value=1, span=(0, 10), raw="0123456789", extractor="x", confidence=0.9)
    b = Extraction(value=2, span=(0, 10), raw="0123456789", extractor="y", confidence=0.5)
    assert resolve_overlaps([b, a]) == [a]  # higher confidence wins
    c = Extraction(value=3, span=(2, 12), raw="0123456789", extractor="x", confidence=0.9)
    d = Extraction(value=4, span=(3, 13), raw="1234567890", extractor="y", confidence=0.9)
    assert resolve_overlaps([d, c]) == [c]  # equal length/confidence -> earlier start wins
