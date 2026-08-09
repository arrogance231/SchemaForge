"""Tests for the ten corruption operators.

Iterates the real ``OPERATORS`` registry and the real schema registry rather
than hardcoding names, so a missing operator or schema fails these tests
instead of being silently skipped.
"""

import copy
import random
import re

import schemaforge  # noqa: F401  (importing the package populates the registry)
from schemaforge.hardexamples.operators import OPERATORS
from schemaforge.hardexamples.seeds import build_seed
from schemaforge.registry import all_schemas


def _seed_for(schema_name, rng_seed):
    return build_seed(schema_name, random.Random(rng_seed))


def test_severity_zero_is_identity_for_all_operators():
    for name, operator in OPERATORS.items():
        for i, spec in enumerate(all_schemas()):
            text, gold = _seed_for(spec.name, 200 + i)
            new_text, new_gold, tags = operator(text, gold, random.Random(300 + i), 0.0)
            assert new_text == text, (name, spec.name)
            assert new_gold == gold, (name, spec.name)
            assert tags == [], (name, spec.name)


def test_operators_never_raise_on_any_schema_seed():
    for name, operator in OPERATORS.items():
        for i, spec in enumerate(all_schemas()):
            text, gold = _seed_for(spec.name, 400 + i)
            for severity in (0.0, 0.5, 1.0):
                operator(text, gold, random.Random(500 + i), severity)  # must not raise


def test_operators_never_mutate_the_caller_gold():
    for name, operator in OPERATORS.items():
        for i, spec in enumerate(all_schemas()):
            text, gold = _seed_for(spec.name, 600 + i)
            snapshot = copy.deepcopy(gold)
            operator(text, gold, random.Random(700 + i), 1.0)
            assert gold == snapshot, name


def test_nest_output_gold_validates():
    for i, spec in enumerate(all_schemas()):
        text, gold = _seed_for(spec.name, 800 + i)
        new_text, new_gold, tags = OPERATORS["nest"](text, gold, random.Random(900 + i), 1.0)
        spec.model.model_validate(new_gold)  # raises if invalid


def _medical_note_seed(rng, predicate):
    """Return the first medical_note seed satisfying ``predicate`` (bounded)."""
    for _ in range(200):
        text, gold = build_seed("medical_note", rng)
        if predicate(text, gold):
            return text, gold
    raise AssertionError("no qualifying medical_note seed found")


def test_abbreviate_medical_note_keeps_canonical_gold():
    text, gold = _medical_note_seed(random.Random(1234), lambda t, g: g["diagnosis"] == "Hypertension")
    assert gold["diagnosis"] == "Hypertension"
    new_text, new_gold, tags = OPERATORS["abbreviate"](text, gold, random.Random(7), 1.0)
    assert "HTN" in new_text
    assert "Hypertension" not in new_text
    assert new_gold["diagnosis"] == "Hypertension"
    assert tags == ["abbreviate"]


def test_implicit_age_removes_digits_keeps_gold():
    text, gold = _medical_note_seed(random.Random(4321), lambda t, g: re.search(r"\bage \d+\b", t) is not None)
    assert re.search(r"\bage \d+\b", text)
    age = gold["patient_age"]
    new_text, new_gold, tags = OPERATORS["implicit"](text, gold, random.Random(9), 1.0)
    assert "celebrated" in new_text
    assert f"age {age}" not in new_text
    assert new_gold["patient_age"] == age
    assert tags == ["implicit"]


def test_every_operator_changes_text_at_severity_one():
    for name, operator in OPERATORS.items():
        changed = False
        for i, spec in enumerate(all_schemas()):
            text, gold = _seed_for(spec.name, 1000 + i)
            new_text, _, _ = operator(text, gold, random.Random(2000 + i), 1.0)
            if new_text != text:
                changed = True
                break
        assert changed, f"operator {name!r} did nothing at severity 1.0"
