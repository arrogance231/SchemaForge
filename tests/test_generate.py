"""Tests for seed coverage and deterministic hard-example generation."""

import random

import pytest

import schemaforge  # noqa: F401  (importing the package populates the registry)
from schemaforge.evaluation.json_utils import flatten
from schemaforge.hardexamples.generate import (
    _split_list_arg,
    apply_operators,
    generate_dataset,
    main,
    serialize_records,
)
from schemaforge.hardexamples.seeds import _BUILDERS, build_seed
from schemaforge.registry import all_schemas, get_schema, held_out_schemas, training_schemas


def test_seed_builders_cover_every_registered_schema():
    assert {spec.name for spec in all_schemas()} == set(_BUILDERS)


def test_every_schema_seed_gold_validates():
    for i, spec in enumerate(all_schemas()):
        text, gold = build_seed(spec.name, random.Random(i))
        spec.model.model_validate(gold)  # raises if invalid
        assert isinstance(text, str) and text, spec.name


def test_every_gold_string_value_appears_in_clean_text():
    for i, spec in enumerate(all_schemas()):
        text, gold = build_seed(spec.name, random.Random(10 + i))
        for path, value in flatten(gold).items():
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, str):
                        assert item in text, (spec.name, path, item)
            elif isinstance(value, str):
                assert value in text, (spec.name, path, value)


def test_same_seed_produces_byte_identical_output():
    a = generate_dataset(["invoice", "medical_note"], 3, [0.0, 0.5, 1.0], ["ocr_noise", "typo"], seed=7)
    b = generate_dataset(["invoice", "medical_note"], 3, [0.0, 0.5, 1.0], ["ocr_noise", "typo"], seed=7)
    assert serialize_records(a) == serialize_records(b)
    assert a[0].tags == b[0].tags


def test_different_seed_produces_different_output():
    a = generate_dataset(["invoice", "medical_note"], 3, [0.0, 0.5, 1.0], ["ocr_noise", "typo"], seed=7)
    b = generate_dataset(["invoice", "medical_note"], 3, [0.0, 0.5, 1.0], ["ocr_noise", "typo"], seed=8)
    assert serialize_records(a) != serialize_records(b)


def test_generated_records_carry_operator_and_severity_tags():
    records = generate_dataset(["invoice", "medical_note"], 2, [0.5, 1.0], ["ocr_noise", "typo"], seed=3)
    assert len(records) == 2 * 2 * 2
    for record in records:
        assert "ocr_noise" in record.tags
        assert "typo" in record.tags
        assert any(tag.startswith("severity=") for tag in record.tags)


def test_apply_operators_accumulates_tags_and_validates_gold():
    spec = get_schema("medical_note")
    text, gold = build_seed("medical_note", random.Random(5))
    new_text, new_gold, tags = apply_operators(
        text, gold, spec, ["abbreviate", "delabel"], random.Random(6), 1.0
    )
    assert set(tags) == {"abbreviate", "delabel"}
    spec.model.model_validate(new_gold)


def test_apply_operators_unknown_operator_raises():
    spec = get_schema("invoice")
    text, gold = build_seed("invoice", random.Random(1))
    with pytest.raises(ValueError):
        apply_operators(text, gold, spec, ["not_an_operator"], random.Random(2), 0.5)


def test_generate_unknown_schema_raises():
    with pytest.raises(KeyError):
        generate_dataset(["does_not_exist"], 1, [0.5], ["ocr_noise"], seed=1)


def test_split_train_rejects_held_out_schema_by_name():
    with pytest.raises(ValueError) as excinfo:
        generate_dataset(["insurance_claim"], 1, [0.0], [], seed=1, split="train")
    assert "insurance_claim" in str(excinfo.value)


def test_split_train_defaults_to_training_schemas_only():
    records = generate_dataset([], 1, [0.0], [], seed=1, split="train")
    names = {record.schema for record in records}
    assert names == {spec.name for spec in training_schemas()}
    assert names.isdisjoint({spec.name for spec in held_out_schemas()})


def test_split_eval_defaults_to_exactly_the_held_out_schemas():
    records = generate_dataset([], 1, [0.0], [], seed=1, split="eval")
    names = {record.schema for record in records}
    assert names == {spec.name for spec in held_out_schemas()}


def test_split_any_defaults_to_every_registered_schema():
    records = generate_dataset([], 1, [0.0], [], seed=1, split="any")
    assert {record.schema for record in records} == {spec.name for spec in all_schemas()}


def test_split_train_allows_explicit_training_schemas():
    records = generate_dataset(["invoice", "medical_note"], 1, [0.0], [], seed=1, split="train")
    assert {record.schema for record in records} == {"invoice", "medical_note"}


def test_clean_generation_produces_uncorrupted_seeds():
    records = generate_dataset(["medical_note"], 1, [0.0], [], seed=1)
    assert len(records) == 1
    assert records[0].tags == ["clean", "severity=0.0"]
    get_schema("medical_note").model.model_validate(records[0].reference)


def test_split_list_arg_accepts_space_and_comma_forms():
    assert _split_list_arg(["invoice", "medical_note"]) == ["invoice", "medical_note"]
    assert _split_list_arg(["invoice,medical_note"]) == ["invoice", "medical_note"]
    assert _split_list_arg(["invoice,", " medical_note"]) == ["invoice", "medical_note"]
    assert _split_list_arg(["invoice, ,medical_note"]) == ["invoice", "medical_note"]
    assert _split_list_arg([]) == []
    assert _split_list_arg(None) == []


def test_cli_writes_missing_parent_dir_and_equivalent_arg_forms_are_identical(tmp_path):
    common = ["--n", "2", "--severities", "0.0,0.5", "--operators", "ocr_noise,typo", "--seed", "11"]
    out_a = tmp_path / "nested" / "missing" / "smoke_a.jsonl"
    out_b = tmp_path / "nested" / "missing" / "smoke_b.jsonl"
    main(["--schemas", "invoice", "medical_note", *common, "--out", str(out_a)])
    main(["--schemas", "invoice,medical_note", *common, "--out", str(out_b)])
    assert out_a.is_file() and out_a.read_text(encoding="utf-8").strip()
    assert out_a.read_bytes() == out_b.read_bytes()
