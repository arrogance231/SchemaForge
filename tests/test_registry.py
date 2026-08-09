"""Tests for the schema registry and the field-ownership invariant."""

import pytest

import schemaforge  # noqa: F401  (importing the package populates the registry)
from schemaforge.registry import (
    all_schemas,
    get_schema,
    held_out_schemas,
    leaf_paths,
    training_schemas,
)
from schemaforge.schemas.medical_note import MedicalNote


def test_all_schemas_registered():
    assert len(all_schemas()) == 12
    assert len(training_schemas()) == 9


def test_exactly_three_held_out():
    held = held_out_schemas()
    assert len(held) == 3
    assert {spec.name for spec in held} == {"insurance_claim", "conversation", "kg_triple"}
    all_names = {spec.name for spec in all_schemas()}
    assert {spec.name for spec in training_schemas()} == all_names - {spec.name for spec in held}


def test_field_ownership_invariant_holds_for_every_schema():
    for spec in all_schemas():
        leaves = leaf_paths(spec.model)
        assert spec.deterministic_fields.isdisjoint(spec.semantic_fields), spec.name
        assert spec.deterministic_fields | spec.semantic_fields == leaves, spec.name


def test_required_ontologies():
    assert len(get_schema("medical_note").ontology) >= 8
    assert len(get_schema("support_ticket").ontology) >= 8
    assert len(get_schema("insurance_claim").ontology) >= 6


def test_every_schema_stays_small():
    """Schemas stay small: 5-9 top-level fields, and at most 12 leaves once
    nested line-item/turn models are expanded. The bound is on schema size, not
    on nesting -- nested list-of-object fields are intended (see invoice and
    receipt line_items) and legitimately push the leaf count above the
    top-level count.
    """
    for spec in all_schemas():
        assert 5 <= len(spec.model.model_fields) <= 9, f"{spec.name} top-level"
        assert len(leaf_paths(spec.model)) <= 12, f"{spec.name} leaves"


def test_leaf_paths_collapses_lists_with_bracket():
    leaves = leaf_paths(MedicalNote)
    assert "medications[].name" in leaves
    assert "medications[].dosage" in leaves
    assert "medications" not in leaves


def test_leaf_paths_unwraps_optional():
    from schemaforge.schemas.invoice import Invoice

    leaves = leaf_paths(Invoice)
    assert "invoice_date" in leaves  # Optional[date] unwraps to a leaf
    assert "invoice_number" in leaves


def test_get_schema_unknown_raises():
    with pytest.raises(KeyError):
        get_schema("does_not_exist")


def test_get_schema_known():
    assert get_schema("invoice").name == "invoice"
