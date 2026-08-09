"""Tests for brace-aware JSON extraction and flattening (part 1 foundation)."""

from schemaforge.evaluation.json_utils import extract_json, flatten, parse_json


def test_nested_object_survives_balanced_scan():
    text = 'noise {"a":{"b":1},"c":2} tail'
    extracted = extract_json(text)
    # The old V1 behaviour (text.find("}")) would have truncated at the inner
    # closing brace, producing exactly the string below — assert we do NOT do that.
    assert extracted == '{"a":{"b":1},"c":2}'
    assert extracted != '{"a":{"b":1}'
    assert parse_json(text) == {"a": {"b": 1}, "c": 2}


def test_braces_inside_string_literals_ignored():
    assert parse_json('{"a":"}"}') == {"a": "}"}


def test_escaped_quotes_inside_string_literals():
    assert parse_json(r'{"a":"say \"hi\""}') == {"a": 'say "hi"'}


def test_fenced_json_block_parses():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('```\n{"a": {"b": 2}}\n```') == {"a": {"b": 2}}


def test_unbalanced_input_returns_none_without_raising():
    assert extract_json('noise {"a":') is None
    assert extract_json('no braces at all') is None
    assert extract_json('{"a":[1,2}') is None  # closes the object, not the array
    assert parse_json('{"a": 1') is None


def test_flatten_list_of_objects_collapses_index():
    parsed = {"line_items": [{"qty": 2}, {"qty": 3}]}
    assert flatten(parsed) == {"line_items[].qty": [2, 3]}


def test_flatten_nested_dicts_use_dots():
    assert flatten({"a": {"b": {"c": 1}}}) == {"a.b.c": 1}


def test_flatten_none_retained_empty_values_dropped():
    assert flatten({"a": None, "b": {}, "c": []}) == {"a": None}


def test_flatten_keeps_scalar_list_leaf_intact():
    assert flatten({"tags": ["x", "y"]}) == {"tags": ["x", "y"]}


def test_flatten_one_element_repeated_path_stays_list():
    # A [] path maps to a list even with exactly one element (a bare scalar here
    # would silently drop partial credit in evaluate_record).
    assert flatten({"medications": [{"name": "a"}]}) == {"medications[].name": ["a"]}
    assert flatten({"medications": [{"name": "a"}, {"name": "b"}]}) == {
        "medications[].name": ["a", "b"]
    }
