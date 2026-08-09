"""Brace-aware JSON extraction and leaf-path flattening.

Fixes the V1 defect where the eval scripts (``src/03_eval.py``,
``src/04_eval_public_dataset.py``) cut at the first ``}`` via
``text.find("{")`` / ``text.find("}")``, truncating any nested object.  This
module instead extracts the substring through the *matching* closing brace with
a real balanced-brace scanner, then flattens the result to the dotted leaf
paths used by :mod:`schemaforge.registry`.
"""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*$")


def _strip_fences(text: str) -> str:
    """Return ``text`` with a surrounding ``` / ```json fence removed.

    If no fence is present the text is returned unchanged.  When a fence is
    found, only the content between the first fence line and its closing fence
    line is returned.
    """
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if _FENCE_RE.match(line):
            for j in range(i + 1, len(lines)):
                if _FENCE_RE.match(lines[j]):
                    return "\n".join(lines[i + 1 : j])
            return "\n".join(lines[i + 1 :])
    return text


def extract_json(text: str) -> str | None:
    """Return the substring of ``text`` spanning the first balanced JSON object.

    A ```json / ``` fence is stripped first when present.  The scan begins at
    the first ``{`` and tracks nesting depth for both ``{}`` and ``[]``,
    ignoring braces inside string literals and honouring backslash escapes
    inside those literals.  Returns ``None`` when no balanced object is found.
    """
    body = _strip_fences(text)
    start = body.find("{")
    if start == -1:
        return None
    depth_curly = 0
    depth_square = 0
    in_string = False
    escaped = False
    for i in range(start, len(body)):
        ch = body[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth_curly += 1
        elif ch == "}":
            depth_curly -= 1
            if depth_curly == 0 and depth_square == 0:
                return body[start : i + 1]
        elif ch == "[":
            depth_square += 1
        elif ch == "]":
            depth_square -= 1
    return None


def parse_json(text: str) -> dict | None:
    """Parse the first balanced JSON object in ``text``.

    Extracts the object with :func:`extract_json` and decodes it with
    ``json.loads``.  Returns ``None`` (never raises) when extraction fails, the
    extraction is not a JSON object, or ``json.loads`` fails.
    """
    block = extract_json(text)
    if block is None:
        return None
    try:
        parsed = json.loads(block)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def _flatten_into(obj: Any, prefix: str, out: dict[str, list[Any]]) -> None:
    """Accumulate leaf paths of ``obj`` into ``out`` (path -> list of values)."""
    if isinstance(obj, dict):
        if not obj:
            return
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else key
            _flatten_into(value, path, out)
        return
    if isinstance(obj, list):
        if not obj:
            return
        if all(isinstance(item, dict) for item in obj):
            for item in obj:
                _flatten_into(item, prefix + "[]", out)
            return
        out.setdefault(prefix, []).append(obj)
        return
    out.setdefault(prefix, []).append(obj)


def flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten a parsed JSON object to dotted leaf paths.

    Mirrors the ``registry.leaf_paths`` convention: nested dicts recurse with a
    dot (``parent.child``) and list-of-object elements collapse their index to
    ``[]``, so ``{"line_items": [{"qty": 2}, {"qty": 3}]}`` becomes
    ``{"line_items[].qty": [2, 3]}``.

    Type convention: a leaf path containing ``[]`` ALWAYS maps to a ``list`` of
    its element values, in document order, regardless of how many elements it
    has (including exactly one), so ``{"medications": [{"name": "a"}]}`` becomes
    ``{"medications[].name": ["a"]}``.  A leaf path with no ``[]`` maps to its
    scalar value.  A path whose value is itself a JSON list of scalars with no
    ``[]`` in the path (the elements are not objects, e.g.
    ``{"tags": ["x", "y"]}``) keeps that list as its value.  Empty dict/list
    values produce no leaf and ``None`` values are retained as leaves with value
    ``None``.  A ``None`` object flattens to an empty dict.
    """
    if obj is None:
        return {}
    out: dict[str, list[Any]] = {}
    _flatten_into(obj, prefix, out)
    return {
        path: values if "[]" in path else values[0]
        for path, values in out.items()
    }
