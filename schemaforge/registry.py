"""Schema registry: ``SchemaSpec``, field-path utilities, and the global registry.

Every schema module defines a ``SPEC`` and calls ``register(SPEC)`` at import
time. Importing ``schemaforge.schemas`` (which ``schemaforge/__init__.py`` does
automatically) populates the registry, so a bare ``import schemaforge`` is
sufficient to make ``all_schemas()`` non-empty.

The core invariant enforced here is the field-ownership split that drives the
hybrid deterministic + semantic architecture: every leaf field of a model must
be owned by exactly one of the two passes (``deterministic_fields`` or
``semantic_fields``).
"""

from __future__ import annotations

import types as _types
import typing
from dataclasses import dataclass
from typing import Any, Mapping

from pydantic import BaseModel

__all__ = [
    "SchemaSpec",
    "leaf_paths",
    "normalize_via_ontology",
    "register",
    "get_schema",
    "all_schemas",
    "training_schemas",
    "held_out_schemas",
]


def _is_base_model(t: Any) -> bool:
    """Return True if ``t`` is a pydantic v2 BaseModel subclass."""
    return isinstance(t, type) and issubclass(t, BaseModel)


def _unwrap_optional(t: Any) -> Any:
    """Unwrap ``Optional[X]`` / ``X | None`` to ``X``; return ``t`` unchanged otherwise."""
    origin = typing.get_origin(t)
    if origin is None or origin not in (typing.Union, _types.UnionType):
        return t
    args = typing.get_args(t)
    non_none = [a for a in args if a is not type(None)]
    if len(non_none) == 1:
        return non_none[0]
    return t


def _collect_leaf_paths(model: type[BaseModel], prefix: str, out: set[str]) -> None:
    """Recursively accumulate dotted leaf paths of ``model`` into ``out``.

    ``prefix`` carries a trailing dot so children join as ``parent.child`` or
    ``items[].child`` for list-nested models.
    """
    for name, field in model.model_fields.items():
        ann = _unwrap_optional(field.annotation)
        if typing.get_origin(ann) is list:
            args = typing.get_args(ann)
            if args:
                inner = _unwrap_optional(args[0])
                if _is_base_model(inner):
                    _collect_leaf_paths(inner, f"{prefix}{name}[].", out)
                    continue
        if _is_base_model(ann):
            _collect_leaf_paths(ann, f"{prefix}{name}.", out)
            continue
        out.add(f"{prefix}{name}")


def leaf_paths(model: type[BaseModel]) -> frozenset[str]:
    """Return the set of dotted leaf field paths of a pydantic v2 model.

    - A nested ``BaseModel`` field recurses with a dot: ``parent.child``.
    - A ``list[SubModel]`` field collapses its index: ``items[].qty``.
    - ``Optional[X]`` is unwrapped before classification.
    - Scalar fields (``str``/``int``/``float``/``bool``/``date``/``Decimal``)
      and ``dict``/``list[str]`` fields are leaves themselves.
    """
    out: set[str] = set()
    _collect_leaf_paths(model, "", out)
    return frozenset(out)


def normalize_via_ontology(spec: SchemaSpec, value: Any) -> Any:
    """Return the canonical value for ``value`` if it is an ontology key.

    The lookup is case-insensitive and whitespace-trimmed on both the given
    value and the ontology keys. Non-string values and values that are not
    ontology keys are returned unchanged.
    """
    if value is None or not spec.ontology:
        return value
    key = str(value).strip().casefold()
    for surface, canonical in spec.ontology.items():
        if surface.strip().casefold() == key:
            return canonical
    return value


@dataclass(frozen=True)
class SchemaSpec:
    """Metadata binding a schema to its two field-ownership classes.

    ``deterministic_fields`` are the dotted leaf paths the regex/rule pre-pass
    resolves near-perfectly (IDs, dates, emails, phones, currency). ``semantic_fields``
    are the dotted leaf paths that require language understanding. Every leaf path
    of ``model`` must be owned by exactly one of the two sets.
    """

    name: str
    model: type[BaseModel]
    deterministic_fields: frozenset[str]
    semantic_fields: frozenset[str]
    ontology: Mapping[str, str]
    held_out: bool = False

    def __post_init__(self) -> None:
        leaves = leaf_paths(self.model)
        det = frozenset(self.deterministic_fields)
        sem = frozenset(self.semantic_fields)
        overlap = det & sem
        unowned = leaves - (det | sem)
        not_in_model = (det | sem) - leaves
        if overlap or unowned or not_in_model:
            problems: list[str] = []
            if overlap:
                problems.append("owned by both passes: " + ", ".join(sorted(overlap)))
            if unowned:
                problems.append("not owned by either pass: " + ", ".join(sorted(unowned)))
            if not_in_model:
                problems.append("not a leaf path of the model: " + ", ".join(sorted(not_in_model)))
            raise ValueError(
                f"SchemaSpec {self.name!r} violates the field-ownership invariant "
                f"(deterministic | semantic must equal leaf_paths(model), disjoint): "
                + "; ".join(problems)
            )


_REGISTRY: dict[str, SchemaSpec] = {}


def register(spec: SchemaSpec) -> None:
    """Register a ``SchemaSpec`` by name; raise ``ValueError`` on duplicates."""
    if spec.name in _REGISTRY:
        raise ValueError(f"duplicate schema name in registry: {spec.name!r}")
    _REGISTRY[spec.name] = spec


def get_schema(name: str) -> SchemaSpec:
    """Return the registered ``SchemaSpec`` for ``name``.

    Raises ``KeyError`` listing all known schema names when ``name`` is not
    registered.
    """
    try:
        return _REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY)) if _REGISTRY else "(none registered)"
        raise KeyError(f"unknown schema {name!r}; registered schemas: {known}") from None


def all_schemas() -> list[SchemaSpec]:
    """Return every registered ``SchemaSpec``, sorted by name."""
    return sorted(_REGISTRY.values(), key=lambda spec: spec.name)


def training_schemas() -> list[SchemaSpec]:
    """Return registered specs with ``held_out is False``, sorted by name."""
    return sorted((spec for spec in _REGISTRY.values() if not spec.held_out), key=lambda spec: spec.name)


def held_out_schemas() -> list[SchemaSpec]:
    """Return registered specs with ``held_out is True``, sorted by name."""
    return sorted((spec for spec in _REGISTRY.values() if spec.held_out), key=lambda spec: spec.name)
