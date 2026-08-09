"""SchemaForge: hybrid deterministic + semantic structured extraction engine.

Importing this package imports :mod:`schemaforge.schemas`, which registers all
schema modules, so the registry is always populated on ``import schemaforge``.
"""

from schemaforge import schemas as _schemas  # noqa: F401  (populates the registry on import)
from schemaforge import evaluation  # noqa: F401  (exposes the evaluation package)
from schemaforge.registry import (
    SchemaSpec,
    all_schemas,
    get_schema,
    held_out_schemas,
    training_schemas,
)

__all__ = [
    "SchemaSpec",
    "get_schema",
    "all_schemas",
    "training_schemas",
    "held_out_schemas",
]
