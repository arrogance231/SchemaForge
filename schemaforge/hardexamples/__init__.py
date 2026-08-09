"""Hard-example generation for SchemaForge V2.

Provides clean seed generation (:mod:`seeds`), the ten corruption operators
(:mod:`operators`) and deterministic dataset generation (:mod:`generate`).
"""

from schemaforge.hardexamples.generate import (
    apply_operators,
    generate_dataset,
    main,
    serialize_records,
)
from schemaforge.hardexamples.operators import OPERATORS, Operator
from schemaforge.hardexamples.seeds import build_seed

__all__ = [
    "Operator",
    "OPERATORS",
    "build_seed",
    "apply_operators",
    "generate_dataset",
    "serialize_records",
    "main",
]
