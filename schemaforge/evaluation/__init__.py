"""SchemaForge V2 field-level evaluation package.

Provides brace-aware JSON extraction (:mod:`json_utils`), value comparison and
micro-averaged metrics (:mod:`metrics`), and a slicing evaluation harness
(:mod:`harness`).
"""

from schemaforge.evaluation.harness import EvalRecord, evaluate, format_table, load_records
from schemaforge.evaluation.json_utils import extract_json, flatten, parse_json
from schemaforge.evaluation.metrics import (
    RecordResult,
    aggregate,
    evaluate_record,
    values_equal,
)

__all__ = [
    "extract_json",
    "parse_json",
    "flatten",
    "values_equal",
    "evaluate_record",
    "RecordResult",
    "aggregate",
    "EvalRecord",
    "load_records",
    "evaluate",
    "format_table",
]
