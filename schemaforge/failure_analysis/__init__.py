"""SchemaForge V2 failure-analysis package.

Automatic error classifier for research direction §7: buckets every
field-level discrepancy produced by :mod:`schemaforge.evaluation.metrics` into
the 8 named categories plus the ``unclassified_mismatch`` catch-all, and
aggregates them into per-category/per-schema/per-operator counts with worst-N
examples for inspection.
"""

from schemaforge.failure_analysis.analyze import (
    FailureInstance,
    FailureReport,
    build_report,
    classify_record,
)

__all__ = [
    "FailureInstance",
    "classify_record",
    "FailureReport",
    "build_report",
]
