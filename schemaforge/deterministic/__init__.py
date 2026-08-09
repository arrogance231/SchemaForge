"""SchemaForge V2 deterministic pre-pass (research direction §2, stage 1).

This package contains the regex/rule extractors (:mod:`extractors`) and the
schema-aware pre-pass (:mod:`prepass`) that resolves
``SchemaSpec.deterministic_fields`` from raw text.  It is the hybrid pipeline's
stage 1 and, by design, the primary baseline the model is benchmarked against
(§6), so the extractors here are tuned rather than strawmen.
"""

from schemaforge.deterministic import extractors, prepass  # noqa: F401

__all__ = ["extractors", "prepass"]
