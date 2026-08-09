"""SchemaForge V2 hybrid merge pipeline (research direction §6, "hybrid (rules → SchemaForge)").

The deterministic pre-pass owns ``SchemaSpec.deterministic_fields``; the model
owns the ``semantic_fields`` leaves plus any deterministic leaf the pre-pass
failed to resolve.  :mod:`pipeline` folds the two into ONE nested JSON
prediction, with the pre-pass winning every conflict on the fields it owns.
"""

from schemaforge.hybrid.pipeline import merge_prediction, rules_only_prediction

__all__ = ["merge_prediction", "rules_only_prediction"]
