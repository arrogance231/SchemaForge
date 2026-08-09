"""Knowledge-graph triple schema (held out): relation-extraction targets.

Designated held-out so the model never trains on this shape; it exercises a
triple-shaped subject/predicate/object structure rather than a flat field list.
"""

from datetime import date

from pydantic import BaseModel

from schemaforge.registry import SchemaSpec, register


class KgTriple(BaseModel):
    subject: str | None = None
    predicate: str | None = None
    object: str | None = None
    subject_type: str | None = None
    object_type: str | None = None
    context: str | None = None
    confidence: float | None = None
    extracted_date: date | None = None


SPEC = SchemaSpec(
    name="kg_triple",
    model=KgTriple,
    deterministic_fields=frozenset({"extracted_date"}),
    semantic_fields=frozenset(
        {
            "subject",
            "predicate",
            "object",
            "subject_type",
            "object_type",
            "context",
            "confidence",
        }
    ),
    ontology={},
    held_out=True,
)

register(SPEC)
