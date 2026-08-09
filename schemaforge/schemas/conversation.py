"""Conversation schema (held out): multi-turn dialogue extraction targets.

Designated held-out so the model never trains on this shape; it exercises
multi-turn nested structure (``turns[]``) rather than a flat field list.
"""

from datetime import date

from pydantic import BaseModel

from schemaforge.registry import SchemaSpec, register


class Turn(BaseModel):
    speaker: str | None = None
    text: str | None = None
    timestamp: str | None = None


class Conversation(BaseModel):
    channel: str | None = None
    started_date: date | None = None
    customer_name: str | None = None
    resolved: bool | None = None
    turns: list[Turn] | None = None


SPEC = SchemaSpec(
    name="conversation",
    model=Conversation,
    deterministic_fields=frozenset({"started_date", "turns[].timestamp"}),
    semantic_fields=frozenset(
        {"channel", "customer_name", "resolved", "turns[].speaker", "turns[].text"}
    ),
    ontology={},
    held_out=True,
)

register(SPEC)
