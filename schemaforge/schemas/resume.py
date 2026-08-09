"""Resume schema: candidate-profile extraction targets."""

from pydantic import BaseModel

from schemaforge.registry import SchemaSpec, register


class Resume(BaseModel):
    candidate_name: str | None = None
    email: str | None = None
    phone: str | None = None
    job_title: str | None = None
    years_experience: float | None = None
    highest_degree: str | None = None
    current_employer: str | None = None
    location: str | None = None


ONTOLOGY = {
    "Corp.": "Corporation",
    "Inc.": "Incorporated",
    "Intl.": "International",
    "Tech.": "Technologies",
}

SPEC = SchemaSpec(
    name="resume",
    model=Resume,
    deterministic_fields=frozenset({"email", "phone"}),
    semantic_fields=frozenset(
        {
            "candidate_name",
            "job_title",
            "years_experience",
            "highest_degree",
            "current_employer",
            "location",
        }
    ),
    ontology=ONTOLOGY,
)

register(SPEC)
