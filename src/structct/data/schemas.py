from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Expression(BaseModel):
    unit: str = ""
    value: float = 0.0
    operator: Literal["<", ">", "<=", ">=", "==", "!="] = "=="
    subject_name: str | None = ""


class Entity(BaseModel):
    id: str
    type: str
    text: str
    code_expr: list[Expression] | None = None


class Relation(BaseModel):
    id: str
    type: str
    arg1_id: str
    arg2_id: str


class RawGraph(BaseModel):
    entities: list[Entity] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return len(self.entities) == 0

    @property
    def has_nodes_without_relations(self) -> bool:
        return len(self.entities) > 0 and len(self.relations) == 0


class RawSample(BaseModel):
    topic_id: int
    year: int
    nct_id: str
    label: int
    patient_graph: RawGraph
    inc_trial_graph: RawGraph
    exc_trial_graph: RawGraph
    patient_description: str | None = None
    trial_title: str | None = None
    trial_summary: str | None = None
    trial_inc_text: str | None = None
    trial_exc_text: str | None = None

    @property
    def group_key(self) -> tuple[int, int]:
        return (self.topic_id, self.year)
