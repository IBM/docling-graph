from __future__ import annotations

from enum import Enum

from pydantic import BaseModel

from docling_graph.core.extractors.density import calculate_dynamic_density


class NoteKind(str, Enum):
    A = "a"
    B = "b"


class ChildModel(BaseModel):
    raw_text: str | None = None


class ParentModel(BaseModel):
    items: list[ChildModel] = []
    notes: list[str] = []
    tags: list[NoteKind] = []


def test_density_penalties_and_clamp():
    density = calculate_dynamic_density(ParentModel, max_depth=5)
    assert 1.2 <= density <= 3.0
    assert round(density, 2) == 2.1


def test_density_depth_guard_with_cycle():
    class Node(BaseModel):
        children: list[Node] = []

    density = calculate_dynamic_density(Node, max_depth=2)
    assert 1.2 <= density <= 3.0
