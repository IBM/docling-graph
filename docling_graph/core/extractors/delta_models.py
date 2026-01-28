"""Delta-operation models for registry-based diff extraction."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DeleteOperation(BaseModel):
    """Explicit delete operation for an entity by id."""

    id: str = Field(..., description="Entity identifier to delete.")
    entity_type: str | None = Field(
        default=None,
        description="Optional entity type discriminator (best-effort).",
    )
    reason: str | None = Field(
        default=None,
        description="Optional rationale for deletion.",
    )


class EvidenceItem(BaseModel):
    """Evidence snippet pointing to source text for new entities."""

    entity_id: str = Field(..., description="ID of the newly created entity.")
    snippet: str = Field(..., description="Short quote or pointer from the chunk.")
    source: str | None = Field(
        default=None,
        description="Optional source label (e.g., chunk or page reference).",
    )


class DeltaOperation(BaseModel):
    """
    Delta operations for incremental extraction.

    - data: partial data matching the target schema (upserts only)
    - deletes: explicit delete operations
    - evidence: required snippets for new entities
    """

    model_config = ConfigDict(extra="forbid")

    data: dict[str, Any] = Field(
        default_factory=dict,
        description="Partial data matching target schema (upserts only).",
    )
    deletes: list[DeleteOperation] = Field(
        default_factory=list,
        description="Explicit delete operations (by id).",
    )
    evidence: list[EvidenceItem] = Field(
        default_factory=list,
        description="Evidence snippets for newly created entities.",
    )
