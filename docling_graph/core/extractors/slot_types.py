"""
Core data models for slot-based bottom-up extraction.

This module defines the data structures used for progressive knowledge graph
extraction optimized for small LLMs (≈1B–3B parameters).
"""

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Optional

from pydantic import BaseModel, Field


@dataclass
class SourceLocation:
    """
    Provenance information linking an extracted entity to its source in the document.

    All fields are assigned by code from slot metadata, never by the LLM.
    """

    page_no: int
    """Page number (1-indexed)."""

    element_type: str
    """Type of document element: 'table', 'figure', 'text', etc."""

    element_index: int
    """Index of the element within its type on the page."""

    bbox: tuple[float, float, float, float] | None = None
    """Bounding box coordinates (x0, y0, x1, y1) if available."""

    table_row: int | None = None
    """Row index for table elements."""

    table_col: int | None = None
    """Column index for table elements."""

    text_snippet: str = ""
    """Short excerpt from the source content for human verification."""

    charspan: tuple[int, int] | None = None
    """Character span (start, end) within the element if available."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "page_no": self.page_no,
            "element_type": self.element_type,
            "element_index": self.element_index,
            "bbox": self.bbox,
            "table_row": self.table_row,
            "table_col": self.table_col,
            "text_snippet": self.text_snippet[:100],  # Truncate for storage
            "charspan": self.charspan,
        }


@dataclass
class ExtractionSlot:
    """
    A unit of content from the document to be processed by the LLM.

    Slots are enumerated from DoclingDocument structure (tables, figures, text blocks)
    and provide stable identity and provenance for extracted entities.
    """

    slot_id: str
    """Unique identifier for this slot (e.g., 'table0row5', 'figure2', 'textp1b3')."""

    surrogatekey: str
    """Stable surrogate key derived from document fingerprint and slot metadata."""

    element_type: str
    """Type of element: 'table_row', 'figure', 'text_block'."""

    page_no: int
    """Page number where this slot appears."""

    bbox: tuple[float, float, float, float] | None = None
    """Bounding box if available."""

    table_index: int | None = None
    """Table index for table row slots."""

    row_index: int | None = None
    """Row index for table row slots."""

    block_index: int | None = None
    """Block index for text slots."""

    charspan: tuple[int, int] | None = None
    """Character span within the element."""

    content: str = ""
    """The actual text content to be processed."""

    rawdata: dict[str, Any] | None = None
    """Raw structured data (e.g., table cells) if applicable."""

    text_snippet: str = ""
    """Short excerpt for display/debugging."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "slot_id": self.slot_id,
            "surrogatekey": self.surrogatekey,
            "element_type": self.element_type,
            "page_no": self.page_no,
            "bbox": self.bbox,
            "table_index": self.table_index,
            "row_index": self.row_index,
            "block_index": self.block_index,
            "charspan": self.charspan,
            "content": self.content[:500],  # Truncate for storage
            "text_snippet": self.text_snippet[:100],
        }


class EntityEvidence(BaseModel):
    """
    Evidence provided by the LLM for an extracted entity.

    This is returned by the LLM but does not determine identity.
    """

    quote: str = Field(default="", description="Exact short snippet from source")
    local_span: dict[str, int] | None = Field(
        default=None,
        description="Character span within the slot content {start: int, end: int}"
    )


class ExtractedEntity(BaseModel):
    """
    Base class for entities extracted from slots.

    Subclasses should add domain-specific fields. The LLM returns these fields
    plus optional evidence, but never surrogatekey or sourcelocation.
    """

    evidence: EntityEvidence | None = Field(
        default=None,
        description="Evidence for this extraction (optional, provided by LLM)"
    )

    # These fields are assigned by code after LLM extraction
    surrogatekey: str | None = Field(
        default=None,
        description="Stable identifier assigned by code from slot metadata"
    )

    sourcelocation: SourceLocation | None = Field(
        default=None,
        description="Provenance assigned by code from slot metadata"
    )


class SlotEntityBatch(BaseModel):
    """
    Response format for a single slot in batched extraction.

    The LLM must return exactly one entry per input slot, with the same slotid.
    """

    slotid: str = Field(description="Must match the input slot_id exactly")
    entities: list[dict[str, Any]] = Field(
        default_factory=list,
        description="0..N entities extracted from this slot"
    )


class SlotBatchResponse(BaseModel):
    """
    Complete response from batched slot extraction.

    The LLM returns one SlotEntityBatch per input slot.
    """

    slots: list[SlotEntityBatch] = Field(
        description="One entry per input slot, in any order"
    )


@dataclass
class EntityIdentity:
    """
    Helper for assigning stable identity to extracted entities.

    Implements the surrogate key generation logic with multiple fallback strategies.
    """

    @staticmethod
    def generate_surrogate(
        slot: ExtractionSlot,
        entity_index: int = 0,
        total_entities: int = 1,
        evidence: EntityEvidence | None = None,
    ) -> str:
        """
        Generate a stable surrogate key for an entity.

        Strategy:
        1. If only one entity in slot: use slot.surrogatekey
        2. If multiple entities:
           a. Prefer charspan-based suffix if available
           b. Fall back to hash of evidence quote
           c. Last resort: use entity index

        Args:
            slot: The extraction slot
            entity_index: Index of this entity within the slot (0-based)
            total_entities: Total number of entities in this slot
            evidence: Optional evidence from LLM

        Returns:
            Stable surrogate key string
        """
        base_key = slot.surrogatekey

        # Single entity: use slot key directly
        if total_entities == 1:
            return base_key

        # Multiple entities: need stable suffix
        suffix = None

        # Strategy 1: Use charspan if available
        if evidence and evidence.local_span:
            start = evidence.local_span.get("start", 0)
            end = evidence.local_span.get("end", 0)
            suffix = f"span{start}-{end}"

        # Strategy 2: Hash evidence quote
        elif evidence and evidence.quote:
            quote_hash = hashlib.sha256(evidence.quote.encode()).hexdigest()[:8]
            suffix = f"hash{quote_hash}"

        # Strategy 3: Use index (least stable, but deterministic)
        else:
            suffix = f"idx{entity_index}"

        return f"{base_key}#{suffix}"

    @staticmethod
    def create_source_location(slot: ExtractionSlot) -> SourceLocation:
        """
        Create a SourceLocation from slot metadata.

        Args:
            slot: The extraction slot

        Returns:
            SourceLocation with provenance information
        """
        # Determine element index based on slot type
        element_index = 0
        if slot.table_index is not None:
            element_index = slot.table_index
        elif slot.block_index is not None:
            element_index = slot.block_index

        return SourceLocation(
            page_no=slot.page_no,
            element_type=slot.element_type,
            element_index=element_index,
            bbox=slot.bbox,
            table_row=slot.row_index,
            table_col=None,  # Not tracked at slot level
            text_snippet=slot.text_snippet,
            charspan=slot.charspan,
        )


def compute_stable_surrogate(
    doc_fingerprint: str,
    page_no: int,
    element_type: str,
    bbox: tuple[float, float, float, float] | None = None,
    indices: dict[str, int] | None = None,
) -> str:
    """
    Compute a stable surrogate key from document and element metadata.

    This function ensures that the same slot in the same document always
    gets the same surrogate key across multiple runs.

    Args:
        doc_fingerprint: Document-level fingerprint
        page_no: Page number
        element_type: Type of element
        bbox: Bounding box if available
        indices: Additional indices (table_idx, row_idx, block_idx, etc.)

    Returns:
        Stable surrogate key string
    """
    # Build components for hashing
    components = [
        doc_fingerprint,
        str(page_no),
        element_type,
    ]

    # Add bbox if available (rounded to avoid floating point issues)
    if bbox:
        bbox_str = ",".join(f"{x:.2f}" for x in bbox)
        components.append(bbox_str)

    # Add indices if provided
    if indices:
        for key in sorted(indices.keys()):
            components.append(f"{key}={indices[key]}")

    # Create hash
    combined = "|".join(components)
    hash_value = hashlib.sha256(combined.encode()).hexdigest()[:16]

    return f"{doc_fingerprint[:8]}_{element_type}_{hash_value}"
