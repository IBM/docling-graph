"""
SlotEnumerator: Convert DoclingDocument into extraction slots.

This module implements the pure parser step that enumerates slots from
DoclingDocument structure without any LLM calls.
"""

import hashlib
import logging
from typing import Any, Optional

from docling_core.types.doc import DoclingDocument, TableCell

from .slot_types import ExtractionSlot, compute_stable_surrogate

logger = logging.getLogger(__name__)


class SlotEnumerator:
    """
    Enumerate extraction slots from a DoclingDocument.

    This is a pure parser step that creates stable, traceable slots from
    document structure (tables, figures, text blocks) without LLM involvement.
    """

    def __init__(self, min_text_length: int = 20) -> None:
        """
        Initialize the slot enumerator.

        Args:
            min_text_length: Minimum character length for text blocks (default: 20)
        """
        self.min_text_length = min_text_length

    def enumerate_slots(
        self, doc: DoclingDocument
    ) -> tuple[list[ExtractionSlot], dict[str, str]]:
        """
        Enumerate all extraction slots from a DoclingDocument.

        Args:
            doc: The DoclingDocument to process

        Returns:
            Tuple of (slots, surrogate_to_slot_id_map) where:
            - slots: List of ExtractionSlot objects
            - surrogate_to_slot_id_map: Mapping from surrogate key to slot_id
        """
        logger.info("Starting slot enumeration from DoclingDocument")

        # Compute document fingerprint
        doc_fingerprint = self._compute_doc_fingerprint(doc)
        logger.info(f"Document fingerprint: {doc_fingerprint}")

        # Enumerate slots by type
        slots: list[ExtractionSlot] = []

        # 1. Table row slots
        table_slots = self._enumerate_table_slots(doc, doc_fingerprint)
        slots.extend(table_slots)
        logger.info(f"Enumerated {len(table_slots)} table row slots")

        # 2. Figure slots
        figure_slots = self._enumerate_figure_slots(doc, doc_fingerprint)
        slots.extend(figure_slots)
        logger.info(f"Enumerated {len(figure_slots)} figure slots")

        # 3. Text block slots
        text_slots = self._enumerate_text_slots(doc, doc_fingerprint)
        slots.extend(text_slots)
        logger.info(f"Enumerated {len(text_slots)} text block slots")

        # Build surrogate-to-slot_id mapping
        surrogate_map = {slot.surrogatekey: slot.slot_id for slot in slots}

        logger.info(f"Total slots enumerated: {len(slots)}")
        return slots, surrogate_map

    def _compute_doc_fingerprint(self, doc: DoclingDocument) -> str:
        """
        Compute a stable fingerprint for the document.

        Priority:
        1. Use doc.origin.binaryhash if available
        2. Fall back to hashing early-page text

        Args:
            doc: The DoclingDocument

        Returns:
            Stable document fingerprint (hex string)
        """
        # Try to use binary hash from origin
        if hasattr(doc, "origin") and doc.origin and hasattr(doc.origin, "binary_hash"):
            binary_hash = doc.origin.binary_hash
            if binary_hash:
                # Convert to string if it's an int
                hash_str = str(binary_hash) if isinstance(binary_hash, int) else binary_hash
                return hash_str[:16]  # Use first 16 chars

        # Fallback: hash early-page text
        early_text = []

        # Collect text from first few pages (up to 3)
        for page_no in sorted(doc.pages.keys())[:3]:
            page_md = doc.export_to_markdown(page_no=page_no)
            early_text.append(page_md[:500])  # First 500 chars per page

        combined_text = "\n".join(early_text)
        hash_value = hashlib.sha256(combined_text.encode()).hexdigest()

        return hash_value[:16]

    def _enumerate_table_slots(
        self, doc: DoclingDocument, fingerprint: str
    ) -> list[ExtractionSlot]:
        """
        Enumerate table row slots from document tables.

        Creates one slot per data row (skips header rows).

        Args:
            doc: The DoclingDocument
            fingerprint: Document fingerprint

        Returns:
            List of table row slots
        """
        slots: list[ExtractionSlot] = []

        if not hasattr(doc, "tables") or not doc.tables:
            return slots

        for table_idx, table in enumerate(doc.tables):
            # Get table provenance
            page_no = 1  # Default
            bbox = None

            if table.prov and len(table.prov) > 0:
                page_no = table.prov[0].page_no
                if hasattr(table.prov[0], "bbox") and table.prov[0].bbox:
                    bbox_obj = table.prov[0].bbox
                    bbox = (bbox_obj.l, bbox_obj.t, bbox_obj.r, bbox_obj.b)

            # Process table data
            if not hasattr(table, "data") or not table.data:
                continue

            # Identify header rows (cells with column_header flag)
            header_rows = set()
            for cell in table.data.table_cells:
                if hasattr(cell, "column_header") and cell.column_header:
                    header_rows.add(cell.start_row_offset_idx)

            # Group cells by row
            rows_data: dict[int, list[TableCell]] = {}
            for cell in table.data.table_cells:
                row_idx = cell.start_row_offset_idx
                if row_idx not in rows_data:
                    rows_data[row_idx] = []
                rows_data[row_idx].append(cell)

            # Create slots for data rows (skip headers)
            for row_idx in sorted(rows_data.keys()):
                if row_idx in header_rows:
                    continue  # Skip header rows

                cells = rows_data[row_idx]

                # Build row content (join cell texts)
                cell_texts = []
                for cell in sorted(cells, key=lambda c: c.start_col_offset_idx):
                    if hasattr(cell, "text") and cell.text:
                        cell_texts.append(cell.text.strip())

                content = " | ".join(cell_texts)

                if not content:
                    continue  # Skip empty rows

                # Generate slot ID and surrogate key
                slot_id = f"table{table_idx}row{row_idx}"

                surrogatekey = compute_stable_surrogate(
                    doc_fingerprint=fingerprint,
                    page_no=page_no,
                    element_type="table_row",
                    bbox=bbox,
                    indices={"table_idx": table_idx, "row_idx": row_idx},
                )

                # Create slot
                slot = ExtractionSlot(
                    slot_id=slot_id,
                    surrogatekey=surrogatekey,
                    element_type="table_row",
                    page_no=page_no,
                    bbox=bbox,
                    table_index=table_idx,
                    row_index=row_idx,
                    content=content,
                    text_snippet=content[:100],
                )

                slots.append(slot)

        return slots

    def _enumerate_figure_slots(
        self, doc: DoclingDocument, fingerprint: str
    ) -> list[ExtractionSlot]:
        """
        Enumerate figure slots from document pictures.

        Creates one slot per figure with resolved caption.

        Args:
            doc: The DoclingDocument
            fingerprint: Document fingerprint

        Returns:
            List of figure slots
        """
        slots: list[ExtractionSlot] = []

        if not hasattr(doc, "pictures") or not doc.pictures:
            return slots

        for fig_idx, figure in enumerate(doc.pictures):
            # Get figure provenance
            page_no = 1  # Default
            bbox = None

            if figure.prov and len(figure.prov) > 0:
                page_no = figure.prov[0].page_no
                if hasattr(figure.prov[0], "bbox") and figure.prov[0].bbox:
                    bbox_obj = figure.prov[0].bbox
                    bbox = (bbox_obj.l, bbox_obj.t, bbox_obj.r, bbox_obj.b)

            # Resolve caption text
            caption_text = ""
            if hasattr(figure, "captions") and figure.captions:
                # Captions are references to text items
                caption_texts: list[str] = []
                for caption_ref in figure.captions:
                    # Try to resolve caption from doc.texts
                    if hasattr(doc, "texts") and doc.texts:
                        for text_item in doc.texts:
                            # Check if self_ref matches caption_ref (use object equality)
                            # Note: self_ref is str, caption_ref is RefItem, but they can be compared
                            if hasattr(text_item, "self_ref") and text_item.self_ref == caption_ref:  # type: ignore[comparison-overlap]
                                if hasattr(text_item, "text") and text_item.text:
                                    caption_texts.append(text_item.text)
                                break

                caption_text = " ".join(caption_texts)

            # Use caption as content (or placeholder if no caption)
            content = caption_text if caption_text else f"[Figure {fig_idx}]"

            # Generate slot ID and surrogate key
            slot_id = f"figure{fig_idx}"

            surrogatekey = compute_stable_surrogate(
                doc_fingerprint=fingerprint,
                page_no=page_no,
                element_type="figure",
                bbox=bbox,
                indices={"fig_idx": fig_idx},
            )

            # Create slot
            slot = ExtractionSlot(
                slot_id=slot_id,
                surrogatekey=surrogatekey,
                element_type="figure",
                page_no=page_no,
                bbox=bbox,
                content=content,
                text_snippet=content[:100],
            )

            slots.append(slot)

        return slots

    def _enumerate_text_slots(
        self, doc: DoclingDocument, fingerprint: str
    ) -> list[ExtractionSlot]:
        """
        Enumerate text block slots from document texts.

        Groups text items by page and creates slots for each block.

        Args:
            doc: The DoclingDocument
            fingerprint: Document fingerprint

        Returns:
            List of text block slots
        """
        slots: list[ExtractionSlot] = []

        if not hasattr(doc, "texts") or not doc.texts:
            return slots

        # Group text items by page
        page_texts: dict[int, list[tuple[int, Any]]] = {}

        for text_idx, text_item in enumerate(doc.texts):
            if not hasattr(text_item, "text") or not text_item.text:
                continue

            # Get page number
            page_no = 1  # Default
            if text_item.prov and len(text_item.prov) > 0:
                page_no = text_item.prov[0].page_no

            if page_no not in page_texts:
                page_texts[page_no] = []

            page_texts[page_no].append((text_idx, text_item))

        # Create slots for each page's text blocks
        for page_no in sorted(page_texts.keys()):
            texts = page_texts[page_no]

            for block_idx, (text_idx, text_item) in enumerate(texts):
                content = text_item.text.strip()

                # Skip very short blocks
                if len(content) < self.min_text_length:
                    continue

                # Get bbox and charspan if available
                bbox = None
                charspan = None

                if text_item.prov and len(text_item.prov) > 0:
                    prov = text_item.prov[0]

                    if hasattr(prov, "bbox") and prov.bbox:
                        bbox_obj = prov.bbox
                        bbox = (bbox_obj.l, bbox_obj.t, bbox_obj.r, bbox_obj.b)

                    if hasattr(prov, "charspan") and prov.charspan:
                        charspan = (prov.charspan.start, prov.charspan.end)

                # Generate slot ID and surrogate key
                slot_id = f"textp{page_no}b{block_idx}"

                surrogatekey = compute_stable_surrogate(
                    doc_fingerprint=fingerprint,
                    page_no=page_no,
                    element_type="text_block",
                    bbox=bbox,
                    indices={"block_idx": block_idx},
                )

                # Create slot
                slot = ExtractionSlot(
                    slot_id=slot_id,
                    surrogatekey=surrogatekey,
                    element_type="text_block",
                    page_no=page_no,
                    bbox=bbox,
                    block_index=block_idx,
                    charspan=charspan,
                    content=content,
                    text_snippet=content[:100],
                )

                slots.append(slot)

        return slots
