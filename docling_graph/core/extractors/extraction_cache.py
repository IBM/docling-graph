"""
In-memory extraction cache for slot-based extraction.

This module provides a run-scoped, per-slot cache that enables partial cache hits
even when batch composition changes. The cache is explicitly cleared at end-of-document.
"""

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class CacheKey:
    """
    Complete cache key for slot extraction results.

    Includes everything that changes output to ensure correctness.
    """

    doc_fingerprint: str
    """Document fingerprint (from SlotEnumerator)."""

    slot_id: str
    """Slot identifier (e.g., 'table0row5', 'figure2')."""

    entity_type: str
    """Entity type name (e.g., 'Material', 'Component')."""

    prompt_version: str
    """Prompt version identifier (e.g., 'v1', 'v2')."""

    model_id: str
    """Full model identifier (e.g., 'ollama/llama3.2:3b')."""

    schema_hash: str
    """Hash of the entity schema (schema changes = new extractor)."""

    def to_string(self) -> str:
        """Convert to string key for dict lookup."""
        return (
            f"{self.doc_fingerprint}|{self.slot_id}|{self.entity_type}|"
            f"{self.prompt_version}|{self.model_id}|{self.schema_hash}"
        )

    @staticmethod
    def compute_schema_hash(schema: dict | str) -> str:
        """
        Compute a stable hash of the entity schema.

        Args:
            schema: Schema as dict or JSON string

        Returns:
            Hex hash string (first 16 chars)
        """
        if isinstance(schema, dict):
            schema_str = json.dumps(schema, sort_keys=True)
        else:
            schema_str = schema

        return hashlib.sha256(schema_str.encode()).hexdigest()[:16]


@dataclass
class CacheEntry:
    """
    Cached extraction result for a single slot.

    Stores the raw entities list (before identity assignment) to enable
    reuse across different extraction runs.
    """

    entities: list[dict[str, Any]]
    """List of entity dictionaries (raw LLM output)."""

    hit_count: int = 0
    """Number of times this entry was retrieved."""


class ExtractionCache:
    """
    In-memory cache for slot extraction results.

    This is a run-scoped cache that:
    - Stores results per slot (not per batch) for partial hit support
    - Uses complete cache keys to ensure correctness
    - Tracks hit/miss statistics
    - Can be explicitly cleared at end-of-document
    """

    def __init__(self) -> None:
        """Initialize empty cache."""
        self._cache: dict[str, CacheEntry] = {}
        self._hits = 0
        self._misses = 0
        self._stores = 0

        logger.info("ExtractionCache initialized (in-memory, run-scoped)")

    def get(
        self,
        doc_fingerprint: str,
        slot_id: str,
        entity_type: str,
        prompt_version: str,
        model_id: str,
        schema_hash: str,
    ) -> list[dict[str, Any]] | None:
        """
        Retrieve cached entities for a slot.

        Args:
            doc_fingerprint: Document fingerprint
            slot_id: Slot identifier
            entity_type: Entity type name
            prompt_version: Prompt version
            model_id: Model identifier
            schema_hash: Schema hash

        Returns:
            List of entity dictionaries if cached, None otherwise
        """
        key = CacheKey(
            doc_fingerprint=doc_fingerprint,
            slot_id=slot_id,
            entity_type=entity_type,
            prompt_version=prompt_version,
            model_id=model_id,
            schema_hash=schema_hash,
        )

        key_str = key.to_string()

        if key_str in self._cache:
            entry = self._cache[key_str]
            entry.hit_count += 1
            self._hits += 1

            logger.debug(
                f"Cache HIT: {slot_id} for {entity_type} "
                f"(hit #{entry.hit_count})"
            )

            return entry.entities
        else:
            self._misses += 1
            logger.debug(f"Cache MISS: {slot_id} for {entity_type}")
            return None

    def set(
        self,
        doc_fingerprint: str,
        slot_id: str,
        entity_type: str,
        prompt_version: str,
        model_id: str,
        schema_hash: str,
        entities: list[dict[str, Any]],
    ) -> None:
        """
        Store entities for a slot in cache.

        Args:
            doc_fingerprint: Document fingerprint
            slot_id: Slot identifier
            entity_type: Entity type name
            prompt_version: Prompt version
            model_id: Model identifier
            schema_hash: Schema hash
            entities: List of entity dictionaries to cache
        """
        key = CacheKey(
            doc_fingerprint=doc_fingerprint,
            slot_id=slot_id,
            entity_type=entity_type,
            prompt_version=prompt_version,
            model_id=model_id,
            schema_hash=schema_hash,
        )

        key_str = key.to_string()

        self._cache[key_str] = CacheEntry(entities=entities)
        self._stores += 1

        logger.debug(f"Cache STORE: {slot_id} for {entity_type}")

    def clear(self) -> None:
        """
        Clear all cached entries.

        Should be called at end-of-document to free memory.
        """
        entries_cleared = len(self._cache)
        self._cache.clear()

        logger.info(
            f"Cache cleared: {entries_cleared} entries removed "
            f"(hits: {self._hits}, misses: {self._misses}, stores: {self._stores})"
        )

        # Reset statistics
        self._hits = 0
        self._misses = 0
        self._stores = 0

    def get_stats(self) -> dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        total_requests = self._hits + self._misses
        hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0.0

        return {
            "entries": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "stores": self._stores,
            "hit_rate_percent": round(hit_rate, 2),
            "total_requests": total_requests,
        }

    def __len__(self) -> int:
        """Return number of cached entries."""
        return len(self._cache)

    def __contains__(self, key_str: str) -> bool:
        """Check if a key exists in cache."""
        return key_str in self._cache

# Made with Bob
