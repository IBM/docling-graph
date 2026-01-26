"""
Identity and provenance management for extracted entities.

This module handles the assignment of stable surrogate keys and source locations
to entities extracted from slots, ensuring traceability and deterministic identity.
"""

import logging
from typing import Any, Type

from pydantic import BaseModel

from .slot_types import (
    EntityEvidence,
    EntityIdentity,
    ExtractionSlot,
    SourceLocation,
)

logger = logging.getLogger(__name__)


class IdentityManager:
    """
    Manages identity assignment and provenance tracking for extracted entities.

    Ensures that:
    1. Every entity gets a stable surrogatekey derived from slot metadata
    2. Every entity has complete sourcelocation provenance
    3. IDs are tracked in a registry to detect duplicates
    """

    def __init__(self) -> None:
        """Initialize the identity manager with an empty registry."""
        self.id_registry: dict[str, list[str]] = {}
        """Registry mapping model names to lists of surrogate keys."""

    def assign_identity(
        self,
        entity: BaseModel,
        slot: ExtractionSlot,
        entity_index: int,
        total_entities: int,
        model_name: str,
    ) -> BaseModel:
        """
        Assign surrogatekey and sourcelocation to an extracted entity.

        This method modifies the entity in-place by setting:
        - surrogatekey: Stable identifier from slot metadata
        - sourcelocation: Provenance information

        Args:
            entity: The extracted entity (must have surrogatekey and sourcelocation fields)
            slot: The extraction slot this entity came from
            entity_index: Index of this entity within the slot (0-based)
            total_entities: Total number of entities extracted from this slot
            model_name: Name of the entity model class (for registry tracking)

        Returns:
            The entity with identity assigned

        Raises:
            ValueError: If entity doesn't have required fields
        """
        # Validate entity has required fields
        if not hasattr(entity, "surrogatekey"):
            raise ValueError(
                f"Entity of type {type(entity).__name__} must have 'surrogatekey' field"
            )
        if not hasattr(entity, "sourcelocation"):
            raise ValueError(
                f"Entity of type {type(entity).__name__} must have 'sourcelocation' field"
            )

        # Extract evidence if available
        evidence = None
        if hasattr(entity, "evidence") and entity.evidence:
            evidence = entity.evidence

        # Generate surrogate key
        surrogatekey = EntityIdentity.generate_surrogate(
            slot=slot,
            entity_index=entity_index,
            total_entities=total_entities,
            evidence=evidence,
        )

        # Create source location
        sourcelocation = EntityIdentity.create_source_location(slot)

        # Assign to entity (using setattr for dynamic attributes)
        entity.surrogatekey = surrogatekey
        entity.sourcelocation = sourcelocation

        # Track in registry
        if model_name not in self.id_registry:
            self.id_registry[model_name] = []

        if surrogatekey in self.id_registry[model_name]:
            logger.warning(
                f"Duplicate surrogate key detected: {surrogatekey} for {model_name}"
            )
        else:
            self.id_registry[model_name].append(surrogatekey)

        logger.debug(
            f"Assigned identity to {model_name}: {surrogatekey} "
            f"(slot: {slot.slot_id}, entity {entity_index + 1}/{total_entities})"
        )

        return entity

    def assign_identities_batch(
        self,
        entities: list[BaseModel],
        slot: ExtractionSlot,
        model_name: str,
    ) -> list[BaseModel]:
        """
        Assign identities to a batch of entities from the same slot.

        This is a convenience method that calls assign_identity for each entity.

        Args:
            entities: List of entities extracted from the slot
            slot: The extraction slot
            model_name: Name of the entity model class

        Returns:
            List of entities with identities assigned
        """
        total_entities = len(entities)

        for idx, entity in enumerate(entities):
            self.assign_identity(
                entity=entity,
                slot=slot,
                entity_index=idx,
                total_entities=total_entities,
                model_name=model_name,
            )

        return entities

    def get_registry_stats(self) -> dict[str, int]:
        """
        Get statistics about the ID registry.

        Returns:
            Dictionary mapping model names to entity counts
        """
        return {model: len(ids) for model, ids in self.id_registry.items()}

    def check_duplicate(self, surrogatekey: str, model_name: str) -> bool:
        """
        Check if a surrogate key already exists in the registry.

        Args:
            surrogatekey: The surrogate key to check
            model_name: The model name

        Returns:
            True if the key exists, False otherwise
        """
        if model_name not in self.id_registry:
            return False
        return surrogatekey in self.id_registry[model_name]

    def clear_registry(self) -> None:
        """Clear the ID registry (useful for testing or reprocessing)."""
        self.id_registry.clear()
        logger.info("ID registry cleared")


def create_entity_with_identity(
    entity_data: dict[str, Any],
    entity_class: Type[BaseModel],
    slot: ExtractionSlot,
    entity_index: int,
    total_entities: int,
) -> BaseModel:
    """
    Create an entity instance with identity assigned.

    This is a helper function that combines entity instantiation with
    identity assignment in a single step.

    Args:
        entity_data: Dictionary of entity field values (from LLM)
        entity_class: The Pydantic model class
        slot: The extraction slot
        entity_index: Index of this entity within the slot
        total_entities: Total entities from this slot

    Returns:
        Entity instance with identity assigned
    """
    # Extract evidence if present
    evidence = None
    if "evidence" in entity_data:
        evidence_data = entity_data["evidence"]
        if evidence_data:
            evidence = EntityEvidence(**evidence_data)

    # Generate surrogate key
    surrogatekey = EntityIdentity.generate_surrogate(
        slot=slot,
        entity_index=entity_index,
        total_entities=total_entities,
        evidence=evidence,
    )

    # Create source location
    sourcelocation = EntityIdentity.create_source_location(slot)

    # Add identity fields to entity data
    entity_data_with_identity = {
        **entity_data,
        "surrogatekey": surrogatekey,
        "sourcelocation": sourcelocation,
    }

    # Create entity instance
    entity = entity_class(**entity_data_with_identity)

    return entity
