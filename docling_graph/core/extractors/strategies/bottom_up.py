"""
Bottom-up extraction strategy for small LLMs.

This strategy extracts knowledge graphs progressively (leaves → root) by:
1. Enumerating document slots (tables, figures, text blocks)
2. Analyzing schema hierarchy to determine extraction order
3. Extracting entities level-by-level with focused prompts
4. Assembling the final graph from extracted entities + references
"""

import hashlib
import json
import logging
import time
from typing import Any, List, Tuple, Type, cast

from docling_core.types.doc import DoclingDocument
from pydantic import BaseModel
from pydantic.fields import FieldInfo
from rich import print as rich_print

from ....llm_clients.base import BaseLlmClient
from ....protocols import Backend, is_llm_backend
from ...utils.dict_merger import merge_pydantic_models
from ..document_processor import DocumentProcessor
from ..extraction_cache import ExtractionCache
from ..extraction_metrics import ExtractionMetrics, StageMetrics
from ..extractor_base import BaseExtractor
from ..identity_manager import IdentityManager, create_entity_with_identity
from ..rate_limiter import ParallelExecutor, RateLimiter, RetryConfig, RetryHandler
from ..slot_enumerator import SlotEnumerator
from ..slot_types import ExtractionSlot, SlotBatchResponse
from ..token_budgeter import TokenBudgeter

logger = logging.getLogger(__name__)


class HierarchyAnalyzer:
    """
    Analyzes Pydantic schema hierarchy to determine extraction order.

    Identifies entity types and their dependencies to enable bottom-up
    extraction (leaves first, then parents that reference them).
    """

    def __init__(self, root_model: Type[BaseModel]) -> None:
        """
        Initialize hierarchy analyzer.

        Args:
            root_model: The root Pydantic model to analyze
        """
        self.root_model = root_model
        self.entity_types: dict[str, Type[BaseModel]] = {}
        self.dependencies: dict[str, set[str]] = {}

        self._analyze_schema()

    def _analyze_schema(self) -> None:
        """Analyze the schema to identify entity types and dependencies."""
        # Start with root model
        self._analyze_model(self.root_model, self.root_model.__name__)

    def _analyze_model(self, model: Type[BaseModel], model_name: str) -> None:
        """
        Recursively analyze a model to find entity types.

        Args:
            model: The Pydantic model to analyze
            model_name: Name of the model
        """
        if model_name in self.entity_types:
            return  # Already analyzed

        self.entity_types[model_name] = model
        self.dependencies[model_name] = set()

        # Analyze fields
        for _field_name, field_info in model.model_fields.items():
            field_type = field_info.annotation

            # Handle Optional, List, etc.
            origin = getattr(field_type, "__origin__", None)

            if origin is list:
                # List[SomeModel]
                args = getattr(field_type, "__args__", ())
                if args and isinstance(args[0], type) and issubclass(args[0], BaseModel):
                    referenced_model = args[0]
                    referenced_name = referenced_model.__name__

                    # Check if this is an edge (reference) or nested entity
                    is_edge = self._is_edge_field(field_info)

                    if is_edge:
                        # This is a reference to another entity
                        self.dependencies[model_name].add(referenced_name)

                    # Recursively analyze referenced model
                    self._analyze_model(referenced_model, referenced_name)

            elif isinstance(field_type, type) and issubclass(field_type, BaseModel):
                # Direct model reference
                referenced_name = field_type.__name__

                is_edge = self._is_edge_field(field_info)

                if is_edge:
                    self.dependencies[model_name].add(referenced_name)

                self._analyze_model(field_type, referenced_name)

    def _is_edge_field(self, field_info: FieldInfo) -> bool:
        """
        Check if a field represents an edge (reference) vs nested entity.

        Args:
            field_info: Pydantic field info

        Returns:
            True if this is an edge field
        """
        # Check for edge_label in json_schema_extra
        if field_info.json_schema_extra:
            if isinstance(field_info.json_schema_extra, dict):
                return "edge_label" in field_info.json_schema_extra

        return False

    def get_extraction_levels(self) -> List[List[str]]:
        """
        Get entity types grouped by extraction level (bottom-up).

        Returns:
            List of levels, where each level is a list of entity type names.
            Level 0 contains leaf entities (no dependencies).
            Higher levels contain entities that depend on lower levels.
        """
        levels: List[List[str]] = []
        remaining = set(self.entity_types.keys())
        extracted: set[str] = set()

        while remaining:
            # Find entities whose dependencies are all extracted
            current_level = []
            for entity_name in remaining:
                deps = self.dependencies.get(entity_name, set())
                if deps.issubset(extracted):
                    current_level.append(entity_name)

            if not current_level:
                # Circular dependency or error - extract remaining
                logger.warning(
                    f"Circular dependency detected. Remaining entities: {remaining}"
                )
                current_level = list(remaining)

            levels.append(current_level)
            extracted.update(current_level)
            remaining -= set(current_level)

        return levels


class BottomUpStrategy(BaseExtractor):
    """
    Bottom-up extraction strategy optimized for small LLMs.

    Extracts knowledge graphs progressively by:
    1. Enumerating document slots from DoclingDocument structure
    2. Analyzing schema hierarchy
    3. Extracting entities level-by-level (leaves → root)
    4. Assembling graph from extracted entities
    """

    def __init__(
        self,
        backend: Backend,
        docling_config: str = "ocr",
        min_text_length: int = 20,
        enable_caching: bool = False,
        enable_rate_limiting: bool = True,
        max_retries: int = 3,
        enable_metrics: bool = True,
    ) -> None:
        """
        Initialize bottom-up extraction strategy.

        Args:
            backend: LLM backend (must be LLM, not VLM)
            docling_config: Docling pipeline configuration
            min_text_length: Minimum text length for text slots
            enable_caching: Enable extraction caching
            enable_rate_limiting: Enable rate limiting based on provider limits
            max_retries: Maximum number of retries for failed API calls
            enable_metrics: Enable metrics collection
        """
        super().__init__()

        if not is_llm_backend(backend):
            raise ValueError(
                "BottomUpStrategy requires an LLM backend. "
                "VLM backends are not supported for slot-based extraction."
            )

        self.backend = backend
        self.doc_processor = DocumentProcessor(docling_config=docling_config)
        self.slot_enumerator = SlotEnumerator(min_text_length=min_text_length)
        self.identity_manager = IdentityManager()
        self.enable_caching = enable_caching
        self.enable_metrics = enable_metrics

        # Initialize cache if enabled
        self.cache: ExtractionCache | None = None
        if enable_caching:
            self.cache = ExtractionCache()
            rich_print("[blue][BottomUpStrategy][/blue] Caching enabled")

        # Get LLM client for token budgeting and rate limiting
        if hasattr(backend, "client") and isinstance(backend.client, BaseLlmClient):
            self.llm_client = backend.client
            self.token_budgeter = TokenBudgeter(
                provider=self.llm_client.provider,
                model=self.llm_client.model,
            )

            # Initialize rate limiter from provider config
            rate_limit_rpm = None
            if enable_rate_limiting:
                try:
                    from ....llm_clients.config import get_provider_config
                    provider_config = get_provider_config(self.llm_client.provider)
                    if provider_config:
                        rate_limit_rpm = provider_config.rate_limit_rpm
                except Exception as e:
                    logger.warning(f"Could not load rate limit config: {e}")

            self.rate_limiter = RateLimiter(rpm=rate_limit_rpm)
            self.retry_handler = RetryHandler(
                config=RetryConfig(max_retries=max_retries)
            )

            if rate_limit_rpm:
                rich_print(
                    f"[blue][BottomUpStrategy][/blue] Rate limiting enabled: "
                    f"{rate_limit_rpm} RPM"
                )
        else:
            raise ValueError("Backend must have a BaseLlmClient instance")

        # Prompt version for cache keys
        self.prompt_version = "v1"

        # Metrics tracking
        self.metrics: ExtractionMetrics | None = None

        rich_print(
            f"[blue][BottomUpStrategy][/blue] Initialized with LLM backend: "
            f"[cyan]{self.backend.__class__.__name__}[/cyan]"
        )

    def extract(
        self, source: str, template: Type[BaseModel]
    ) -> Tuple[List[BaseModel], DoclingDocument | None]:
        """
        Extract structured data using bottom-up strategy.

        Args:
            source: Path to the source document
            template: Pydantic model template (root model)

        Returns:
            Tuple of (extracted_models, docling_document)
        """
        try:
            rich_print("[blue][BottomUpStrategy][/blue] Starting bottom-up extraction")

            # Initialize metrics if enabled
            doc_fingerprint = "unknown"
            if self.enable_metrics:
                # Get doc fingerprint from first slot (will be set during enumeration)
                doc_fingerprint = "temp"
                self.metrics = ExtractionMetrics(doc_fingerprint=doc_fingerprint)

            # Step 1: Convert to DoclingDocument
            rich_print("[blue][BottomUpStrategy][/blue] Converting document...")
            conv_start = time.time()
            document = self.doc_processor.convert_to_docling_doc(source)
            if self.metrics:
                self.metrics.conversion_time = time.time() - conv_start

            # Step 2: Enumerate slots
            rich_print("[blue][BottomUpStrategy][/blue] Enumerating extraction slots...")
            enum_start = time.time()
            slots, _surrogate_map = self.slot_enumerator.enumerate_slots(document)
            if self.metrics:
                self.metrics.enumeration_time = time.time() - enum_start
                self.metrics.total_slots = len(slots)
                # Update doc fingerprint from actual slots
                if slots:
                    self.metrics.doc_fingerprint = slots[0].surrogatekey.split(":")[0]

            rich_print(
                f"[blue][BottomUpStrategy][/blue] Enumerated {len(slots)} slots "
                f"({len([s for s in slots if s.element_type == 'table_row'])} table rows, "
                f"{len([s for s in slots if s.element_type == 'figure'])} figures, "
                f"{len([s for s in slots if s.element_type == 'text_block'])} text blocks)"
            )

            # Step 3: Analyze hierarchy
            rich_print("[blue][BottomUpStrategy][/blue] Analyzing schema hierarchy...")
            hier_start = time.time()
            analyzer = HierarchyAnalyzer(template)
            levels = analyzer.get_extraction_levels()
            if self.metrics:
                self.metrics.hierarchy_analysis_time = time.time() - hier_start

            rich_print(
                f"[blue][BottomUpStrategy][/blue] Identified {len(levels)} extraction levels: "
                f"{[len(level) for level in levels]} entities per level"
            )

            # Step 4: Extract entities level-by-level
            all_entities: dict[str, List[BaseModel]] = {}

            for level_idx, entity_types in enumerate(levels):
                rich_print(
                    f"[blue][BottomUpStrategy][/blue] Processing level {level_idx + 1}/{len(levels)}: "
                    f"{entity_types}"
                )

                for entity_type_name in entity_types:
                    entity_class = analyzer.entity_types[entity_type_name]

                    rich_print(
                        f"[blue][BottomUpStrategy][/blue] Extracting {entity_type_name}..."
                    )

                    # Start stage metrics
                    stage_metrics = None
                    if self.metrics:
                        stage_metrics = self.metrics.start_stage(entity_type_name)
                        stage_metrics.slots_total = len(slots)

                    entities = self._extract_entity_type(
                        entity_class=entity_class,
                        entity_type_name=entity_type_name,
                        slots=slots,
                        stage_metrics=stage_metrics,
                    )

                    all_entities[entity_type_name] = entities

                    # Complete stage metrics
                    if stage_metrics:
                        stage_metrics.entities_extracted = len(entities)
                        stage_metrics.mark_complete()

                    rich_print(
                        f"[green][BottomUpStrategy][/green] Extracted {len(entities)} "
                        f"{entity_type_name} entities"
                    )

            # Step 5: Assemble final model
            rich_print("[blue][BottomUpStrategy][/blue] Assembling final model...")
            asm_start = time.time()
            final_model, unassigned = self._assemble_model(template, all_entities)
            if self.metrics:
                self.metrics.assembly_time = time.time() - asm_start

                # Update stage metrics with assignment info
                for entity_type, entities in all_entities.items():
                    stage = self.metrics.get_stage(entity_type)
                    if stage:
                        if entity_type in unassigned:
                            stage.entities_unassigned = unassigned[entity_type]
                            stage.entities_assigned = len(entities) - unassigned[entity_type]
                        else:
                            stage.entities_assigned = len(entities)

                self.metrics.mark_complete()

            # Report unassigned entities (gap visibility)
            if unassigned:
                rich_print(
                    f"[yellow][BottomUpStrategy][/yellow] Warning: {len(unassigned)} "
                    f"entity types not assigned to root model fields:"
                )
                for entity_type, count in unassigned.items():
                    rich_print(f"  • {entity_type}: {count} entities")

            if final_model:
                rich_print(
                    "[green][BottomUpStrategy][/green] Successfully assembled final model"
                )

                # Clear cache at end of document
                if self.cache:
                    stats = self.cache.get_stats()
                    rich_print(
                        f"[blue][BottomUpStrategy][/blue] Cache stats: "
                        f"{stats['hit_rate_percent']}% hit rate "
                        f"({stats['hits']}/{stats['total_requests']} requests)"
                    )
                    self.cache.clear()

                # Print metrics summary
                if self.metrics:
                    self.metrics.print_summary()

                return [final_model], document
            else:
                rich_print(
                    "[yellow][BottomUpStrategy][/yellow] Failed to assemble final model"
                )

                # Still print metrics even on failure
                if self.metrics:
                    self.metrics.mark_complete()
                    self.metrics.print_summary()

                # Clear cache even on failure
                if self.cache:
                    self.cache.clear()

                return [], document

        except Exception as e:
            rich_print(f"[red][BottomUpStrategy][/red] Extraction error: {e}")
            logger.exception("Bottom-up extraction failed")

            # Clear cache on error
            if self.cache:
                self.cache.clear()

            return [], None

    def _extract_entity_type(
        self,
        entity_class: Type[BaseModel],
        entity_type_name: str,
        slots: List[ExtractionSlot],
        stage_metrics: StageMetrics | None = None,
    ) -> List[BaseModel]:
        """
        Extract all entities of a specific type from slots.

        Args:
            entity_class: The Pydantic model class for this entity type
            entity_type_name: Name of the entity type
            slots: All available extraction slots
            stage_metrics: Optional metrics tracker for this stage

        Returns:
            List of extracted entities with identity assigned
        """
        # Get entity schema
        schema = entity_class.model_json_schema()
        schema_json = json.dumps(schema)

        # Compute schema hash for cache key
        schema_hash = hashlib.sha256(schema_json.encode()).hexdigest()[:16]

        # Get doc fingerprint from first slot (all slots share same doc)
        doc_fingerprint = slots[0].surrogatekey.split(":")[0] if slots else "unknown"

        # Get model ID from LLM client
        model_id = getattr(self.llm_client, "model_name", "unknown")

        # Prompt version (increment when prompt logic changes)
        prompt_version = "v1"

        # Separate cached and uncached slots
        cached_entities: List[BaseModel] = []
        uncached_slots: List[ExtractionSlot] = []

        if self.enable_caching and self.cache:
            for slot in slots:
                cached_result = self.cache.get(
                    doc_fingerprint=doc_fingerprint,
                    slot_id=slot.slot_id,
                    entity_type=entity_type_name,
                    prompt_version=prompt_version,
                    model_id=model_id,
                    schema_hash=schema_hash,
                )

                if cached_result is not None:
                    # Use cached entities (convert from dict to BaseModel)
                    for entity_dict in cached_result:
                        # Reconstruct entity from dict
                        entity = entity_class(**entity_dict)
                        cached_entities.append(entity)
                else:
                    # Need to extract this slot
                    uncached_slots.append(slot)

            if cached_entities:
                rich_print(
                    f"[cyan][BottomUpStrategy][/cyan] Cache hit: {len(cached_entities)} entities "
                    f"from {len(slots) - len(uncached_slots)} slots"
                )

            # Update metrics
            if stage_metrics:
                stage_metrics.slots_cached = len(slots) - len(uncached_slots)
                stage_metrics.slots_extracted = len(uncached_slots)
        else:
            uncached_slots = slots
            if stage_metrics:
                stage_metrics.slots_extracted = len(slots)

        # Create batches for uncached slots
        if uncached_slots:
            slot_dicts = [
                {"slot_id": slot.slot_id, "content": slot.content}
                for slot in uncached_slots
            ]

            batches = self.token_budgeter.create_batches(slot_dicts, schema_json)

            # Update metrics
            if stage_metrics:
                stage_metrics.batches_processed = len(batches)

            rich_print(
                f"[blue][BottomUpStrategy][/blue] Processing {len(batches)} batches "
                f"for {entity_type_name} ({len(uncached_slots)} uncached slots)"
            )

            for batch_idx, batch in enumerate(batches):
                rich_print(
                    f"[blue][BottomUpStrategy][/blue] Batch {batch_idx + 1}/{len(batches)} "
                    f"({len(batch)} slots)"
                )

                try:
                    # Extract from batch
                    start_time = time.time()
                    response = self.llm_client.extract_from_slots_batch(
                        slots=batch,
                        entity_schema=schema_json,
                        entity_type_name=entity_type_name,
                    )
                    extraction_time = time.time() - start_time

                    # Update metrics
                    if stage_metrics:
                        stage_metrics.api_calls += 1
                        # TODO: Extract token counts from response metadata if available

                    # Parse response
                    batch_response = SlotBatchResponse(**response)

                    # Process each slot's entities
                    for slot_result in batch_response.slots:
                        # Find the corresponding slot
                        found_slot = next(
                            (s for s in uncached_slots if s.slot_id == slot_result.slotid),
                            None
                        )

                        if not found_slot:
                            logger.warning(
                                f"Slot {slot_result.slotid} not found in slot list"
                            )
                            continue

                        # Create entities with identity
                        slot_entities: List[BaseModel] = []
                        for entity_idx, entity_data in enumerate(slot_result.entities):
                            try:
                                entity = create_entity_with_identity(
                                    entity_data=entity_data,
                                    entity_class=entity_class,
                                    slot=found_slot,
                                    entity_index=entity_idx,
                                    total_entities=len(slot_result.entities),
                                )
                                slot_entities.append(entity)
                                cached_entities.append(entity)
                            except Exception as e:
                                logger.error(
                                    f"Failed to create entity from slot {found_slot.slot_id}: {e}"
                                )

                        # Cache the slot's entities (convert to dicts)
                        if self.enable_caching and self.cache:
                            entities_as_dicts = [
                                entity.model_dump() if hasattr(entity, "model_dump") else entity.dict()
                                for entity in slot_entities
                            ]
                            self.cache.set(
                                doc_fingerprint=doc_fingerprint,
                                slot_id=found_slot.slot_id,
                                entity_type=entity_type_name,
                                prompt_version=prompt_version,
                                model_id=model_id,
                                schema_hash=schema_hash,
                                entities=entities_as_dicts,
                            )

                    rich_print(
                        f"[green][BottomUpStrategy][/green] Batch {batch_idx + 1} complete: "
                        f"{len(batch_response.slots)} slots processed in {extraction_time:.2f}s"
                    )

                except Exception as e:
                    logger.error(f"Batch {batch_idx + 1} extraction failed: {e}")
                    rich_print(
                        f"[red][BottomUpStrategy][/red] Batch {batch_idx + 1} failed: {e}"
                    )

                    # Update metrics
                    if stage_metrics:
                        stage_metrics.api_failures += 1

        return cached_entities

    def _assemble_model(
        self,
        template: Type[BaseModel],
        all_entities: dict[str, List[BaseModel]],
    ) -> tuple[BaseModel | None, dict[str, int]]:
        """
        Assemble the final root model from extracted entities.

        This creates an instance of the root template by populating it with
        the extracted entities organized by type. Also tracks unassigned entities
        for gap visibility.

        Args:
            template: The root Pydantic model class
            all_entities: Dictionary mapping entity type names to lists of entities

        Returns:
            Tuple of (assembled_model, unassigned_entities) where:
            - assembled_model: Root model instance or None if assembly fails
            - unassigned_entities: Dict mapping entity type names to counts of unassigned entities
        """
        try:
            # Build data dictionary for root model
            model_data: dict[str, Any] = {}
            assigned_types: set[str] = set()

            # Map entity types to fields in the root model
            for field_name, field_info in template.model_fields.items():
                field_type = field_info.annotation

                # Handle List[EntityType]
                origin = getattr(field_type, "__origin__", None)
                if origin is list:
                    args = getattr(field_type, "__args__", ())
                    if args and isinstance(args[0], type) and issubclass(args[0], BaseModel):
                        entity_type_name = args[0].__name__
                        if entity_type_name in all_entities:
                            model_data[field_name] = all_entities[entity_type_name]
                            assigned_types.add(entity_type_name)
                        else:
                            model_data[field_name] = []

                # Handle direct EntityType
                elif isinstance(field_type, type) and issubclass(field_type, BaseModel):
                    entity_type_name = field_type.__name__
                    if all_entities.get(entity_type_name):
                        # Take first entity
                        model_data[field_name] = all_entities[entity_type_name][0]
                        assigned_types.add(entity_type_name)

            # Track unassigned entities (gap visibility)
            unassigned: dict[str, int] = {}
            for entity_type, entities in all_entities.items():
                if entity_type not in assigned_types and entities:
                    unassigned[entity_type] = len(entities)

            # Create root model instance
            root_model = template(**model_data)

            return root_model, unassigned

        except Exception as e:
            logger.error(f"Failed to assemble root model: {e}")
            return None, {}
