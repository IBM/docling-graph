"""
Many-to-one extraction strategy.
Processes entire document and returns single consolidated model.
"""

import json
import time
from typing import Any, Dict, Iterable, List, Tuple, Type, cast

from docling_core.types.doc import DoclingDocument
from pydantic import BaseModel, ValidationError
from rich import print as rich_print

from ....llm_clients.prompts import get_context_aware_prompt, get_extraction_prompt
from ....protocols import (
    Backend,
    ExtractionBackendProtocol,
    TextExtractionBackendProtocol,
    get_backend_type,
    is_llm_backend,
    is_vlm_backend,
)
from ...utils.dict_merger import deep_merge_dicts, merge_pydantic_models
from ..delta_models import DeltaOperation
from ..document_processor import DocumentProcessor
from ..extractor_base import BaseExtractor


class ManyToOneStrategy(BaseExtractor):
    """Many-to-one extraction strategy.

    Extracts one consolidated model from an entire document
    using Protocol-based backend type checking (VLM or LLM).
    """

    def __init__(
        self,
        backend: Backend,
        docling_config: str = "default",
        use_chunking: bool = True,
        llm_consolidation: bool = False,
        chunker_config: dict | None = None,
    ) -> None:
        """
        Initialize the extraction strategy with a backend and document processor.

        Args:
            backend: Extraction backend (VLM or LLM)
            docling_config: Docling pipeline config ("ocr" or "vision")
            llm_consolidation: If True, run a final LLM pass to merge results.
            use_chunking: Use structure-aware chunking instead of page-by-page (default: True)
            chunker_config: Configuration for HybridChunker. Example:
                {
                    "tokenizer_name": "mistralai/Mistral-7B-v0.1",
                    "max_tokens": 8000,
                    "merge_peers": True
                }
                If None and use_chunking=True, uses default tokenizer with backend's context limit.
        """
        super().__init__()  # Initialize base extractor with trace_data attribute
        self.backend = backend
        self.llm_consolidation = llm_consolidation
        self.use_chunking = use_chunking

        # Cache protocol checks (optimization: avoid repeated isinstance checks)
        self._is_llm = is_llm_backend(self.backend)
        self._is_vlm = is_vlm_backend(self.backend)
        self._backend_type = get_backend_type(self.backend)

        # Auto-configure chunker based on backend if not provided
        # Note: schema_size will be set dynamically in extract() method
        if use_chunking and chunker_config is None:
            # Provide minimal config - will be updated with schema_size later
            if hasattr(backend, "client"):
                provider = getattr(backend.client, "provider", None)
                model_id = getattr(backend.client, "model_id", None)
                model_config = getattr(backend.client, "model_config", None)
                if provider:
                    chunker_config = {
                        "provider": provider,
                        "model": model_id,
                        "model_config": model_config,
                    }
                else:
                    # Fallback: use context limit if available
                    context_limit = (
                        getattr(model_config, "context_limit", None)
                        if model_config is not None
                        else getattr(backend.client, "context_limit", 8000)
                    )
                    if not isinstance(context_limit, int):
                        context_limit = 8000
                    max_tokens = max(1, int(context_limit - 2048 - 100))
                    chunker_config = {"max_tokens": max_tokens}
            else:
                chunker_config = {"max_tokens": 2048}

        self.doc_processor = DocumentProcessor(
            docling_config=docling_config,
            chunker_config=chunker_config if use_chunking else None,
        )

        rich_print(
            f"[blue][ManyToOneStrategy][/blue] Initialized with {self._backend_type.upper()} backend: "
            f"[cyan]{self.backend.__class__.__name__}[/cyan]\n"
            f"  • Chunking: {'enabled' if self.use_chunking else 'disabled'}\n"
            f"  • LLM Consolidation: {'enabled' if self.llm_consolidation and self._is_llm else 'disabled'}"
        )

    # Public extraction entry point
    def extract(
        self, source: str, template: Type[BaseModel]
    ) -> Tuple[List[BaseModel], DoclingDocument | None]:
        """Extract structured data using a many-to-one strategy.

        - VLM backend: Extracts all pages and merges the results.
        - LLM backend: Uses structure-aware chunking (if enabled) or falls back to page-by-page.

        Returns:
            Tuple containing:
                - A list containing a single merged model instance, or an empty list on failure.
                - The DoclingDocument object used during extraction (or None if extraction failed).
        """
        try:
            # Use cached protocol checks (optimization)
            if self._is_vlm:
                rich_print("[blue][ManyToOneStrategy][/blue] Using VLM backend for extraction")
                return self._extract_with_vlm(
                    cast(ExtractionBackendProtocol, self.backend), source, template
                )
            elif self._is_llm:
                rich_print("[blue][ManyToOneStrategy][/blue] Using LLM backend for extraction")
                return self._extract_with_llm(
                    cast(TextExtractionBackendProtocol, self.backend), source, template
                )
            else:
                backend_class = self.backend.__class__.__name__
                raise TypeError(
                    f"Backend '{backend_class}' does not implement a recognized extraction protocol. "
                    "Expected either a VLM or LLM backend."
                )
        except Exception as e:
            rich_print(f"[red][ManyToOneStrategy][/red] Extraction error: {e}")
            return [], None

    # VLM backend extraction
    def _extract_with_vlm(
        self, backend: ExtractionBackendProtocol, source: str, template: Type[BaseModel]
    ) -> Tuple[List[BaseModel], DoclingDocument | None]:
        """Extract using a Vision-Language Model (VLM) backend, merging page-level models."""
        try:
            rich_print("[blue][ManyToOneStrategy][/blue] Running VLM extraction...")
            models = backend.extract_from_document(source, template)

            if not models:
                rich_print(
                    "[yellow][ManyToOneStrategy][/yellow] No models extracted by VLM backend"
                )
                return [], None

            if len(models) == 1:
                rich_print(
                    "[blue][ManyToOneStrategy][/blue] Single-page document extracted successfully"
                )
                return models, None

            # Merge multiple page-level models
            rich_print(
                f"[blue][ManyToOneStrategy][/blue] Merging [cyan]{len(models)}[/cyan] extracted page models..."
            )
            merged_model = merge_pydantic_models(models, template)

            if merged_model:
                rich_print(
                    "[green][ManyToOneStrategy][/green] Successfully merged all VLM page models"
                )
                return [merged_model], None
            else:
                rich_print(
                    "[yellow][ManyToOneStrategy][/yellow] Merge failed — "
                    "returning all page models (zero data loss: preserving partial results)"
                )
                return models, None

        except Exception as e:
            rich_print(
                f"[red][ManyToOneStrategy][/red] VLM extraction failed: {e}. "
                "Returning empty list (catastrophic failure: no data to preserve)."
            )
            import traceback

            rich_print(f"[red]Traceback:[/red]\n{traceback.format_exc()}")
            return [], None

    # LLM backend extraction
    def _extract_with_llm(
        self, backend: TextExtractionBackendProtocol, source: str, template: Type[BaseModel]
    ) -> Tuple[List[BaseModel], DoclingDocument | None]:
        """Extract using an LLM backend with intelligent strategy selection."""
        try:
            document = self.doc_processor.convert_to_docling_doc(source)

            model_config = getattr(backend.client, "model_config", None)
            context_limit = getattr(backend.client, "context_limit", 8000)
            if not isinstance(context_limit, int):
                context_limit = 8000
            max_output_tokens = getattr(model_config, "max_output_tokens", 2048)
            if not isinstance(max_output_tokens, int):
                max_output_tokens = 2048

            tokenizer_fn = None
            if self.doc_processor.chunker and self.doc_processor.chunker.tokenizer is not None:
                tokenizer = self.doc_processor.chunker.tokenizer
                if hasattr(tokenizer, "count_tokens"):
                    tokenizer_fn = tokenizer.count_tokens

            def count_tokens(text: str) -> int:
                estimate = max(1, int(len(text) / 3.5))
                if tokenizer_fn:
                    return max(int(tokenizer_fn(text)), estimate)
                return estimate

            # Use chunking if enabled
            if self.use_chunking:
                full_markdown = self.doc_processor.extract_full_markdown(document)
                schema_json = json.dumps(template.model_json_schema())
                schema_token_count = count_tokens(schema_json)
                density = min(1.0, 0.2 + (schema_token_count / 5000))

                prompt = get_extraction_prompt(
                    markdown_content="",
                    schema_json=schema_json,
                    is_partial=False,
                    model_config=model_config,
                )
                prompt_overhead = count_tokens(prompt["system"]) + count_tokens(prompt["user"])
                content_tokens = count_tokens(full_markdown)
                input_tokens = prompt_overhead + content_tokens
                estimated_output_tokens = int(content_tokens * density)

                if input_tokens < context_limit and estimated_output_tokens < max_output_tokens:
                    rich_print(
                        "[blue][ManyToOneStrategy][/blue] Document fits context/output budget "
                        f"(input={int(input_tokens)}, output≈{estimated_output_tokens}) — "
                        "using full-document extraction"
                    )
                    try:
                        models = self._extract_full_document(
                            backend, full_markdown, template
                        )
                    except Exception as e:
                        rich_print(
                            "[yellow][ManyToOneStrategy][/yellow] Full-document extraction failed "
                            f"({e}). Falling back to diff extraction."
                        )
                        models = []
                    if models:
                        return models, document
                    rich_print(
                        "[yellow][ManyToOneStrategy][/yellow] Full-document extraction returned no model. "
                        "Falling back to diff extraction."
                    )

                models = self._extract_with_chunks(backend, document, template)
                return models, document

            # Fallback to legacy page-by-page or full-doc extraction
            if hasattr(backend.client, "context_limit"):
                full_markdown = self.doc_processor.extract_full_markdown(document)
                estimated_tokens = len(full_markdown) / 3.5

                if estimated_tokens < (context_limit * 0.9):
                    rich_print(
                        f"[blue][ManyToOneStrategy][/blue] Document fits context "
                        f"({int(estimated_tokens)} tokens) — using full-document extraction"
                    )
                    models = self._extract_full_document(backend, full_markdown, template)
                    return models, document
                else:
                    rich_print(
                        f"[yellow][ManyToOneStrategy][/yellow] Document too large "
                        f"({int(estimated_tokens)} tokens) — using page-by-page fallback"
                    )
                    models = self._extract_pages_and_merge(backend, document, template)
                    return models, document
            else:
                full_markdown = self.doc_processor.extract_full_markdown(document)
                models = self._extract_full_document(backend, full_markdown, template)
                return models, document

        except Exception as e:
            rich_print(f"[red][ManyToOneStrategy][/red] LLM extraction failed: {e}")
            return [], None

    # Chunk-based extraction
    def _extract_with_chunks(  # noqa: C901
        self,
        backend: TextExtractionBackendProtocol,
        document: DoclingDocument,
        template: Type[BaseModel],
    ) -> List[BaseModel]:
        """Extract using structure-aware chunks with adaptive batching."""
        try:
            schema_json = json.dumps(template.model_json_schema())
            delta_schema_json = json.dumps(DeltaOperation.model_json_schema())
            if self.doc_processor.chunker:
                self.doc_processor.chunker.update_schema_config(schema_json)

            chunks = self.doc_processor.extract_chunks(document)
            total_chunks = len(chunks)

            context_limit = getattr(backend.client, "context_limit", 3500)
            if not isinstance(context_limit, int):
                context_limit = 3500

            tokenizer_fn = None
            if self.doc_processor.chunker and self.doc_processor.chunker.tokenizer is not None:
                tokenizer = self.doc_processor.chunker.tokenizer
                if hasattr(tokenizer, "count_tokens"):
                    tokenizer_fn = tokenizer.count_tokens
                    rich_print(
                        "[blue][ManyToOneStrategy][/blue] Using real tokenizer from DocumentChunker"
                    )

            model_config = getattr(backend.client, "model_config", None)
            reserved_output_tokens = getattr(model_config, "max_output_tokens", 2048)
            if not isinstance(reserved_output_tokens, int):
                reserved_output_tokens = 2048

            def count_tokens(text: str) -> int:
                estimate = max(1, int(len(text) / 3.5))
                if tokenizer_fn:
                    return max(int(tokenizer_fn(text)), estimate)
                return estimate

            schema_token_count = count_tokens(schema_json)
            current_density = min(1.0, 0.2 + (schema_token_count / 5000))

            def iter_values(value: Any) -> Iterable[Any]:
                if isinstance(value, BaseModel):
                    yield value
                    for _field_name, field_value in value:
                        yield from iter_values(field_value)
                elif isinstance(value, list):
                    for item in value:
                        yield from iter_values(item)
                elif isinstance(value, dict):
                    for item in value.values():
                        yield from iter_values(item)

            def collect_registry_entries(model: BaseModel) -> Dict[str, Dict[str, str]]:
                entries: Dict[str, Dict[str, str]] = {}
                for value in iter_values(model):
                    if isinstance(value, BaseModel):
                        entity_id = getattr(value, "id", None)
                        if not entity_id:
                            continue
                        if entity_id in entries:
                            continue
                        label = None
                        for attr in ("name", "title", "label", "type"):
                            candidate = getattr(value, attr, None)
                            if isinstance(candidate, str) and candidate.strip():
                                label = candidate.strip()
                                break
                        entries[entity_id] = {
                            "id": entity_id,
                            "label": label or entity_id,
                            "type": value.__class__.__name__,
                        }
                return entries

            def prune_registry(
                registry_list: List[Dict[str, str]],
                last_seen: Dict[str, int],
                prompt_budget_tokens: int,
            ) -> tuple[str, List[str], int]:
                if not registry_list:
                    return "[]", [], 2

                def sort_key(item: Dict[str, str]) -> int:
                    return last_seen.get(item["id"], 0)

                registry_list = sorted(registry_list, key=sort_key, reverse=True)
                pruned_ids: List[str] = []

                registry_str = json.dumps(registry_list, ensure_ascii=True)
                registry_tokens = count_tokens(registry_str)
                original_tokens = registry_tokens

                if registry_tokens <= prompt_budget_tokens:
                    return registry_str, pruned_ids, registry_tokens

                while registry_list and registry_tokens > prompt_budget_tokens:
                    removed = registry_list.pop()
                    pruned_ids.append(removed["id"])
                    registry_str = json.dumps(registry_list, ensure_ascii=True)
                    registry_tokens = count_tokens(registry_str)

                if pruned_ids:
                    rich_print(
                        "[yellow][ManyToOneStrategy][/yellow] "
                        f"Registry overflow (size {original_tokens}). "
                        f"Pruned {len(pruned_ids)} oldest entities from context."
                    )

                return registry_str, pruned_ids, registry_tokens

            def extract_ids_from_data(data: Any, ids: set[str]) -> None:
                if isinstance(data, dict):
                    if "id" in data and isinstance(data["id"], str):
                        ids.add(data["id"])
                    for value in data.values():
                        extract_ids_from_data(value, ids)
                elif isinstance(data, list):
                    for item in data:
                        extract_ids_from_data(item, ids)

            def apply_deletes(master_dict: Dict[str, Any], deletes: list) -> None:
                def delete_in_value(value: Any, target_id: str, entity_type: str | None) -> Any:
                    if isinstance(value, list):
                        new_list = []
                        for item in value:
                            if isinstance(item, dict):
                                item_id = item.get("id")
                                if item_id == target_id:
                                    if entity_type:
                                        item_type = (
                                            item.get("type")
                                            or item.get("entity_type")
                                            or item.get("__class__")
                                        )
                                        if item_type and item_type != entity_type:
                                            new_list.append(item)
                                    continue
                                new_list.append(delete_in_value(item, target_id, entity_type))
                            else:
                                new_list.append(delete_in_value(item, target_id, entity_type))
                        return new_list
                    if isinstance(value, dict):
                        for key, item in list(value.items()):
                            value[key] = delete_in_value(item, target_id, entity_type)
                        return value
                    return value

                for delete_op in deletes:
                    target_id = getattr(delete_op, "id", None)
                    if not target_id:
                        continue
                    entity_type = getattr(delete_op, "entity_type", None)
                    delete_in_value(master_dict, target_id, entity_type)

            def coerce_list_strings(data: Dict[str, Any], loc: tuple[Any, ...]) -> bool:
                if not loc:
                    return False
                cursor: Any = data
                for key in loc[:-1]:
                    if isinstance(key, int):
                        if not isinstance(cursor, list) or key >= len(cursor):
                            return False
                        cursor = cursor[key]
                    else:
                        if not isinstance(cursor, dict) or key not in cursor:
                            return False
                        cursor = cursor[key]
                last = loc[-1]
                if isinstance(last, int):
                    if not isinstance(cursor, list) or last >= len(cursor):
                        return False
                    value = cursor[last]
                    if isinstance(value, list) and all(isinstance(v, str) for v in value):
                        cursor[last] = " ".join(v.strip() for v in value if v)
                        return True
                    return False
                if not isinstance(cursor, dict) or last not in cursor:
                    return False
                value = cursor[last]
                if isinstance(value, list) and all(isinstance(v, str) for v in value):
                    cursor[last] = " ".join(v.strip() for v in value if v)
                    return True
                return False

            def attempt_delta_fixups(delta_data: Dict[str, Any], exc: ValidationError) -> bool:
                fixed_any = False
                for err in exc.errors():
                    if err.get("type") != "string_type":
                        continue
                    loc = err.get("loc")
                    if not isinstance(loc, tuple):
                        continue
                    input_value = err.get("input")
                    if input_value is None:
                        input_value = err.get("input_value")
                    if not isinstance(input_value, list) or not all(
                        isinstance(v, str) for v in input_value
                    ):
                        continue
                    fixed_any = coerce_list_strings(delta_data, loc) or fixed_any
                return fixed_any

            def apply_delta(
                master_state: BaseModel,
                delta: DeltaOperation,
                context_tag: str,
            ) -> BaseModel | None:
                try:
                    master_dict = master_state.model_dump()
                    deep_merge_dicts(master_dict, delta.data, context_tag=context_tag)
                    if delta.deletes:
                        apply_deletes(master_dict, delta.deletes)
                    return template.model_validate(master_dict)
                except Exception as e:
                    rich_print(f"[yellow][ManyToOneStrategy][/yellow] Failed to apply delta: {e}")
                    return None

            try:
                master_state: BaseModel = template()
            except Exception:
                master_state = template.model_construct()

            last_seen_step: Dict[str, int] = {}
            extracted_models: List[BaseModel] = []

            from ....pipeline.trace import ExtractionData

            extraction_id = 0
            step_index = 0
            chunk_index = 0

            rich_print(
                f"[blue][ManyToOneStrategy][/blue] Starting sequential extraction "
                f"({total_chunks} chunks)..."
            )

            max_retries = 2

            def process_batch(
                batch_chunks: List[str],
                batch_indices: List[int],
                density: float,
            ) -> bool:
                nonlocal extraction_id, step_index, current_density, master_state, extracted_models

                batch_text = "\n\n".join(batch_chunks)
                batch_label = (
                    f"batch {step_index + 1} (chunks {batch_indices[0] + 1}-{batch_indices[-1] + 1})"
                    if len(batch_indices) > 1
                    else f"batch {step_index + 1} (chunk {batch_indices[0] + 1})"
                )

                registry_entries = collect_registry_entries(master_state)
                registry_ids = set(registry_entries.keys())
                registry_list = list(registry_entries.values())
                registry_budget = int(context_limit * 0.5)
                registry_str, pruned_ids, registry_tokens = prune_registry(
                    registry_list, last_seen_step, registry_budget
                )

                base_prompt = get_context_aware_prompt(
                    markdown_content="",
                    schema_json=schema_json,
                    registry_content=registry_str,
                    delta_schema_json=delta_schema_json,
                    is_partial=True,
                    model_config=model_config,
                )
                base_prompt_tokens = count_tokens(base_prompt["system"]) + count_tokens(
                    base_prompt["user"]
                )
                batch_tokens = count_tokens(batch_text)
                input_tokens = base_prompt_tokens + batch_tokens
                if input_tokens >= context_limit:
                    rich_print(
                        "[yellow][ManyToOneStrategy][/yellow] Batch exceeds context window; "
                        "processing with tighter batch size."
                    )

                attempt = 0
                while attempt <= max_retries:
                    start_time = time.time()
                    error = None
                    delta = None

                    try:
                        delta = backend.extract_with_context(
                            markdown=batch_text,
                            template=template,
                            registry_str=registry_str,
                            context=batch_label,
                            is_partial=True,
                        )
                    except Exception as e:
                        error = str(e)

                    extraction_time = time.time() - start_time

                    if delta:
                        # Validate delta data against template
                        if delta.data:
                            try:
                                template.model_validate(delta.data)
                            except ValidationError as e:
                                if attempt_delta_fixups(delta.data, e):
                                    try:
                                        template.model_validate(delta.data)
                                    except ValidationError as retry_error:
                                        error = (
                                            "Delta data validation failed after fixups: "
                                            f"{retry_error}"
                                        )
                                        delta = None
                                else:
                                    error = f"Delta data validation failed: {e}"
                                    delta = None

                    if delta:
                        new_master = apply_delta(master_state, delta, context_tag=batch_label)
                        if new_master is None:
                            error = "Failed to apply delta to master graph."
                            delta = None

                    if delta and error is None and new_master is not None:
                        master_state = new_master
                        extracted_models = [master_state]
                        extracted_ids: set[str] = set()
                        extract_ids_from_data(delta.data, extracted_ids)

                        for entity_id in extracted_ids:
                            last_seen_step[entity_id] = step_index

                        evidence_ids = {e.entity_id for e in delta.evidence}
                        new_ids = [eid for eid in extracted_ids if eid not in registry_ids]
                        evidence_present = all(eid in evidence_ids for eid in new_ids)
                        if new_ids and not evidence_present:
                            rich_print(
                                "[yellow][ManyToOneStrategy][/yellow] Missing evidence for "
                                f"new entities: {', '.join(sorted(set(new_ids) - evidence_ids))}"
                            )

                        if self.trace_data:
                            extraction_data = ExtractionData(
                                extraction_id=extraction_id,
                                source_type="chunk",
                                source_id=batch_indices[0],
                                parsed_model=delta,
                                extraction_time=extraction_time,
                                error=error,
                                metadata={
                                    "type": "delta",
                                    "registry_size": registry_tokens,
                                    "density_used": current_density,
                                    "pruned_ids": pruned_ids,
                                    "evidence_present": evidence_present,
                                },
                            )
                            self.trace_data.extractions.append(extraction_data)
                            extraction_id += 1

                        step_index += 1
                        return True

                    attempt += 1
                    if attempt > max_retries:
                        rich_print(
                            "[yellow][ManyToOneStrategy][/yellow] "
                            f"{batch_label} failed after retries: {error}"
                        )
                        return False

                    current_density = min(1.0, current_density * 2)
                    if len(batch_chunks) > 1:
                        mid = max(1, len(batch_chunks) // 2)
                        left_chunks = batch_chunks[:mid]
                        right_chunks = batch_chunks[mid:]
                        left_indices = batch_indices[:mid]
                        right_indices = batch_indices[mid:]
                        rich_print(
                            "[yellow][ManyToOneStrategy][/yellow] "
                            f"{batch_label} failed. Backing off to smaller batches "
                            f"({len(left_chunks)} + {len(right_chunks)})."
                        )
                        left_ok = process_batch(
                            left_chunks, left_indices, current_density
                        )
                        right_ok = True
                        if right_chunks:
                            right_ok = process_batch(
                                right_chunks, right_indices, current_density
                            )
                        return left_ok and right_ok

                return False

            while chunk_index < total_chunks:
                batch_chunks: List[str] = []
                batch_indices: List[int] = []
                batch_text = ""

                while chunk_index < total_chunks:
                    candidate = chunks[chunk_index]
                    candidate_text = candidate
                    candidate_batch = (
                        f"{batch_text}\n\n{candidate_text}" if batch_text else candidate_text
                    )
                    batch_tokens = count_tokens(candidate_batch)
                    output_tokens = int(batch_tokens * current_density)

                    registry_entries = collect_registry_entries(master_state)
                    registry_list = list(registry_entries.values())
                    registry_budget = int(context_limit * 0.5)
                    registry_str, _pruned_ids, _registry_tokens = prune_registry(
                        registry_list, last_seen_step, registry_budget
                    )

                    base_prompt = get_context_aware_prompt(
                        markdown_content="",
                        schema_json=schema_json,
                        registry_content=registry_str,
                        delta_schema_json=delta_schema_json,
                        is_partial=True,
                        model_config=model_config,
                    )
                    base_prompt_tokens = count_tokens(base_prompt["system"]) + count_tokens(
                        base_prompt["user"]
                    )
                    input_tokens = base_prompt_tokens + batch_tokens

                    if input_tokens >= context_limit or output_tokens >= reserved_output_tokens:
                        if batch_chunks:
                            break
                        batch_chunks.append(candidate_text)
                        batch_indices.append(chunk_index)
                        chunk_index += 1
                        break

                    batch_chunks.append(candidate_text)
                    batch_indices.append(chunk_index)
                    batch_text = candidate_batch
                    chunk_index += 1

                if not batch_chunks:
                    break

                process_batch(batch_chunks, batch_indices, current_density)

            if not extracted_models:
                rich_print(
                    "[yellow][ManyToOneStrategy][/yellow] No models extracted from any batch. "
                    "Returning empty list (zero data loss: no partial data to preserve)."
                )
                return []

            if self.trace_data:
                try:
                    from ....core.converters.graph_converter import GraphConverter
                    from ....pipeline.trace import GraphData

                    converter = GraphConverter(
                        add_reverse_edges=False,
                        validate_graph=True,
                    )
                    graph, metadata = converter.pydantic_list_to_graph([master_state])
                    self.trace_data.intermediate_graphs.append(
                        GraphData(
                            graph_id=len(self.trace_data.intermediate_graphs),
                            source_type="chunk",
                            source_id=step_index,
                            graph=graph,
                            pydantic_model=master_state,
                            node_count=metadata.node_count,
                            edge_count=metadata.edge_count,
                        )
                    )
                except Exception as e:
                    rich_print(
                        f"[yellow][ManyToOneStrategy][/yellow] Failed to snapshot master graph: {e}"
                    )

            return extracted_models

        except Exception as e:
            rich_print(
                f"[red][ManyToOneStrategy][/red] Batch extraction failed: {e}. "
                "Returning empty list (catastrophic failure: no data to preserve)."
            )
            import traceback

            rich_print(f"[red]Traceback:[/red]\n{traceback.format_exc()}")
            return []

    # Full-document extraction (LLM)
    def _extract_full_document(
        self, backend: TextExtractionBackendProtocol, full_markdown: str, template: Type[BaseModel]
    ) -> List[BaseModel]:
        """Extract a single consolidated model from full document markdown."""
        try:
            model = backend.extract_from_markdown(
                markdown=full_markdown,
                template=template,
                context="full document",
                is_partial=False,
            )

            if model:
                rich_print(
                    "[green][ManyToOneStrategy][/green] Successfully extracted consolidated model from full document"
                )
                return [model]
            else:
                rich_print(
                    "[yellow][ManyToOneStrategy][/yellow] Full-document extraction returned no model"
                )
                return []

        except Exception as e:
            rich_print(
                f"[red][ManyToOneStrategy][/red] Full-document extraction failed: {e}. "
                "Returning empty list (catastrophic failure: no data to preserve)."
            )
            import traceback

            rich_print(f"[red]Traceback:[/red]\n{traceback.format_exc()}")
            return []

    # Page-by-page extraction + merging (LLM)
    def _extract_pages_and_merge(
        self,
        backend: TextExtractionBackendProtocol,
        document: DoclingDocument,
        template: Type[BaseModel],
    ) -> List[BaseModel]:
        """Extract individual page models and intelligently merge them into one."""
        try:
            page_markdowns = self.doc_processor.extract_page_markdowns(document)
            total_pages = len(page_markdowns)

            rich_print(
                f"[blue][ManyToOneStrategy][/blue] Starting page-by-page extraction ({total_pages} pages)..."
            )

            extracted_models: List[BaseModel] = []

            for page_num, page_md in enumerate(page_markdowns, 1):
                rich_print(
                    f"[blue][ManyToOneStrategy][/blue] Extracting from page {page_num}/{total_pages}"
                )

                model = backend.extract_from_markdown(
                    markdown=page_md,
                    template=template,
                    context=f"page {page_num}",
                    is_partial=True,
                )

                if model:
                    extracted_models.append(model)
                else:
                    rich_print(
                        f"[yellow][ManyToOneStrategy][/yellow] Page {page_num} returned no model"
                    )

            if not extracted_models:
                rich_print(
                    "[yellow][ManyToOneStrategy][/yellow] No models extracted from any page. "
                    "Returning empty list (zero data loss: no partial data to preserve)."
                )
                return []

            if len(extracted_models) == 1:
                rich_print(
                    "[blue][ManyToOneStrategy][/blue] Single page extracted — no merge needed"
                )
                return extracted_models

            rich_print(
                f"[blue][ManyToOneStrategy][/blue] Programmatically merging "
                f"[cyan]{len(extracted_models)}[/cyan] page models..."
            )
            programmatic_model = merge_pydantic_models(extracted_models, template)

            if not programmatic_model:
                rich_print(
                    "[yellow][ManyToOneStrategy][/yellow] Programmatic merge failed. "
                    "Returning all extracted page models (zero data loss: preserving partial results)."
                )
                return extracted_models

            # Consolidation step (use cached protocol check)
            if self.llm_consolidation and self._is_llm:
                rich_print(
                    "[blue]Programmatic merge complete. Starting LLM consolidation pass...[/blue]"
                )
                final_model = cast(
                    TextExtractionBackendProtocol, self.backend
                ).consolidate_from_pydantic_models(
                    raw_models=extracted_models,
                    programmatic_model=programmatic_model,
                    template=template,
                )
                if final_model:
                    return [final_model]
                rich_print(
                    "[yellow][ManyToOneStrategy][/yellow] LLM consolidation failed. "
                    "Falling back to programmatic merge (zero data loss: preserving merged result)."
                )
                return [programmatic_model]
            else:
                return [programmatic_model]

        except Exception as e:
            rich_print(
                f"[red][ManyToOneStrategy][/red] Page-by-page extraction failed: {e}. "
                "Returning empty list (catastrophic failure: no data to preserve)."
            )
            import traceback

            rich_print(f"[red]Traceback:[/red]\n{traceback.format_exc()}")
            return []
