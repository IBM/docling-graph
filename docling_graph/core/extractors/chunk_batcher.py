"""
Adaptive chunk batching for efficient LLM extraction.

Groups multiple chunks into batches that fit within context window,
using real tokenizer counts over the full prompt.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, List, Optional

from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer
from rich import print as rich_print

from docling_graph.llm_clients.config import ModelConfigLike, get_merge_threshold_for_provider
from docling_graph.llm_clients.prompts import get_extraction_prompt

logger = logging.getLogger(__name__)


@dataclass
class ChunkBatch:
    """A batch of chunks to send to LLM in a single call."""

    batch_id: int
    """Batch sequence number."""

    chunks: List[str]
    """List of chunk texts in this batch."""

    total_tokens: int
    """Estimated content tokens (prompt tokens minus static overhead)."""

    prompt_tokens: int
    """Full prompt tokens for this batch (system + user + content)."""

    chunk_indices: List[int]
    """Original chunk indices from document."""

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    @property
    def combined_text(self) -> str:
        separator = "\n\n---CHUNK BOUNDARY---\n\n"
        return separator.join(
            [f"[Chunk {i + 1}/{len(self.chunks)}]\n{chunk}" for i, chunk in enumerate(self.chunks)]
        )


class ChunkBatcher:
    """
    Intelligently batch chunks with real tokenizer integration.

    Computes full prompt tokens for combined chunks and enforces context limits
    with fixed output reservation and safety margin.
    """

    def __init__(
        self,
        context_limit: int,
        schema_json: str,
        tokenizer: HuggingFaceTokenizer | OpenAITokenizer,
        reserved_output_tokens: int = 2048,
        safety_margin_tokens: int = 100,
        merge_threshold: float | None = None,
        provider: str | None = None,
        is_partial: bool = True,
        model_config: ModelConfigLike | None = None,
    ) -> None:
        """
        Initialize batcher with context constraints and provider configuration.

        Args:
            context_limit: Total context window in tokens
            schema_json: Pydantic schema JSON string
            tokenizer: Tokenizer instance used for prompt token counting
            reserved_output_tokens: Tokens reserved for model responses
            safety_margin_tokens: Fixed buffer for protocol/tokenizer overhead
            merge_threshold: Merge chunks if batch is <this% of available context
            provider: LLM provider name (openai, gemini, etc.)
            is_partial: Use partial (chunk) prompt template when True
            model_config: Optional model config for prompt tailoring
        """
        self.context_limit = context_limit
        self.schema_json = schema_json or "{}"
        self.tokenizer = tokenizer
        self.reserved_output_tokens = reserved_output_tokens
        self.safety_margin_tokens = safety_margin_tokens
        self.is_partial = is_partial
        self.model_config = model_config

        self.provider_name = provider or "unknown"
        self.merge_threshold = (
            merge_threshold
            if merge_threshold is not None
            else get_merge_threshold_for_provider(self.provider_name)
        )

        self.prompt_budget = context_limit - reserved_output_tokens - safety_margin_tokens
        self.static_overhead_tokens = self._prompt_tokens("")
        self.available_tokens = self.prompt_budget - self.static_overhead_tokens

        if self.available_tokens <= 0:
            logger.warning(
                "No available tokens for content: context=%s overhead=%s reserved=%s margin=%s",
                context_limit,
                self.static_overhead_tokens,
                reserved_output_tokens,
                safety_margin_tokens,
            )

        rich_print(
            f"[blue][ChunkBatcher][/blue] Initialized with:\n"
            f" • Provider: [cyan]{self.provider_name}[/cyan]\n"
            f" • Context limit: [yellow]{context_limit:,}[/yellow] tokens\n"
            f" • Prompt budget: [cyan]{self.prompt_budget:,}[/cyan] tokens\n"
            f" • Static overhead: [cyan]{self.static_overhead_tokens:,}[/cyan] tokens\n"
            f" • Available for content: [cyan]{self.available_tokens:,}[/cyan] tokens\n"
            f" • Merge threshold: {self.merge_threshold * 100:.0f}%"
        )

    def _prompt_tokens(self, markdown_content: str) -> int:
        prompt = get_extraction_prompt(
            markdown_content=markdown_content,
            schema_json=self.schema_json,
            is_partial=self.is_partial,
            model_config=self.model_config,
        )
        system_tokens = self.tokenizer.count_tokens(prompt["system"])
        user_tokens = self.tokenizer.count_tokens(prompt["user"])
        return system_tokens + user_tokens

    def _content_tokens(self, markdown_content: str) -> tuple[int, int]:
        prompt_tokens = self._prompt_tokens(markdown_content)
        content_tokens = max(0, prompt_tokens - self.static_overhead_tokens)
        return content_tokens, prompt_tokens

    def _build_combined_text(self, chunks: List[str]) -> str:
        separator = "\n\n---CHUNK BOUNDARY---\n\n"
        return separator.join(
            [f"[Chunk {i + 1}/{len(chunks)}]\n{chunk}" for i, chunk in enumerate(chunks)]
        )

    def batch_chunks(
        self,
        chunks: List[str],
        tokenizer_fn: Callable[[str], int] | None = None,
    ) -> List[ChunkBatch]:
        """
        Batch chunks to fit context window efficiently using prompt token counts.

        Args:
            chunks: List of chunk texts

        Returns:
            List of ChunkBatch objects ready for LLM extraction
        """
        if not chunks:
            return []

        batches: List[ChunkBatch] = []
        current_batch_chunks: List[str] = []
        current_batch_indices: List[int] = []
        current_content_tokens = 0
        current_prompt_tokens = 0

        for chunk_idx, chunk_text in enumerate(chunks):
            candidate_chunks = [*current_batch_chunks, chunk_text]
            combined_text = self._build_combined_text(candidate_chunks)
            candidate_content_tokens, candidate_prompt_tokens = self._content_tokens(combined_text)

            if current_batch_chunks and candidate_prompt_tokens > self.prompt_budget:
                batches.append(
                    ChunkBatch(
                        batch_id=len(batches),
                        chunks=current_batch_chunks.copy(),
                        total_tokens=current_content_tokens,
                        prompt_tokens=current_prompt_tokens,
                        chunk_indices=current_batch_indices.copy(),
                    )
                )
                current_batch_chunks = [chunk_text]
                current_batch_indices = [chunk_idx]
                combined_text = self._build_combined_text(current_batch_chunks)
                current_content_tokens, current_prompt_tokens = self._content_tokens(combined_text)
            else:
                current_batch_chunks = candidate_chunks
                current_batch_indices.append(chunk_idx)
                current_content_tokens = candidate_content_tokens
                current_prompt_tokens = candidate_prompt_tokens

        if current_batch_chunks:
            batches.append(
                ChunkBatch(
                    batch_id=len(batches),
                    chunks=current_batch_chunks,
                    total_tokens=current_content_tokens,
                    prompt_tokens=current_prompt_tokens,
                    chunk_indices=current_batch_indices,
                )
            )

        merged_batches = self._merge_undersized_batches(batches)
        self._log_batching_summary(
            total_chunks=len(chunks),
            batches=merged_batches,
            total_tokens=sum(b.total_tokens for b in merged_batches),
        )
        return merged_batches

    def _merge_undersized_batches(self, batches: List[ChunkBatch]) -> List[ChunkBatch]:
        if len(batches) <= 1:
            return batches

        threshold_tokens = int(self.available_tokens * self.merge_threshold)
        merged: List[ChunkBatch] = []

        i = 0
        while i < len(batches):
            current = batches[i]

            if current.total_tokens >= threshold_tokens:
                merged.append(current)
                i += 1
                continue

            combined_chunks = current.chunks.copy()
            combined_indices = current.chunk_indices.copy()
            combined_text = self._build_combined_text(combined_chunks)
            combined_content_tokens, combined_prompt_tokens = self._content_tokens(combined_text)

            j = i + 1
            while j < len(batches):
                next_batch = batches[j]
                candidate_chunks = combined_chunks + next_batch.chunks
                candidate_text = self._build_combined_text(candidate_chunks)
                candidate_content_tokens, candidate_prompt_tokens = self._content_tokens(
                    candidate_text
                )

                if candidate_prompt_tokens > self.prompt_budget:
                    break

                combined_chunks = candidate_chunks
                combined_indices = combined_indices + next_batch.chunk_indices
                combined_content_tokens = candidate_content_tokens
                combined_prompt_tokens = candidate_prompt_tokens
                j += 1

            merged.append(
                ChunkBatch(
                    batch_id=len(merged),
                    chunks=combined_chunks,
                    total_tokens=combined_content_tokens,
                    prompt_tokens=combined_prompt_tokens,
                    chunk_indices=combined_indices,
                )
            )
            i = j

        return merged

    def _log_batching_summary(
        self, total_chunks: int, batches: List[ChunkBatch], total_tokens: int
    ) -> None:
        reduction = (total_chunks - len(batches)) / max(1, total_chunks) * 100
        avg_batch_size = sum(b.chunk_count for b in batches) / max(1, len(batches))
        avg_utilization = (
            total_tokens / (len(batches) * self.available_tokens) * 100 if batches else 0
        )

        summary = (
            f"[blue][ChunkBatcher][/blue] Batching summary:\n"
            f"  • Total chunks: [cyan]{total_chunks}[/cyan]\n"
            f"  • Batches created: [yellow]{len(batches)}[/yellow] ([green]-{reduction:.0f}%[/green] API calls)\n"
            f"  • Avg chunks/batch: {avg_batch_size:.1f}\n"
            f"  • Context utilization: {avg_utilization:.1f}%"
        )

        rich_print(summary)

        for batch in batches:
            utilization = (
                batch.total_tokens / self.available_tokens * 100 if self.available_tokens else 0
            )
            batch_info = (
                f"  └─ Batch {batch.batch_id}: "
                f"{batch.chunk_count} chunks "
                f"({batch.total_tokens:,} content tokens, {utilization:.0f}% utilized)"
            )
            rich_print(batch_info)
