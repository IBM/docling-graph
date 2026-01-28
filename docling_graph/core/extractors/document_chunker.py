"""
Structure-preserving document chunker using Docling's HybridChunker.

Preserves:
- Tables (not split across chunks)
- Lists (kept intact)
- Hierarchical structure (sections with headers)
- Semantic boundaries

Configurable per LLM provider tokenizer.
"""

import json
import logging
import re
import time
from typing import List, Optional, Union

from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer
from docling_core.types.doc import DoclingDocument
from rich import print as rich_print
from transformers import AutoTokenizer

from ...llm_clients.config import (
    ModelConfigLike,
    get_tokenizer_for_provider,
    resolve_effective_model_config,
)
from ...llm_clients.prompts import get_extraction_prompt

logger = logging.getLogger(__name__)


class DocumentChunker:
    """Structure-preserving document chunker using Docling's HybridChunker."""

    def __init__(
        self,
        tokenizer_name: str | None = None,
        max_tokens: int | None = None,
        provider: str | None = None,
        model: str | None = None,
        model_config: ModelConfigLike | None = None,
        merge_peers: bool = True,
        schema_json: str | None = None,
        reserved_output_tokens: int | None = None,
        safety_margin_tokens: int = 100,
        is_partial: bool = True,
    ) -> None:
        """
        Initialize the chunker with smart defaults based on provider or custom tokenizer.

        Uses LiteLLM metadata for context limits and precise prompt token math.

        Args:
            tokenizer_name: Name of the tokenizer to use
            max_tokens: Maximum tokens per chunk (if None, calculated from provider)
            provider: LLM provider name (e.g., "watsonx", "openai")
            model: Model name (optional, improves chunk sizing)
            model_config: Resolved model config (optional, avoids metadata lookup)
            merge_peers: Whether to merge peer sections in chunking
            schema_json: Pydantic schema JSON string for prompt sizing
            reserved_output_tokens: Output tokens reserved for model responses
            safety_margin_tokens: Fixed buffer for protocol/tokenizer overhead
            is_partial: Use partial (chunk) prompt template when True
        """
        # region agent log
        try:
            with open(
                "/home/ayoub/github/docling-graph/.cursor/debug.log",
                "a",
                encoding="utf-8",
            ) as log_file:
                log_file.write(
                    json.dumps(
                        {
                            "sessionId": "debug-session",
                            "runId": "pre-fix",
                            "hypothesisId": "H1",
                            "location": "document_chunker.py:53",
                            "message": "DocumentChunker init start",
                            "data": {
                                "tokenizer_name": tokenizer_name,
                                "max_tokens": max_tokens,
                                "provider": provider,
                                "model": model,
                            },
                            "timestamp": int(time.time() * 1000),
                        }
                    )
                    + "\n"
                )
        except Exception:
            pass
        # endregion
        self.tokenizer: Union[HuggingFaceTokenizer, OpenAITokenizer] | None = None
        self.chunker: HybridChunker | None = None
        self.model_config: ModelConfigLike | None = model_config
        self.context_limit: int | None = None
        self.reserved_output_tokens = (
            2048 if reserved_output_tokens is None else reserved_output_tokens
        )
        self.safety_margin_tokens = safety_margin_tokens
        self.is_partial = is_partial
        self.schema_json = schema_json or "{}"

        if tokenizer_name is None and provider is None and max_tokens is None and model is None:
            tokenizer_name = "sentence-transformers/all-MiniLM-L6-v2"
            max_tokens = 5120
            self.tokenizer = None
            self.chunker = None
            self.max_tokens = max_tokens
            self.original_max_tokens = max_tokens
            self.tokenizer_name = tokenizer_name
            self.merge_peers = merge_peers
            try:
                with open(
                    "/home/ayoub/github/docling-graph/.cursor/debug.log",
                    "a",
                    encoding="utf-8",
                ) as log_file:
                    log_file.write(
                        json.dumps(
                            {
                                "sessionId": "debug-session",
                                "runId": "pre-fix",
                                "hypothesisId": "H2",
                                "location": "document_chunker.py:78",
                                "message": "Skipped tokenizer init (lazy defaults)",
                                "data": {
                                    "tokenizer_name": tokenizer_name,
                                    "max_tokens": max_tokens,
                                },
                                "timestamp": int(time.time() * 1000),
                            }
                        )
                        + "\n"
                    )
            except Exception:
                pass
            return

        if self.model_config is not None:
            self.context_limit = self.model_config.context_limit
            if reserved_output_tokens is None and hasattr(self.model_config, "max_output_tokens"):
                self.reserved_output_tokens = self.model_config.max_output_tokens
        elif provider and model:
            self.model_config = resolve_effective_model_config(provider, model)
            self.context_limit = self.model_config.context_limit
            if reserved_output_tokens is None:
                self.reserved_output_tokens = self.model_config.max_output_tokens

        if tokenizer_name is None and provider is not None:
            tokenizer_name = get_tokenizer_for_provider(provider)
        elif tokenizer_name is None:
            tokenizer_name = "sentence-transformers/all-MiniLM-L6-v2"

        temp_max_tokens = max_tokens or self.context_limit or 5120

        if tokenizer_name != "tiktoken":
            try:
                with open(
                    "/home/ayoub/github/docling-graph/.cursor/debug.log",
                    "a",
                    encoding="utf-8",
                ) as log_file:
                    log_file.write(
                        json.dumps(
                            {
                                "sessionId": "debug-session",
                                "runId": "pre-fix",
                                "hypothesisId": "H2",
                                "location": "document_chunker.py:86",
                                "message": "Initializing HF tokenizer",
                                "data": {"tokenizer_name": tokenizer_name},
                                "timestamp": int(time.time() * 1000),
                            }
                        )
                        + "\n"
                    )
            except Exception:
                pass
            hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            self.tokenizer = HuggingFaceTokenizer(
                tokenizer=hf_tokenizer,
                max_tokens=temp_max_tokens,
            )
        else:
            try:
                import tiktoken

                if model:
                    tt_tokenizer = tiktoken.encoding_for_model(model)
                else:
                    tt_tokenizer = tiktoken.get_encoding("cl100k_base")
                self.tokenizer = OpenAITokenizer(
                    tokenizer=tt_tokenizer,
                    max_tokens=temp_max_tokens,
                )
            except ImportError:
                rich_print(
                    "[yellow][DocumentChunker][/yellow] tiktoken not installed, "
                    "falling back to HuggingFace tokenizer"
                )
                hf_tokenizer = AutoTokenizer.from_pretrained(
                    "sentence-transformers/all-MiniLM-L6-v2"
                )
                self.tokenizer = HuggingFaceTokenizer(
                    tokenizer=hf_tokenizer,
                    max_tokens=temp_max_tokens,
                )

        if max_tokens is None and self.context_limit is not None:
            max_tokens = self._calculate_max_tokens(
                context_limit=self.context_limit, schema_json=self.schema_json
            )
        elif max_tokens is None:
            max_tokens = temp_max_tokens

        if self.tokenizer is not None and hasattr(self.tokenizer, "max_tokens"):
            self.tokenizer.max_tokens = max_tokens

        try:
            with open(
                "/home/ayoub/github/docling-graph/.cursor/debug.log",
                "a",
                encoding="utf-8",
            ) as log_file:
                log_file.write(
                    json.dumps(
                        {
                            "sessionId": "debug-session",
                            "runId": "pre-fix",
                            "hypothesisId": "H1",
                            "location": "document_chunker.py:74",
                            "message": "DocumentChunker resolved defaults",
                            "data": {
                                "tokenizer_name": tokenizer_name,
                                "max_tokens": max_tokens,
                                "provider": provider,
                            },
                            "timestamp": int(time.time() * 1000),
                        }
                    )
                    + "\n"
                )
        except Exception:
            pass

        # Step 4: Create HybridChunker instance
        self.chunker = HybridChunker(
            tokenizer=self.tokenizer,
            merge_peers=merge_peers,
        )
        # region agent log
        try:
            with open(
                "/home/ayoub/github/docling-graph/.cursor/debug.log",
                "a",
                encoding="utf-8",
            ) as log_file:
                log_file.write(
                    json.dumps(
                        {
                            "sessionId": "debug-session",
                            "runId": "pre-fix",
                            "hypothesisId": "H2",
                            "location": "document_chunker.py:127",
                            "message": "HybridChunker initialized",
                            "data": {"merge_peers": merge_peers, "max_tokens": max_tokens},
                            "timestamp": int(time.time() * 1000),
                        }
                    )
                    + "\n"
                )
        except Exception:
            pass
        # endregion

        self.max_tokens = max_tokens
        self.original_max_tokens = max_tokens  # Store original for schema adjustments
        self.tokenizer_name = tokenizer_name
        self.merge_peers = merge_peers

        rich_print(
            f"[blue][DocumentChunker][/blue] Initialized with:\n"
            f" • Tokenizer: [cyan]{tokenizer_name}[/cyan]\n"
            f" • Max tokens/chunk: [yellow]{max_tokens}[/yellow]\n"
            f" • Merge peers: {merge_peers}"
        )

    def update_schema_config(self, schema_json: str) -> None:
        """
        Update chunker configuration based on schema JSON.

        Adjusts max_tokens to reserve space for schema in context window,
        preventing context overflow when schema is large.

        Args:
            schema_json: The JSON schema string
        """
        if not self.tokenizer:
            # region agent log
            try:
                with open(
                    "/home/ayoub/github/docling-graph/.cursor/debug.log",
                    "a",
                    encoding="utf-8",
                ) as log_file:
                    log_file.write(
                        json.dumps(
                            {
                                "sessionId": "debug-session",
                                "runId": "pre-fix",
                                "hypothesisId": "H3",
                                "location": "document_chunker.py:158",
                                "message": "update_schema_config missing tokenizer",
                                "data": {"schema_size": len(schema_json)},
                                "timestamp": int(time.time() * 1000),
                            }
                        )
                        + "\n"
                    )
            except Exception:
                pass
            # endregion
            logger.warning("No tokenizer available for schema config update")
            return

        self.schema_json = schema_json or "{}"
        if self.context_limit is None:
            logger.warning("No context limit available for schema config update")
            return

        self.max_tokens = self._calculate_max_tokens(
            context_limit=self.context_limit, schema_json=self.schema_json
        )

        # Update the tokenizer's max_tokens
        if self.tokenizer is not None and hasattr(self.tokenizer, "max_tokens"):
            self.tokenizer.max_tokens = self.max_tokens

        # Update the chunker's tokenizer max_tokens as well
        if (
            self.chunker is not None
            and hasattr(self.chunker, "tokenizer")
            and hasattr(self.chunker.tokenizer, "max_tokens")
        ):
            self.chunker.tokenizer.max_tokens = self.max_tokens
        # region agent log
        try:
            with open(
                "/home/ayoub/github/docling-graph/.cursor/debug.log",
                "a",
                encoding="utf-8",
            ) as log_file:
                log_file.write(
                    json.dumps(
                        {
                            "sessionId": "debug-session",
                            "runId": "pre-fix",
                            "hypothesisId": "H3",
                            "location": "document_chunker.py:193",
                            "message": "update_schema_config complete",
                            "data": {
                                "schema_size": len(self.schema_json),
                                "max_tokens": self.max_tokens,
                                "original_max_tokens": self.original_max_tokens,
                                "chunker_is_none": self.chunker is None,
                            },
                            "timestamp": int(time.time() * 1000),
                        }
                    )
                    + "\n"
                )
        except Exception:
            pass
        # endregion

        # Note: chunker.max_tokens is a read-only property derived from tokenizer.max_tokens
        # so we don't need to (and can't) set it directly

        rich_print(
            "[blue][DocumentChunker][/blue] Schema config updated:\n"
            f" • Schema size: {len(self.schema_json)} bytes\n"
            f" • Adjusted max_tokens: {self.max_tokens} (was {self.original_max_tokens})"
        )

    @staticmethod
    def calculate_recommended_max_tokens(
        tokenizer: Union[HuggingFaceTokenizer, OpenAITokenizer],
        context_limit: int,
        schema_json: str,
        is_partial: bool = True,
        reserved_output_tokens: int = 2048,
        safety_margin_tokens: int = 100,
        model_config: ModelConfigLike | None = None,
    ) -> int:
        """
        Calculate recommended max_tokens for a given context window.

        Formula:
        available = context_limit - static_overhead - reserved_output - safety_margin

        static_overhead = tokens(system_prompt) + tokens(user_prompt_skeleton)

        Args:
            tokenizer: Tokenizer instance used for counting prompt tokens
            context_limit: Total context window (e.g., 8000 for Mistral-Large)
            schema_json: Pydantic schema JSON string
            is_partial: Whether to use the partial prompt template
            reserved_output_tokens: Tokens reserved for the LLM response
            safety_margin_tokens: Fixed buffer for protocol/tokenizer overhead
            model_config: Optional model config for prompt tailoring

        Returns:
            Recommended max_tokens value for chunker
        """
        prompt = get_extraction_prompt(
            markdown_content="",
            schema_json=schema_json,
            is_partial=is_partial,
            model_config=model_config,
        )
        system_tokens = tokenizer.count_tokens(prompt["system"])
        user_tokens = tokenizer.count_tokens(prompt["user"])
        static_overhead = system_tokens + user_tokens
        available = context_limit - static_overhead - reserved_output_tokens - safety_margin_tokens
        return max(1, int(available))

    def _count_tokens(self, text: str) -> int:
        if not self.tokenizer:
            raise ValueError("Tokenizer not initialized.")
        return self.tokenizer.count_tokens(text)

    def _prompt_overhead_tokens(self, schema_json: str) -> int:
        prompt = get_extraction_prompt(
            markdown_content="",
            schema_json=schema_json,
            is_partial=self.is_partial,
            model_config=self.model_config,
        )
        return self._count_tokens(prompt["system"]) + self._count_tokens(prompt["user"])

    def _calculate_max_tokens(self, context_limit: int, schema_json: str) -> int:
        overhead = self._prompt_overhead_tokens(schema_json)
        available = (
            context_limit - overhead - self.reserved_output_tokens - self.safety_margin_tokens
        )
        if available <= 0:
            logger.warning(
                "Calculated negative/zero available tokens: context=%s overhead=%s reserved=%s margin=%s",
                context_limit,
                overhead,
                self.reserved_output_tokens,
                self.safety_margin_tokens,
            )
        return max(1, int(available))

    def chunk_document(self, document: DoclingDocument) -> List[str]:
        """
        Chunk a DoclingDocument into structure-aware text chunks.

        Args:
            document: Parsed DoclingDocument from DocumentConverter

        Returns:
            List of contextualized text chunks, ready for LLM consumption
        """
        chunks = []

        if self.chunker is None:
            raise ValueError("Chunker not initialized.")

        # Chunk the document using HybridChunker
        chunk_iter = self.chunker.chunk(dl_doc=document)

        for chunk in chunk_iter:
            # Use contextualized text (includes metadata like headers, section captions)
            # This is essential for LLM extraction to understand chunk context
            enriched_text = self.chunker.contextualize(chunk=chunk)
            chunks.append(enriched_text)

        return chunks

    def chunk_document_with_stats(self, document: DoclingDocument) -> tuple[List[str], dict]:
        """
        Chunk document and return tokenization statistics.
        Useful for debugging/optimization to understand chunk distribution.

        Args:
            document: Parsed DoclingDocument

        Returns:
            Tuple of (chunks, stats) where stats contains:
            - total_chunks: number of chunks
            - chunk_tokens: list of token counts per chunk
            - avg_tokens: average tokens per chunk
            - max_tokens_in_chunk: maximum tokens in any chunk
            - total_tokens: sum of all chunk tokens
        """
        chunks = []
        chunk_tokens = []

        if self.chunker is None:
            raise ValueError("Chunker not initialized.")

        chunk_iter = self.chunker.chunk(dl_doc=document)

        for chunk in chunk_iter:
            enriched_text = self.chunker.contextualize(chunk=chunk)
            chunks.append(enriched_text)

            # Count tokens for this chunk
            if self.tokenizer is None:
                raise ValueError("Tokenizer not initialized.")
            num_tokens = self.tokenizer.count_tokens(enriched_text)
            chunk_tokens.append(num_tokens)

        stats = {
            "total_chunks": len(chunks),
            "chunk_tokens": chunk_tokens,
            "avg_tokens": sum(chunk_tokens) / len(chunk_tokens) if chunk_tokens else 0,
            "max_tokens_in_chunk": max(chunk_tokens) if chunk_tokens else 0,
            "total_tokens": sum(chunk_tokens),
        }

        return chunks, stats

    def chunk_text_fallback(self, text: str) -> List[str]:
        """
        Fallback chunker for raw text when DoclingDocument unavailable.

        This is a simple token-based splitter that respects sentence boundaries.
        For best results, always use chunk_document() with a DoclingDocument.

        Args:
            text: Raw text string (e.g., plain Markdown)

        Returns:
            List of text chunks
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized.")

        if self.tokenizer.count_tokens(text) <= self.max_tokens:
            return [text]

        segments = [seg for seg in re.split(r"(?<=[.!?])\s+|\n\n|\n", text) if seg]
        chunks: list[str] = []
        current_segments: list[str] = []

        for segment in segments:
            candidate_segments = [*current_segments, segment]
            candidate_text = " ".join(candidate_segments).strip()
            if not candidate_text:
                continue
            candidate_tokens = self.tokenizer.count_tokens(candidate_text)

            if candidate_tokens <= self.max_tokens or not current_segments:
                current_segments = candidate_segments
                continue

            chunks.append(" ".join(current_segments).strip())
            current_segments = [segment]

        if current_segments:
            chunks.append(" ".join(current_segments).strip())

        return chunks

    def get_config_summary(self) -> dict:
        """Get current chunker configuration as dictionary."""
        return {
            "tokenizer_name": self.tokenizer_name,
            "max_tokens": self.max_tokens,
            "merge_peers": self.merge_peers,
            "tokenizer_class": self.tokenizer.__class__.__name__,
        }
