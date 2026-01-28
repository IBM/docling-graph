"""
LLM (Language Model) extraction backend.
Handles document extraction using LLM models (local or API) with model-aware prompting.
"""

import gc
import json
import logging
from types import SimpleNamespace
from typing import List, Optional, Type

from pydantic import BaseModel, ValidationError
from rich import print as rich_print

from ....llm_clients.config import ModelCapability, ModelConfigLike, detect_model_capability
from ....llm_clients.prompts import (
    get_consolidation_prompt,
    get_context_aware_prompt,
    get_extraction_prompt,
)
from ....protocols import LLMClientProtocol
from ..delta_models import DeltaOperation

logger = logging.getLogger(__name__)


class LlmBackend:
    """Backend for LLM-based extraction with model-aware prompting and multi-turn consolidation."""

    def __init__(self, llm_client: LLMClientProtocol) -> None:
        """
        Initialize LLM backend with a client and model configuration.

        Args:
            llm_client (LLMClientProtocol): LLM client instance
        """
        self.client = llm_client

        raw_model_config = getattr(llm_client, "model_config", None)
        if raw_model_config:
            self.model_config: ModelConfigLike = raw_model_config
        else:
            # Fallback: dynamically detect capability when a full model_config
            # is not available (e.g., in tests or lightweight clients).
            context_limit = getattr(llm_client, "context_limit", 8192)
            # Use only the public `model` attribute; tests expect an empty
            # string when it is unset instead of a mocked model_id.
            model_name = getattr(llm_client, "model", "") or ""
            capability = detect_model_capability(context_limit, model_name, None)
            self.model_config = SimpleNamespace(
                capability=capability,
                context_limit=context_limit,
                supports_chain_of_density=capability == ModelCapability.ADVANCED,
            )
        model_attr = getattr(llm_client, "model", None) or getattr(llm_client, "model_id", None)

        provider_id = getattr(self.model_config, "provider_id", "unknown")
        model_id = getattr(self.model_config, "model_id", "unknown")
        logger.info(
            "Loaded model config: provider=%s, model=%s, capability=%s",
            provider_id,
            model_id,
            self.model_config.capability.value,
        )

        chain_of_density = getattr(self.model_config, "supports_chain_of_density", False)
        context_limit = self.model_config.context_limit
        context_limit_display = (
            f"{context_limit:,}" if isinstance(context_limit, int) else str(context_limit)
        )
        rich_print(
            f"[blue][LlmBackend][/blue] Initialized with:\n"
            f"  • Client: [cyan]{self.client.__class__.__name__}[/cyan]\n"
            f"  • Model: [cyan]{model_attr or 'unknown'}[/cyan]\n"
            f"  • Model capability: [yellow]{self.model_config.capability.value}[/yellow]\n"
            f"  • Context limit: {context_limit_display} tokens\n"
            f"  • Chain of Density: {'enabled' if chain_of_density else 'disabled'}"
        )

    def extract_from_markdown(
        self,
        markdown: str,
        template: Type[BaseModel],
        context: str = "document",
        is_partial: bool = False,
    ) -> BaseModel | None:
        """
        Extract structured data from markdown content using LLM with model-aware prompting.

        Args:
            markdown (str): Markdown content to extract from.
            template (Type[BaseModel]): Pydantic model template.
            context (str): Context description for the extraction (e.g., "page 1", "full document").
            is_partial (bool): If True, use the partial/chunk-based prompt.

        Returns:
            Optional[BaseModel]: Extracted and validated Pydantic model instance, or None if failed.
        """
        rich_print(
            f"[blue][LlmBackend][/blue] Extracting from {context} ([cyan]{len(markdown)}[/cyan] chars)"
        )

        # Validation for empty markdown
        if not markdown or len(markdown.strip()) == 0:
            rich_print(
                f"[red]Error:[/red] Extracted markdown is empty for {context}. Cannot proceed with LLM extraction."
            )
            return None

        try:
            # Get the Pydantic schema as JSON
            schema_json = json.dumps(template.model_json_schema(), indent=2)

            # Generate prompt with model configuration
            prompt = get_extraction_prompt(
                markdown_content=markdown,
                schema_json=schema_json,
                is_partial=is_partial,
                model_config=self.model_config,
            )

            # Call LLM with correct method name
            parsed_json = self.client.get_json_response(prompt=prompt, schema_json=schema_json)

            if not parsed_json:
                rich_print(
                    f"[yellow]Warning:[/yellow] No valid JSON returned from LLM for {context}"
                )
                return None

            # Use model_validate for proper Pydantic validation
            try:
                validated_model = template.model_validate(parsed_json)
                rich_print(f"[blue][LlmBackend][/blue] Successfully extracted data from {context}")
                return validated_model

            except ValidationError as e:
                # Detailed error reporting
                rich_print(
                    f"[blue][LlmBackend][/blue] [yellow]Validation Error for {context}:[/yellow]"
                )
                rich_print("  The data extracted by the LLM does not match your Pydantic template.")
                rich_print("[red]Details:[/red]")
                for error in e.errors():
                    loc = " -> ".join(map(str, error["loc"]))
                    rich_print(f"  - [bold magenta]{loc}[/bold magenta]: [red]{error['msg']}[/red]")
                rich_print(f"\n[yellow]Extracted Data (raw):[/yellow]\n{parsed_json}\n")
                return None

        except Exception as e:
            rich_print(
                f"[red]Error during LLM extraction for {context}:[/red] {type(e).__name__}: {e}"
            )
            return None

    def extract_with_context(
        self,
        markdown: str,
        template: Type[BaseModel],
        registry_str: str,
        context: str = "document",
        is_partial: bool = True,
    ) -> DeltaOperation | None:
        """
        Extract delta operations using context-aware prompting.

        Args:
            markdown (str): Markdown content to extract from.
            template (Type[BaseModel]): Pydantic model template.
            registry_str (str): Compact registry string for entity IDs.
            context (str): Context description for the extraction.
            is_partial (bool): If True, use the partial/chunk-based prompt.

        Returns:
            Optional[DeltaOperation]: Validated delta operation model, or None if failed.
        """
        rich_print(
            f"[blue][LlmBackend][/blue] Extracting delta from {context} "
            f"([cyan]{len(markdown)}[/cyan] chars)"
        )

        if not markdown or len(markdown.strip()) == 0:
            rich_print(
                f"[red]Error:[/red] Extracted markdown is empty for {context}. Cannot proceed."
            )
            return None

        try:
            schema_json = json.dumps(template.model_json_schema(), indent=2)
            delta_schema_json = json.dumps(DeltaOperation.model_json_schema(), indent=2)

            prompt = get_context_aware_prompt(
                markdown_content=markdown,
                schema_json=schema_json,
                registry_content=registry_str,
                delta_schema_json=delta_schema_json,
                is_partial=is_partial,
                model_config=self.model_config,
            )

            parsed_json = self.client.get_json_response(
                prompt=prompt, schema_json=delta_schema_json
            )

            if not parsed_json:
                rich_print(
                    f"[yellow]Warning:[/yellow] No valid JSON returned from LLM for {context}"
                )
                return None

            try:
                validated_delta = DeltaOperation.model_validate(parsed_json)
                rich_print(f"[blue][LlmBackend][/blue] Successfully extracted delta from {context}")
                return validated_delta

            except ValidationError as e:
                rich_print(
                    f"[blue][LlmBackend][/blue] [yellow]Validation Error for {context}:[/yellow]"
                )
                rich_print("  The delta extracted by the LLM does not match the delta schema.")
                rich_print("[red]Details:[/red]")
                for error in e.errors():
                    loc = " -> ".join(map(str, error["loc"]))
                    rich_print(f"  - [bold magenta]{loc}[/bold magenta]: [red]{error['msg']}[/red]")
                rich_print(f"\n[yellow]Extracted Data (raw):[/yellow]\n{parsed_json}\n")
                return None

        except Exception as e:
            rich_print(
                f"[red]Error during delta extraction for {context}:[/red] {type(e).__name__}: {e}"
            )
            return None

    def consolidate_from_pydantic_models(
        self,
        raw_models: List[BaseModel],
        programmatic_model: BaseModel,
        template: Type[BaseModel],
    ) -> BaseModel | None:
        """
        Uses an LLM to consolidate multiple extracted models with multi-turn support.

        Handles both single-prompt and Chain of Density (multi-turn) consolidation
        based on model capability.

        Args:
            raw_models: The list of raw models from each batch/page.
            programmatic_model: The programmatically merged model (as a draft).
            template: The Pydantic model template to validate against.

        Returns:
            A single, validated, LLM-consolidated Pydantic model, or None if failed.
        """
        capability_str = self.model_config.capability.value if self.model_config else "unknown"
        rich_print(
            f"[blue][LlmBackend][/blue] Consolidating {len(raw_models)} models "
            f"(capability: {capability_str})..."
        )

        try:
            schema_json = json.dumps(template.model_json_schema(), indent=2)

            # Get prompt(s) - may be string or list for Chain of Density
            prompts = get_consolidation_prompt(
                schema_json=schema_json,
                raw_models=raw_models,
                programmatic_model=programmatic_model,
                model_config=self.model_config,
            )

            # Handle multi-turn consolidation (Chain of Density)
            if isinstance(prompts, list):
                rich_print(
                    f"[blue][LlmBackend][/blue] Using Chain of Density ({len(prompts)} stages)..."
                )

                # Stage 1: Initial merge
                stage1_result = self.client.get_json_response(
                    prompt=prompts[0], schema_json=schema_json
                )

                if not stage1_result:
                    rich_print("[yellow]Warning:[/yellow] Stage 1 consolidation failed")
                    return None

                # Stage 2: Refinement (inject stage1 result)
                raw_jsons = "\n\n---\n\n".join(m.model_dump_json(indent=2) for m in raw_models)
                stage2_prompt = prompts[1].format(
                    schema=schema_json,
                    stage1_result=json.dumps(stage1_result, indent=2),
                    originals=raw_jsons,
                )

                final_result = self.client.get_json_response(
                    prompt=stage2_prompt, schema_json=schema_json
                )

                if not final_result:
                    rich_print(
                        "[yellow]Warning:[/yellow] Stage 2 refinement failed, using stage 1 result"
                    )
                    parsed_json = stage1_result
                else:
                    parsed_json = final_result

            else:
                # Single-turn consolidation
                parsed_json = self.client.get_json_response(prompt=prompts, schema_json=schema_json)

            if not parsed_json:
                rich_print("[yellow]Warning:[/yellow] LLM consolidation returned no valid JSON.")
                return None

            # Use model_validate for proper Pydantic validation
            try:
                validated_model = template.model_validate(parsed_json)
                rich_print(
                    "[blue][LlmBackend][/blue] Successfully consolidated and validated model."
                )
                return validated_model

            except ValidationError as e:
                rich_print(
                    "[blue][LlmBackend][/blue] [yellow]Validation Error during consolidation:[/yellow]"
                )
                rich_print(
                    "  The data consolidated by the LLM does not match your Pydantic template."
                )
                rich_print("[red]Details:[/red]")

                for error in e.errors():
                    loc = " -> ".join(map(str, error["loc"]))
                    rich_print(f"  - [bold magenta]{loc}[/bold magenta]: [red]{error['msg']}[/red]")
                rich_print(f"\n[yellow]Consolidated Data (raw):[/yellow]\n{parsed_json}\n")
                return None

        except Exception as e:
            rich_print(f"[red]Error during LLM consolidation:[/red] {type(e).__name__}: {e}")
            return None

    def cleanup(self) -> None:
        """
        Clean up LLM client resources.

        Note: Most LLM clients use stateless HTTP APIs and don't require cleanup.
        This method is provided for consistency with VlmBackend and handles any
        clients that may have cleanup methods.
        """
        try:
            # Release the client reference
            if hasattr(self, "client"):
                # If the client has its own cleanup method, call it
                # Use getattr to avoid type checker issues with protocol
                cleanup_fn = getattr(self.client, "cleanup", None)
                if callable(cleanup_fn):
                    cleanup_fn()
                del self.client

            # Force garbage collection
            gc.collect()

            rich_print("[blue][LlmBackend][/blue] [green]Cleaned up resources[/green]")

        except Exception as e:
            rich_print(f"[blue][LlmBackend][/blue] [yellow]Warning during cleanup:[/yellow] {e}")
