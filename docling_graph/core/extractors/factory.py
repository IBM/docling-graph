"""
Factory for creating extractors based on configuration.
"""

from typing import Any, Literal

from rich import print as rich_print

from ...llm_clients.base import BaseLlmClient
from ...protocols import Backend
from .backends.llm_backend import LlmBackend
from .backends.vlm_backend import VlmBackend
from .extractor_base import BaseExtractor
from .strategies.bottom_up import BottomUpStrategy
from .strategies.many_to_one import ManyToOneStrategy
from .strategies.one_to_one import OneToOneStrategy


class ExtractorFactory:
    """Factory for creating the right extractor combination."""

    @staticmethod
    def create_extractor(
        processing_mode: Literal["one-to-one", "many-to-one", "bottom-up"],
        backend_name: Literal["vlm", "llm"],
        model_name: str | None = None,
        llm_client: BaseLlmClient | None = None,
        docling_config: str = "ocr",
        use_chunking: bool = True,
        llm_consolidation: bool = False,
        min_text_length: int = 20,
        enable_caching: bool = False,
        enable_rate_limiting: bool = True,
        max_retries: int = 3,
        enable_metrics: bool = True,
    ) -> BaseExtractor:
        """
        Create an extractor based on configuration.

        Args:
            processing_mode (str): 'one-to-one', 'many-to-one', or 'bottom-up'
            backend_name (str): 'vlm' or 'llm'
            model_name (str): Model name for VLM (optional)
            llm_client (BaseLlmClient): LLM client instance (optional)
            docling_config (str): Docling pipeline configuration ('default' or 'vlm')
            llm_consolidation (bool): Whether to use LLM consolidation.
            use_chunking (bool): Whether to use chunking.
            min_text_length (int): Minimum text length for bottom-up text slots (default: 20)
            enable_caching (bool): Enable extraction caching for bottom-up (default: False)
            enable_rate_limiting (bool): Enable rate limiting for bottom-up (default: True)
            max_retries (int): Maximum retries for failed API calls in bottom-up (default: 3)
            enable_metrics (bool): Enable metrics collection for bottom-up (default: True)

        Returns:
            BaseExtractor: Configured extractor instance.
        """
        rich_print("[blue][ExtractorFactory][/blue] Creating extractor:")
        rich_print(f" • Mode: [cyan]{processing_mode}[/cyan]")
        rich_print(f" • Type: [cyan]{backend_name}[/cyan]")
        rich_print(f" • Docling: [cyan]{docling_config}[/cyan]")
        # --- ADDED PRINT STATEMENTS START ---
        if backend_name == "llm":
            rich_print(f" • Consolidation: [cyan]{llm_consolidation}[/cyan]")
        rich_print(f" • Chunking: [cyan]{use_chunking}[/cyan]")
        # --- ADDED PRINT STATEMENTS END ---

        # Create backend instance
        backend_obj: Backend
        if backend_name == "vlm":
            if not model_name:
                raise ValueError("VLM requires model_name parameter")
            backend_obj = VlmBackend(model_name=model_name)
        elif backend_name == "llm":
            if not llm_client:
                raise ValueError("LLM requires llm_client parameter")
            backend_obj = LlmBackend(llm_client=llm_client)
        else:
            raise ValueError(f"Unknown backend: {backend_name}")

        # Create strategy with docling_config
        extractor: BaseExtractor

        if processing_mode == "one-to-one":
            # OneToOneStrategy only takes backend and docling_config
            # It doesn't use chunking or consolidation args
            extractor = OneToOneStrategy(
                backend=backend_obj,
                docling_config=docling_config,
            )
        elif processing_mode == "many-to-one":
            # Build args specifically for ManyToOne
            strategy_args: dict[str, Any] = {
                "backend": backend_obj,
                "docling_config": docling_config,
                "use_chunking": use_chunking,
            }
            if backend_name == "llm":
                strategy_args["llm_consolidation"] = llm_consolidation

            extractor = ManyToOneStrategy(**strategy_args)
        elif processing_mode == "bottom-up":
            # BottomUpStrategy requires LLM backend
            if backend_name != "llm":
                raise ValueError("Bottom-up mode requires LLM backend")
            if not llm_client:
                raise ValueError("Bottom-up mode requires llm_client parameter")

            rich_print(f" • Min text length: [cyan]{min_text_length}[/cyan]")
            rich_print(f" • Caching: [cyan]{enable_caching}[/cyan]")
            rich_print(f" • Rate limiting: [cyan]{enable_rate_limiting}[/cyan]")
            rich_print(f" • Max retries: [cyan]{max_retries}[/cyan]")
            rich_print(f" • Metrics: [cyan]{enable_metrics}[/cyan]")

            extractor = BottomUpStrategy(
                backend=backend_obj,
                docling_config=docling_config,
                min_text_length=min_text_length,
                enable_caching=enable_caching,
                enable_rate_limiting=enable_rate_limiting,
                max_retries=max_retries,
                enable_metrics=enable_metrics,
            )
        else:
            raise ValueError(f"Unknown processing_mode: {processing_mode}")

        rich_print(
            f"[blue][ExtractorFactory][/blue] Created [green]{extractor.__class__.__name__}[/green]"
        )
        return extractor
