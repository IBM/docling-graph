"""
Factory for creating extractors based on configuration.
"""

from rich import print

from .strategies.one_to_one import OneToOneStrategy
from .strategies.many_to_one import ManyToOneStrategy
from .backends.vlm_backend import VlmBackend
from .backends.llm_backend import LlmBackend
from ..llm_clients.base import BaseLlmClient


class ExtractorFactory:
    """Factory for creating the right extractor combination."""
    
    @staticmethod
    def create_extractor(processing_mode: str, model_type: str,
                        model_name: str = None, llm_client: BaseLlmClient = None,
                        docling_config: str = "default"):  # ADD THIS PARAMETER
        """
        Create an extractor based on configuration.
        
        Args:
            processing_mode (str): 'one-to-one' or 'many-to-one'
            model_type (str): 'vlm' or 'llm'
            model_name (str): Model name for VLM (optional)
            llm_client (BaseLlmClient): LLM client instance (optional)
            docling_config (str): Docling pipeline configuration ('default' or 'vlm')
        
        Returns:
            BaseExtractor: Configured extractor instance.
        """
        print(f"[ExtractorFactory] Creating extractor:")
        print(f"  Mode: [cyan]{processing_mode}[/cyan]")
        print(f"  Type: [cyan]{model_type}[/cyan]")
        print(f"  Docling: [cyan]{docling_config}[/cyan]")
        
        # Create backend
        if model_type == "vlm":
            if not model_name:
                raise ValueError("VLM requires model_name parameter")
            backend = VlmBackend(model_name=model_name)
        elif model_type == "llm":
            if not llm_client:
                raise ValueError("LLM requires llm_client parameter")
            backend = LlmBackend(llm_client=llm_client)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        # Create strategy with docling_config
        if processing_mode == "one-to-one":
            extractor = OneToOneStrategy(backend=backend, docling_config=docling_config)
        elif processing_mode == "many-to-one":
            extractor = ManyToOneStrategy(backend=backend, docling_config=docling_config)
        else:
            raise ValueError(f"Unknown processing_mode: {processing_mode}")
        
        print(f"[ExtractorFactory] Created [green]{extractor.__class__.__name__}[/green]")
        return extractor
