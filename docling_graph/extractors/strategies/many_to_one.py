"""
Many-to-one extraction strategy.
Processes entire document and returns single consolidated model.
"""

from pydantic import BaseModel
from typing import Type, List
from rich import print

from ...protocols import is_vlm_backend, is_llm_backend, get_backend_type
from ..document_processor import DocumentProcessor
from ..extractor_base import BaseExtractor
from ..utils import merge_pydantic_models


class ManyToOneStrategy(BaseExtractor):
    """Many-to-one extraction strategy.

    Extracts one consolidated model from entire document using Protocol-based type checking.
    """

    def __init__(self, backend, docling_config: str = "default"):
        """Initialize with a backend (VlmBackend or LlmBackend).

        Args:
            backend: Extraction backend instance implementing either
                     ExtractionBackendProtocol or TextExtractionBackendProtocol.
            docling_config: Docling pipeline configuration ('ocr' or 'vision').
        """
        self.backend = backend
        self.doc_processor = DocumentProcessor(docling_config=docling_config)

        backend_type = get_backend_type(backend)
        print(f"[blue][ManyToOneStrategy][/blue] Initialized with {backend_type.upper()} backend: "
              f"[cyan]{backend.__class__.__name__}[/cyan]")

    def extract(self, source: str, template: Type[BaseModel]) -> List[BaseModel]:
        """Extract data using many-to-one strategy.

        For VLM: Extracts all pages then intelligently merges them
        For LLM: Two approaches:
            1. Try full document extraction first (fast path)
            2. If too large, extract page-by-page and merge (fallback)

        Args:
            source: Path to the source document.
            template: Pydantic model template to extract into.

        Returns:
            List containing single merged model instance.
        """
        # Use Protocol-based type checking
        if is_vlm_backend(self.backend):
            return self._extract_with_vlm(source, template)
        elif is_llm_backend(self.backend):
            return self._extract_with_llm(source, template)
        else:
            backend_class = self.backend.__class__.__name__
            raise TypeError(
                f"Backend {backend_class} does not implement required protocols. "
                f"Must implement either ExtractionBackendProtocol or TextExtractionBackendProtocol."
            )

    def _extract_with_vlm(self, source: str, template: Type[BaseModel]) -> List[BaseModel]:
        """Extract using VLM backend with page-by-page merging."""
        print("[blue][ManyToOneStrategy][/blue] Using VLM for multi-page document")

        models = self.backend.extract_from_document(source, template)

        if not models:
            print("[yellow]Warning:[/yellow] VLM extraction returned no models")
            return []

        if len(models) == 1:
            print("[blue][ManyToOneStrategy][/blue] Single page extracted")
            return models

        # Merge multiple page extractions
        print(f"[blue][ManyToOneStrategy][/blue] Merging [cyan]{len(models)}[/cyan] page extractions...")
        merged_model = merge_pydantic_models(models, template)

        if merged_model:
            print("[blue][ManyToOneStrategy][/blue] Successfully merged all pages")
            return [merged_model]
        else:
            print("[yellow]Warning:[/yellow] Merge failed, returning first page")
            return [models[0]]

    def _extract_with_llm(self, source: str, template: Type[BaseModel]) -> List[BaseModel]:
        """Extract using LLM backend with intelligent strategy selection.

        Strategy:
            1. Try full document extraction (if context allows)
            2. If document too large, extract page-by-page and merge
        """
        document = self.doc_processor.convert_to_markdown(source)

        # Check if we should try full document or page-by-page
        # Use hasattr to check for context_limit (Protocol doesn't enforce it as property)
        if hasattr(self.backend.client, 'context_limit'):
            context_limit = self.backend.client.context_limit
            full_markdown = self.doc_processor.extract_full_markdown(document)
            estimated_tokens = len(full_markdown) / 3.5  # Rough estimate

            if estimated_tokens < (context_limit * 0.9):  # Use 90% as safe threshold
                # Try full document extraction
                print(f"[blue][ManyToOneStrategy][/blue] Document fits context "
                      f"({int(estimated_tokens)} tokens), processing as whole")
                return self._extract_full_document(full_markdown, template)
            else:
                # Document too large, use page-by-page with merging
                print(f"[blue][ManyToOneStrategy][/blue] Document too large "
                      f"({int(estimated_tokens)} tokens), using page-by-page extraction")
                return self._extract_pages_and_merge(document, template)
        else:
            # No context limit info, try full document
            full_markdown = self.doc_processor.extract_full_markdown(document)
            return self._extract_full_document(full_markdown, template)

    def _extract_full_document(self, full_markdown: str, template: Type[BaseModel]) -> List[BaseModel]:
        """Extract from complete document markdown."""
        model = self.backend.extract_from_markdown(
            markdown=full_markdown,
            template=template,
            context="full document"
        )

        if model:
            print("[blue][ManyToOneStrategy][/blue] Extracted consolidated model from full document")
            return [model]
        else:
            print("[yellow]Warning:[/yellow] Failed to extract from full document")
            return []

    def _extract_pages_and_merge(self, document, template: Type[BaseModel]) -> List[BaseModel]:
        """Extract from individual pages and merge results."""
        page_markdowns = self.doc_processor.extract_page_markdowns(document)
        print(f"[blue][ManyToOneStrategy][/blue] Extracting from {len(page_markdowns)} pages individually...")

        extracted_models = []
        for page_num, page_md in enumerate(page_markdowns, 1):
            print(f"[blue][ManyToOneStrategy][/blue] Processing page {page_num}/{len(page_markdowns)}")

            model = self.backend.extract_from_markdown(
                markdown=page_md,
                template=template,
                context=f"page {page_num}"
            )

            if model:
                extracted_models.append(model)

        if not extracted_models:
            print("[yellow]Warning:[/yellow] No pages extracted successfully")
            return []

        if len(extracted_models) == 1:
            print("[blue][ManyToOneStrategy][/blue] Single page extracted")
            return extracted_models

        # Merge all page extractions
        print(f"[blue][ManyToOneStrategy][/blue] Merging [cyan]{len(extracted_models)}[/cyan] page extractions...")
        merged_model = merge_pydantic_models(extracted_models, template)

        if merged_model:
            print("[blue][ManyToOneStrategy][/blue] Successfully merged all pages")
            return [merged_model]
        else:
            print("[yellow]Warning:[/yellow] Merge failed, returning first page")
            return [extracted_models[0]]
