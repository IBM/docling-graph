"""
Base extractor interface for all extraction strategies.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Type

from docling_core.types.doc import DoclingDocument
from pydantic import BaseModel


class BaseExtractor(ABC):
    """Abstract base class for all extraction strategies."""

    @abstractmethod
    def extract(
        self, source: str, template: Type[BaseModel]
    ) -> Tuple[List[BaseModel], DoclingDocument | None]:
        """
        Extracts structured data from a source document based on a Pydantic template.

        Args:
            source (str): The file path to the document.
            template (Type[BaseModel]): The Pydantic model to extract into.

        Returns:
            Tuple[List[BaseModel], Optional[DoclingDocument]]: A tuple containing:
                - List of Pydantic model instances:
                    - For "One-to-One", this list may contain N models (one per page).
                    - For "Many-to-One", this list will contain 1 model.
                - The DoclingDocument object used during extraction (or None if extraction failed).
        """
