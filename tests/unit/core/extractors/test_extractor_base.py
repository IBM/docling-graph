"""
Tests for extractor base class.
"""

from abc import ABC
from typing import List, Tuple, Type

import pytest
from docling_core.types.doc import DoclingDocument
from pydantic import BaseModel

from docling_graph.core.extractors.extractor_base import BaseExtractor


# Test Models
class SampleExtractModel(BaseModel):
    """Sample model for testing."""

    name: str
    value: int


class ConcreteExtractor(BaseExtractor):
    """Concrete implementation for testing."""

    def extract(
        self, source: str, template: Type[BaseModel]
    ) -> Tuple[List[BaseModel], DoclingDocument | None]:
        """Simple extract implementation."""
        return [template(name="test", value=1)], None


class TestBaseExtractor:
    """Test BaseExtractor abstract class."""

    def test_base_extractor_is_abstract(self):
        """Should not be able to instantiate BaseExtractor."""
        with pytest.raises(TypeError):
            BaseExtractor()

    def test_concrete_extractor_can_be_instantiated(self):
        """Concrete implementation should be instantiable."""
        extractor = ConcreteExtractor()
        assert extractor is not None

    def test_extract_method_is_abstract(self):
        """extract method should be abstract."""
        assert hasattr(BaseExtractor, "extract")
        assert hasattr(BaseExtractor.extract, "__isabstractmethod__")
        assert BaseExtractor.extract.__isabstractmethod__ is True

    def test_extract_method_signature(self):
        """Extract method should accept source and template."""
        extractor = ConcreteExtractor()
        models, _document = extractor.extract("test.pdf", SampleExtractModel)

        assert isinstance(models, list)
        assert len(models) > 0

    def test_extract_returns_tuple(self):
        """Extract should return tuple of (list of models, document)."""
        extractor = ConcreteExtractor()
        result = extractor.extract("test.pdf", SampleExtractModel)

        assert isinstance(result, tuple)
        assert len(result) == 2
        models, _document = result
        assert isinstance(models, list)
        assert all(isinstance(m, BaseModel) for m in models)
