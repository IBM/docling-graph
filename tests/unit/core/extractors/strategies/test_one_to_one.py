"""
Tests for one-to-one extraction strategy.
"""

from typing import List
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from docling_graph.core.extractors.strategies.one_to_one import OneToOneStrategy


class SampleModel(BaseModel):
    """Sample model for testing."""

    name: str
    value: int


@pytest.fixture
def mock_vlm_backend():
    """Create mock VLM backend."""
    backend = MagicMock()
    backend.extract_from_document = MagicMock(return_value=[SampleModel(name="test", value=1)])
    return backend


@pytest.fixture
def mock_llm_backend():
    """Create mock LLM backend."""
    backend = MagicMock()
    backend.extract_from_markdown = MagicMock(return_value=SampleModel(name="test", value=1))
    backend.client = MagicMock()
    return backend


class TestOneToOneStrategyInitialization:
    """Test one-to-one strategy initialization."""

    @patch("docling_graph.core.extractors.strategies.one_to_one.DocumentProcessor")
    @patch("docling_graph.core.extractors.strategies.one_to_one.get_backend_type")
    def test_initialization_with_backend(self, mock_get_type, mock_doc_proc, mock_vlm_backend):
        """Should initialize with backend."""
        mock_get_type.return_value = "vlm"

        strategy = OneToOneStrategy(backend=mock_vlm_backend)

        assert strategy.backend is mock_vlm_backend

    @patch("docling_graph.core.extractors.strategies.one_to_one.DocumentProcessor")
    @patch("docling_graph.core.extractors.strategies.one_to_one.get_backend_type")
    def test_initialization_creates_doc_processor(
        self, mock_get_type, mock_doc_proc, mock_vlm_backend
    ):
        """Should create document processor."""
        mock_get_type.return_value = "vlm"

        strategy = OneToOneStrategy(backend=mock_vlm_backend)

        assert hasattr(strategy, "doc_processor")


class TestOneToOneStrategyExtract:
    """Test one-to-one extraction."""

    @patch("docling_graph.core.extractors.strategies.one_to_one.is_vlm_backend")
    @patch("docling_graph.core.extractors.strategies.one_to_one.is_llm_backend")
    @patch("docling_graph.core.extractors.strategies.one_to_one.DocumentProcessor")
    @patch("docling_graph.core.extractors.strategies.one_to_one.get_backend_type")
    def test_extract_with_vlm_backend(
        self, mock_get_type, mock_doc_proc, mock_is_llm, mock_is_vlm, mock_vlm_backend
    ):
        """Should extract using VLM backend."""
        mock_get_type.return_value = "vlm"
        mock_is_vlm.return_value = True
        mock_is_llm.return_value = False

        strategy = OneToOneStrategy(backend=mock_vlm_backend)
        models, document = strategy.extract("test.pdf", SampleModel)

        assert isinstance(models, list)
        mock_vlm_backend.extract_from_document.assert_called_once()

    @patch("docling_graph.core.extractors.strategies.one_to_one.DocumentProcessor")
    @patch("docling_graph.core.extractors.strategies.one_to_one.get_backend_type")
    def test_extract_returns_tuple(self, mock_get_type, mock_doc_proc, mock_vlm_backend):
        """Should return tuple of (models, document)."""
        mock_get_type.return_value = "vlm"

        with patch(
            "docling_graph.core.extractors.strategies.one_to_one.is_vlm_backend", return_value=True
        ):
            with patch(
                "docling_graph.core.extractors.strategies.one_to_one.is_llm_backend",
                return_value=False,
            ):
                strategy = OneToOneStrategy(backend=mock_vlm_backend)
                result = strategy.extract("test.pdf", SampleModel)

        assert isinstance(result, tuple)
        assert len(result) == 2

    @patch("docling_graph.core.extractors.strategies.one_to_one.DocumentProcessor")
    @patch("docling_graph.core.extractors.strategies.one_to_one.get_backend_type")
    def test_extract_default_backend_fallback(self, mock_get_type, mock_doc_proc, mock_vlm_backend):
        """Should fallback to VLM when backend type is unknown."""
        mock_get_type.return_value = "unknown"
        mock_vlm_backend.extract_from_document.return_value = [SampleModel(name="test", value=1)]

        with patch(
            "docling_graph.core.extractors.strategies.one_to_one.is_vlm_backend", return_value=True
        ):
            with patch(
                "docling_graph.core.extractors.strategies.one_to_one.is_llm_backend",
                return_value=False,
            ):
                strategy = OneToOneStrategy(backend=mock_vlm_backend)
                models, document = strategy.extract("test.pdf", SampleModel)

        assert isinstance(models, list)
        assert len(models) > 0
