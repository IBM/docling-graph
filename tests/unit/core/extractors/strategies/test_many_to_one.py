"""
Tests for many-to-one extraction strategy.
"""

from typing import List
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from docling_graph.core.extractors.strategies.many_to_one import ManyToOneStrategy


class SampleModel(BaseModel):
    """Sample model for testing."""

    name: str
    value: int


@pytest.fixture
def mock_vlm_backend():
    """Create mock VLM backend."""
    backend = MagicMock()
    backend.extract_from_document = MagicMock(
        return_value=[SampleModel(name="test1", value=1), SampleModel(name="test2", value=2)]
    )
    return backend


@pytest.fixture
def mock_llm_backend():
    """Create mock LLM backend."""
    backend = MagicMock()
    backend.extract_from_markdown = MagicMock(return_value=SampleModel(name="merged", value=99))
    backend.client = MagicMock()
    backend.client.context_limit = 4096
    return backend


class TestManyToOneStrategyInitialization:
    """Test many-to-one strategy initialization."""

    @patch("docling_graph.core.extractors.strategies.many_to_one.DocumentProcessor")
    @patch("docling_graph.core.extractors.strategies.many_to_one.get_backend_type")
    def test_initialization(self, mock_get_type, mock_doc_proc, mock_vlm_backend):
        """Should initialize correctly."""
        mock_get_type.return_value = "vlm"

        strategy = ManyToOneStrategy(backend=mock_vlm_backend)

        assert strategy.backend is mock_vlm_backend


class TestManyToOneStrategyExtract:
    """Test many-to-one extraction."""

    @patch("docling_graph.core.extractors.strategies.many_to_one.is_vlm_backend")
    @patch("docling_graph.core.extractors.strategies.many_to_one.is_llm_backend")
    @patch("docling_graph.core.extractors.strategies.many_to_one.DocumentProcessor")
    @patch("docling_graph.core.extractors.strategies.many_to_one.get_backend_type")
    def test_extract_with_vlm_backend(
        self, mock_get_type, mock_doc_proc, mock_is_llm, mock_is_vlm, mock_vlm_backend
    ):
        """Should extract and merge using VLM backend."""
        mock_get_type.return_value = "vlm"
        mock_is_vlm.return_value = True
        mock_is_llm.return_value = False

        strategy = ManyToOneStrategy(backend=mock_vlm_backend)

        with patch(
            "docling_graph.core.extractors.strategies.many_to_one.merge_pydantic_models"
        ) as mock_merge:
            mock_merge.return_value = SampleModel(name="merged", value=3)
            result = strategy.extract("test.pdf", SampleModel)

        assert isinstance(result, list)

    @patch("docling_graph.core.extractors.strategies.many_to_one.DocumentProcessor")
    @patch("docling_graph.core.extractors.strategies.many_to_one.get_backend_type")
    def test_extract_returns_single_model(self, mock_get_type, mock_doc_proc, mock_vlm_backend):
        """Should return list with single merged model."""
        mock_get_type.return_value = "vlm"

        with patch(
            "docling_graph.core.extractors.strategies.many_to_one.is_vlm_backend", return_value=True
        ):
            with patch(
                "docling_graph.core.extractors.strategies.many_to_one.is_llm_backend",
                return_value=False,
            ):
                with patch(
                    "docling_graph.core.extractors.strategies.many_to_one.merge_pydantic_models"
                ) as mock_merge:
                    mock_merge.return_value = SampleModel(name="merged", value=99)

                    strategy = ManyToOneStrategy(backend=mock_vlm_backend)
                    result = strategy.extract("test.pdf", SampleModel)

        assert len(result) == 1
        assert result[0].name == "merged"

    @patch("docling_graph.core.extractors.strategies.many_to_one.DocumentProcessor")
    @patch("docling_graph.core.extractors.strategies.many_to_one.get_backend_type")
    def test_extract_handles_error(self, mock_get_type, mock_doc_proc):
        """Should handle extraction error."""
        mock_get_type.return_value = "unknown"
        mock_backend = MagicMock()

        strategy = ManyToOneStrategy(backend=mock_backend)
        result = strategy.extract("test.pdf", SampleModel)

        # Should return empty list on error
        assert result == []
