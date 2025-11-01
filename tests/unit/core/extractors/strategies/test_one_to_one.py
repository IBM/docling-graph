"""
Unit tests for OneToOneStrategy.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest
from pydantic import BaseModel, Field

from docling_graph.core.extractors.strategies.one_to_one import OneToOneStrategy


class SampleModel(BaseModel):
    """Sample Pydantic model for testing."""
    name: str
    page_number: int


class TestOneToOneStrategyInitialization:
    """Tests for OneToOneStrategy initialization."""

    def test_init_with_vlm_backend(self):
        """Test initialization with VLM backend."""
        mock_backend = Mock()
        strategy = OneToOneStrategy(backend=mock_backend, docling_config="ocr")
        
        assert strategy.backend == mock_backend
        assert hasattr(strategy, "doc_processor")

    def test_init_with_llm_backend(self):
        """Test initialization with LLM backend."""
        mock_backend = Mock()
        strategy = OneToOneStrategy(backend=mock_backend, docling_config="vision")
        
        assert strategy.backend == mock_backend

    @patch("docling_graph.core.extractors.strategies.one_to_one.DocumentProcessor")
    def test_init_creates_document_processor(self, mock_processor):
        """Test that initialization creates DocumentProcessor."""
        mock_backend = Mock()
        mock_processor_instance = Mock()
        mock_processor.return_value = mock_processor_instance
        
        strategy = OneToOneStrategy(backend=mock_backend, docling_config="ocr")
        
        mock_processor.assert_called_once_with(docling_config="ocr")
        assert strategy.doc_processor == mock_processor_instance

    def test_init_default_docling_config(self):
        """Test default docling_config value."""
        mock_backend = Mock()
        with patch("docling_graph.core.extractors.strategies.one_to_one.DocumentProcessor"):
            strategy = OneToOneStrategy(backend=mock_backend)


class TestOneToOneStrategyVLMExtraction:
    """Tests for extraction with VLM backend."""

    @patch("docling_graph.core.extractors.strategies.one_to_one.is_vlm_backend")
    @patch("docling_graph.core.extractors.strategies.one_to_one.DocumentProcessor")
    def test_extract_vlm_success(self, mock_processor_class, mock_is_vlm):
        """Test successful VLM extraction."""
        mock_is_vlm.return_value = True
        mock_backend = Mock()
        mock_model1 = SampleModel(name="Page 1", page_number=1)
        mock_model2 = SampleModel(name="Page 2", page_number=2)
        mock_backend.extract_from_document.return_value = [mock_model1, mock_model2]
        
        strategy = OneToOneStrategy(backend=mock_backend)
        result = strategy.extract(source="test.pdf", template=SampleModel)
        
        assert len(result) == 2
        assert result[0].name == "Page 1"
        assert result[1].name == "Page 2"
        mock_backend.extract_from_document.assert_called_once_with("test.pdf", SampleModel)

    @patch("docling_graph.core.extractors.strategies.one_to_one.is_vlm_backend")
    @patch("docling_graph.core.extractors.strategies.one_to_one.DocumentProcessor")
    def test_extract_vlm_empty_result(self, mock_processor_class, mock_is_vlm):
        """Test VLM extraction with no results."""
        mock_is_vlm.return_value = True
        mock_backend = Mock()
        mock_backend.extract_from_document.return_value = []
        
        strategy = OneToOneStrategy(backend=mock_backend)
        result = strategy.extract(source="test.pdf", template=SampleModel)
        
        assert result == []

    @patch("docling_graph.core.extractors.strategies.one_to_one.is_vlm_backend")
    @patch("docling_graph.core.extractors.strategies.one_to_one.DocumentProcessor")
    def test_extract_vlm_with_none_results(self, mock_processor_class, mock_is_vlm):
        """Test VLM extraction filters out None results."""
        mock_is_vlm.return_value = True
        mock_backend = Mock()
        mock_model = SampleModel(name="Valid", page_number=1)
        mock_backend.extract_from_document.return_value = [mock_model, None, mock_model]
        
        strategy = OneToOneStrategy(backend=mock_backend)
        result = strategy.extract(source="test.pdf", template=SampleModel)
        
        # None should be filtered out
        assert len(result) == 2
        assert all(r is not None for r in result)


class TestOneToOneStrategyLLMExtraction:
    """Tests for extraction with LLM backend."""

    @patch("docling_graph.core.extractors.strategies.one_to_one.is_vlm_backend")
    @patch("docling_graph.core.extractors.strategies.one_to_one.is_llm_backend")
    @patch("docling_graph.core.extractors.strategies.one_to_one.DocumentProcessor")
    def test_extract_llm_success(self, mock_processor_class, mock_is_llm, mock_is_vlm):
        """Test successful LLM extraction."""
        mock_is_vlm.return_value = False
        mock_is_llm.return_value = True
        
        # Setup mock processor
        mock_processor_instance = Mock()
        mock_processor_instance.process_document.return_value = ["Page 1 content", "Page 2 content"]
        mock_processor_class.return_value = mock_processor_instance
        
        # Setup mock backend
        mock_backend = Mock()
        mock_model1 = SampleModel(name="Page 1", page_number=1)
        mock_model2 = SampleModel(name="Page 2", page_number=2)
        mock_backend.extract_from_markdown.side_effect = [mock_model1, mock_model2]
        
        strategy = OneToOneStrategy(backend=mock_backend)
        result = strategy.extract(source="test.pdf", template=SampleModel)
        
        assert len(result) == 2
        assert result[0].name == "Page 1"
        assert result[1].name == "Page 2"
        assert mock_backend.extract_from_markdown.call_count == 2

    @patch("docling_graph.core.extractors.strategies.one_to_one.is_vlm_backend")
    @patch("docling_graph.core.extractors.strategies.one_to_one.is_llm_backend")
    @patch("docling_graph.core.extractors.strategies.one_to_one.DocumentProcessor")
    def test_extract_llm_with_none_results(self, mock_processor_class, mock_is_llm, mock_is_vlm):
        """Test LLM extraction filters out None results."""
        mock_is_vlm.return_value = False
        mock_is_llm.return_value = True
        
        mock_processor_instance = Mock()
        mock_processor_instance.process_document.return_value = ["Page 1", "Page 2", "Page 3"]
        mock_processor_class.return_value = mock_processor_instance
        
        mock_backend = Mock()
        mock_model = SampleModel(name="Valid", page_number=1)
        mock_backend.extract_from_markdown.side_effect = [mock_model, None, mock_model]
        
        strategy = OneToOneStrategy(backend=mock_backend)
        result = strategy.extract(source="test.pdf", template=SampleModel)
        
        # None should be filtered out
        assert len(result) == 2

    @patch("docling_graph.core.extractors.strategies.one_to_one.is_vlm_backend")
    @patch("docling_graph.core.extractors.strategies.one_to_one.is_llm_backend")
    @patch("docling_graph.core.extractors.strategies.one_to_one.DocumentProcessor")
    def test_extract_llm_with_page_context(self, mock_processor_class, mock_is_llm, mock_is_vlm):
        """Test that LLM extraction includes page context."""
        mock_is_vlm.return_value = False
        mock_is_llm.return_value = True
        
        mock_processor_instance = Mock()
        mock_processor_instance.process_document.return_value = ["Content"]
        mock_processor_class.return_value = mock_processor_instance
        
        mock_backend = Mock()
        mock_backend.extract_from_markdown.return_value = SampleModel(name="Test", page_number=1)
        
        strategy = OneToOneStrategy(backend=mock_backend)
        strategy.extract(source="test.pdf", template=SampleModel)
        
        # Check that context includes page information
        call_kwargs = mock_backend.extract_from_markdown.call_args[1]
        assert "context" in call_kwargs
        assert "page" in call_kwargs["context"].lower()


class TestOneToOneStrategyErrorHandling:
    """Tests for error handling."""

    @patch("docling_graph.core.extractors.strategies.one_to_one.is_vlm_backend")
    @patch("docling_graph.core.extractors.strategies.one_to_one.is_llm_backend")
    @patch("docling_graph.core.extractors.strategies.one_to_one.DocumentProcessor")
    def test_extract_with_unknown_backend(self, mock_processor_class, mock_is_llm, mock_is_vlm):
        """Test extraction with unknown backend type."""
        mock_is_vlm.return_value = False
        mock_is_llm.return_value = False
        mock_backend = Mock()
        
        strategy = OneToOneStrategy(backend=mock_backend)
        
        with pytest.raises(ValueError, match="Unknown backend type"):
            strategy.extract(source="test.pdf", template=SampleModel)

    @patch("docling_graph.core.extractors.strategies.one_to_one.is_vlm_backend")
    @patch("docling_graph.core.extractors.strategies.one_to_one.DocumentProcessor")
    def test_extract_vlm_with_exception(self, mock_processor_class, mock_is_vlm):
        """Test VLM extraction when backend raises exception."""
        mock_is_vlm.return_value = True
        mock_backend = Mock()
        mock_backend.extract_from_document.side_effect = RuntimeError("Extraction failed")
        
        strategy = OneToOneStrategy(backend=mock_backend)
        
        with pytest.raises(RuntimeError, match="Extraction failed"):
            strategy.extract(source="test.pdf", template=SampleModel)

    @patch("docling_graph.core.extractors.strategies.one_to_one.is_vlm_backend")
    @patch("docling_graph.core.extractors.strategies.one_to_one.is_llm_backend")
    @patch("docling_graph.core.extractors.strategies.one_to_one.DocumentProcessor")
    def test_extract_llm_document_processing_error(
        self, mock_processor_class, mock_is_llm, mock_is_vlm
    ):
        """Test LLM extraction when document processing fails."""
        mock_is_vlm.return_value = False
        mock_is_llm.return_value = True
        
        mock_processor_instance = Mock()
        mock_processor_instance.process_document.side_effect = RuntimeError("Processing failed")
        mock_processor_class.return_value = mock_processor_instance
        
        strategy = OneToOneStrategy(backend=Mock())
        
        with pytest.raises(RuntimeError, match="Processing failed"):
            strategy.extract(source="test.pdf", template=SampleModel)


class TestOneToOneStrategyCleanup:
    """Tests for cleanup method."""

    @patch("docling_graph.core.extractors.strategies.one_to_one.DocumentProcessor")
    def test_cleanup_calls_backend_cleanup(self, mock_processor_class):
        """Test cleanup calls backend cleanup method."""
        mock_backend = Mock()
        mock_backend.cleanup = Mock()
        
        strategy = OneToOneStrategy(backend=mock_backend)
        strategy.cleanup()
        
        mock_backend.cleanup.assert_called_once()

    @patch("docling_graph.core.extractors.strategies.one_to_one.DocumentProcessor")
    def test_cleanup_calls_processor_cleanup(self, mock_processor_class):
        """Test cleanup calls processor cleanup method."""
        mock_processor_instance = Mock()
        mock_processor_instance.cleanup = Mock()
        mock_processor_class.return_value = mock_processor_instance
        
        strategy = OneToOneStrategy(backend=Mock())
        strategy.cleanup()
        
        mock_processor_instance.cleanup.assert_called_once()

    @patch("docling_graph.core.extractors.strategies.one_to_one.DocumentProcessor")
    def test_cleanup_handles_missing_cleanup_methods(self, mock_processor_class):
        """Test cleanup when components don't have cleanup method."""
        mock_backend = Mock(spec=[])  # No cleanup method
        mock_processor_instance = Mock(spec=[])
        mock_processor_class.return_value = mock_processor_instance
        
        strategy = OneToOneStrategy(backend=mock_backend)
        # Should not raise
        strategy.cleanup()

    @patch("docling_graph.core.extractors.strategies.one_to_one.DocumentProcessor")
    def test_cleanup_handles_errors(self, mock_processor_class):
        """Test cleanup handles errors gracefully."""
        mock_backend = Mock()
        mock_backend.cleanup.side_effect = RuntimeError("Cleanup failed")
        
        strategy = OneToOneStrategy(backend=mock_backend)
        # Should not raise
        strategy.cleanup()
