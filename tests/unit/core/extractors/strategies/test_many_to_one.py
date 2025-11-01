"""
Unit tests for ManyToOneStrategy.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest
from pydantic import BaseModel, Field

from docling_graph.core.extractors.strategies.many_to_one import ManyToOneStrategy


class SampleModel(BaseModel):
    """Sample Pydantic model for testing."""
    title: str
    summary: str
    page_count: int = 0


class TestManyToOneStrategyInitialization:
    """Tests for ManyToOneStrategy initialization."""

    def test_init_with_vlm_backend(self):
        """Test initialization with VLM backend."""
        mock_backend = Mock()
        strategy = ManyToOneStrategy(backend=mock_backend, docling_config="ocr")
        
        assert strategy.backend == mock_backend
        assert hasattr(strategy, "doc_processor")

    def test_init_with_llm_backend(self):
        """Test initialization with LLM backend."""
        mock_backend = Mock()
        strategy = ManyToOneStrategy(backend=mock_backend, docling_config="vision")
        
        assert strategy.backend == mock_backend

    @patch("docling_graph.core.extractors.strategies.many_to_one.DocumentProcessor")
    def test_init_creates_document_processor(self, mock_processor):
        """Test that initialization creates DocumentProcessor."""
        mock_backend = Mock()
        mock_processor_instance = Mock()
        mock_processor.return_value = mock_processor_instance
        
        strategy = ManyToOneStrategy(backend=mock_backend, docling_config="ocr")
        
        mock_processor.assert_called_once_with(docling_config="ocr")
        assert strategy.doc_processor == mock_processor_instance


class TestManyToOneStrategyVLMExtraction:
    """Tests for extraction with VLM backend."""

    @patch("docling_graph.core.extractors.strategies.many_to_one.is_vlm_backend")
    @patch("docling_graph.core.extractors.strategies.many_to_one.DocumentProcessor")
    def test_extract_vlm_success_single_result(self, mock_processor_class, mock_is_vlm):
        """Test successful VLM extraction returning single model."""
        mock_is_vlm.return_value = True
        mock_backend = Mock()
        mock_model = SampleModel(title="Document", summary="Summary", page_count=5)
        mock_backend.extract_from_document.return_value = [mock_model]
        
        strategy = ManyToOneStrategy(backend=mock_backend)
        result = strategy.extract(source="test.pdf", template=SampleModel)
        
        assert len(result) == 1
        assert result[0].title == "Document"
        mock_backend.extract_from_document.assert_called_once_with("test.pdf", SampleModel)

    @patch("docling_graph.core.extractors.strategies.many_to_one.merge_pydantic_models")
    @patch("docling_graph.core.extractors.strategies.many_to_one.is_vlm_backend")
    @patch("docling_graph.core.extractors.strategies.many_to_one.DocumentProcessor")
    def test_extract_vlm_multiple_results_merged(
        self, mock_processor_class, mock_is_vlm, mock_merge
    ):
        """Test VLM extraction with multiple results that get merged."""
        mock_is_vlm.return_value = True
        mock_backend = Mock()
        mock_model1 = SampleModel(title="Part 1", summary="First part", page_count=2)
        mock_model2 = SampleModel(title="Part 2", summary="Second part", page_count=3)
        mock_backend.extract_from_document.return_value = [mock_model1, mock_model2]
        
        merged_model = SampleModel(title="Complete", summary="Merged summary", page_count=5)
        mock_merge.return_value = merged_model
        
        strategy = ManyToOneStrategy(backend=mock_backend)
        result = strategy.extract(source="test.pdf", template=SampleModel)
        
        assert len(result) == 1
        assert result[0].title == "Complete"
        mock_merge.assert_called_once_with([mock_model1, mock_model2], SampleModel)

    @patch("docling_graph.core.extractors.strategies.many_to_one.is_vlm_backend")
    @patch("docling_graph.core.extractors.strategies.many_to_one.DocumentProcessor")
    def test_extract_vlm_empty_result(self, mock_processor_class, mock_is_vlm):
        """Test VLM extraction with no results."""
        mock_is_vlm.return_value = True
        mock_backend = Mock()
        mock_backend.extract_from_document.return_value = []
        
        strategy = ManyToOneStrategy(backend=mock_backend)
        result = strategy.extract(source="test.pdf", template=SampleModel)
        
        assert result == []


class TestManyToOneStrategyLLMExtraction:
    """Tests for extraction with LLM backend."""

    @patch("docling_graph.core.extractors.strategies.many_to_one.is_vlm_backend")
    @patch("docling_graph.core.extractors.strategies.many_to_one.is_llm_backend")
    @patch("docling_graph.core.extractors.strategies.many_to_one.DocumentProcessor")
    def test_extract_llm_success(self, mock_processor_class, mock_is_llm, mock_is_vlm):
        """Test successful LLM extraction."""
        mock_is_vlm.return_value = False
        mock_is_llm.return_value = True
        
        # Setup mock processor to return full document markdown
        mock_processor_instance = Mock()
        mock_document = Mock()
        mock_processor_instance.convert_to_markdown.return_value = mock_document
        mock_processor_instance.extract_full_markdown.return_value = "Full document content"
        mock_processor_class.return_value = mock_processor_instance
        
        # Setup mock backend
        mock_backend = Mock()
        mock_model = SampleModel(title="Document", summary="Summary", page_count=10)
        mock_backend.extract_from_markdown.return_value = mock_model
        
        strategy = ManyToOneStrategy(backend=mock_backend)
        result = strategy.extract(source="test.pdf", template=SampleModel)
        
        assert len(result) == 1
        assert result[0].title == "Document"
        mock_backend.extract_from_markdown.assert_called_once()

    @patch("docling_graph.core.extractors.strategies.many_to_one.is_vlm_backend")
    @patch("docling_graph.core.extractors.strategies.many_to_one.is_llm_backend")
    @patch("docling_graph.core.extractors.strategies.many_to_one.DocumentProcessor")
    def test_extract_llm_with_context(self, mock_processor_class, mock_is_llm, mock_is_vlm):
        """Test that LLM extraction includes full document context."""
        mock_is_vlm.return_value = False
        mock_is_llm.return_value = True
        
        mock_processor_instance = Mock()
        mock_document = Mock()
        mock_processor_instance.convert_to_markdown.return_value = mock_document
        mock_processor_instance.extract_full_markdown.return_value = "Content"
        mock_processor_class.return_value = mock_processor_instance
        
        mock_backend = Mock()
        mock_backend.extract_from_markdown.return_value = SampleModel(
            title="Test", summary="Test", page_count=1
        )
        
        strategy = ManyToOneStrategy(backend=mock_backend)
        strategy.extract(source="test.pdf", template=SampleModel)
        
        # Check that context indicates full document
        call_kwargs = mock_backend.extract_from_markdown.call_args[1]
        assert "context" in call_kwargs
        assert "document" in call_kwargs["context"].lower()

    @patch("docling_graph.core.extractors.strategies.many_to_one.is_vlm_backend")
    @patch("docling_graph.core.extractors.strategies.many_to_one.is_llm_backend")
    @patch("docling_graph.core.extractors.strategies.many_to_one.DocumentProcessor")
    def test_extract_llm_returns_none(self, mock_processor_class, mock_is_llm, mock_is_vlm):
        """Test LLM extraction when backend returns None."""
        mock_is_vlm.return_value = False
        mock_is_llm.return_value = True
        
        mock_processor_instance = Mock()
        mock_document = Mock()
        mock_processor_instance.convert_to_markdown.return_value = mock_document
        mock_processor_instance.extract_full_markdown.return_value = "Content"
        mock_processor_class.return_value = mock_processor_instance
        
        mock_backend = Mock()
        mock_backend.extract_from_markdown.return_value = None
        
        strategy = ManyToOneStrategy(backend=mock_backend)
        result = strategy.extract(source="test.pdf", template=SampleModel)
        
        assert result == []

    @patch("docling_graph.core.extractors.strategies.many_to_one.chunk_text")
    @patch("docling_graph.core.extractors.strategies.many_to_one.is_vlm_backend")
    @patch("docling_graph.core.extractors.strategies.many_to_one.is_llm_backend")
    @patch("docling_graph.core.extractors.strategies.many_to_one.DocumentProcessor")
    def test_extract_llm_with_chunking(
        self, mock_processor_class, mock_is_llm, mock_is_vlm, mock_chunk
    ):
        """Test LLM extraction with text chunking for large documents."""
        mock_is_vlm.return_value = False
        mock_is_llm.return_value = True
        
        mock_processor_instance = Mock()
        mock_document = Mock()
        mock_processor_instance.convert_to_markdown.return_value = mock_document
        mock_processor_instance.extract_full_markdown.return_value = "Very long content" * 1000
        mock_processor_class.return_value = mock_processor_instance
        
        # Mock chunking
        mock_chunk.return_value = ["Chunk 1", "Chunk 2"]
        
        mock_backend = Mock()
        mock_model1 = SampleModel(title="Part 1", summary="Summary 1", page_count=5)
        mock_model2 = SampleModel(title="Part 2", summary="Summary 2", page_count=5)
        mock_backend.extract_from_markdown.side_effect = [mock_model1, mock_model2]
        
        strategy = ManyToOneStrategy(backend=mock_backend)
        result = strategy.extract(source="test.pdf", template=SampleModel)
        
        # Should process chunks
        assert mock_backend.extract_from_markdown.call_count == 2

    @patch("docling_graph.core.extractors.strategies.many_to_one.merge_pydantic_models")
    @patch("docling_graph.core.extractors.strategies.many_to_one.chunk_text")
    @patch("docling_graph.core.extractors.strategies.many_to_one.is_vlm_backend")
    @patch("docling_graph.core.extractors.strategies.many_to_one.is_llm_backend")
    @patch("docling_graph.core.extractors.strategies.many_to_one.DocumentProcessor")
    def test_extract_llm_merges_chunked_results(
        self, mock_processor_class, mock_is_llm, mock_is_vlm, mock_chunk, mock_merge
    ):
        """Test that chunked LLM results are merged."""
        mock_is_vlm.return_value = False
        mock_is_llm.return_value = True
        
        mock_processor_instance = Mock()
        mock_document = Mock()
        mock_processor_instance.convert_to_markdown.return_value = mock_document
        mock_processor_instance.extract_full_markdown.return_value = "Content"
        mock_processor_class.return_value = mock_processor_instance
        
        mock_chunk.return_value = ["Chunk 1", "Chunk 2"]
        
        mock_backend = Mock()
        mock_model1 = SampleModel(title="Part 1", summary="S1", page_count=3)
        mock_model2 = SampleModel(title="Part 2", summary="S2", page_count=4)
        mock_backend.extract_from_markdown.side_effect = [mock_model1, mock_model2]
        
        merged_model = SampleModel(title="Complete", summary="Merged", page_count=7)
        mock_merge.return_value = merged_model
        
        strategy = ManyToOneStrategy(backend=mock_backend)
        result = strategy.extract(source="test.pdf", template=SampleModel)
        
        assert len(result) == 1
        assert result[0].title == "Complete"
        mock_merge.assert_called_once()


class TestManyToOneStrategyErrorHandling:
    """Tests for error handling."""

    @patch("docling_graph.core.extractors.strategies.many_to_one.is_vlm_backend")
    @patch("docling_graph.core.extractors.strategies.many_to_one.is_llm_backend")
    @patch("docling_graph.core.extractors.strategies.many_to_one.DocumentProcessor")
    def test_extract_with_unknown_backend(self, mock_processor_class, mock_is_llm, mock_is_vlm):
        """Test extraction with unknown backend type."""
        mock_is_vlm.return_value = False
        mock_is_llm.return_value = False
        mock_backend = Mock()
        
        strategy = ManyToOneStrategy(backend=mock_backend)
        
        with pytest.raises(ValueError, match="Unknown backend type"):
            strategy.extract(source="test.pdf", template=SampleModel)

    @patch("docling_graph.core.extractors.strategies.many_to_one.is_vlm_backend")
    @patch("docling_graph.core.extractors.strategies.many_to_one.DocumentProcessor")
    def test_extract_vlm_with_exception(self, mock_processor_class, mock_is_vlm):
        """Test VLM extraction when backend raises exception."""
        mock_is_vlm.return_value = True
        mock_backend = Mock()
        mock_backend.extract_from_document.side_effect = RuntimeError("Extraction failed")
        
        strategy = ManyToOneStrategy(backend=mock_backend)
        
        with pytest.raises(RuntimeError, match="Extraction failed"):
            strategy.extract(source="test.pdf", template=SampleModel)

    @patch("docling_graph.core.extractors.strategies.many_to_one.is_vlm_backend")
    @patch("docling_graph.core.extractors.strategies.many_to_one.is_llm_backend")
    @patch("docling_graph.core.extractors.strategies.many_to_one.DocumentProcessor")
    def test_extract_llm_document_conversion_error(
        self, mock_processor_class, mock_is_llm, mock_is_vlm
    ):
        """Test LLM extraction when document conversion fails."""
        mock_is_vlm.return_value = False
        mock_is_llm.return_value = True
        
        mock_processor_instance = Mock()
        mock_processor_instance.convert_to_markdown.side_effect = RuntimeError("Conversion failed")
        mock_processor_class.return_value = mock_processor_instance
        
        strategy = ManyToOneStrategy(backend=Mock())
        
        with pytest.raises(RuntimeError, match="Conversion failed"):
            strategy.extract(source="test.pdf", template=SampleModel)


class TestManyToOneStrategyCleanup:
    """Tests for cleanup method."""

    @patch("docling_graph.core.extractors.strategies.many_to_one.DocumentProcessor")
    def test_cleanup_calls_backend_cleanup(self, mock_processor_class):
        """Test cleanup calls backend cleanup method."""
        mock_backend = Mock()
        mock_backend.cleanup = Mock()
        
        strategy = ManyToOneStrategy(backend=mock_backend)
        strategy.cleanup()
        
        mock_backend.cleanup.assert_called_once()

    @patch("docling_graph.core.extractors.strategies.many_to_one.DocumentProcessor")
    def test_cleanup_calls_processor_cleanup(self, mock_processor_class):
        """Test cleanup calls processor cleanup method."""
        mock_processor_instance = Mock()
        mock_processor_instance.cleanup = Mock()
        mock_processor_class.return_value = mock_processor_instance
        
        strategy = ManyToOneStrategy(backend=Mock())
        strategy.cleanup()
        
        mock_processor_instance.cleanup.assert_called_once()

    @patch("docling_graph.core.extractors.strategies.many_to_one.DocumentProcessor")
    def test_cleanup_handles_missing_cleanup_methods(self, mock_processor_class):
        """Test cleanup when components don't have cleanup method."""
        mock_backend = Mock(spec=[])  # No cleanup method
        mock_processor_instance = Mock(spec=[])
        mock_processor_class.return_value = mock_processor_instance
        
        strategy = ManyToOneStrategy(backend=mock_backend)
        # Should not raise
        strategy.cleanup()
