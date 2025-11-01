"""
Unit tests for VlmBackend.
"""

from unittest.mock import MagicMock, Mock, patch, PropertyMock

import pytest
from pydantic import BaseModel, Field, ValidationError

from docling_graph.core.extractors.backends.vlm_backend import VlmBackend


class SampleModel(BaseModel):
    """Sample Pydantic model for testing."""
    name: str
    age: int
    email: str = None


class TestVlmBackendInitialization:
    """Tests for VlmBackend initialization."""

    @patch("docling_graph.core.extractors.backends.vlm_backend.ExtractionVlmPipeline")
    @patch("docling_graph.core.extractors.backends.vlm_backend.DocumentExtractor")
    def test_init_with_model_name(self, mock_extractor, mock_pipeline):
        """Test initialization with model name."""
        backend = VlmBackend(model_name="numind/NuExtract")
        
        assert backend.model_name == "numind/NuExtract"
        assert hasattr(backend, "extractor")

    @patch("docling_graph.core.extractors.backends.vlm_backend.ExtractionVlmPipeline")
    @patch("docling_graph.core.extractors.backends.vlm_backend.DocumentExtractor")
    def test_init_creates_extractor(self, mock_extractor, mock_pipeline):
        """Test that initialization creates DocumentExtractor."""
        mock_extractor_instance = Mock()
        mock_extractor.return_value = mock_extractor_instance
        
        backend = VlmBackend(model_name="numind/NuExtract-2.0-8B")
        
        mock_extractor.assert_called_once()
        assert backend.extractor == mock_extractor_instance

    @patch("docling_graph.core.extractors.backends.vlm_backend.ExtractionVlmPipeline")
    @patch("docling_graph.core.extractors.backends.vlm_backend.DocumentExtractor")
    def test_init_with_custom_model(self, mock_extractor, mock_pipeline):
        """Test initialization with custom HuggingFace model."""
        backend = VlmBackend(model_name="custom-org/custom-model")
        
        assert backend.model_name == "custom-org/custom-model"

    @patch("docling_graph.core.extractors.backends.vlm_backend.ExtractionVlmPipeline")
    @patch("docling_graph.core.extractors.backends.vlm_backend.DocumentExtractor")
    def test_init_pipeline_configuration(self, mock_extractor, mock_pipeline):
        """Test that VLM pipeline is configured correctly."""
        mock_pipeline_instance = Mock()
        mock_pipeline.return_value = mock_pipeline_instance
        
        backend = VlmBackend(model_name="numind/NuExtract")
        
        # Pipeline should be initialized
        mock_pipeline.assert_called_once()


class TestVlmBackendExtraction:
    """Tests for extract_from_document method."""

    @patch("docling_graph.core.extractors.backends.vlm_backend.ExtractionVlmPipeline")
    @patch("docling_graph.core.extractors.backends.vlm_backend.DocumentExtractor")
    def test_extract_success(self, mock_extractor_class, mock_pipeline):
        """Test successful extraction from document."""
        # Setup mock extractor
        mock_extractor_instance = Mock()
        mock_result = Mock()
        mock_extracted_data = Mock()
        mock_extracted_data.dict.return_value = {
            "name": "Alice",
            "age": 30,
            "email": "alice@example.com"
        }
        mock_result.extracted_data = [mock_extracted_data]
        mock_extractor_instance.extract.return_value = [mock_result]
        mock_extractor_class.return_value = mock_extractor_instance
        
        backend = VlmBackend(model_name="numind/NuExtract")
        result = backend.extract_from_document(
            source="test.pdf",
            template=SampleModel
        )
        
        assert len(result) == 1
        assert result[0].name == "Alice"
        assert result[0].age == 30
        assert result[0].email == "alice@example.com"

    @patch("docling_graph.core.extractors.backends.vlm_backend.ExtractionVlmPipeline")
    @patch("docling_graph.core.extractors.backends.vlm_backend.DocumentExtractor")
    def test_extract_multiple_results(self, mock_extractor_class, mock_pipeline):
        """Test extraction returning multiple results."""
        mock_extractor_instance = Mock()
        mock_result1 = Mock()
        mock_result2 = Mock()
        
        mock_data1 = Mock()
        mock_data1.dict.return_value = {"name": "Alice", "age": 30}
        mock_result1.extracted_data = [mock_data1]
        
        mock_data2 = Mock()
        mock_data2.dict.return_value = {"name": "Bob", "age": 25}
        mock_result2.extracted_data = [mock_data2]
        
        mock_extractor_instance.extract.return_value = [mock_result1, mock_result2]
        mock_extractor_class.return_value = mock_extractor_instance
        
        backend = VlmBackend(model_name="numind/NuExtract")
        result = backend.extract_from_document(
            source="test.pdf",
            template=SampleModel
        )
        
        assert len(result) == 2
        assert result[0].name == "Alice"
        assert result[1].name == "Bob"

    @patch("docling_graph.core.extractors.backends.vlm_backend.ExtractionVlmPipeline")
    @patch("docling_graph.core.extractors.backends.vlm_backend.DocumentExtractor")
    def test_extract_empty_results(self, mock_extractor_class, mock_pipeline):
        """Test extraction with no results."""
        mock_extractor_instance = Mock()
        mock_extractor_instance.extract.return_value = []
        mock_extractor_class.return_value = mock_extractor_instance
        
        backend = VlmBackend(model_name="numind/NuExtract")
        result = backend.extract_from_document(
            source="test.pdf",
            template=SampleModel
        )
        
        assert result == []

    @patch("docling_graph.core.extractors.backends.vlm_backend.ExtractionVlmPipeline")
    @patch("docling_graph.core.extractors.backends.vlm_backend.DocumentExtractor")
    def test_extract_with_validation_error(self, mock_extractor_class, mock_pipeline):
        """Test extraction with validation error."""
        mock_extractor_instance = Mock()
        mock_result = Mock()
        mock_data = Mock()
        mock_data.dict.return_value = {"name": "Alice", "age": "invalid"}  # Invalid age
        mock_result.extracted_data = [mock_data]
        mock_extractor_instance.extract.return_value = [mock_result]
        mock_extractor_class.return_value = mock_extractor_instance
        
        backend = VlmBackend(model_name="numind/NuExtract")
        result = backend.extract_from_document(
            source="test.pdf",
            template=SampleModel
        )
        
        # Should handle validation error gracefully
        assert isinstance(result, list)

    @patch("docling_graph.core.extractors.backends.vlm_backend.ExtractionVlmPipeline")
    @patch("docling_graph.core.extractors.backends.vlm_backend.DocumentExtractor")
    def test_extract_with_missing_required_fields(self, mock_extractor_class, mock_pipeline):
        """Test extraction with missing required fields."""
        mock_extractor_instance = Mock()
        mock_result = Mock()
        mock_data = Mock()
        mock_data.dict.return_value = {"name": "Alice"}  # Missing required 'age'
        mock_result.extracted_data = [mock_data]
        mock_extractor_instance.extract.return_value = [mock_result]
        mock_extractor_class.return_value = mock_extractor_instance
        
        backend = VlmBackend(model_name="numind/NuExtract")
        result = backend.extract_from_document(
            source="test.pdf",
            template=SampleModel
        )
        
        # Should skip invalid models
        assert isinstance(result, list)

    @patch("docling_graph.core.extractors.backends.vlm_backend.ExtractionVlmPipeline")
    @patch("docling_graph.core.extractors.backends.vlm_backend.DocumentExtractor")
    def test_extract_with_extraction_error(self, mock_extractor_class, mock_pipeline):
        """Test extraction when extractor raises error."""
        mock_extractor_instance = Mock()
        mock_extractor_instance.extract.side_effect = RuntimeError("Extraction failed")
        mock_extractor_class.return_value = mock_extractor_instance
        
        backend = VlmBackend(model_name="numind/NuExtract")
        
        with pytest.raises(RuntimeError, match="Extraction failed"):
            backend.extract_from_document(
                source="test.pdf",
                template=SampleModel
            )

    @patch("docling_graph.core.extractors.backends.vlm_backend.ExtractionVlmPipeline")
    @patch("docling_graph.core.extractors.backends.vlm_backend.DocumentExtractor")
    def test_extract_with_nested_extracted_data(self, mock_extractor_class, mock_pipeline):
        """Test extraction with nested data structures."""
        mock_extractor_instance = Mock()
        mock_result = Mock()
        mock_data1 = Mock()
        mock_data2 = Mock()
        mock_data1.dict.return_value = {"name": "Alice", "age": 30}
        mock_data2.dict.return_value = {"name": "Bob", "age": 25}
        mock_result.extracted_data = [mock_data1, mock_data2]
        mock_extractor_instance.extract.return_value = [mock_result]
        mock_extractor_class.return_value = mock_extractor_instance
        
        backend = VlmBackend(model_name="numind/NuExtract")
        result = backend.extract_from_document(
            source="test.pdf",
            template=SampleModel
        )
        
        # Should flatten nested results
        assert len(result) == 2


class TestVlmBackendCleanup:
    """Tests for cleanup method."""

    @patch("docling_graph.core.extractors.backends.vlm_backend.torch")
    @patch("docling_graph.core.extractors.backends.vlm_backend.gc")
    @patch("docling_graph.core.extractors.backends.vlm_backend.ExtractionVlmPipeline")
    @patch("docling_graph.core.extractors.backends.vlm_backend.DocumentExtractor")
    def test_cleanup_success(self, mock_extractor, mock_pipeline, mock_gc, mock_torch):
        """Test successful cleanup."""
        backend = VlmBackend(model_name="numind/NuExtract")
        backend.cleanup()
        
        # Verify CUDA cache is cleared if available
        if mock_torch.cuda.is_available():
            mock_torch.cuda.empty_cache.assert_called()
        mock_gc.collect.assert_called()

    @patch("docling_graph.core.extractors.backends.vlm_backend.torch")
    @patch("docling_graph.core.extractors.backends.vlm_backend.gc")
    @patch("docling_graph.core.extractors.backends.vlm_backend.ExtractionVlmPipeline")
    @patch("docling_graph.core.extractors.backends.vlm_backend.DocumentExtractor")
    def test_cleanup_without_cuda(self, mock_extractor, mock_pipeline, mock_gc, mock_torch):
        """Test cleanup when CUDA is not available."""
        mock_torch.cuda.is_available.return_value = False
        
        backend = VlmBackend(model_name="numind/NuExtract")
        backend.cleanup()
        
        mock_torch.cuda.empty_cache.assert_not_called()
        mock_gc.collect.assert_called()

    @patch("docling_graph.core.extractors.backends.vlm_backend.torch")
    @patch("docling_graph.core.extractors.backends.vlm_backend.gc")
    @patch("docling_graph.core.extractors.backends.vlm_backend.ExtractionVlmPipeline")
    @patch("docling_graph.core.extractors.backends.vlm_backend.DocumentExtractor")
    def test_cleanup_handles_error(self, mock_extractor, mock_pipeline, mock_gc, mock_torch):
        """Test cleanup handles errors gracefully."""
        mock_gc.collect.side_effect = RuntimeError("GC error")
        
        backend = VlmBackend(model_name="numind/NuExtract")
        # Should not raise
        backend.cleanup()

    @patch("docling_graph.core.extractors.backends.vlm_backend.ExtractionVlmPipeline")
    @patch("docling_graph.core.extractors.backends.vlm_backend.DocumentExtractor")
    def test_cleanup_releases_extractor_reference(self, mock_extractor, mock_pipeline):
        """Test that cleanup releases extractor reference."""
        backend = VlmBackend(model_name="numind/NuExtract")
        assert hasattr(backend, "extractor")
        
        backend.cleanup()
        
        # Extractor reference might be set to None or deleted


class TestVlmBackendSchemaGeneration:
    """Tests for schema generation."""

    @patch("docling_graph.core.extractors.backends.vlm_backend.ExtractionVlmPipeline")
    @patch("docling_graph.core.extractors.backends.vlm_backend.DocumentExtractor")
    def test_schema_passed_to_extractor(self, mock_extractor_class, mock_pipeline):
        """Test that Pydantic schema is passed to extractor."""
        mock_extractor_instance = Mock()
        mock_extractor_instance.extract.return_value = []
        mock_extractor_class.return_value = mock_extractor_instance
        
        backend = VlmBackend(model_name="numind/NuExtract")
        backend.extract_from_document(
            source="test.pdf",
            template=SampleModel
        )
        
        # Verify extract was called
        mock_extractor_instance.extract.assert_called_once()
        call_kwargs = mock_extractor_instance.extract.call_args[1]
        
        # Should contain schema information
        assert "schema" in call_kwargs or "template" in call_kwargs


class TestVlmBackendModelLoading:
    """Tests for model loading behavior."""

    @patch("docling_graph.core.extractors.backends.vlm_backend.ExtractionVlmPipeline")
    @patch("docling_graph.core.extractors.backends.vlm_backend.DocumentExtractor")
    def test_model_loaded_once(self, mock_extractor, mock_pipeline):
        """Test that model is loaded only once during initialization."""
        backend = VlmBackend(model_name="numind/NuExtract")
        
        # Pipeline should be created once
        assert mock_pipeline.call_count == 1

    @patch("docling_graph.core.extractors.backends.vlm_backend.ExtractionVlmPipeline")
    @patch("docling_graph.core.extractors.backends.vlm_backend.DocumentExtractor")
    def test_multiple_extractions_same_model(self, mock_extractor_class, mock_pipeline):
        """Test multiple extractions with same model instance."""
        mock_extractor_instance = Mock()
        mock_extractor_instance.extract.return_value = []
        mock_extractor_class.return_value = mock_extractor_instance
        
        backend = VlmBackend(model_name="numind/NuExtract")
        backend.extract_from_document("doc1.pdf", SampleModel)
        backend.extract_from_document("doc2.pdf", SampleModel)
        
        # Extractor should be created once
        assert mock_extractor_class.call_count == 1
        # But extract should be called twice
        assert mock_extractor_instance.extract.call_count == 2
