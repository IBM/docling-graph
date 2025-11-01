"""
Unit tests for pipeline module helpers.
"""

from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest
from pydantic import BaseModel

from docling_graph.pipeline import (
    _load_template_class,
    run_pipeline,
)


class SampleModel(BaseModel):
    """Sample model for testing."""
    name: str
    value: int


class TestLoadTemplateClass:
    """Tests for _load_template_class function."""

    def test_load_builtin_model(self):
        """Test loading a built-in model."""
        # Use this test module's SampleModel
        template = _load_template_class("test_pipeline.SampleModel")
        
        # Should return the class
        assert template is not None

    def test_load_invalid_module_raises_error(self):
        """Test loading from non-existent module."""
        with pytest.raises(Exception):  # ModuleNotFoundError or similar
            _load_template_class("nonexistent_module.Model")

    def test_load_invalid_class_raises_error(self):
        """Test loading non-existent class."""
        with pytest.raises(Exception):
            _load_template_class("test_pipeline.NonexistentModel")

    def test_load_template_is_pydantic_model(self):
        """Test that loaded template is a Pydantic model."""
        template = _load_template_class("test_pipeline.SampleModel")
        
        # Should be instantiable with correct fields
        instance = template(name="test", value=42)
        assert instance.name == "test"
        assert instance.value == 42

    def test_template_string_format(self):
        """Test that template string must be dotted path."""
        # Valid dotted path
        template = _load_template_class("test_pipeline.SampleModel")
        assert template is not None


class TestRunPipeline:
    """Tests for run_pipeline function."""

    @patch("docling_graph.pipeline.GraphConverter")
    @patch("docling_graph.pipeline._load_template_class")
    @patch("docling_graph.pipeline.get_client")
    def test_pipeline_with_minimal_config(self, mock_get_client, mock_load_template, mock_converter):
        """Test pipeline with minimal configuration."""
        # Setup mocks
        mock_load_template.return_value = SampleModel
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        mock_converter_instance = Mock()
        mock_converter.return_value = mock_converter_instance
        mock_converter_instance.pydantic_list_to_graph.return_value = (Mock(), Mock())
        
        config = {
            "source": "test.pdf",
            "template": "test_pipeline.SampleModel",
            "processing_mode": "many-to-one",
            "backend_type": "llm",
            "inference": "local",
        }
        
        # Should not raise error
        result = run_pipeline(config)
        assert result is not None

    @patch("docling_graph.pipeline._load_template_class")
    def test_pipeline_invalid_template_raises_error(self, mock_load_template):
        """Test pipeline with invalid template."""
        mock_load_template.side_effect = ImportError("Cannot load template")
        
        config = {
            "source": "test.pdf",
            "template": "invalid.Template",
        }
        
        with pytest.raises(ImportError):
            run_pipeline(config)

    @patch("docling_graph.pipeline.GraphConverter")
    @patch("docling_graph.pipeline._load_template_class")
    @patch("docling_graph.pipeline.get_client")
    def test_pipeline_missing_required_config(self, mock_get_client, mock_load_template, mock_converter):
        """Test pipeline with missing required configuration."""
        config = {
            # Missing source and template
            "backend_type": "llm",
        }
        
        with pytest.raises(KeyError):
            run_pipeline(config)

    @patch("docling_graph.pipeline.GraphConverter")
    @patch("docling_graph.pipeline._load_template_class")
    @patch("docling_graph.pipeline.get_client")
    def test_pipeline_with_all_options(
        self, mock_get_client, mock_load_template, mock_converter
    ):
        """Test pipeline with all configuration options."""
        mock_load_template.return_value = SampleModel
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        mock_converter_instance = Mock()
        mock_converter.return_value = mock_converter_instance
        mock_graph = Mock()
        mock_metadata = Mock()
        mock_converter_instance.pydantic_list_to_graph.return_value = (mock_graph, mock_metadata)
        
        config = {
            "source": "test.pdf",
            "template": "test_pipeline.SampleModel",
            "processing_mode": "many-to-one",
            "backend_type": "llm",
            "inference": "local",
            "model": "llama3",
            "provider": "ollama",
            "export_format": "csv",
            "output_dir": Path("outputs"),
            "reverse_edges": True,
            "docling_config": "ocr",
        }
        
        result = run_pipeline(config)
        
        # Verify pipeline was executed
        mock_load_template.assert_called_once()

    @patch("docling_graph.pipeline.GraphConverter")
    @patch("docling_graph.pipeline._load_template_class")
    @patch("docling_graph.pipeline.get_client")
    def test_pipeline_returns_graph_and_metadata(
        self, mock_get_client, mock_load_template, mock_converter
    ):
        """Test that pipeline returns graph and metadata."""
        mock_load_template.return_value = SampleModel
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        mock_converter_instance = Mock()
        mock_converter.return_value = mock_converter_instance
        mock_graph = Mock()
        mock_metadata = Mock()
        mock_converter_instance.pydantic_list_to_graph.return_value = (mock_graph, mock_metadata)
        
        config = {
            "source": "test.pdf",
            "template": "test_pipeline.SampleModel",
        }
        
        result = run_pipeline(config)
        
        # Result should be tuple of (graph, metadata)
        assert result is not None
        assert len(result) == 2

    @patch("docling_graph.pipeline.GraphConverter")
    @patch("docling_graph.pipeline._load_template_class")
    @patch("docling_graph.pipeline.get_client")
    def test_pipeline_vlm_backend(self, mock_get_client, mock_load_template, mock_converter):
        """Test pipeline with VLM backend."""
        mock_load_template.return_value = SampleModel
        mock_converter_instance = Mock()
        mock_converter.return_value = mock_converter_instance
        mock_converter_instance.pydantic_list_to_graph.return_value = (Mock(), Mock())
        
        config = {
            "source": "test.pdf",
            "template": "test_pipeline.SampleModel",
            "backend_type": "vlm",
            "model": "numind/NuExtract",
        }
        
        result = run_pipeline(config)
        assert result is not None


class TestPipelineIntegration:
    """Integration tests for pipeline."""

    @patch("docling_graph.pipeline.GraphConverter")
    @patch("docling_graph.pipeline._load_template_class")
    @patch("docling_graph.pipeline.get_client")
    def test_pipeline_end_to_end(self, mock_get_client, mock_load_template, mock_converter):
        """Test pipeline end-to-end."""
        mock_load_template.return_value = SampleModel
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        mock_converter_instance = Mock()
        mock_converter.return_value = mock_converter_instance
        
        # Mock the graph and metadata
        mock_graph = Mock()
        mock_graph.number_of_nodes.return_value = 5
        mock_graph.number_of_edges.return_value = 4
        mock_metadata = Mock()
        mock_converter_instance.pydantic_list_to_graph.return_value = (mock_graph, mock_metadata)
        
        config = {
            "source": "test.pdf",
            "template": "test_pipeline.SampleModel",
            "processing_mode": "many-to-one",
            "backend_type": "llm",
            "inference": "local",
        }
        
        graph, metadata = run_pipeline(config)
        
        # Verify graph was created
        assert graph.number_of_nodes() == 5
        assert graph.number_of_edges() == 4
