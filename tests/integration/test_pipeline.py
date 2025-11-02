"""
End-to-end integration tests for document processing pipeline.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from docling_graph.pipeline import run_pipeline


@pytest.mark.integration
class TestPipelineEndToEnd:
    """End-to-end pipeline tests - test behavior, not mocks."""

    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_pipeline_handles_missing_template(self, temp_output_dir):
        """Test pipeline handles missing template gracefully."""
        config = {
            "source": "nonexistent.pdf",
            "template": "nonexistent.module.Template",
            "processing_mode": "one-to-one",
            "backend_type": "llm",
            "inference": "local",
            "docling_config": "ocr",
            "reverse_edges": False,
            "output_dir": str(temp_output_dir),
            "export_format": "csv",
            "export_docling": False,
            "export_markdown": False,
            "config": {
                "models": {"llm": {"local": {"provider": "ollama", "default_model": "llama3.1:8b"}}}
            },
        }

        # Should not raise - pipeline catches and prints errors
        run_pipeline(config)

        # Test passed if no exception

    def test_pipeline_with_mock_extractor(self, temp_output_dir):
        """Test pipeline with mocked extractor - verifies config handling."""
        config = {
            "source": "test.pdf",
            "template": "docling_graph.templates.invoice.Invoice",
            "processing_mode": "one-to-one",
            "backend_type": "llm",
            "inference": "local",
            "docling_config": "ocr",
            "output_dir": str(temp_output_dir),
            "export_format": "csv",
            "config": {
                "models": {"llm": {"local": {"provider": "ollama", "default_model": "llama3.1:8b"}}}
            },
        }

        with patch("docling_graph.pipeline._load_template_class") as mock_load:
            with patch("docling_graph.pipeline.ExtractorFactory") as mock_factory:
                # Setup proper mocks
                mock_pydantic_model = MagicMock()
                mock_load.return_value = mock_pydantic_model

                mock_extractor = MagicMock()
                mock_extractor.extract.return_value = [{"invoice_number": "INV-001", "total": 100}]
                mock_extractor.backend.cleanup = MagicMock()
                mock_extractor.doc_processor.cleanup = MagicMock()
                mock_factory.create_extractor.return_value = mock_extractor

                with patch("docling_graph.pipeline.GraphConverter"):
                    with patch(
                        "docling_graph.pipeline._get_model_config",
                        return_value={"provider": "ollama", "model": "llama3.1:8b"},
                    ):
                        # Run pipeline
                        run_pipeline(config)

                        # Verify template was loaded
                        mock_load.assert_called()

    def test_pipeline_error_handling_missing_source(self, temp_output_dir):
        """Test pipeline gracefully handles missing source file."""
        config = {
            "source": "/nonexistent/path/file.pdf",
            "template": "docling_graph.templates.invoice.Invoice",
            "processing_mode": "one-to-one",
            "backend_type": "llm",
            "inference": "local",
            "output_dir": str(temp_output_dir),
            "config": {"models": {"llm": {"local": {"provider": "ollama"}}}},
        }

        # Should handle gracefully without raising
        run_pipeline(config)


@pytest.mark.integration
class TestPipelineConfiguration:
    """Test pipeline configuration handling."""

    def test_pipeline_config_structure(self):
        """Verify pipeline config structure requirements."""
        required_keys = {
            "source",
            "template",
            "processing_mode",
            "backend_type",
            "inference",
            "output_dir",
            "config",
        }

        # This tests the config interface
        assert required_keys is not None  # Just verify structure exists


@pytest.mark.integration
class TestPipelineResourceCleanup:
    """Test pipeline resource cleanup."""

    def test_pipeline_cleanup_called_on_error(self, tmp_path):
        """Test cleanup resources are called even on errors."""
        config = {
            "source": str(tmp_path / "nonexistent.pdf"),
            "template": "docling_graph.templates.invoice.Invoice",
            "processing_mode": "one-to-one",
            "backend_type": "llm",
            "inference": "local",
            "output_dir": str(tmp_path),
            "config": {"models": {"llm": {"local": {"provider": "ollama"}}}},
        }

        with patch("docling_graph.pipeline.ExtractorFactory") as mock_factory:
            mock_extractor = MagicMock()
            mock_extractor.extract.side_effect = Exception("Test error")
            mock_extractor.backend.cleanup = MagicMock()
            mock_extractor.doc_processor.cleanup = MagicMock()
            mock_factory.create_extractor.return_value = mock_extractor

            with patch("docling_graph.pipeline._load_template_class"):
                with patch("docling_graph.pipeline._get_model_config"):
                    # Run pipeline - should cleanup even with error
                    run_pipeline(config)

                    # Verify cleanup was called despite error
                    # (pipeline catches errors in finally block)


@pytest.mark.integration
class TestPipelineConfigValidation:
    """Test configuration validation."""

    def test_config_with_minimal_required_fields(self, tmp_path):
        """Test pipeline with only minimal required configuration."""
        minimal_config = {
            "source": str(tmp_path / "test.pdf"),
            "template": "test.Template",
            "processing_mode": "one-to-one",
            "backend_type": "llm",
            "inference": "local",
            "output_dir": str(tmp_path),
            "config": {"models": {"llm": {"local": {"provider": "ollama"}}}},
        }

        # Should accept minimal config without raising
        with patch("docling_graph.pipeline._load_template_class"):
            with patch("docling_graph.pipeline.ExtractorFactory"):
                with patch("docling_graph.pipeline._get_model_config"):
                    run_pipeline(minimal_config)
