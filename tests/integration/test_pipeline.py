"""
Pipeline integration tests.

Tests the docling-graph pipeline module directly, focusing on
workflow execution, configuration handling, and component integration.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from docling_graph.pipeline import run_pipeline

from .conftest import Document, Invoice, Person


@pytest.mark.integration
@pytest.mark.slow
class TestPipelineBasicExecution:
    """Test basic pipeline execution."""

    def test_pipeline_accepts_minimal_config(self, temp_dir):
        """Test pipeline runs with minimal configuration."""
        config = {
            "source": str(temp_dir / "test.pdf"),
            "template": "tests.integration.conftest.Person",
            "processing_mode": "one-to-one",
            "backend_type": "llm",
            "inference": "local",
            "export_docling": False,
        }

        pdf_path = temp_dir / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n")

        with patch("docling_graph.pipeline.ExtractorFactory"):
            with patch("docling_graph.pipeline._initialize_llm_client"):
                with patch("docling_graph.pipeline.GraphConverter"):
                    with patch("docling_graph.pipeline.CSVExporter"):
                        with patch("docling_graph.pipeline.JSONExporter"):
                            # Should not raise
                            run_pipeline(config)

    def test_pipeline_accepts_full_config(self, temp_dir):
        """Test pipeline runs with full configuration."""
        config = {
            "source": str(temp_dir / "test.pdf"),
            "template": "tests.integration.conftest.Invoice",
            "processing_mode": "many-to-one",
            "backend_type": "llm",
            "inference": "local",
            "model_override": "custom-model",
            "provider_override": "custom-provider",
            "docling_config": "ocr",
            "output_dir": str(temp_dir / "output"),
            "export_format": "cypher",
            "reverse_edges": True,
            "export_docling": True,
            "export_docling_json": True,
            "export_markdown": True,
            "export_per_page_markdown": True,
        }

        pdf_path = temp_dir / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n")

        with patch("docling_graph.pipeline.ExtractorFactory"):
            with patch("docling_graph.pipeline._initialize_llm_client"):
                with patch("docling_graph.pipeline.GraphConverter"):
                    with patch("docling_graph.pipeline.CypherExporter"):
                        with patch("docling_graph.pipeline.JSONExporter"):
                            with patch("docling_graph.pipeline.ReportGenerator"):
                                with patch("docling_graph.pipeline.InteractiveVisualizer"):
                                    # Should not raise
                                    run_pipeline(config)


@pytest.mark.integration
@pytest.mark.slow
class TestPipelineComponentIntegration:
    """Test pipeline component integration."""

    def test_pipeline_calls_extractor_factory(self, temp_dir):
        """Test pipeline calls ExtractorFactory with correct arguments."""
        config = {
            "source": str(temp_dir / "test.pdf"),
            "template": "tests.integration.conftest.Person",
            "processing_mode": "one-to-one",
            "backend_type": "llm",
            "inference": "local",
            "export_docling": False,
        }

        pdf_path = temp_dir / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n")

        with patch("docling_graph.pipeline.ExtractorFactory") as mock_factory:
            with patch("docling_graph.pipeline._initialize_llm_client"):
                with patch("docling_graph.pipeline.GraphConverter"):
                    with patch("docling_graph.pipeline.CSVExporter"):
                        with patch("docling_graph.pipeline.JSONExporter"):
                            run_pipeline(config)

                            # Factory should be called
                            mock_factory.assert_called_once()

    def test_pipeline_calls_graph_converter(self, temp_dir):
        """Test pipeline calls GraphConverter."""
        config = {
            "source": str(temp_dir / "test.pdf"),
            "template": "tests.integration.conftest.Person",
            "processing_mode": "one-to-one",
            "backend_type": "llm",
            "inference": "local",
            "export_docling": False,
        }

        pdf_path = temp_dir / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n")

        with patch("docling_graph.pipeline.ExtractorFactory"):
            with patch("docling_graph.pipeline._initialize_llm_client"):
                with patch("docling_graph.pipeline.GraphConverter") as mock_converter:
                    with patch("docling_graph.pipeline.CSVExporter"):
                        with patch("docling_graph.pipeline.JSONExporter"):
                            run_pipeline(config)

                            # Converter should be instantiated
                            mock_converter.assert_called_once()

    def test_pipeline_calls_json_exporter_always(self, temp_dir):
        """Test pipeline always calls JSONExporter."""
        config = {
            "source": str(temp_dir / "test.pdf"),
            "template": "tests.integration.conftest.Document",
            "processing_mode": "many-to-one",
            "backend_type": "vlm",
            "inference": "local",
            "export_format": "cypher",
            "export_docling": False,
        }

        pdf_path = temp_dir / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n")

        with patch("docling_graph.pipeline.ExtractorFactory"):
            with patch("docling_graph.pipeline.GraphConverter"):
                with patch("docling_graph.pipeline.JSONExporter") as mock_json:
                    with patch("docling_graph.pipeline.CypherExporter"):
                        run_pipeline(config)

                        # JSON export should always be called
                        mock_json.assert_called_once()

    def test_pipeline_calls_format_specific_exporter(self, temp_dir):
        """Test pipeline calls format-specific exporter."""
        config = {
            "source": str(temp_dir / "test.pdf"),
            "template": "tests.integration.conftest.Person",
            "processing_mode": "one-to-one",
            "backend_type": "llm",
            "inference": "local",
            "export_format": "csv",
            "export_docling": False,
        }

        pdf_path = temp_dir / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n")

        with patch("docling_graph.pipeline.ExtractorFactory"):
            with patch("docling_graph.pipeline._initialize_llm_client"):
                with patch("docling_graph.pipeline.GraphConverter"):
                    with patch("docling_graph.pipeline.CSVExporter") as mock_csv:
                        with patch("docling_graph.pipeline.JSONExporter"):
                            run_pipeline(config)

                            # CSV export should be called
                            mock_csv.assert_called_once()


@pytest.mark.integration
@pytest.mark.slow
class TestPipelineTemplateHandling:
    """Test pipeline template loading and handling."""

    def test_pipeline_loads_person_template(self, temp_dir):
        """Test pipeline loads Person template."""
        config = {
            "source": str(temp_dir / "test.pdf"),
            "template": "tests.integration.conftest.Person",
            "processing_mode": "one-to-one",
            "backend_type": "llm",
            "inference": "local",
            "export_docling": False,
        }

        pdf_path = temp_dir / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n")

        with patch("docling_graph.pipeline.ExtractorFactory"):
            with patch("docling_graph.pipeline._initialize_llm_client"):
                with patch("docling_graph.pipeline.GraphConverter"):
                    with patch("docling_graph.pipeline.CSVExporter"):
                        with patch("docling_graph.pipeline.JSONExporter"):
                            # Should not raise
                            run_pipeline(config)

    def test_pipeline_loads_invoice_template(self, temp_dir):
        """Test pipeline loads Invoice template."""
        config = {
            "source": str(temp_dir / "test.pdf"),
            "template": "tests.integration.conftest.Invoice",
            "processing_mode": "one-to-one",
            "backend_type": "llm",
            "inference": "local",
            "export_docling": False,
        }

        pdf_path = temp_dir / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n")

        with patch("docling_graph.pipeline.ExtractorFactory"):
            with patch("docling_graph.pipeline._initialize_llm_client"):
                with patch("docling_graph.pipeline.GraphConverter"):
                    with patch("docling_graph.pipeline.CSVExporter"):
                        with patch("docling_graph.pipeline.JSONExporter"):
                            # Should not raise
                            run_pipeline(config)

    def test_pipeline_loads_document_template(self, temp_dir):
        """Test pipeline loads Document template."""
        config = {
            "source": str(temp_dir / "test.pdf"),
            "template": "tests.integration.conftest.Document",
            "processing_mode": "many-to-one",
            "backend_type": "llm",
            "inference": "local",
            "export_docling": False,
        }

        pdf_path = temp_dir / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n")

        with patch("docling_graph.pipeline.ExtractorFactory"):
            with patch("docling_graph.pipeline._initialize_llm_client"):
                with patch("docling_graph.pipeline.GraphConverter"):
                    with patch("docling_graph.pipeline.CSVExporter"):
                        with patch("docling_graph.pipeline.JSONExporter"):
                            # Should not raise
                            run_pipeline(config)


@pytest.mark.integration
@pytest.mark.slow
class TestPipelineBackendConfiguration:
    """Test pipeline backend configuration."""

    def test_pipeline_configures_llm_local_backend(self, temp_dir):
        """Test pipeline configures LLM local backend."""
        config = {
            "source": str(temp_dir / "test.pdf"),
            "template": "tests.integration.conftest.Person",
            "processing_mode": "one-to-one",
            "backend_type": "llm",
            "inference": "local",
            "export_docling": False,
        }

        pdf_path = temp_dir / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n")

        with patch("docling_graph.pipeline.ExtractorFactory"):
            with patch("docling_graph.pipeline._initialize_llm_client") as mock_llm:
                with patch("docling_graph.pipeline.GraphConverter"):
                    with patch("docling_graph.pipeline.CSVExporter"):
                        with patch("docling_graph.pipeline.JSONExporter"):
                            run_pipeline(config)

                            # LLM client should be initialized
                            mock_llm.assert_called_once()

    def test_pipeline_configures_llm_remote_backend(self, temp_dir):
        """Test pipeline configures LLM remote backend."""
        config = {
            "source": str(temp_dir / "test.pdf"),
            "template": "tests.integration.conftest.Person",
            "processing_mode": "one-to-one",
            "backend_type": "llm",
            "inference": "remote",
            "export_docling": False,
        }

        pdf_path = temp_dir / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n")

        with patch("docling_graph.pipeline.ExtractorFactory"):
            with patch("docling_graph.pipeline._initialize_llm_client") as mock_llm:
                with patch("docling_graph.pipeline.GraphConverter"):
                    with patch("docling_graph.pipeline.CSVExporter"):
                        with patch("docling_graph.pipeline.JSONExporter"):
                            run_pipeline(config)

                            # LLM client should be initialized
                            mock_llm.assert_called_once()

    def test_pipeline_configures_vlm_backend(self, temp_dir):
        """Test pipeline configures VLM backend."""
        config = {
            "source": str(temp_dir / "test.pdf"),
            "template": "tests.integration.conftest.Invoice",
            "processing_mode": "one-to-one",
            "backend_type": "vlm",
            "inference": "local",
            "export_docling": False,
        }

        pdf_path = temp_dir / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n")

        with patch("docling_graph.pipeline.ExtractorFactory"):
            with patch("docling_graph.pipeline.GraphConverter"):
                with patch("docling_graph.pipeline.CSVExporter"):
                    with patch("docling_graph.pipeline.JSONExporter"):
                        # Should not raise
                        run_pipeline(config)


@pytest.mark.integration
@pytest.mark.slow
class TestPipelineExportFormats:
    """Test pipeline export format handling."""

    def test_pipeline_exports_to_csv(self, temp_dir):
        """Test pipeline exports graph to CSV."""
        config = {
            "source": str(temp_dir / "test.pdf"),
            "template": "tests.integration.conftest.Person",
            "processing_mode": "one-to-one",
            "backend_type": "llm",
            "inference": "local",
            "export_format": "csv",
            "export_docling": False,
        }

        pdf_path = temp_dir / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n")

        with patch("docling_graph.pipeline.ExtractorFactory"):
            with patch("docling_graph.pipeline._initialize_llm_client"):
                with patch("docling_graph.pipeline.GraphConverter"):
                    with patch("docling_graph.pipeline.CSVExporter") as mock_csv:
                        with patch("docling_graph.pipeline.JSONExporter"):
                            run_pipeline(config)
                            mock_csv.assert_called_once()

    def test_pipeline_exports_to_json(self, temp_dir):
        """Test pipeline exports graph to JSON."""
        config = {
            "source": str(temp_dir / "test.pdf"),
            "template": "tests.integration.conftest.Person",
            "processing_mode": "one-to-one",
            "backend_type": "llm",
            "inference": "local",
            "export_format": "json",
            "export_docling": False,
        }

        pdf_path = temp_dir / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n")

        with patch("docling_graph.pipeline.ExtractorFactory"):
            with patch("docling_graph.pipeline._initialize_llm_client"):
                with patch("docling_graph.pipeline.GraphConverter"):
                    with patch("docling_graph.pipeline.JSONExporter") as mock_json:
                        run_pipeline(config)
                        # JSON is always called
                        mock_json.assert_called_once()

    def test_pipeline_exports_to_cypher(self, temp_dir):
        """Test pipeline exports graph to Cypher."""
        config = {
            "source": str(temp_dir / "test.pdf"),
            "template": "tests.integration.conftest.Invoice",
            "processing_mode": "many-to-one",
            "backend_type": "llm",
            "inference": "local",
            "export_format": "cypher",
            "export_docling": False,
        }

        pdf_path = temp_dir / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n")

        with patch("docling_graph.pipeline.ExtractorFactory"):
            with patch("docling_graph.pipeline._initialize_llm_client"):
                with patch("docling_graph.pipeline.GraphConverter"):
                    with patch("docling_graph.pipeline.CypherExporter") as mock_cypher:
                        with patch("docling_graph.pipeline.JSONExporter"):
                            run_pipeline(config)
                            mock_cypher.assert_called_once()


@pytest.mark.integration
class TestPipelineOutputDirectory:
    """Test pipeline output directory creation."""

    def test_pipeline_creates_output_directory(self, temp_dir):
        """Test pipeline creates output directory."""
        output_dir = temp_dir / "output"

        config = {
            "source": str(temp_dir / "test.pdf"),
            "template": "tests.integration.conftest.Person",
            "processing_mode": "one-to-one",
            "backend_type": "llm",
            "inference": "local",
            "output_dir": str(output_dir),
            "export_docling": False,
        }

        pdf_path = temp_dir / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n")

        with patch("docling_graph.pipeline.ExtractorFactory"):
            with patch("docling_graph.pipeline._initialize_llm_client"):
                with patch("docling_graph.pipeline.GraphConverter"):
                    with patch("docling_graph.pipeline.CSVExporter"):
                        with patch("docling_graph.pipeline.JSONExporter"):
                            run_pipeline(config)

        # Output directory should exist
        assert output_dir.exists()

    def test_pipeline_handles_nested_output_directory(self, temp_dir):
        """Test pipeline creates nested output directories."""
        output_dir = temp_dir / "deep" / "nested" / "output"

        config = {
            "source": str(temp_dir / "test.pdf"),
            "template": "tests.integration.conftest.Person",
            "processing_mode": "one-to-one",
            "backend_type": "llm",
            "inference": "local",
            "output_dir": str(output_dir),
            "export_docling": False,
        }

        pdf_path = temp_dir / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n")

        with patch("docling_graph.pipeline.ExtractorFactory"):
            with patch("docling_graph.pipeline._initialize_llm_client"):
                with patch("docling_graph.pipeline.GraphConverter"):
                    with patch("docling_graph.pipeline.CSVExporter"):
                        with patch("docling_graph.pipeline.JSONExporter"):
                            run_pipeline(config)

        # Nested directory should be created
        assert output_dir.exists()
