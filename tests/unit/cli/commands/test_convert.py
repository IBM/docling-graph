"""
Tests for convert command.
"""

from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest
import typer

from docling_graph.cli.commands.convert import convert_command


class TestConvertCommand:
    """Test convert command functionality."""

    @patch("docling_graph.cli.commands.convert.run_pipeline")
    @patch("docling_graph.cli.commands.convert.load_config")
    def test_convert_command_with_all_defaults(self, mock_load_config, mock_run_pipeline, tmp_path):
        """Should process document with default configuration."""
        # Create temporary document
        doc_path = tmp_path / "test.pdf"
        doc_path.write_text("test")

        mock_load_config.return_value = {
            "defaults": {
                "processing_mode": "many-to-one",
                "backend_type": "llm",
                "inference": "local",
                "export_format": "csv",
            },
            "docling": {
                "pipeline": "ocr",
                "export": {"docling_json": True, "markdown": True, "per_page_markdown": False},
            },
        }

        with patch(
            "docling_graph.cli.commands.convert.validate_processing_mode",
            return_value="many-to-one",
        ):
            with patch(
                "docling_graph.cli.commands.convert.validate_backend_type", return_value="llm"
            ):
                with patch(
                    "docling_graph.cli.commands.convert.validate_inference", return_value="local"
                ):
                    with patch(
                        "docling_graph.cli.commands.convert.validate_docling_config",
                        return_value="ocr",
                    ):
                        with patch(
                            "docling_graph.cli.commands.convert.validate_export_format",
                            return_value="csv",
                        ):
                            with patch(
                                "docling_graph.cli.commands.convert.validate_vlm_constraints"
                            ):
                                convert_command(
                                    source=doc_path,
                                    template="templates.invoice.Invoice",
                                )

        mock_run_pipeline.assert_called_once()

    @patch("docling_graph.cli.commands.convert.load_config")
    def test_convert_command_cli_overrides_config(self, mock_load_config, tmp_path):
        """Should allow CLI arguments to override config file."""
        doc_path = tmp_path / "test.pdf"
        doc_path.write_text("test")

        mock_load_config.return_value = {
            "defaults": {
                "processing_mode": "many-to-one",
                "backend_type": "llm",
                "inference": "remote",
                "export_format": "csv",
            },
            "docling": {"pipeline": "ocr", "export": {}},
        }

        with patch("docling_graph.cli.commands.convert.run_pipeline") as mock_pipeline:
            with patch(
                "docling_graph.cli.commands.convert.validate_processing_mode",
                return_value="one-to-one",
            ):
                with patch(
                    "docling_graph.cli.commands.convert.validate_backend_type", return_value="llm"
                ):
                    with patch(
                        "docling_graph.cli.commands.convert.validate_inference",
                        return_value="local",
                    ):
                        with patch(
                            "docling_graph.cli.commands.convert.validate_docling_config",
                            return_value="ocr",
                        ):
                            with patch(
                                "docling_graph.cli.commands.convert.validate_export_format",
                                return_value="csv",
                            ):
                                with patch(
                                    "docling_graph.cli.commands.convert.validate_vlm_constraints"
                                ):
                                    convert_command(
                                        source=doc_path,
                                        template="templates.invoice.Invoice",
                                        processing_mode="one-to-one",
                                        inference="local",
                                    )

            # Verify overrides were applied
            called_config = mock_pipeline.call_args[0][0]
            assert called_config["processing_mode"] == "one-to-one"
            assert called_config["inference"] == "local"

    @patch(
        "docling_graph.cli.commands.convert.run_pipeline", side_effect=Exception("Pipeline error")
    )
    @patch("docling_graph.cli.commands.convert.load_config")
    def test_convert_command_pipeline_error_exits(
        self, mock_load_config, mock_run_pipeline, tmp_path
    ):
        """Should exit with error on pipeline failure."""
        doc_path = tmp_path / "test.pdf"
        doc_path.write_text("test")

        mock_load_config.return_value = {
            "defaults": {
                "processing_mode": "many-to-one",
                "backend_type": "llm",
                "inference": "local",
                "export_format": "csv",
            },
            "docling": {"pipeline": "ocr", "export": {}},
        }

        with patch(
            "docling_graph.cli.commands.convert.validate_processing_mode",
            return_value="many-to-one",
        ):
            with patch(
                "docling_graph.cli.commands.convert.validate_backend_type", return_value="llm"
            ):
                with patch(
                    "docling_graph.cli.commands.convert.validate_inference", return_value="local"
                ):
                    with patch(
                        "docling_graph.cli.commands.convert.validate_docling_config",
                        return_value="ocr",
                    ):
                        with patch(
                            "docling_graph.cli.commands.convert.validate_export_format",
                            return_value="csv",
                        ):
                            with patch(
                                "docling_graph.cli.commands.convert.validate_vlm_constraints"
                            ):
                                with pytest.raises(typer.Exit) as exc_info:
                                    convert_command(
                                        source=doc_path,
                                        template="templates.invoice.Invoice",
                                    )
                                assert exc_info.value.exit_code == 1

    @patch("docling_graph.cli.commands.convert.run_pipeline")
    @patch("docling_graph.cli.commands.convert.load_config")
    def test_convert_command_docling_exports(self, mock_load_config, mock_run_pipeline, tmp_path):
        """Should respect Docling export flags."""
        doc_path = tmp_path / "test.pdf"
        doc_path.write_text("test")

        mock_load_config.return_value = {
            "defaults": {
                "processing_mode": "many-to-one",
                "backend_type": "llm",
                "inference": "local",
                "export_format": "csv",
            },
            "docling": {"pipeline": "ocr", "export": {"docling_json": True, "markdown": False}},
        }

        with patch(
            "docling_graph.cli.commands.convert.validate_processing_mode",
            return_value="many-to-one",
        ):
            with patch(
                "docling_graph.cli.commands.convert.validate_backend_type", return_value="llm"
            ):
                with patch(
                    "docling_graph.cli.commands.convert.validate_inference", return_value="local"
                ):
                    with patch(
                        "docling_graph.cli.commands.convert.validate_docling_config",
                        return_value="ocr",
                    ):
                        with patch(
                            "docling_graph.cli.commands.convert.validate_export_format",
                            return_value="csv",
                        ):
                            with patch(
                                "docling_graph.cli.commands.convert.validate_vlm_constraints"
                            ):
                                convert_command(
                                    source=doc_path,
                                    template="templates.invoice.Invoice",
                                    export_markdown=False,
                                )

        called_config = mock_run_pipeline.call_args[0][0]
        assert called_config["export_markdown"] is False
