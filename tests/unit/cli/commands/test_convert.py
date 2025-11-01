"""
Unit tests for convert command module.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import typer

from docling_graph.cli.commands.convert import convert_command


class TestConvertCommand:
    """Tests for convert_command function."""

    @patch("docling_graph.cli.commands.convert.run_pipeline")
    @patch("docling_graph.cli.commands.convert.load_config")
    @patch("docling_graph.cli.commands.convert.rich_print")
    def test_convert_command_basic(self, mock_print, mock_load_config, mock_run_pipeline, temp_dir):
        """Test basic convert command with minimal arguments."""
        # Create a dummy source file
        source_file = temp_dir / "test.pdf"
        source_file.write_text("dummy pdf")

        mock_load_config.return_value = {
            "defaults": {
                "processing_mode": "many-to-one",
                "backend_type": "llm",
                "inference": "local",
                "export_format": "csv",
            },
            "docling": {"pipeline": "ocr", "export": {}},
            "models": {},
        }

        convert_command(
            source=source_file,
            template="templates.Invoice",
        )

        # Verify pipeline was called
        mock_run_pipeline.assert_called_once()
        run_config = mock_run_pipeline.call_args[0][0]

        assert run_config["source"] == str(source_file)
        assert run_config["template"] == "templates.Invoice"
        assert run_config["processing_mode"] == "many-to-one"

    @patch("docling_graph.cli.commands.convert.run_pipeline")
    @patch("docling_graph.cli.commands.convert.load_config")
    @patch("docling_graph.cli.commands.convert.rich_print")
    def test_convert_command_with_overrides(
        self, mock_print, mock_load_config, mock_run_pipeline, temp_dir
    ):
        """Test convert command with CLI argument overrides."""
        source_file = temp_dir / "test.pdf"
        source_file.write_text("dummy")

        mock_load_config.return_value = {
            "defaults": {
                "processing_mode": "many-to-one",
                "backend_type": "llm",
                "inference": "local",
                "export_format": "csv",
            },
            "docling": {"pipeline": "ocr", "export": {}},
            "models": {},
        }

        convert_command(
            source=source_file,
            template="templates.Resume",
            processing_mode="one-to-one",
            backend_type="vlm",
            inference="local",
            export_format="json",
            model="custom-model",
            provider="ollama",
            reverse_edges=True,
        )

        run_config = mock_run_pipeline.call_args[0][0]

        # Verify overrides were applied
        assert run_config["processing_mode"] == "one-to-one"
        assert run_config["backend_type"] == "vlm"
        assert run_config["export_format"] == "json"
        assert run_config["model_override"] == "custom-model"
        assert run_config["provider_override"] == "ollama"
        assert run_config["reverse_edges"] is True

    @patch("docling_graph.cli.commands.convert.run_pipeline")
    @patch("docling_graph.cli.commands.convert.load_config")
    @patch("docling_graph.cli.commands.convert.rich_print")
    def test_convert_command_docling_export_options(
        self, mock_print, mock_load_config, mock_run_pipeline, temp_dir
    ):
        """Test convert command with Docling export options."""
        source_file = temp_dir / "test.pdf"
        source_file.write_text("dummy")

        mock_load_config.return_value = {
            "defaults": {"processing_mode": "many-to-one", "backend_type": "llm"},
            "docling": {
                "pipeline": "ocr",
                "export": {
                    "docling_json": True,
                    "markdown": True,
                    "per_page_markdown": False,
                },
            },
            "models": {},
        }

        convert_command(
            source=source_file,
            template="templates.Invoice",
            export_docling_json=True,
            export_markdown=True,
            export_per_page=True,
        )

        run_config = mock_run_pipeline.call_args[0][0]

        assert run_config["export_docling_json"] is True
        assert run_config["export_markdown"] is True
        assert run_config["export_per_page_markdown"] is True

    @patch("docling_graph.cli.commands.convert.validate_vlm_constraints")
    @patch("docling_graph.cli.commands.convert.load_config")
    @patch("docling_graph.cli.commands.convert.rich_print")
    def test_convert_command_vlm_validation(
        self, mock_print, mock_load_config, mock_validate, temp_dir
    ):
        """Test that VLM constraints are validated."""
        source_file = temp_dir / "test.pdf"
        source_file.write_text("dummy")

        mock_load_config.return_value = {
            "defaults": {"processing_mode": "many-to-one", "backend_type": "vlm"},
            "docling": {"pipeline": "ocr", "export": {}},
            "models": {},
        }

        # Simulate validation failure
        mock_validate.side_effect = typer.Exit(code=1)

        with pytest.raises(typer.Exit):
            convert_command(
                source=source_file,
                template="templates.Invoice",
                backend_type="vlm",
                inference="remote",  # Invalid for VLM
            )

    @patch("docling_graph.cli.commands.convert.run_pipeline")
    @patch("docling_graph.cli.commands.convert.load_config")
    @patch("docling_graph.cli.commands.convert.rich_print")
    def test_convert_command_pipeline_error(
        self, mock_print, mock_load_config, mock_run_pipeline, temp_dir
    ):
        """Test convert command when pipeline raises error."""
        source_file = temp_dir / "test.pdf"
        source_file.write_text("dummy")

        mock_load_config.return_value = {
            "defaults": {},
            "docling": {"pipeline": "ocr", "export": {}},
            "models": {},
        }
        mock_run_pipeline.side_effect = Exception("Pipeline failed")

        with pytest.raises(typer.Exit) as exc_info:
            convert_command(source=source_file, template="templates.Invoice")

        assert exc_info.value.exit_code == 1

    @patch("docling_graph.cli.commands.convert.run_pipeline")
    @patch("docling_graph.cli.commands.convert.load_config")
    @patch("docling_graph.cli.commands.convert.rich_print")
    def test_convert_command_custom_output_dir(
        self, mock_print, mock_load_config, mock_run_pipeline, temp_dir
    ):
        """Test convert command with custom output directory."""
        source_file = temp_dir / "test.pdf"
        source_file.write_text("dummy")
        output_dir = temp_dir / "custom_output"

        mock_load_config.return_value = {
            "defaults": {},
            "docling": {"pipeline": "ocr", "export": {}},
            "models": {},
        }

        convert_command(
            source=source_file,
            template="templates.Invoice",
            output_dir=output_dir,
        )

        run_config = mock_run_pipeline.call_args[0][0]
        assert run_config["output_dir"] == str(output_dir)
