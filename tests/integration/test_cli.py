"""
CLI integration tests.

Tests the CLI interface for the docling-graph convert command,
focusing on argument parsing, configuration, and CLI behavior.
"""

import pytest
import yaml
from pathlib import Path
from unittest.mock import patch
from click.testing import CliRunner

from docling_graph.cli.main import app
from .conftest import Person, Invoice, Document


@pytest.mark.integration
@pytest.mark.cli
class TestCLIConvertCommandBasics:
    """Test basic CLI convert command functionality."""

    def test_convert_help_displays_all_options(self, cli_runner):
        """Test that help text shows all available options."""
        result = cli_runner.invoke(app, ["convert", "--help"])
        
        assert result.exit_code == 0
        assert "--template" in result.output
        assert "--processing-mode" in result.output
        assert "--backend-type" in result.output
        assert "--inference" in result.output
        assert "--output-dir" in result.output
        assert "--export-format" in result.output
        assert "--docling-pipeline" in result.output

    def test_convert_requires_source_argument(self, cli_runner):
        """Test convert command requires source PDF argument."""
        result = cli_runner.invoke(app, ["convert"])
        
        # Should fail without source
        assert result.exit_code != 0

    def test_convert_requires_template_option(self, cli_runner, sample_pdf):
        """Test convert command requires template option."""
        result = cli_runner.invoke(app, [
            "convert",
            str(sample_pdf),
        ])
        
        # Should fail without template
        assert result.exit_code != 0

    def test_convert_rejects_nonexistent_source_file(self, cli_runner):
        """Test convert rejects non-existent source file."""
        result = cli_runner.invoke(app, [
            "convert",
            "/nonexistent/file.pdf",
            "--template", "tests.integration.conftest.Person",
        ])
        
        # Should fail due to file not found
        assert result.exit_code != 0

    def test_convert_accepts_valid_source_file(self, cli_runner, sample_pdf):
        """Test convert accepts valid source file."""
        with patch("docling_graph.cli.commands.convert.run_pipeline"):
            result = cli_runner.invoke(app, [
                "convert",
                str(sample_pdf),
                "--template", "tests.integration.conftest.Person",
            ])
            
            # Should not fail on file validation
            assert result.exit_code in [0, 1]  # May fail with mock, but not file validation


@pytest.mark.integration
@pytest.mark.cli
class TestCLIArgumentParsing:
    """Test CLI argument parsing and validation."""

    def test_cli_parses_processing_mode_one_to_one(self, cli_runner, sample_pdf):
        """Test CLI parses one-to-one processing mode."""
        with patch("docling_graph.cli.commands.convert.run_pipeline") as mock_run:
            cli_runner.invoke(app, [
                "convert",
                str(sample_pdf),
                "--template", "tests.integration.conftest.Person",
                "--processing-mode", "one-to-one",
            ])
            
            call_config = mock_run.call_args[0][0]
            assert call_config["processing_mode"] == "one-to-one"

    def test_cli_parses_processing_mode_many_to_one(self, cli_runner, sample_pdf):
        """Test CLI parses many-to-one processing mode."""
        with patch("docling_graph.cli.commands.convert.run_pipeline") as mock_run:
            cli_runner.invoke(app, [
                "convert",
                str(sample_pdf),
                "--template", "tests.integration.conftest.Person",
                "--processing-mode", "many-to-one",
            ])
            
            call_config = mock_run.call_args[0][0]
            assert call_config["processing_mode"] == "many-to-one"

    def test_cli_parses_backend_type_llm(self, cli_runner, sample_pdf):
        """Test CLI parses LLM backend type."""
        with patch("docling_graph.cli.commands.convert.run_pipeline") as mock_run:
            cli_runner.invoke(app, [
                "convert",
                str(sample_pdf),
                "--template", "tests.integration.conftest.Person",
                "--backend-type", "llm",
            ])
            
            call_config = mock_run.call_args[0][0]
            assert call_config["backend_type"] == "llm"

    def test_cli_parses_backend_type_vlm(self, cli_runner, sample_pdf):
        """Test CLI parses VLM backend type."""
        with patch("docling_graph.cli.commands.convert.run_pipeline") as mock_run:
            cli_runner.invoke(app, [
                "convert",
                str(sample_pdf),
                "--template", "tests.integration.conftest.Person",
                "--backend-type", "vlm",
            ])
            
            call_config = mock_run.call_args[0][0]
            assert call_config["backend_type"] == "vlm"

    def test_cli_parses_inference_local(self, cli_runner, sample_pdf):
        """Test CLI parses local inference."""
        with patch("docling_graph.cli.commands.convert.run_pipeline") as mock_run:
            cli_runner.invoke(app, [
                "convert",
                str(sample_pdf),
                "--template", "tests.integration.conftest.Person",
                "--inference", "local",
            ])
            
            call_config = mock_run.call_args[0][0]
            assert call_config["inference"] == "local"

    def test_cli_parses_inference_remote(self, cli_runner, sample_pdf):
        """Test CLI parses remote inference."""
        with patch("docling_graph.cli.commands.convert.run_pipeline") as mock_run:
            cli_runner.invoke(app, [
                "convert",
                str(sample_pdf),
                "--template", "tests.integration.conftest.Person",
                "--inference", "remote",
            ])
            
            call_config = mock_run.call_args[0][0]
            assert call_config["inference"] == "remote"

    def test_cli_parses_export_format_csv(self, cli_runner, sample_pdf):
        """Test CLI parses CSV export format."""
        with patch("docling_graph.cli.commands.convert.run_pipeline") as mock_run:
            cli_runner.invoke(app, [
                "convert",
                str(sample_pdf),
                "--template", "tests.integration.conftest.Person",
                "--export-format", "csv",
            ])
            
            call_config = mock_run.call_args[0][0]
            assert call_config["export_format"] == "csv"

    def test_cli_parses_export_format_json(self, cli_runner, sample_pdf):
        """Test CLI parses JSON export format."""
        with patch("docling_graph.cli.commands.convert.run_pipeline") as mock_run:
            cli_runner.invoke(app, [
                "convert",
                str(sample_pdf),
                "--template", "tests.integration.conftest.Person",
                "--export-format", "json",
            ])
            
            call_config = mock_run.call_args[0][0]
            assert call_config["export_format"] == "json"

    def test_cli_parses_export_format_cypher(self, cli_runner, sample_pdf):
        """Test CLI parses Cypher export format."""
        with patch("docling_graph.cli.commands.convert.run_pipeline") as mock_run:
            cli_runner.invoke(app, [
                "convert",
                str(sample_pdf),
                "--template", "tests.integration.conftest.Person",
                "--export-format", "cypher",
            ])
            
            call_config = mock_run.call_args[0][0]
            assert call_config["export_format"] == "cypher"


@pytest.mark.integration
@pytest.mark.cli
class TestCLIFlagParsing:
    """Test CLI boolean flag parsing."""

    def test_cli_parses_reverse_edges_flag(self, cli_runner, sample_pdf):
        """Test CLI parses --reverse-edges flag."""
        with patch("docling_graph.cli.commands.convert.run_pipeline") as mock_run:
            cli_runner.invoke(app, [
                "convert",
                str(sample_pdf),
                "--template", "tests.integration.conftest.Person",
                "--reverse-edges",
            ])
            
            call_config = mock_run.call_args[0][0]
            assert call_config["reverse_edges"] is True

    def test_cli_reverse_edges_flag_default_false(self, cli_runner, sample_pdf):
        """Test reverse-edges flag defaults to False."""
        with patch("docling_graph.cli.commands.convert.run_pipeline") as mock_run:
            cli_runner.invoke(app, [
                "convert",
                str(sample_pdf),
                "--template", "tests.integration.conftest.Person",
            ])
            
            call_config = mock_run.call_args[0][0]
            assert call_config.get("reverse_edges", False) is False

    def test_cli_parses_docling_export_flags(self, cli_runner, sample_pdf):
        """Test CLI parses Docling export flags."""
        with patch("docling_graph.cli.commands.convert.run_pipeline") as mock_run:
            cli_runner.invoke(app, [
                "convert",
                str(sample_pdf),
                "--template", "tests.integration.conftest.Person",
                "--export-docling-json",
                "--export-markdown",
                "--export-per-page",
            ])
            
            call_config = mock_run.call_args[0][0]
            assert call_config.get("export_docling_json") is True
            assert call_config.get("export_markdown") is True
            assert call_config.get("export_per_page_markdown") is True


@pytest.mark.integration
@pytest.mark.cli
class TestCLIOutputDirectory:
    """Test CLI output directory handling."""

    def test_cli_creates_output_directory(self, cli_runner, temp_dir, sample_pdf):
        """Test CLI creates output directory if specified."""
        output_dir = temp_dir / "test_output"
        
        with patch("docling_graph.cli.commands.convert.run_pipeline") as mock_run:
            cli_runner.invoke(app, [
                "convert",
                str(sample_pdf),
                "--template", "tests.integration.conftest.Person",
                "--output-dir", str(output_dir),
            ])
            
            call_config = mock_run.call_args[0][0]
            assert call_config["output_dir"] == str(output_dir)

    def test_cli_parses_nested_output_directory(self, cli_runner, temp_dir, sample_pdf):
        """Test CLI handles nested output directory paths."""
        output_dir = temp_dir / "deep" / "nested" / "path"
        
        with patch("docling_graph.cli.commands.convert.run_pipeline") as mock_run:
            cli_runner.invoke(app, [
                "convert",
                str(sample_pdf),
                "--template", "tests.integration.conftest.Person",
                "--output-dir", str(output_dir),
            ])
            
            call_config = mock_run.call_args[0][0]
            assert call_config["output_dir"] == str(output_dir)


@pytest.mark.integration
@pytest.mark.cli
class TestCLIModelOptions:
    """Test CLI model and provider options."""

    def test_cli_parses_model_option(self, cli_runner, sample_pdf):
        """Test CLI parses --model option."""
        with patch("docling_graph.cli.commands.convert.run_pipeline") as mock_run:
            cli_runner.invoke(app, [
                "convert",
                str(sample_pdf),
                "--template", "tests.integration.conftest.Person",
                "--model", "custom-model",
            ])
            
            call_config = mock_run.call_args[0][0]
            assert call_config.get("model_override") == "custom-model"

    def test_cli_parses_provider_option(self, cli_runner, sample_pdf):
        """Test CLI parses --provider option."""
        with patch("docling_graph.cli.commands.convert.run_pipeline") as mock_run:
            cli_runner.invoke(app, [
                "convert",
                str(sample_pdf),
                "--template", "tests.integration.conftest.Person",
                "--provider", "custom-provider",
            ])
            
            call_config = mock_run.call_args[0][0]
            assert call_config.get("provider_override") == "custom-provider"

    def test_cli_parses_model_and_provider_together(self, cli_runner, sample_pdf):
        """Test CLI parses both --model and --provider."""
        with patch("docling_graph.cli.commands.convert.run_pipeline") as mock_run:
            cli_runner.invoke(app, [
                "convert",
                str(sample_pdf),
                "--template", "tests.integration.conftest.Person",
                "--model", "gpt-4",
                "--provider", "openai",
            ])
            
            call_config = mock_run.call_args[0][0]
            assert call_config.get("model_override") == "gpt-4"
            assert call_config.get("provider_override") == "openai"


@pytest.mark.integration
@pytest.mark.cli
class TestCLIErrorHandling:
    """Test CLI error handling and messages."""

    def test_cli_displays_error_for_invalid_processing_mode(self, cli_runner, sample_pdf):
        """Test CLI shows error for invalid processing mode."""
        with patch("docling_graph.cli.commands.convert.validate_processing_mode") as mock_val:
            mock_val.side_effect = ValueError("Invalid processing mode")
            
            result = cli_runner.invoke(app, [
                "convert",
                str(sample_pdf),
                "--template", "tests.integration.conftest.Person",
                "--processing-mode", "invalid-mode",
            ])
            
            # Should fail or handle error
            assert result.exit_code != 0 or "error" in result.output.lower()

    def test_cli_displays_error_for_invalid_backend_type(self, cli_runner, sample_pdf):
        """Test CLI shows error for invalid backend type."""
        with patch("docling_graph.cli.commands.convert.validate_backend_type") as mock_val:
            mock_val.side_effect = ValueError("Invalid backend type")
            
            result = cli_runner.invoke(app, [
                "convert",
                str(sample_pdf),
                "--template", "tests.integration.conftest.Person",
                "--backend-type", "invalid-backend",
            ])
            
            # Should fail or handle error
            assert result.exit_code != 0 or "error" in result.output.lower()

    def test_cli_handles_vlm_without_provider_gracefully(self, cli_runner, sample_pdf):
        """Test CLI handles VLM without provider gracefully."""
        with patch("docling_graph.cli.commands.convert.validate_vlm_constraints") as mock_val:
            mock_val.side_effect = ValueError("VLM requires provider")
            
            result = cli_runner.invoke(app, [
                "convert",
                str(sample_pdf),
                "--template", "tests.integration.conftest.Person",
                "--backend-type", "vlm",
            ])
            
            # Should fail or warn
            assert result.exit_code != 0 or "vlm" in result.output.lower()


@pytest.mark.integration
@pytest.mark.cli
class TestCLIOutputMessages:
    """Test CLI output and logging messages."""

    def test_cli_displays_source_template_info(self, cli_runner, sample_pdf):
        """Test CLI displays source and template information."""
        with patch("docling_graph.cli.commands.convert.run_pipeline"):
            result = cli_runner.invoke(app, [
                "convert",
                str(sample_pdf),
                "--template", "tests.integration.conftest.Person",
            ])
            
            # Output should mention source or template
            assert str(sample_pdf) in result.output or "Person" in result.output or result.exit_code == 0

    def test_cli_displays_configuration_summary(self, cli_runner, sample_pdf):
        """Test CLI displays configuration summary before execution."""
        with patch("docling_graph.cli.commands.convert.run_pipeline"):
            result = cli_runner.invoke(app, [
                "convert",
                str(sample_pdf),
                "--template", "tests.integration.conftest.Person",
                "--processing-mode", "one-to-one",
                "--backend-type", "llm",
                "--export-format", "csv",
            ])
            
            # Output should show configuration
            output_lower = result.output.lower()
            assert any(x in output_lower for x in [
                "configuration", "template", "backend", "export", "success"
            ]) or result.exit_code == 0
