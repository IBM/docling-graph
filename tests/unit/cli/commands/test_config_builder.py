"""
Unit tests for config_builder module.
"""

from unittest.mock import MagicMock, patch

import pytest
import typer

from docling_graph.cli.commands.config_builder import (
    build_config_interactive,
    print_next_steps,
    _prompt_defaults,
    _prompt_docling,
    _prompt_models,
    _prompt_output,
)


class TestPromptDefaults:
    """Tests for _prompt_defaults function."""

    @patch("docling_graph.cli.commands.config_builder.typer.prompt")
    @patch("docling_graph.cli.commands.config_builder.rich_print")
    def test_prompt_defaults_llm_remote(self, mock_print, mock_prompt):
        """Test prompting defaults for LLM remote configuration."""
        # Mock user inputs
        mock_prompt.side_effect = [
            "one-to-one",  # processing_mode
            "llm",  # backend_type
            "remote",  # inference
            "csv",  # export_format
        ]

        result = _prompt_defaults()

        assert result["processing_mode"] == "one-to-one"
        assert result["backend_type"] == "llm"
        assert result["inference"] == "remote"
        assert result["export_format"] == "csv"
        assert mock_prompt.call_count == 4

    @patch("docling_graph.cli.commands.config_builder.typer.prompt")
    @patch("docling_graph.cli.commands.config_builder.rich_print")
    def test_prompt_defaults_vlm_forces_local(self, mock_print, mock_prompt):
        """Test that VLM backend forces local inference."""
        mock_prompt.side_effect = [
            "many-to-one",  # processing_mode
            "vlm",  # backend_type
            # No inference prompt - should be forced to local
            "cypher",  # export_format
        ]

        result = _prompt_defaults()

        assert result["backend_type"] == "vlm"
        assert result["inference"] == "local"
        # Prompt should only be called 3 times (no inference prompt)
        assert mock_prompt.call_count == 3


class TestPromptDocling:
    """Tests for _prompt_docling function."""

    @patch("docling_graph.cli.commands.config_builder.typer.confirm")
    @patch("docling_graph.cli.commands.config_builder.typer.prompt")
    @patch("docling_graph.cli.commands.config_builder.rich_print")
    def test_prompt_docling_with_exports(self, mock_print, mock_prompt, mock_confirm):
        """Test docling configuration with export options."""
        mock_prompt.return_value = "ocr"
        mock_confirm.side_effect = [True, True, False]  # docling_json, markdown, per_page

        result = _prompt_docling()

        assert result["pipeline"] == "ocr"
        assert result["export"]["docling_json"] is True
        assert result["export"]["markdown"] is True
        assert result["export"]["per_page_markdown"] is False

    @patch("docling_graph.cli.commands.config_builder.typer.confirm")
    @patch("docling_graph.cli.commands.config_builder.typer.prompt")
    @patch("docling_graph.cli.commands.config_builder.rich_print")
    def test_prompt_docling_vision_pipeline(self, mock_print, mock_prompt, mock_confirm):
        """Test selecting vision pipeline."""
        mock_prompt.return_value = "vision"
        mock_confirm.side_effect = [False, False, True]

        result = _prompt_docling()

        assert result["pipeline"] == "vision"
        assert result["export"]["docling_json"] is False
        assert result["export"]["per_page_markdown"] is True


class TestPromptModels:
    """Tests for _prompt_models function."""

    @patch("docling_graph.cli.commands.config_builder._prompt_llm_local_models")
    @patch("docling_graph.cli.commands.config_builder.rich_print")
    def test_prompt_models_llm_local(self, mock_print, mock_prompt_llm_local):
        """Test model prompting for LLM local configuration."""
        mock_prompt_llm_local.return_value = (
            "numind/NuExtract",  # vlm
            "llama3:8b",  # llm_local
            "gpt-4",  # llm_remote
            "openai",  # remote_provider
            "ollama",  # local_provider
        )

        result = _prompt_models("llm", "local")

        assert result["llm"]["local"]["default_model"] == "llama3:8b"
        assert result["llm"]["local"]["provider"] == "ollama"
        assert "providers" in result["llm"]

    @patch("docling_graph.cli.commands.config_builder._prompt_vlm_models")
    @patch("docling_graph.cli.commands.config_builder.rich_print")
    def test_prompt_models_vlm(self, mock_print, mock_prompt_vlm):
        """Test model prompting for VLM configuration."""
        mock_prompt_vlm.return_value = (
            "numind/NuExtract",  # vlm
            "llama3:8b",  # llm_local
            "gpt-4",  # llm_remote
            "openai",  # remote_provider
            "docling",  # local_provider
        )

        result = _prompt_models("vlm", "local")

        assert result["vlm"]["local"]["default_model"] == "numind/NuExtract"
        assert result["vlm"]["local"]["provider"] == "docling"


class TestPromptOutput:
    """Tests for _prompt_output function."""

    @patch("docling_graph.cli.commands.config_builder.typer.confirm")
    @patch("docling_graph.cli.commands.config_builder.typer.prompt")
    @patch("docling_graph.cli.commands.config_builder.rich_print")
    def test_prompt_output_with_visualizations(self, mock_print, mock_prompt, mock_confirm):
        """Test output configuration with visualizations enabled."""
        mock_prompt.return_value = "outputs"
        mock_confirm.side_effect = [True, True]  # visualizations, markdown

        result = _prompt_output()

        assert result["default_directory"] == "outputs"
        assert result["create_visualizations"] is True
        assert result["create_markdown"] is True

    @patch("docling_graph.cli.commands.config_builder.typer.confirm")
    @patch("docling_graph.cli.commands.config_builder.typer.prompt")
    @patch("docling_graph.cli.commands.config_builder.rich_print")
    def test_prompt_output_without_visualizations(self, mock_print, mock_prompt, mock_confirm):
        """Test output configuration with visualizations disabled."""
        mock_prompt.return_value = "my_outputs"
        mock_confirm.side_effect = [False, False]

        result = _prompt_output()

        assert result["default_directory"] == "my_outputs"
        assert result["create_visualizations"] is False
        assert result["create_markdown"] is False


class TestBuildConfigInteractive:
    """Tests for build_config_interactive function."""

    @patch("docling_graph.cli.commands.config_builder._prompt_output")
    @patch("docling_graph.cli.commands.config_builder._prompt_models")
    @patch("docling_graph.cli.commands.config_builder._prompt_docling")
    @patch("docling_graph.cli.commands.config_builder._prompt_defaults")
    @patch("docling_graph.cli.commands.config_builder.rich_print")
    def test_build_config_interactive_complete(
        self, mock_print, mock_defaults, mock_docling, mock_models, mock_output
    ):
        """Test building complete configuration interactively."""
        # Mock all subsections
        mock_defaults.return_value = {
            "processing_mode": "many-to-one",
            "backend_type": "llm",
            "inference": "local",
            "export_format": "csv",
        }
        mock_docling.return_value = {
            "pipeline": "ocr",
            "export": {"docling_json": True, "markdown": True, "per_page_markdown": False},
        }
        mock_models.return_value = {
            "vlm": {"local": {"default_model": "numind/NuExtract", "provider": "docling"}},
            "llm": {
                "local": {"default_model": "llama3:8b", "provider": "ollama"},
                "remote": {"default_model": "gpt-4", "provider": "openai"},
                "providers": {},
            },
        }
        mock_output.return_value = {
            "default_directory": "outputs",
            "create_visualizations": True,
            "create_markdown": True,
        }

        result = build_config_interactive()

        # Verify structure
        assert "defaults" in result
        assert "docling" in result
        assert "models" in result
        assert "output" in result

        # Verify content
        assert result["defaults"]["processing_mode"] == "many-to-one"
        assert result["docling"]["pipeline"] == "ocr"
        assert result["models"]["llm"]["local"]["provider"] == "ollama"
        assert result["output"]["default_directory"] == "outputs"


class TestPrintNextSteps:
    """Tests for print_next_steps function."""

    @patch("docling_graph.cli.commands.config_builder.rich_print")
    def test_print_next_steps_ollama(self, mock_print):
        """Test printing next steps for Ollama configuration."""
        config = {
            "defaults": {"inference": "local", "backend_type": "llm"},
            "models": {
                "llm": {
                    "local": {"default_model": "llama3:8b", "provider": "ollama"},
                }
            },
        }

        print_next_steps(config)

        # Check that relevant instructions were printed
        calls = [str(call) for call in mock_print.call_args_list]
        assert any("ollama serve" in str(call) for call in calls)
        assert any("ollama pull" in str(call) for call in calls)

    @patch("docling_graph.cli.commands.config_builder.rich_print")
    def test_print_next_steps_vllm(self, mock_print):
        """Test printing next steps for vLLM configuration."""
        config = {
            "defaults": {"inference": "local", "backend_type": "llm"},
            "models": {
                "llm": {
                    "local": {"default_model": "llama-3.1-8b", "provider": "vllm"},
                }
            },
        }

        print_next_steps(config)

        calls = [str(call) for call in mock_print.call_args_list]
        assert any("vllm serve" in str(call) for call in calls)

    @patch("docling_graph.cli.commands.config_builder.rich_print")
    def test_print_next_steps_remote(self, mock_print):
        """Test printing next steps for remote API configuration."""
        config = {
            "defaults": {"inference": "remote", "backend_type": "llm"},
            "models": {
                "llm": {
                    "remote": {"default_model": "gpt-4", "provider": "openai"},
                }
            },
        }

        print_next_steps(config)

        calls = [str(call) for call in mock_print.call_args_list]
        assert any("API" in str(call) and "KEY" in str(call) for call in calls)


class TestPromptHelpers:
    """Tests for helper prompt functions."""

    @patch("docling_graph.cli.commands.config_builder.typer.prompt")
    @patch("docling_graph.cli.commands.config_builder.rich_print")
    def test_prompt_llm_local_models_ollama(self, mock_print, mock_prompt):
        """Test prompting for Ollama local models."""
        from docling_graph.cli.commands.config_builder import _prompt_llm_local_models

        mock_prompt.side_effect = ["ollama", "llama3:8b"]

        result = _prompt_llm_local_models()

        assert result[1] == "llama3:8b"  # llm_local
        assert result[4] == "ollama"  # local_provider

    @patch("docling_graph.cli.commands.config_builder.typer.prompt")
    @patch("docling_graph.cli.commands.config_builder.rich_print")
    def test_prompt_llm_remote_models(self, mock_print, mock_prompt):
        """Test prompting for remote LLM models."""
        from docling_graph.cli.commands.config_builder import _prompt_llm_remote_models

        mock_prompt.side_effect = ["mistral", "mistral-large-latest"]

        result = _prompt_llm_remote_models()

        assert result[2] == "mistral-large-latest"  # llm_remote
        assert result[3] == "mistral"  # remote_provider
