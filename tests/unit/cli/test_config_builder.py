"""
Tests for interactive configuration builder.
"""

import pytest
from unittest.mock import patch, MagicMock
from typing import Dict, Any

from docling_graph.cli.commands.config_builder import (
    build_config_interactive,
    _prompt_defaults,
    _prompt_docling,
    _prompt_models,
    _prompt_output,
)


class TestPromptDefaults:
    """Test default settings prompt."""

    @patch("typer.prompt")
    def test_prompt_defaults_returns_dict(self, mock_prompt):
        """Should return dictionary with default settings."""
        mock_prompt.side_effect = [
            "one-to-one",  # processing_mode
            "llm",  # backend_type
            "local",  # inference
            "csv",  # export_format
        ]

        result = _prompt_defaults()

        assert isinstance(result, dict)
        assert "processing_mode" in result
        assert "backend_type" in result
        assert "inference" in result
        assert "export_format" in result

    @patch("typer.prompt")
    def test_prompt_defaults_vlm_sets_local_inference(self, mock_prompt):
        """Should force local inference for VLM backend."""
        mock_prompt.side_effect = [
            "one-to-one",  # processing_mode
            "vlm",  # backend_type
            "csv",  # export_format
        ]

        result = _prompt_defaults()

        assert result["backend_type"] == "vlm"
        assert result["inference"] == "local"


class TestPromptDocling:
    """Test Docling configuration prompt."""

    @patch("typer.prompt")
    @patch("typer.confirm")
    def test_prompt_docling_returns_config(self, mock_confirm, mock_prompt):
        """Should return Docling configuration dictionary."""
        mock_prompt.return_value = "ocr"
        mock_confirm.side_effect = [True, True, False]

        result = _prompt_docling()

        assert isinstance(result, dict)
        assert "pipeline" in result
        assert "export" in result
        assert result["pipeline"] == "ocr"

    @patch("typer.prompt")
    @patch("typer.confirm")
    def test_prompt_docling_export_settings(self, mock_confirm, mock_prompt):
        """Should include export settings in result."""
        mock_prompt.return_value = "vision"
        mock_confirm.side_effect = [True, False, True]

        result = _prompt_docling()

        assert result["export"]["docling_json"] is True
        assert result["export"]["markdown"] is False
        assert result["export"]["per_page_markdown"] is True


class TestPromptModels:
    """Test model configuration prompt."""

    @patch("typer.prompt")
    @patch("typer.confirm")
    def test_prompt_models_llm_local(self, mock_confirm, mock_prompt):
        """Should configure local LLM model."""
        mock_prompt.side_effect = [
            "vllm",  # local_provider
            "llama-3.1-8b",  # llm_model
        ]

        result = _prompt_models("llm", "local")

        assert "vlm" in result
        assert "llm" in result
        assert result["llm"]["local"]["provider"] == "vllm"

    @patch("typer.prompt")
    def test_prompt_models_vlm_backend(self, mock_prompt):
        """Should configure VLM model."""
        mock_prompt.return_value = "numind/NuExtract-2.0-8B"

        result = _prompt_models("vlm", "local")

        assert "vlm" in result
        assert result["vlm"]["local"]["provider"] == "docling"


class TestPromptOutput:
    """Test output configuration prompt."""

    @patch("typer.prompt")
    @patch("typer.confirm")
    def test_prompt_output_returns_config(self, mock_confirm, mock_prompt):
        """Should return output configuration."""
        mock_prompt.return_value = "outputs"
        mock_confirm.side_effect = [True, True]

        result = _prompt_output()

        assert isinstance(result, dict)
        assert "default_directory" in result
        assert "create_visualizations" in result
        assert "create_markdown" in result


class TestBuildConfigInteractive:
    """Test complete interactive config building."""

    @patch("typer.prompt")
    @patch("typer.confirm")
    def test_build_config_interactive_returns_complete_config(
        self, mock_confirm, mock_prompt
    ):
        """Should return complete configuration dictionary."""
        # Setup mock responses for all prompts
        mock_prompt.side_effect = [
            # _prompt_defaults
            "one-to-one",
            "llm",
            "remote",
            "csv",
            # _prompt_docling
            "ocr",
            # _prompt_models
            "mistral",
            "mistral-small-latest",
            # _prompt_output
            "outputs",
        ]
        mock_confirm.side_effect = [True, True, False, True, True]

        result = build_config_interactive()

        assert isinstance(result, dict)
        assert "defaults" in result
        assert "docling" in result
        assert "models" in result
        assert "output" in result

    @patch("typer.prompt")
    @patch("typer.confirm")
    def test_build_config_has_all_required_sections(
        self, mock_confirm, mock_prompt
    ):
        """Should have all required configuration sections."""
        mock_prompt.side_effect = [
            "one-to-one", "llm", "local", "csv",  # defaults
            "vision",  # docling
            "ollama", "llama3:8b",  # models - local
            "outputs",  # output
        ]
        mock_confirm.side_effect = [True, True, False, True, True]

        result = build_config_interactive()

        assert result["defaults"]["processing_mode"] == "one-to-one"
        assert result["defaults"]["backend_type"] == "llm"
        assert result["docling"]["pipeline"] == "vision"
        assert result["output"]["default_directory"] == "outputs"
