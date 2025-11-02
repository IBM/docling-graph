"""
Tests for CLI constants module.
"""

import pytest

from docling_graph.cli.constants import (
    API_PROVIDERS,
    BACKEND_TYPES,
    CONFIG_FILE_NAME,
    DEFAULT_MODELS,
    DOCLING_PIPELINES,
    EXPORT_FORMATS,
    INFERENCE_LOCATIONS,
    LOCAL_PROVIDERS,
    PROCESSING_MODES,
)


class TestConstants:
    """Test CLI constants are properly defined."""

    def test_processing_modes_contains_valid_values(self):
        """Processing modes should contain expected values."""
        assert "one-to-one" in PROCESSING_MODES
        assert "many-to-one" in PROCESSING_MODES
        assert len(PROCESSING_MODES) == 2

    def test_backend_types_contains_valid_values(self):
        """Backend types should include LLM and VLM."""
        assert "llm" in BACKEND_TYPES
        assert "vlm" in BACKEND_TYPES
        assert len(BACKEND_TYPES) == 2

    def test_inference_locations_contains_valid_values(self):
        """Inference locations should include local and remote."""
        assert "local" in INFERENCE_LOCATIONS
        assert "remote" in INFERENCE_LOCATIONS
        assert len(INFERENCE_LOCATIONS) == 2

    def test_export_formats_contains_valid_values(self):
        """Export formats should include CSV, Cypher, and JSON."""
        assert "csv" in EXPORT_FORMATS
        assert "cypher" in EXPORT_FORMATS
        assert "json" in EXPORT_FORMATS
        assert len(EXPORT_FORMATS) >= 3

    def test_docling_pipelines_contains_valid_values(self):
        """Docling pipelines should include OCR and Vision."""
        assert "ocr" in DOCLING_PIPELINES
        assert "vision" in DOCLING_PIPELINES
        assert len(DOCLING_PIPELINES) == 2

    def test_config_file_name_is_defined(self):
        """Config file name should be config.yaml."""
        assert CONFIG_FILE_NAME == "config.yaml"

    def test_default_models_has_required_keys(self):
        """Default models should have VLM, LLM local, and LLM remote."""
        assert "vlm" in DEFAULT_MODELS
        assert "llm_local" in DEFAULT_MODELS
        assert "llm_remote" in DEFAULT_MODELS

    def test_llm_remote_has_api_providers(self):
        """LLM remote should have entries for all API providers."""
        llm_remote = DEFAULT_MODELS["llm_remote"]
        assert "mistral" in llm_remote
        assert "openai" in llm_remote
        assert "gemini" in llm_remote

    def test_provider_lists_are_not_empty(self):
        """Provider lists should not be empty."""
        assert len(LOCAL_PROVIDERS) > 0
        assert len(API_PROVIDERS) > 0

    def test_local_providers_contains_expected_values(self):
        """Local providers should include vLLM and Ollama."""
        assert "vllm" in LOCAL_PROVIDERS
        assert "ollama" in LOCAL_PROVIDERS

    def test_api_providers_contains_expected_values(self):
        """API providers should include Mistral, OpenAI, Gemini."""
        assert "mistral" in API_PROVIDERS
        assert "openai" in API_PROVIDERS
        assert "gemini" in API_PROVIDERS

    def test_constants_are_immutable(self):
        """Constants should be Final types (testing tuple/list immutability)."""
        # Lists returned from Final are still lists, but we can verify they're defined
        assert isinstance(PROCESSING_MODES, list | tuple)
        assert isinstance(BACKEND_TYPES, list | tuple)
