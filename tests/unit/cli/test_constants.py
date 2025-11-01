"""
Unit tests for CLI constants.
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
    PROCESSING_MODES,
)


class TestConstants:
    """Tests for CLI constants to ensure they have expected values."""

    def test_config_file_name(self):
        """Test CONFIG_FILE_NAME is defined."""
        assert CONFIG_FILE_NAME == "config.yaml"

    def test_processing_modes(self):
        """Test PROCESSING_MODES contains expected values."""
        assert "one-to-one" in PROCESSING_MODES
        assert "many-to-one" in PROCESSING_MODES
        assert len(PROCESSING_MODES) == 2

    def test_backend_types(self):
        """Test BACKEND_TYPES contains expected values."""
        assert "llm" in BACKEND_TYPES
        assert "vlm" in BACKEND_TYPES
        assert len(BACKEND_TYPES) == 2

    def test_inference_locations(self):
        """Test INFERENCE_LOCATIONS contains expected values."""
        assert "local" in INFERENCE_LOCATIONS
        assert "remote" in INFERENCE_LOCATIONS
        assert len(INFERENCE_LOCATIONS) == 2

    def test_export_formats(self):
        """Test EXPORT_FORMATS contains expected values."""
        assert "csv" in EXPORT_FORMATS
        assert "cypher" in EXPORT_FORMATS
        assert "json" in EXPORT_FORMATS

    def test_docling_pipelines(self):
        """Test DOCLING_PIPELINES contains expected values."""
        assert "ocr" in DOCLING_PIPELINES
        assert "vision" in DOCLING_PIPELINES

    def test_api_providers(self):
        """Test API_PROVIDERS contains expected values."""
        assert "mistral" in API_PROVIDERS
        assert "openai" in API_PROVIDERS
        assert "gemini" in API_PROVIDERS

    def test_default_models_structure(self):
        """Test DEFAULT_MODELS has expected structure."""
        assert "vlm" in DEFAULT_MODELS
        assert "llm_local" in DEFAULT_MODELS
        assert "llm_remote" in DEFAULT_MODELS
        
        # Check llm_remote has providers
        assert isinstance(DEFAULT_MODELS["llm_remote"], dict)
        assert "mistral" in DEFAULT_MODELS["llm_remote"]
        assert "openai" in DEFAULT_MODELS["llm_remote"]
        assert "gemini" in DEFAULT_MODELS["llm_remote"]

    def test_default_models_values_are_strings(self):
        """Test all default model values are strings."""
        assert isinstance(DEFAULT_MODELS["vlm"], str)
        assert isinstance(DEFAULT_MODELS["llm_local"], str)
        
        for provider, model in DEFAULT_MODELS["llm_remote"].items():
            assert isinstance(model, str), f"Model for {provider} should be string"


class TestConstantConsistency:
    """Tests to ensure constants are consistent with validators."""

    def test_processing_modes_are_lowercase(self):
        """Test processing modes are lowercase for validation."""
        for mode in PROCESSING_MODES:
            assert mode == mode.lower()

    def test_backend_types_are_lowercase(self):
        """Test backend types are lowercase."""
        for backend in BACKEND_TYPES:
            assert backend == backend.lower()

    def test_inference_locations_are_lowercase(self):
        """Test inference locations are lowercase."""
        for location in INFERENCE_LOCATIONS:
            assert location == location.lower()

    def test_no_duplicates_in_lists(self):
        """Test that constant lists have no duplicates."""
        assert len(PROCESSING_MODES) == len(set(PROCESSING_MODES))
        assert len(BACKEND_TYPES) == len(set(BACKEND_TYPES))
        assert len(INFERENCE_LOCATIONS) == len(set(INFERENCE_LOCATIONS))
        assert len(EXPORT_FORMATS) == len(set(EXPORT_FORMATS))
        assert len(API_PROVIDERS) == len(set(API_PROVIDERS))
