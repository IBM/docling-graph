"""
Unit tests for ExtractorFactory.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest

from docling_graph.core.extractors.factory import ExtractorFactory
from docling_graph.core.extractors.backends.llm_backend import LlmBackend
from docling_graph.core.extractors.backends.vlm_backend import VlmBackend
from docling_graph.core.extractors.strategies.one_to_one import OneToOneStrategy
from docling_graph.core.extractors.strategies.many_to_one import ManyToOneStrategy


class TestExtractorFactoryVLM:
    """Tests for creating VLM extractors."""

    @patch("docling_graph.core.extractors.factory.VlmBackend")
    @patch("docling_graph.core.extractors.factory.OneToOneStrategy")
    def test_create_vlm_one_to_one(self, mock_strategy, mock_backend):
        """Test creating VLM extractor with one-to-one strategy."""
        mock_backend_instance = MagicMock()
        mock_backend.return_value = mock_backend_instance
        mock_strategy_instance = MagicMock()
        mock_strategy.return_value = mock_strategy_instance

        extractor = ExtractorFactory.create_extractor(
            processing_mode="one-to-one",
            backend_type="vlm",
            model_name="numind/NuExtract",
            docling_config="ocr",
        )

        # Verify VlmBackend was created with model_name
        mock_backend.assert_called_once_with(model_name="numind/NuExtract")
        
        # Verify OneToOneStrategy was created
        mock_strategy.assert_called_once_with(
            backend=mock_backend_instance, docling_config="ocr"
        )
        
        assert extractor == mock_strategy_instance

    @patch("docling_graph.core.extractors.factory.VlmBackend")
    @patch("docling_graph.core.extractors.factory.ManyToOneStrategy")
    def test_create_vlm_many_to_one(self, mock_strategy, mock_backend):
        """Test creating VLM extractor with many-to-one strategy."""
        mock_backend_instance = MagicMock()
        mock_backend.return_value = mock_backend_instance
        mock_strategy_instance = MagicMock()
        mock_strategy.return_value = mock_strategy_instance

        extractor = ExtractorFactory.create_extractor(
            processing_mode="many-to-one",
            backend_type="vlm",
            model_name="numind/NuExtract-2.0-8B",
            docling_config="vision",
        )

        mock_backend.assert_called_once_with(model_name="numind/NuExtract-2.0-8B")
        mock_strategy.assert_called_once_with(
            backend=mock_backend_instance, docling_config="vision"
        )

    def test_create_vlm_missing_model_name(self):
        """Test that creating VLM without model_name raises error."""
        with pytest.raises(ValueError, match="VLM requires model_name"):
            ExtractorFactory.create_extractor(
                processing_mode="one-to-one",
                backend_type="vlm",
                model_name=None,
            )


class TestExtractorFactoryLLM:
    """Tests for creating LLM extractors."""

    @patch("docling_graph.core.extractors.factory.LlmBackend")
    @patch("docling_graph.core.extractors.factory.OneToOneStrategy")
    def test_create_llm_one_to_one(self, mock_strategy, mock_backend):
        """Test creating LLM extractor with one-to-one strategy."""
        mock_llm_client = MagicMock()
        mock_backend_instance = MagicMock()
        mock_backend.return_value = mock_backend_instance
        mock_strategy_instance = MagicMock()
        mock_strategy.return_value = mock_strategy_instance

        extractor = ExtractorFactory.create_extractor(
            processing_mode="one-to-one",
            backend_type="llm",
            llm_client=mock_llm_client,
            docling_config="ocr",
        )

        mock_backend.assert_called_once_with(llm_client=mock_llm_client)
        mock_strategy.assert_called_once_with(
            backend=mock_backend_instance, docling_config="ocr"
        )

    @patch("docling_graph.core.extractors.factory.LlmBackend")
    @patch("docling_graph.core.extractors.factory.ManyToOneStrategy")
    def test_create_llm_many_to_one(self, mock_strategy, mock_backend):
        """Test creating LLM extractor with many-to-one strategy."""
        mock_llm_client = MagicMock()
        mock_backend_instance = MagicMock()
        mock_backend.return_value = mock_backend_instance
        mock_strategy_instance = MagicMock()
        mock_strategy.return_value = mock_strategy_instance

        extractor = ExtractorFactory.create_extractor(
            processing_mode="many-to-one",
            backend_type="llm",
            llm_client=mock_llm_client,
            docling_config="vision",
        )

        mock_backend.assert_called_once_with(llm_client=mock_llm_client)
        mock_strategy.assert_called_once_with(
            backend=mock_backend_instance, docling_config="vision"
        )

    def test_create_llm_missing_client(self):
        """Test that creating LLM without llm_client raises error."""
        with pytest.raises(ValueError, match="LLM requires llm_client"):
            ExtractorFactory.create_extractor(
                processing_mode="one-to-one",
                backend_type="llm",
                llm_client=None,
            )


class TestExtractorFactoryErrors:
    """Tests for error handling in factory."""

    def test_invalid_backend_type(self):
        """Test that invalid backend_type raises error."""
        with pytest.raises(ValueError, match="Unknown backend_type"):
            ExtractorFactory.create_extractor(
                processing_mode="one-to-one",
                backend_type="invalid",
                model_name="test",
            )

    def test_invalid_processing_mode(self):
        """Test that invalid processing_mode raises error."""
        mock_llm_client = MagicMock()
        
        with pytest.raises(ValueError, match="Unknown processing_mode"):
            ExtractorFactory.create_extractor(
                processing_mode="invalid-mode",
                backend_type="llm",
                llm_client=mock_llm_client,
            )

    @patch("docling_graph.core.extractors.factory.VlmBackend")
    def test_vlm_backend_initialization_error(self, mock_backend):
        """Test handling of VLM backend initialization errors."""
        mock_backend.side_effect = RuntimeError("Model not found")
        
        with pytest.raises(RuntimeError, match="Model not found"):
            ExtractorFactory.create_extractor(
                processing_mode="one-to-one",
                backend_type="vlm",
                model_name="invalid/model",
            )


class TestExtractorFactoryDoclingConfig:
    """Tests for docling_config parameter."""

    @patch("docling_graph.core.extractors.factory.VlmBackend")
    @patch("docling_graph.core.extractors.factory.OneToOneStrategy")
    def test_default_docling_config(self, mock_strategy, mock_backend):
        """Test default docling_config value."""
        mock_backend_instance = MagicMock()
        mock_backend.return_value = mock_backend_instance
        mock_strategy_instance = MagicMock()
        mock_strategy.return_value = mock_strategy_instance

        ExtractorFactory.create_extractor(
            processing_mode="one-to-one",
            backend_type="vlm",
            model_name="test-model",
        )

        # Default should be "ocr"
        mock_strategy.assert_called_once_with(
            backend=mock_backend_instance, docling_config="ocr"
        )

    @patch("docling_graph.core.extractors.factory.VlmBackend")
    @patch("docling_graph.core.extractors.factory.ManyToOneStrategy")
    def test_custom_docling_config(self, mock_strategy, mock_backend):
        """Test custom docling_config value."""
        mock_backend_instance = MagicMock()
        mock_backend.return_value = mock_backend_instance
        mock_strategy_instance = MagicMock()
        mock_strategy.return_value = mock_strategy_instance

        ExtractorFactory.create_extractor(
            processing_mode="many-to-one",
            backend_type="vlm",
            model_name="test-model",
            docling_config="vision",
        )

        mock_strategy.assert_called_once_with(
            backend=mock_backend_instance, docling_config="vision"
        )
