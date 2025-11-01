"""
Unit tests for LlmBackend.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest
from pydantic import BaseModel, Field, ValidationError

from docling_graph.core.extractors.backends.llm_backend import LlmBackend


class SampleModel(BaseModel):
    """Sample Pydantic model for testing."""
    name: str
    age: int


class TestLlmBackendInitialization:
    """Tests for LlmBackend initialization."""

    def test_init_with_client(self):
        """Test initialization with LLM client."""
        mock_client = Mock()
        backend = LlmBackend(llm_client=mock_client)
        
        assert backend.client == mock_client

    def test_init_stores_client_reference(self):
        """Test that client reference is properly stored."""
        mock_client = Mock()
        mock_client.__class__.__name__ = "OllamaClient"
        
        backend = LlmBackend(llm_client=mock_client)
        
        assert hasattr(backend, "client")
        assert backend.client.__class__.__name__ == "OllamaClient"


class TestLlmBackendExtraction:
    """Tests for extract_from_markdown method."""

    @patch("docling_graph.core.extractors.backends.llm_backend.get_prompt")
    def test_extract_success(self, mock_get_prompt):
        """Test successful extraction."""
        mock_client = Mock()
        mock_client.get_json_response.return_value = {"name": "Alice", "age": 30}
        mock_get_prompt.return_value = "test prompt"
        
        backend = LlmBackend(llm_client=mock_client)
        result = backend.extract_from_markdown(
            markdown="# Alice\nAge: 30",
            template=SampleModel,
            context="page 1"
        )
        
        assert result is not None
        assert result.name == "Alice"
        assert result.age == 30

    @patch("docling_graph.core.extractors.backends.llm_backend.get_prompt")
    def test_extract_empty_markdown(self, mock_get_prompt):
        """Test extraction with empty markdown."""
        mock_client = Mock()
        backend = LlmBackend(llm_client=mock_client)
        
        result = backend.extract_from_markdown(
            markdown="",
            template=SampleModel,
            context="page 1"
        )
        
        assert result is None
        mock_client.get_json_response.assert_not_called()

    @patch("docling_graph.core.extractors.backends.llm_backend.get_prompt")
    def test_extract_whitespace_only_markdown(self, mock_get_prompt):
        """Test extraction with whitespace-only markdown."""
        mock_client = Mock()
        backend = LlmBackend(llm_client=mock_client)
        
        result = backend.extract_from_markdown(
            markdown="   \n\t  ",
            template=SampleModel,
            context="page 1"
        )
        
        assert result is None

    @patch("docling_graph.core.extractors.backends.llm_backend.get_prompt")
    def test_extract_llm_returns_none(self, mock_get_prompt):
        """Test when LLM returns None."""
        mock_client = Mock()
        mock_client.get_json_response.return_value = None
        mock_get_prompt.return_value = "test prompt"
        
        backend = LlmBackend(llm_client=mock_client)
        result = backend.extract_from_markdown(
            markdown="Test content",
            template=SampleModel
        )
        
        assert result is None

    @patch("docling_graph.core.extractors.backends.llm_backend.get_prompt")
    def test_extract_validation_error(self, mock_get_prompt):
        """Test extraction with validation error."""
        mock_client = Mock()
        mock_client.get_json_response.return_value = {"name": "Alice", "age": "invalid"}
        mock_get_prompt.return_value = "test prompt"
        
        backend = LlmBackend(llm_client=mock_client)
        result = backend.extract_from_markdown(
            markdown="Test content",
            template=SampleModel
        )
        
        assert result is None

    @patch("docling_graph.core.extractors.backends.llm_backend.get_prompt")
    def test_extract_llm_exception(self, mock_get_prompt):
        """Test extraction when LLM raises exception."""
        mock_client = Mock()
        mock_client.get_json_response.side_effect = RuntimeError("API error")
        mock_get_prompt.return_value = "test prompt"
        
        backend = LlmBackend(llm_client=mock_client)
        result = backend.extract_from_markdown(
            markdown="Test content",
            template=SampleModel
        )
        
        assert result is None

    @patch("docling_graph.core.extractors.backends.llm_backend.get_prompt")
    def test_extract_with_context(self, mock_get_prompt):
        """Test that context is passed correctly."""
        mock_client = Mock()
        mock_client.get_json_response.return_value = {"name": "Bob", "age": 25}
        mock_get_prompt.return_value = "test prompt"
        
        backend = LlmBackend(llm_client=mock_client)
        result = backend.extract_from_markdown(
            markdown="Test content",
            template=SampleModel,
            context="full document"
        )
        
        assert result is not None
        assert result.name == "Bob"


class TestLlmBackendPromptGeneration:
    """Tests for prompt generation."""

    @patch("docling_graph.core.extractors.backends.llm_backend.get_prompt")
    @patch("docling_graph.core.extractors.backends.llm_backend.json.dumps")
    def test_prompt_generation_called(self, mock_json_dumps, mock_get_prompt):
        """Test that prompt generation is called correctly."""
        mock_client = Mock()
        mock_client.get_json_response.return_value = {"name": "Alice", "age": 30}
        mock_get_prompt.return_value = "generated prompt"
        mock_json_dumps.return_value = '{"schema": "json"}'
        
        backend = LlmBackend(llm_client=mock_client)
        backend.extract_from_markdown(
            markdown="Test markdown",
            template=SampleModel
        )
        
        # Verify prompt generation was called
        mock_get_prompt.assert_called_once()
        call_kwargs = mock_get_prompt.call_args[1]
        assert "markdown_content" in call_kwargs
        assert "schema_json" in call_kwargs
        assert call_kwargs["markdown_content"] == "Test markdown"


class TestLlmBackendCleanup:
    """Tests for cleanup method."""

    @patch("docling_graph.core.extractors.backends.llm_backend.gc")
    def test_cleanup_success(self, mock_gc):
        """Test successful cleanup."""
        mock_client = Mock()
        backend = LlmBackend(llm_client=mock_client)
        
        backend.cleanup()
        
        mock_gc.collect.assert_called_once()

    @patch("docling_graph.core.extractors.backends.llm_backend.gc")
    def test_cleanup_with_client_cleanup_method(self, mock_gc):
        """Test cleanup when client has cleanup method."""
        mock_client = Mock()
        mock_client.cleanup = Mock()
        
        backend = LlmBackend(llm_client=mock_client)
        backend.cleanup()
        
        mock_client.cleanup.assert_called_once()
        mock_gc.collect.assert_called_once()

    @patch("docling_graph.core.extractors.backends.llm_backend.gc")
    def test_cleanup_handles_error(self, mock_gc):
        """Test cleanup handles errors gracefully."""
        mock_client = Mock()
        mock_client.cleanup = Mock(side_effect=RuntimeError("Cleanup error"))
        
        backend = LlmBackend(llm_client=mock_client)
        # Should not raise
        backend.cleanup()
