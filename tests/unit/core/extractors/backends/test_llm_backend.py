"""
Tests for LLM backend.
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, ValidationError

from docling_graph.core.extractors.backends.llm_backend import LlmBackend


class SampleModel(BaseModel):
    """Sample model for testing."""

    name: str
    value: int


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client."""
    client = MagicMock()
    client.__class__.__name__ = "MockLlmClient"
    client.context_limit = 4096
    client.get_json_response = MagicMock(return_value={"name": "test", "value": 42})
    return client


class TestLlmBackendInitialization:
    """Test LLM backend initialization."""

    def test_initialization_with_client(self, mock_llm_client):
        """Should initialize with LLM client."""
        backend = LlmBackend(llm_client=mock_llm_client)
        assert backend.client is mock_llm_client

    def test_initialization_stores_client(self, mock_llm_client):
        """Should store client reference."""
        backend = LlmBackend(llm_client=mock_llm_client)
        assert hasattr(backend, "client")
        assert backend.client == mock_llm_client


class TestLlmBackendExtractFromMarkdown:
    """Test LLM extraction from markdown."""

    def test_extract_from_markdown_success(self, mock_llm_client):
        """Should extract successfully."""
        backend = LlmBackend(llm_client=mock_llm_client)
        markdown = "# Document\n\nContent here"

        result = backend.extract_from_markdown(markdown, SampleModel)

        assert result is not None
        assert isinstance(result, SampleModel)

    def test_extract_from_markdown_empty_returns_none(self, mock_llm_client):
        """Should return None for empty markdown."""
        backend = LlmBackend(llm_client=mock_llm_client)

        result = backend.extract_from_markdown("", SampleModel)

        assert result is None

    def test_extract_from_markdown_with_context(self, mock_llm_client):
        """Should use context parameter."""
        backend = LlmBackend(llm_client=mock_llm_client)
        markdown = "# Document\n\nContent"

        result = backend.extract_from_markdown(markdown, SampleModel, context="page 1")

        assert result is not None

    def test_extract_calls_llm_client(self, mock_llm_client):
        """Should call LLM client."""
        backend = LlmBackend(llm_client=mock_llm_client)
        markdown = "# Document"

        backend.extract_from_markdown(markdown, SampleModel)

        mock_llm_client.get_json_response.assert_called_once()

    def test_extract_validates_response(self, mock_llm_client):
        """Should validate LLM response against template."""
        mock_llm_client.get_json_response.return_value = {"name": "test", "value": 42}
        backend = LlmBackend(llm_client=mock_llm_client)

        result = backend.extract_from_markdown("# Doc", SampleModel)

        assert result.name == "test"
        assert result.value == 42

    def test_extract_handles_invalid_json(self, mock_llm_client):
        """Should handle invalid JSON from LLM."""
        mock_llm_client.get_json_response.return_value = None
        backend = LlmBackend(llm_client=mock_llm_client)

        result = backend.extract_from_markdown("# Doc", SampleModel)

        assert result is None


class TestLlmBackendCleanup:
    """Test LLM backend cleanup."""

    def test_cleanup_removes_client(self, mock_llm_client):
        """Should remove client reference."""
        backend = LlmBackend(llm_client=mock_llm_client)
        backend.cleanup()

        # Client should be deleted
        assert not hasattr(backend, "client") or backend.client is None

    def test_cleanup_calls_client_cleanup(self, mock_llm_client):
        """Should call client's cleanup method if available."""
        mock_llm_client.cleanup = MagicMock()
        backend = LlmBackend(llm_client=mock_llm_client)

        backend.cleanup()

        mock_llm_client.cleanup.assert_called_once()

    def test_cleanup_handles_missing_client(self):
        """Should handle cleanup without client gracefully."""
        mock_client = MagicMock()
        backend = LlmBackend(llm_client=mock_client)
        del backend.client  # Force missing client

        # Should not raise
        backend.cleanup()
