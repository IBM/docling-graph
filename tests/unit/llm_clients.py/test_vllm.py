"""
Unit tests for VllmClient.
"""

from unittest.mock import Mock, patch

import pytest

from docling_graph.llm_clients.vllm import VllmClient


class TestVllmClientInitialization:
    """Tests for VllmClient initialization."""

    @patch("docling_graph.llm_clients.vllm.OpenAI")
    def test_init_success(self, mock_openai):
        """Test successful initialization."""
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        client = VllmClient(model="llama3:8b", base_url="http://localhost:8000/v1")
        
        assert client.model == "llama3:8b"
        assert client.base_url == "http://localhost:8000/v1"

    @patch("docling_graph.llm_clients.vllm.OpenAI")
    def test_init_default_base_url(self, mock_openai):
        """Test initialization with default base URL."""
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        client = VllmClient(model="llama3:8b")
        
        assert client.base_url == "http://localhost:8000/v1"

    @patch("docling_graph.llm_clients.vllm.OpenAI")
    def test_context_limit(self, mock_openai):
        """Test context limit."""
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        client = VllmClient(model="llama3:8b")
        
        assert client.context_limit == 8000


class TestVllmClientGetJsonResponse:
    """Tests for get_json_response method."""

    @patch("docling_graph.llm_clients.vllm.OpenAI")
    def test_get_json_response_success(self, mock_openai):
        """Test successful JSON response."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '{"result": "success"}'
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        client = VllmClient(model="llama3:8b")
        result = client.get_json_response(
            prompt="Extract data",
            schema_json="{}"
        )
        
        assert result == {"result": "success"}
