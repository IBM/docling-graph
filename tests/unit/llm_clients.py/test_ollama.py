"""
Unit tests for OllamaClient.
"""

from unittest.mock import Mock, patch

import pytest

from docling_graph.llm_clients.ollama import OllamaClient


class TestOllamaClientInitialization:
    """Tests for OllamaClient initialization."""

    @patch("docling_graph.llm_clients.ollama.ollama")
    def test_init_success(self, mock_ollama):
        """Test successful initialization."""
        mock_ollama.show.return_value = {"name": "llama3:8b"}
        
        client = OllamaClient(model="llama3:8b")
        
        assert client.model == "llama3:8b"
        mock_ollama.show.assert_called_once_with("llama3:8b")

    @patch("docling_graph.llm_clients.ollama.ollama", None)
    def test_init_without_ollama_package(self):
        """Test initialization fails without ollama package."""
        with pytest.raises(ImportError, match="could not be imported"):
            OllamaClient(model="llama3:8b")

    @patch("docling_graph.llm_clients.ollama.ollama")
    def test_init_with_model_not_found(self, mock_ollama):
        """Test initialization when model not found."""
        mock_ollama.show.side_effect = Exception("Model not found")
        
        with pytest.raises(Exception, match="Model not found"):
            OllamaClient(model="nonexistent:8b")

    @patch("docling_graph.llm_clients.ollama.ollama")
    def test_context_limit_known_model(self, mock_ollama):
        """Test context limit for known models."""
        mock_ollama.show.return_value = {"name": "llama3.1:8b"}
        
        client = OllamaClient(model="llama3.1:8b")
        
        assert client.context_limit == 128000

    @patch("docling_graph.llm_clients.ollama.ollama")
    def test_context_limit_unknown_model(self, mock_ollama):
        """Test context limit for unknown models defaults."""
        mock_ollama.show.return_value = {"name": "custom:model"}
        
        client = OllamaClient(model="custom:model")
        
        assert client.context_limit == 8000


class TestOllamaClientGetJsonResponse:
    """Tests for get_json_response method."""

    @patch("docling_graph.llm_clients.ollama.ollama")
    def test_get_json_response_dict_prompt(self, mock_ollama):
        """Test JSON response with dict prompt."""
        mock_ollama.show.return_value = {"name": "llama3:8b"}
        mock_ollama.chat.return_value = {
            "message": {"content": '{"name": "Alice", "age": 30}'}
        }
        
        client = OllamaClient(model="llama3:8b")
        result = client.get_json_response(
            prompt={"system": "Extract data", "user": "Extract from text"},
            schema_json="{}"
        )
        
        assert result == {"name": "Alice", "age": 30}

    @patch("docling_graph.llm_clients.ollama.ollama")
    def test_get_json_response_string_prompt(self, mock_ollama):
        """Test JSON response with string prompt."""
        mock_ollama.show.return_value = {"name": "llama3:8b"}
        mock_ollama.chat.return_value = {
            "message": {"content": '{"result": "success"}'}
        }
        
        client = OllamaClient(model="llama3:8b")
        result = client.get_json_response(
            prompt="Extract data",
            schema_json="{}"
        )
        
        assert result == {"result": "success"}

    @patch("docling_graph.llm_clients.ollama.ollama")
    def test_get_json_response_invalid_json(self, mock_ollama):
        """Test handling of invalid JSON response."""
        mock_ollama.show.return_value = {"name": "llama3:8b"}
        mock_ollama.chat.return_value = {
            "message": {"content": "Not valid JSON"}
        }
        
        client = OllamaClient(model="llama3:8b")
        result = client.get_json_response(
            prompt="test",
            schema_json="{}"
        )
        
        assert result == {}

    @patch("docling_graph.llm_clients.ollama.ollama")
    def test_get_json_response_empty_json(self, mock_ollama):
        """Test handling of empty JSON response."""
        mock_ollama.show.return_value = {"name": "llama3:8b"}
        mock_ollama.chat.return_value = {
            "message": {"content": "{}"}
        }
        
        client = OllamaClient(model="llama3:8b")
        result = client.get_json_response(
            prompt="test",
            schema_json="{}"
        )
        
        assert result == {}

    @patch("docling_graph.llm_clients.ollama.ollama")
    def test_get_json_response_api_error(self, mock_ollama):
        """Test handling of API errors."""
        mock_ollama.show.return_value = {"name": "llama3:8b"}
        mock_ollama.chat.side_effect = Exception("API Error")
        
        client = OllamaClient(model="llama3:8b")
        result = client.get_json_response(
            prompt="test",
            schema_json="{}"
        )
        
        assert result == {}
