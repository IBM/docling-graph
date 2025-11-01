"""
Unit tests for OpenAIClient.
"""

from unittest.mock import Mock, patch

import pytest

from docling_graph.llm_clients.openai import OpenAIClient


class TestOpenAIClientInitialization:
    """Tests for OpenAIClient initialization."""

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    @patch("docling_graph.llm_clients.openai.OpenAI")
    def test_init_success(self, mock_openai_class):
        """Test successful initialization."""
        mock_openai_class.return_value = Mock()
        
        client = OpenAIClient(model="gpt-4")
        
        assert client.model == "gpt-4"
        assert client.api_key == "test-key"

    @patch.dict("os.environ", {}, clear=True)
    def test_init_missing_api_key(self):
        """Test initialization fails without API key."""
        with pytest.raises(ValueError, match="OPENAI_API_KEY not set"):
            OpenAIClient(model="gpt-4")

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    @patch("docling_graph.llm_clients.openai.OpenAI")
    def test_context_limit_gpt4(self, mock_openai_class):
        """Test context limit for GPT-4."""
        mock_openai_class.return_value = Mock()
        
        client = OpenAIClient(model="gpt-4")
        
        assert client.context_limit == 8192

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    @patch("docling_graph.llm_clients.openai.OpenAI")
    def test_context_limit_gpt4o(self, mock_openai_class):
        """Test context limit for GPT-4o."""
        mock_openai_class.return_value = Mock()
        
        client = OpenAIClient(model="gpt-4o")
        
        assert client.context_limit == 128000

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    @patch("docling_graph.llm_clients.openai.OpenAI")
    def test_context_limit_gpt35_turbo(self, mock_openai_class):
        """Test context limit for GPT-3.5 Turbo."""
        mock_openai_class.return_value = Mock()
        
        client = OpenAIClient(model="gpt-3.5-turbo")
        
        assert client.context_limit == 16000


class TestOpenAIClientGetJsonResponse:
    """Tests for get_json_response method."""

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    @patch("docling_graph.llm_clients.openai.OpenAI")
    def test_get_json_response_dict_prompt(self, mock_openai_class):
        """Test JSON response with dict prompt."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '{"name": "Bob", "value": 42}'
        mock_client.chat.completions.create.return_value = mock_response
        
        client = OpenAIClient(model="gpt-4")
        result = client.get_json_response(
            prompt={"system": "Extract data", "user": "Extract from text"},
            schema_json="{}"
        )
        
        assert result == {"name": "Bob", "value": 42}

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    @patch("docling_graph.llm_clients.openai.OpenAI")
    def test_get_json_response_string_prompt(self, mock_openai_class):
        """Test JSON response with string prompt."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '{"status": "ok"}'
        mock_client.chat.completions.create.return_value = mock_response
        
        client = OpenAIClient(model="gpt-4")
        result = client.get_json_response(
            prompt="Extract data",
            schema_json="{}"
        )
        
        assert result == {"status": "ok"}

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    @patch("docling_graph.llm_clients.openai.OpenAI")
    def test_get_json_response_invalid_json(self, mock_openai_class):
        """Test handling of invalid JSON response."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Not JSON"
        mock_client.chat.completions.create.return_value = mock_response
        
        client = OpenAIClient(model="gpt-4")
        result = client.get_json_response(prompt="test", schema_json="{}")
        
        assert result == {}

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    @patch("docling_graph.llm_clients.openai.OpenAI")
    def test_get_json_response_api_error(self, mock_openai_class):
        """Test handling of API errors."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        
        client = OpenAIClient(model="gpt-4")
        result = client.get_json_response(prompt="test", schema_json="{}")
        
        assert result == {}
