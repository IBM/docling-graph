"""
Unit tests for MistralClient.
"""

from unittest.mock import Mock, patch

import pytest

from docling_graph.llm_clients.mistral import MistralClient


class TestMistralClientInitialization:
    """Tests for MistralClient initialization."""

    @patch.dict("os.environ", {"MISTRAL_API_KEY": "test-key"})
    @patch("docling_graph.llm_clients.mistral.Mistral")
    def test_init_success(self, mock_mistral_class):
        """Test successful initialization."""
        mock_mistral_class.return_value = Mock()
        
        client = MistralClient(model="mistral-large-latest")
        
        assert client.model == "mistral-large-latest"
        assert client.api_key == "test-key"

    @patch.dict("os.environ", {}, clear=True)
    def test_init_missing_api_key(self):
        """Test initialization fails without API key."""
        with pytest.raises(ValueError, match="MISTRAL_API_KEY not set"):
            MistralClient(model="mistral-large-latest")

    @patch.dict("os.environ", {"MISTRAL_API_KEY": "test-key"})
    @patch("docling_graph.llm_clients.mistral.Mistral")
    def test_context_limit_large(self, mock_mistral_class):
        """Test context limit for Mistral Large."""
        mock_mistral_class.return_value = Mock()
        
        client = MistralClient(model="mistral-large-latest")
        
        assert client.context_limit == 128000

    @patch.dict("os.environ", {"MISTRAL_API_KEY": "test-key"})
    @patch("docling_graph.llm_clients.mistral.Mistral")
    def test_context_limit_small(self, mock_mistral_class):
        """Test context limit for Mistral Small."""
        mock_mistral_class.return_value = Mock()
        
        client = MistralClient(model="mistral-small-latest")
        
        assert client.context_limit == 32000


class TestMistralClientGetJsonResponse:
    """Tests for get_json_response method."""

    @patch.dict("os.environ", {"MISTRAL_API_KEY": "test-key"})
    @patch("docling_graph.llm_clients.mistral.Mistral")
    def test_get_json_response_dict_prompt(self, mock_mistral_class):
        """Test JSON response with dict prompt."""
        mock_client = Mock()
        mock_mistral_class.return_value = mock_client
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '{"status": "ok"}'
        mock_client.chat.complete.return_value = mock_response
        
        client = MistralClient(model="mistral-large-latest")
        result = client.get_json_response(
            prompt={"system": "Extract data", "user": "Extract from text"},
            schema_json="{}"
        )
        
        assert result == {"status": "ok"}

    @patch.dict("os.environ", {"MISTRAL_API_KEY": "test-key"})
    @patch("docling_graph.llm_clients.mistral.Mistral")
    def test_get_json_response_string_prompt(self, mock_mistral_class):
        """Test JSON response with string prompt."""
        mock_client = Mock()
        mock_mistral_class.return_value = mock_client
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '{"result": "done"}'
        mock_client.chat.complete.return_value = mock_response
        
        client = MistralClient(model="mistral-large-latest")
        result = client.get_json_response(
            prompt="Extract data",
            schema_json="{}"
        )
        
        assert result == {"result": "done"}

    @patch.dict("os.environ", {"MISTRAL_API_KEY": "test-key"})
    @patch("docling_graph.llm_clients.mistral.Mistral")
    def test_get_json_response_empty_content(self, mock_mistral_class):
        """Test handling of empty response content."""
        mock_client = Mock()
        mock_mistral_class.return_value = mock_client
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = None
        mock_client.chat.complete.return_value = mock_response
        
        client = MistralClient(model="mistral-large-latest")
        result = client.get_json_response(prompt="test", schema_json="{}")
        
        assert result == {}

    @patch.dict("os.environ", {"MISTRAL_API_KEY": "test-key"})
    @patch("docling_graph.llm_clients.mistral.Mistral")
    def test_get_json_response_invalid_json(self, mock_mistral_class):
        """Test handling of invalid JSON response."""
        mock_client = Mock()
        mock_mistral_class.return_value = mock_client
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Not valid JSON"
        mock_client.chat.complete.return_value = mock_response
        
        client = MistralClient(model="mistral-large-latest")
        result = client.get_json_response(prompt="test", schema_json="{}")
        
        assert result == {}

    @patch.dict("os.environ", {"MISTRAL_API_KEY": "test-key"})
    @patch("docling_graph.llm_clients.mistral.Mistral")
    def test_get_json_response_api_error(self, mock_mistral_class):
        """Test handling of API errors."""
        mock_client = Mock()
        mock_mistral_class.return_value = mock_client
        mock_client.chat.complete.side_effect = Exception("API Error")
        
        client = MistralClient(model="mistral-large-latest")
        result = client.get_json_response(prompt="test", schema_json="{}")
        
        assert result == {}
