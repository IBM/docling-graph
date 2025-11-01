"""
Unit tests for GeminiClient.
"""

from unittest.mock import Mock, patch

import pytest

from docling_graph.llm_clients.gemini import GeminiClient


class TestGeminiClientInitialization:
    """Tests for GeminiClient initialization."""

    @patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"})
    @patch("docling_graph.llm_clients.gemini.genai")
    def test_init_success(self, mock_genai):
        """Test successful initialization."""
        client = GeminiClient(model="gemini-pro")
        
        assert client.model == "gemini-pro"
        assert client.api_key == "test-key"

    @patch.dict("os.environ", {}, clear=True)
    def test_init_missing_api_key(self):
        """Test initialization fails without API key."""
        with pytest.raises(ValueError, match="GOOGLE_API_KEY not set"):
            GeminiClient(model="gemini-pro")

    @patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"})
    @patch("docling_graph.llm_clients.gemini.genai")
    def test_context_limit(self, mock_genai):
        """Test context limit."""
        client = GeminiClient(model="gemini-pro")
        
        assert client.context_limit >= 30000


class TestGeminiClientGetJsonResponse:
    """Tests for get_json_response method."""

    @patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"})
    @patch("docling_graph.llm_clients.gemini.genai")
    def test_get_json_response_success(self, mock_genai):
        """Test successful JSON response."""
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = '{"status": "ok"}'
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model
        
        client = GeminiClient(model="gemini-pro")
        result = client.get_json_response(
            prompt="Extract data",
            schema_json="{}"
        )
        
        assert result == {"status": "ok"}
