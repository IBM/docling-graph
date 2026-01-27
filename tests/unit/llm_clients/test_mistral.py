"""Tests for MistralClient with refactored architecture."""

from unittest.mock import MagicMock, patch

import pytest

from docling_graph.exceptions import ClientError, ConfigurationError
from docling_graph.llm_clients.config import (
    EffectiveModelConfig,
    GenerationDefaults,
    ModelCapability,
    ReliabilityDefaults,
    ResolvedConnection,
)
from docling_graph.llm_clients.mistral import MistralClient


class TestMistralClient:
    """Test suite for MistralClient."""

    def test_provider_id(self):
        """Test provider ID."""
        with patch("docling_graph.llm_clients.mistral.Mistral"):
            client = MistralClient(model_config=_make_effective_config())
            assert client._provider_id() == "mistral"

    @patch("docling_graph.llm_clients.mistral.Mistral")
    def test_client_initialization(self, mock_mistral_class):
        """Test client initialization."""
        mock_mistral_class.return_value = MagicMock()

        client = MistralClient(model_config=_make_effective_config())

        assert client.model == "mistral-large-latest"
        mock_mistral_class.assert_called_once_with(api_key="test-mistral-key")

    def test_missing_api_key(self):
        """Test that missing API key raises ConfigurationError."""
        with pytest.raises(ConfigurationError) as exc_info:
            MistralClient(model_config=_make_effective_config(api_key=None))

        assert "Mistral API key missing" in str(exc_info.value)

    @patch("docling_graph.llm_clients.mistral.Mistral")
    @patch("docling_graph.llm_clients.response_handler.ResponseHandler.parse_json_response")
    def test_get_json_response_success(self, mock_parse, mock_mistral_class):
        """Test successful JSON response."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"name": "Bob", "role": "engineer"}'
        mock_client.chat.complete.return_value = mock_response
        mock_mistral_class.return_value = mock_client

        mock_parse.return_value = {"name": "Bob", "role": "engineer"}

        client = MistralClient(model_config=_make_effective_config())
        result = client.get_json_response(
            prompt={"system": "Extract data", "user": "Bob is an engineer"}, schema_json="{}"
        )

        assert result == {"name": "Bob", "role": "engineer"}
        mock_client.chat.complete.assert_called_once()

    @patch("docling_graph.llm_clients.mistral.Mistral")
    def test_call_api_with_messages(self, mock_mistral_class):
        """Test _call_api with messages."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"result": "success"}'
        mock_response.choices[0].finish_reason = "stop"
        mock_client.chat.complete.return_value = mock_response
        mock_mistral_class.return_value = mock_client

        client = MistralClient(model_config=_make_effective_config())
        messages = [{"role": "user", "content": "test"}]

        response, metadata = client._call_api(messages)

        assert response == '{"result": "success"}'
        assert metadata["finish_reason"] == "stop"
        assert metadata["model"] == "mistral-large-latest"
        mock_client.chat.complete.assert_called_once()

        # Verify call arguments
        call_args = mock_client.chat.complete.call_args
        assert call_args[1]["model"] == "mistral-large-latest"
        assert call_args[1]["messages"] == messages
        assert call_args[1]["temperature"] == 0.1

    @patch("docling_graph.llm_clients.mistral.Mistral")
    def test_call_api_empty_response(self, mock_mistral_class):
        """Test handling of empty response."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = None
        mock_client.chat.complete.return_value = mock_response
        mock_mistral_class.return_value = mock_client

        client = MistralClient(model_config=_make_effective_config())
        messages = [{"role": "user", "content": "test"}]

        with pytest.raises(ClientError) as exc_info:
            client._call_api(messages)

        assert "empty content" in str(exc_info.value).lower()
        assert exc_info.value.details["model"] == "mistral-large-latest"

    @patch("docling_graph.llm_clients.mistral.Mistral")
    def test_call_api_exception_handling(self, mock_mistral_class):
        """Test API exception handling."""
        mock_client = MagicMock()
        mock_client.chat.complete.side_effect = Exception("API Error")
        mock_mistral_class.return_value = mock_client

        client = MistralClient(model_config=_make_effective_config())
        messages = [{"role": "user", "content": "test"}]

        with pytest.raises(ClientError) as exc_info:
            client._call_api(messages)

        assert "API Error" in str(exc_info.value)
        assert exc_info.value.details["model"] == "mistral-large-latest"


def _make_effective_config(api_key: str | None = "test-mistral-key") -> EffectiveModelConfig:
    return EffectiveModelConfig(
        model_id="mistral-large-latest",
        provider_id="mistral",
        provider_model="mistral-large-latest",
        context_limit=128000,
        max_output_tokens=4096,
        capability=ModelCapability.ADVANCED,
        generation=GenerationDefaults(max_tokens=1024, temperature=0.1),
        reliability=ReliabilityDefaults(timeout_s=30, max_retries=0),
        connection=ResolvedConnection(api_key=api_key),
        tokenizer="mistralai/Mistral-7B-Instruct-v0.2",
        content_ratio=0.8,
        merge_threshold=0.9,
        rate_limit_rpm=None,
        supports_batching=True,
    )
