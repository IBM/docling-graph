import json
from unittest.mock import MagicMock, patch

import pytest

from docling_graph.llm_clients.config import (
    EffectiveModelConfig,
    GenerationDefaults,
    ModelCapability,
    ReliabilityDefaults,
    ResolvedConnection,
)
from docling_graph.llm_clients.gemini import GeminiClient


@patch("docling_graph.llm_clients.gemini.genai")
@patch("docling_graph.llm_clients.gemini.types")
def test_gemini_client_init(mock_types, mock_genai):
    """Test Gemini client initialization."""
    mock_genai.Client.return_value = MagicMock()

    client = GeminiClient(model_config=_make_effective_config("test-gemini-key"))

    assert client.model == "gemini-pro"
    assert client.context_limit == 1000000
    mock_genai.Client.assert_called_once_with(api_key="test-gemini-key")


@patch("docling_graph.llm_clients.gemini.genai")
@patch("docling_graph.llm_clients.gemini.types")
def test_gemini_client_init_no_api_key(mock_types, mock_genai):
    """Test that missing API key raises ConfigurationError."""
    from docling_graph.exceptions import ConfigurationError

    mock_genai.Client.return_value = MagicMock()

    with pytest.raises(ConfigurationError, match="Gemini API key missing"):
        GeminiClient(model_config=_make_effective_config(None))


@patch("docling_graph.llm_clients.gemini.genai")
@patch("docling_graph.llm_clients.gemini.types")
def test_get_json_response_dict_prompt(mock_types, mock_genai):
    """Test JSON response with dict-style prompt."""
    mock_client = MagicMock()
    mock_genai.Client.return_value = mock_client

    response_data = {"extracted": "data", "value": 42}
    mock_response = MagicMock()
    mock_response.text = json.dumps(response_data)
    mock_client.models.generate_content.return_value = mock_response

    mock_types.GenerateContentConfig.return_value = MagicMock()

    client = GeminiClient(model_config=_make_effective_config("test-gemini-key"))
    result = client.get_json_response(
        prompt={"system": "Extract info", "user": "Process this"}, schema_json="{}"
    )

    assert result == response_data
    mock_client.models.generate_content.assert_called_once()


@patch("docling_graph.llm_clients.gemini.genai")
@patch("docling_graph.llm_clients.gemini.types")
def test_get_json_response_string_prompt(mock_types, mock_genai):
    """Test JSON response with string prompt."""
    mock_client = MagicMock()
    mock_genai.Client.return_value = mock_client

    response_data = {"status": "ok"}
    mock_response = MagicMock()
    mock_response.text = json.dumps(response_data)
    mock_client.models.generate_content.return_value = mock_response

    mock_types.GenerateContentConfig.return_value = MagicMock()

    client = GeminiClient(model_config=_make_effective_config("test-gemini-key"))
    result = client.get_json_response(prompt="Extract", schema_json="{}")

    assert result == response_data


def _make_effective_config(api_key: str | None) -> EffectiveModelConfig:
    return EffectiveModelConfig(
        model_id="gemini-pro",
        provider_id="gemini",
        provider_model="gemini-pro",
        context_limit=1000000,
        max_output_tokens=8192,
        capability=ModelCapability.ADVANCED,
        generation=GenerationDefaults(max_tokens=1024),
        reliability=ReliabilityDefaults(timeout_s=30, max_retries=0),
        connection=ResolvedConnection(api_key=api_key),
        tokenizer="sentence-transformers/all-MiniLM-L6-v2",
        content_ratio=0.8,
        merge_threshold=0.9,
        rate_limit_rpm=None,
        supports_batching=True,
    )


# NOTE: The following tests were removed as they tested obsolete behavior:
# - test_get_json_response_list_result: Lists are now returned as-is, not wrapped
# - test_get_json_response_scalar_result: Scalar normalization tested in response_handler
# - test_get_json_response_invalid_json: Now raises ClientError instead of returning {}
# - test_get_json_response_api_error: Now raises exceptions instead of returning {}
# These behaviors are properly tested in tests/unit/llm_clients/test_response_handler.py
