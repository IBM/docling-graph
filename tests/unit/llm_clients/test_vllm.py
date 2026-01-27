import json
from unittest.mock import MagicMock, patch

from docling_graph.llm_clients.config import (
    EffectiveModelConfig,
    GenerationDefaults,
    ModelCapability,
    ReliabilityDefaults,
    ResolvedConnection,
)
from docling_graph.llm_clients.vllm import VllmClient


@patch("docling_graph.llm_clients.vllm.OpenAI")
def test_vllm_client_init(mock_openai_class):
    """Test vLLM client initialization."""
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    mock_client.models.list.return_value = [{"id": "llama-7b"}]
    client = VllmClient(model_config=_make_effective_config())

    assert client.model == "llama-7b"
    assert client.base_url == "http://localhost:8000/v1"
    assert client.api_key == "EMPTY"
    assert client.context_limit == 4096
    mock_openai_class.assert_called_once_with(base_url="http://localhost:8000/v1", api_key="EMPTY")


@patch("docling_graph.llm_clients.vllm.OpenAI")
def test_vllm_client_init_custom_url(mock_openai_class):
    """Test vLLM client with custom URL."""
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    mock_client.models.list.return_value = []
    client = VllmClient(
        model_config=_make_effective_config(
            base_url="http://remote-server:8000/v1", api_key="custom-key"
        )
    )

    assert client.base_url == "http://remote-server:8000/v1"
    assert client.api_key == "custom-key"
    mock_openai_class.assert_called_once_with(
        base_url="http://remote-server:8000/v1", api_key="custom-key"
    )


@patch("docling_graph.llm_clients.vllm.OpenAI")
def test_get_json_response_empty_all_null(mock_openai_class):
    """Test handling of all-null JSON."""
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    mock_client.models.list.return_value = []
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = json.dumps({"a": None, "b": None})
    mock_client.chat.completions.create.return_value = mock_response

    client = VllmClient(model_config=_make_effective_config())
    result = client.get_json_response(prompt="test", schema_json="{}")

    assert result == {"a": None, "b": None}


def _make_effective_config(
    base_url: str | None = None, api_key: str | None = "EMPTY"
) -> EffectiveModelConfig:
    return EffectiveModelConfig(
        model_id="llama-7b",
        provider_id="vllm",
        provider_model="llama-7b",
        context_limit=4096,
        max_output_tokens=2048,
        capability=ModelCapability.STANDARD,
        generation=GenerationDefaults(max_tokens=512),
        reliability=ReliabilityDefaults(timeout_s=30, max_retries=0),
        connection=ResolvedConnection(api_key=api_key, base_url=base_url),
        tokenizer="sentence-transformers/all-MiniLM-L6-v2",
        content_ratio=0.8,
        merge_threshold=0.75,
        rate_limit_rpm=None,
        supports_batching=True,
    )
