from unittest.mock import MagicMock, patch

from docling_graph.llm_clients.config import (
    EffectiveModelConfig,
    GenerationDefaults,
    ModelCapability,
    ReliabilityDefaults,
    ResolvedConnection,
)
from docling_graph.llm_clients.openai import OpenAIClient


@patch("docling_graph.llm_clients.openai.OpenAI")
def test_openai_client_init(mock_openai_class):
    """Test OpenAI client initialization."""
    mock_openai_class.return_value = MagicMock()
    model_config = _make_effective_config()
    client = OpenAIClient(model_config=model_config)

    assert client.model == "gpt-4"
    assert client.context_limit == 4096
    mock_openai_class.assert_called_once_with(api_key="test-api-key-123")


def _make_effective_config() -> EffectiveModelConfig:
    return EffectiveModelConfig(
        model_id="gpt-4",
        provider_id="openai",
        provider_model="gpt-4",
        context_limit=4096,
        max_output_tokens=2048,
        capability=ModelCapability.ADVANCED,
        generation=GenerationDefaults(max_tokens=512),
        reliability=ReliabilityDefaults(timeout_s=30, max_retries=0),
        connection=ResolvedConnection(api_key="test-api-key-123"),
        tokenizer="tiktoken",
        content_ratio=0.8,
        merge_threshold=0.9,
        rate_limit_rpm=None,
        supports_batching=True,
    )
