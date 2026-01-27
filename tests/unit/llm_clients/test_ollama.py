from unittest.mock import MagicMock, patch

from docling_graph.llm_clients.config import (
    EffectiveModelConfig,
    GenerationDefaults,
    ModelCapability,
    ReliabilityDefaults,
    ResolvedConnection,
)
from docling_graph.llm_clients.ollama import OllamaClient


@patch("docling_graph.llm_clients.ollama.ollama")
def test_ollama_client_init(mock_ollama):
    """Test Ollama client initialization."""
    mock_client = MagicMock()
    mock_client.show.return_value = {"name": "llama2"}
    mock_ollama.Client.return_value = mock_client

    client = OllamaClient(model_config=_make_effective_config())

    assert client.model == "llama2"
    assert client.context_limit == 4096
    mock_client.show.assert_called_once_with("llama2")


def _make_effective_config() -> EffectiveModelConfig:
    return EffectiveModelConfig(
        model_id="llama2",
        provider_id="ollama",
        provider_model="llama2",
        context_limit=4096,
        max_output_tokens=2048,
        capability=ModelCapability.STANDARD,
        generation=GenerationDefaults(max_tokens=512),
        reliability=ReliabilityDefaults(timeout_s=30, max_retries=0),
        connection=ResolvedConnection(),
        tokenizer="sentence-transformers/all-MiniLM-L6-v2",
        content_ratio=0.8,
        merge_threshold=0.75,
        rate_limit_rpm=None,
        supports_batching=True,
    )
