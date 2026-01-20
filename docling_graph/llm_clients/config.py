"""
Centralized LLM provider configuration registry.

This module consolidates all LLM client constants (model names, context windows,
tokenizer names, etc.) in a single source of truth, eliminating duplication
across mistral.py, openai.py, gemini.py, vllm.py, ollama.py, and chunker.py.

Structure:
- Provider definitions: MISTRAL, OPENAI, ANTHROPIC, IBM, GOOGLE, META
- Model registry per provider: model_name -> context_window mapping
- Tokenizer mappings: provider -> tokenizer_name
- Dynamic lookup: get_model_config(provider, model)
"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class ModelConfig:
    """Configuration for a single LLM model."""

    model_id: str
    context_limit: int
    max_new_tokens: int = 4096  # Maximum tokens the model can generate in response
    description: str = ""
    notes: str = ""

    def __repr__(self) -> str:
        return (
            f"ModelConfig({self.model_id}, context={self.context_limit}, "
            f"max_new_tokens={self.max_new_tokens})"
        )


@dataclass
class ProviderConfig:
    """Configuration for an LLM provider."""

    provider_id: str
    models: Dict[str, ModelConfig]
    tokenizer: str
    content_ratio: float = 0.8  # Ratio of context available for content vs prompt/response

    def get_model(self, model_name: str) -> ModelConfig | None:
        """Get a specific model from this provider."""
        return self.models.get(model_name)

    def list_models(self) -> list[str]:
        """List all available models for this provider."""
        return list(self.models.keys())

    def get_recommended_chunk_size(self, model_name: str, schema_size: int = 0) -> int:
        """
        Get recommended chunk size for a model in this provider.

        Uses dynamic adjustment based on schema complexity to prevent output overflow.

        Args:
            model_name: Name of the model
            schema_size: Size of the Pydantic schema JSON (optional, for dynamic adjustment)

        Returns:
            Recommended max_tokens for DocumentChunker
        """
        model = self.get_model(model_name)
        if not model:
            return 5120  # Default fallback

        # Estimate output density from schema complexity
        # Larger schemas = more extracted data = larger JSON output
        if schema_size > 10000:
            output_ratio = 0.8  # Complex schema: 80% of input becomes output
        elif schema_size > 5000:
            output_ratio = 0.5  # Medium schema: 50% expansion
        elif schema_size > 0:
            output_ratio = 0.3  # Simple schema: 30% expansion
        else:
            # No schema info: use conservative default
            output_ratio = 0.4

        system_prompt_tokens = 500
        safety_buffer = 0.8  # 20% safety margin

        # Strategy 1: Output-constrained sizing
        # Chunk size limited by max_new_tokens to prevent output overflow
        max_safe_chunk = int(model.max_new_tokens / output_ratio * safety_buffer)

        # Strategy 2: Context-constrained sizing
        # Also respect total context window
        max_by_context = int((model.context_limit - system_prompt_tokens) * 0.7)

        # Use the SMALLER of the two (most conservative approach)
        chunk_size = min(max_safe_chunk, max_by_context)

        return max(1024, chunk_size)  # Minimum 1024 tokens


# --- MISTRAL PROVIDER ---
MISTRAL_MODELS = {
    "mistral-large-latest": ModelConfig(
        model_id="mistral-large-latest",
        context_limit=128000,
        max_new_tokens=8192,
        description="Mistral's largest model, optimized for complex reasoning",
        notes="Recommended for comprehensive document extraction",
    ),
    "mistral-medium-latest": ModelConfig(
        model_id="mistral-medium-latest",
        context_limit=128000,
        max_new_tokens=8192,
        description="Mistral's medium model, balanced performance",
        notes="Good general-purpose choice",
    ),
    "mistral-small-latest": ModelConfig(
        model_id="mistral-small-latest",
        context_limit=32000,
        max_new_tokens=4096,
        description="Mistral's smaller model, faster inference",
        notes="Use for real-time or cost-sensitive tasks",
    ),
}

MISTRAL = ProviderConfig(
    provider_id="mistral",
    models=MISTRAL_MODELS,
    tokenizer="mistralai/Mistral-7B-Instruct-v0.2",
    content_ratio=0.8,
)


# --- OPENAI PROVIDER ---
OPENAI_MODELS = {
    "gpt-4o": ModelConfig(
        model_id="gpt-4o",
        context_limit=128000,
        max_new_tokens=16384,
        description="GPT-4 Omni, vision + text, best performance",
        notes="Latest flagship model, supports vision",
    ),
    "gpt-4o-mini": ModelConfig(
        model_id="gpt-4o-mini",
        context_limit=128000,
        max_new_tokens=16384,
        description="Smaller, faster GPT-4o variant",
        notes="Good cost/performance tradeoff",
    ),
    "gpt-4-turbo": ModelConfig(
        model_id="gpt-4-turbo",
        context_limit=128000,
        max_new_tokens=4096,
        description="Previous GPT-4 variant",
        notes="Older but still powerful",
    ),
    "gpt-4": ModelConfig(
        model_id="gpt-4",
        context_limit=8192,
        max_new_tokens=4096,
        description="Base GPT-4, smaller context",
        notes="Limited context window",
    ),
    "gpt-3.5-turbo": ModelConfig(
        model_id="gpt-3.5-turbo",
        context_limit=16000,
        max_new_tokens=4096,
        description="Fast and cost-effective",
        notes="Popular for real-time tasks",
    ),
    "gpt-3.5-turbo-16k": ModelConfig(
        model_id="gpt-3.5-turbo-16k",
        context_limit=16000,
        max_new_tokens=4096,
        description="Extended context version",
        notes="Better for longer documents",
    ),
}

OPENAI = ProviderConfig(
    provider_id="openai",
    models=OPENAI_MODELS,
    tokenizer="tiktoken",  # Special handling required (gpt-4o encoding)
    content_ratio=0.8,
)


# --- GOOGLE GEMINI PROVIDER ---
GEMINI_MODELS = {
    "gemini-2.5-pro": ModelConfig(
        model_id="gemini-2.5-pro",
        context_limit=1000000,
        max_new_tokens=8192,
        description="Google's latest flagship model",
        notes="Massive context window, multimodal",
    ),
    "gemini-2.0-flash": ModelConfig(
        model_id="gemini-2.0-flash",
        context_limit=1000000,
        max_new_tokens=8192,
        description="Fast Gemini 2.0 variant",
        notes="Excellent for real-time extraction",
    ),
    "gemini-1.5-pro": ModelConfig(
        model_id="gemini-1.5-pro",
        context_limit=1000000,
        max_new_tokens=8192,
        description="Gemini 1.5 Pro model",
        notes="Million token context",
    ),
    "gemini-1.5-flash": ModelConfig(
        model_id="gemini-1.5-flash",
        context_limit=1000000,
        max_new_tokens=8192,
        description="Fast Gemini 1.5 variant",
        notes="Fast inference",
    ),
}

GEMINI = ProviderConfig(
    provider_id="google",
    models=GEMINI_MODELS,
    tokenizer="sentence-transformers/all-MiniLM-L6-v2",
    content_ratio=0.8,
)


# --- ANTHROPIC CLAUDE PROVIDER ---
ANTHROPIC_MODELS = {
    "claude-3-opus": ModelConfig(
        model_id="claude-3-opus",
        context_limit=200000,
        max_new_tokens=4096,
        description="Claude 3 Opus, most capable",
        notes="Best for complex reasoning",
    ),
    "claude-3-sonnet": ModelConfig(
        model_id="claude-3-sonnet",
        context_limit=200000,
        max_new_tokens=4096,
        description="Claude 3 Sonnet, balanced",
        notes="Good general-purpose choice",
    ),
    "claude-3-haiku": ModelConfig(
        model_id="claude-3-haiku",
        context_limit=200000,
        max_new_tokens=4096,
        description="Claude 3 Haiku, fastest",
        notes="Use for real-time tasks",
    ),
}

ANTHROPIC = ProviderConfig(
    provider_id="anthropic",
    models=ANTHROPIC_MODELS,
    tokenizer="cl100k_base",  # Anthropic's tokenizer (via external package)
    content_ratio=0.8,
)


# --- WATSONX / REMOTE INFERENCE PROVIDER ---
WATSONX_MODELS = {
    "ibm/granite-4-h-small": ModelConfig(
        model_id="ibm/granite-4-h-small",
        context_limit=128000,
        max_new_tokens=8192,
        description="IBM Granite 4H Small, high-capacity model",
        notes="Large context window, suitable for complex extractions",
    ),
    "meta-llama/llama-3-70b-instruct": ModelConfig(
        model_id="meta-llama/llama-3-70b-instruct",
        context_limit=32768,
        max_new_tokens=4096,
        description="Meta Llama 3 70B via WatsonX",
        notes="Powerful reasoning capabilities",
    ),
    "meta-llama/llama-3-8b-instruct": ModelConfig(
        model_id="meta-llama/llama-3-8b-instruct",
        context_limit=32768,
        max_new_tokens=4096,
        description="Meta Llama 3 8B via WatsonX",
        notes="Efficient and fast",
    ),
    "mistralai/mixtral-8x7b-instruct-v01": ModelConfig(
        model_id="mistralai/mixtral-8x7b-instruct-v01",
        context_limit=32768,
        max_new_tokens=4096,
        description="Mistral Mixtral 8x7B via WatsonX",
        notes="MoE architecture, good performance",
    ),
    "mistralai/mistral-small-3-1-24b-instruct-2503": ModelConfig(
        model_id="mistralai/mistral-small-3-1-24b-instruct-2503",
        context_limit=32768,
        max_new_tokens=4096,
        description="Mistral Small 3.1 24B via WatsonX",
        notes="Balanced model for general use",
    ),
}

WATSONX = ProviderConfig(
    provider_id="ibm",
    models=WATSONX_MODELS,
    tokenizer="ibm-granite/granite-embedding-278m-multilingual",
    content_ratio=0.75,  # More conservative for smaller models
)


# --- META LLAMA PROVIDER ---
META_MODELS = {
    "llama-2-70b": ModelConfig(
        model_id="meta-llama/Llama-2-70b",
        context_limit=4096,
        max_new_tokens=2048,
        description="Meta's Llama 2, 70B parameters",
        notes="Powerful but requires good hardware",
    ),
    "llama-2-13b": ModelConfig(
        model_id="meta-llama/Llama-2-13b",
        context_limit=4096,
        max_new_tokens=2048,
        description="Meta's Llama 2, 13B parameters",
        notes="Good balance of performance and size",
    ),
    "llama-3-8b": ModelConfig(
        model_id="meta-llama/Llama-3-8b",
        context_limit=8192,
        max_new_tokens=4096,
        description="Meta's Llama 3, 8B parameters",
        notes="Improved reasoning from Llama 2",
    ),
    "llama-3.1-8b": ModelConfig(
        model_id="meta-llama/Llama-3.1-8b",
        context_limit=128000,
        max_new_tokens=4096,
        description="Meta's Llama 3.1, 8B parameters",
        notes="Massive context improvement, 128K tokens",
    ),
    "llama-3.1-70b": ModelConfig(
        model_id="meta-llama/Llama-3.1-70b",
        context_limit=128000,
        max_new_tokens=4096,
        description="Meta's Llama 3.1, 70B parameters",
        notes="Most capable Llama version",
    ),
}

META = ProviderConfig(
    provider_id="meta",
    models=META_MODELS,
    tokenizer="meta-llama/Llama-2-7b-hf",
    content_ratio=0.75,
)


# --- VLLM / LOCAL INFERENCE PROVIDER ---
VLLM_MODELS = {
    "meta-llama/Llama-3.1-8B": ModelConfig(
        model_id="meta-llama/Llama-3.1-8B",
        context_limit=128000,
        max_new_tokens=4096,
        description="Llama 3.1 8B via vLLM",
        notes="Local inference, excellent for testing",
    ),
    "mistralai/Mixtral-8x7B-v0.1": ModelConfig(
        model_id="mistralai/Mixtral-8x7B-v0.1",
        context_limit=32000,
        max_new_tokens=4096,
        description="Mistral Mixtral via vLLM",
        notes="MoE architecture, efficient",
    ),
    "qwen/Qwen2-7B": ModelConfig(
        model_id="qwen/Qwen2-7B",
        context_limit=128000,
        max_new_tokens=4096,
        description="Qwen 2 7B via vLLM",
        notes="Good alternative to Llama",
    ),
    "ibm-granite/granite-4.0-1b": ModelConfig(
        model_id="ibm-granite/granite-4.0-1b",
        context_limit=128000,
        max_new_tokens=2048,
        description="IBM Granite 1B for edge devices",
        notes="Tiny model for resource constraints",
    ),
    "ibm-granite/granite-4.0-350m": ModelConfig(
        model_id="ibm-granite/granite-4.0-350m",
        context_limit=32000,
        max_new_tokens=2048,
        description="IBM Granite 350M, lightweight",
        notes="Good balance for local inference",
    ),
}

VLLM = ProviderConfig(
    provider_id="vllm",
    models=VLLM_MODELS,
    tokenizer="sentence-transformers/all-MiniLM-L6-v2",
    content_ratio=0.8,
)


# --- OLLAMA / LOCAL INFERENCE PROVIDER ---
OLLAMA_MODELS = {
    "llama3.1:8b": ModelConfig(
        model_id="llama3.1:8b",
        context_limit=128000,
        max_new_tokens=4096,
        description="Llama 3.1 8B via Ollama",
        notes="Local inference, production-ready",
    ),
    "llama3.2:3b": ModelConfig(
        model_id="llama3.2:3b",
        context_limit=128000,
        max_new_tokens=4096,
        description="Llama 3.2 3B via Ollama",
        notes="Lightweight, fast local inference",
    ),
    "mistral:7b": ModelConfig(
        model_id="mistral:7b",
        context_limit=32000,
        max_new_tokens=4096,
        description="Mistral 7B via Ollama",
        notes="Fast and efficient",
    ),
    "mixtral:8x7b": ModelConfig(
        model_id="mixtral:8x7b",
        context_limit=32000,
        max_new_tokens=4096,
        description="Mixtral 8x7B via Ollama",
        notes="MoE, powerful local inference",
    ),
}

OLLAMA = ProviderConfig(
    provider_id="ollama",
    models=OLLAMA_MODELS,
    tokenizer="sentence-transformers/all-MiniLM-L6-v2",
    content_ratio=0.8,
)


# --- REGISTRY: Map provider_id -> ProviderConfig ---
PROVIDERS: Dict[str, ProviderConfig] = {
    "mistral": MISTRAL,
    "openai": OPENAI,
    "gemini": GEMINI,
    "anthropic": ANTHROPIC,
    "watsonx": WATSONX,
    "meta": META,
    "vllm": VLLM,
    "ollama": OLLAMA,
}


# --- LOOKUP FUNCTIONS ---


def get_provider_config(provider: str) -> ProviderConfig | None:
    """
    Get provider configuration by name.

    Args:
        provider: Provider ID (e.g., "mistral", "openai", "gemini")

    Returns:
        ProviderConfig or None if not found
    """
    return PROVIDERS.get(provider.lower())


def get_model_config(provider: str, model_name: str) -> ModelConfig | None:
    """
    Get model configuration by provider and model name.

    Args:
        provider: Provider ID (e.g., "mistral", "openai")
        model_name: Model name (e.g., "mistral-large-latest")

    Returns:
        ModelConfig or None if not found
    """
    provider_config = get_provider_config(provider)
    if not provider_config:
        return None
    return provider_config.get_model(model_name)


def get_context_limit(provider: str, model: str) -> int:
    """
    Get context window size for a model.

    Args:
        provider: Provider ID
        model: Model name

    Returns:
        Context limit in tokens (defaults to 8000 if not found)
    """
    config = get_model_config(provider, model)
    if config:
        return config.context_limit
    return 8000  # Safe default


def get_tokenizer_for_provider(provider: str) -> str:
    """
    Get recommended tokenizer for a provider.

    Args:
        provider: Provider ID

    Returns:
        Tokenizer name (HuggingFace model or special name like "tiktoken")
    """
    provider_config = get_provider_config(provider)
    if provider_config:
        return provider_config.tokenizer
    return "sentence-transformers/all-MiniLM-L6-v2"  # Default fallback


def get_recommended_chunk_size(provider: str, model: str, schema_size: int = 0) -> int:
    """
    Get recommended chunk size for chunker based on model's context window.

    Args:
        provider: Provider ID
        model: Model name
        schema_size: Size of the Pydantic schema JSON (optional, for dynamic adjustment)

    Returns:
        Recommended max_tokens for DocumentChunker
    """
    provider_config = get_provider_config(provider)
    if provider_config:
        return provider_config.get_recommended_chunk_size(model, schema_size)
    return 5120  # Default fallback


def list_providers() -> list[str]:
    """List all available provider IDs."""
    return list(PROVIDERS.keys())


def list_models(provider: str) -> list[str] | None:
    """List all models for a provider."""
    provider_config = get_provider_config(provider)
    if provider_config:
        return provider_config.list_models()
    return None
