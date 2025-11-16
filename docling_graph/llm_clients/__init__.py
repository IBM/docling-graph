"""
LLM Clients module with lazy imports for optional dependencies.
"""

from typing import Type

from .base import BaseLlmClient

__all__ = ["BaseLlmClient", "get_client"]


def _get_mistral_client() -> Type[BaseLlmClient]:
    """Lazy import MistralClient - only loads if actually used."""
    try:
        from .mistral import MistralClient

        return MistralClient
    except ImportError as e:
        raise ImportError(
            "\nMistral client requires 'mistralai' package.\n"
            "Install with: pip install 'docling-graph[mistral]'\n"
            "Or: pip install mistralai"
        ) from e


def _get_ollama_client() -> Type[BaseLlmClient]:
    """Lazy import OllamaClient - only loads if actually used."""
    try:
        from .ollama import OllamaClient

        return OllamaClient
    except ImportError as e:
        raise ImportError(
            "\nOllama client requires 'ollama' package.\n"
            "Install with: pip install 'docling-graph[ollama]'\n"
            "Or: pip install ollama"
        ) from e


def _get_vllm_client() -> Type[BaseLlmClient]:
    """Lazy import VllmClient - only loads if actually used."""
    try:
        from .vllm import VllmClient

        return VllmClient
    except ImportError as e:
        raise ImportError(
            "\nvLLM client requires 'vllm' package.\n"
            "Install with: pip install 'docling-graph[vllm]'\n"
            "Or: pip install vllm"
        ) from e


def _get_openai_client() -> Type[BaseLlmClient]:
    """Lazy import OpenAIClient - only loads if actually used."""
    try:
        from .openai import OpenAIClient

        return OpenAIClient
    except ImportError as e:
        raise ImportError(
            "\nOpenAI client requires 'openai' package.\n"
            "Install with: pip install 'docling-graph[openai]'\n"
            "Or: pip install openai"
        ) from e


def _get_gemini_client() -> Type[BaseLlmClient]:
    """Lazy import GeminiClient - only loads if actually used."""
    try:
        from .gemini import GeminiClient

        return GeminiClient
    except ImportError as e:
        raise ImportError(
            "\nGemini client requires 'google-genai' package.\n"
            "Install with: pip install 'docling-graph[gemini]'\n"
            "Or: pip install google-genai"
        ) from e


def _get_watsonx_client() -> Type[BaseLlmClient]:
    """Lazy import WatsonxClient - only loads if actually used."""
    try:
        from .watsonx import WatsonxClient

        return WatsonxClient
    except ImportError as e:
        raise ImportError(
            "\nWatsonX client requires 'ibm-watsonx-ai' package.\n"
            "Install with: pip install 'docling-graph[watsonx]'\n"
            "Or: pip install ibm-watsonx-ai"
        ) from e


# Registry mapping provider names to lazy import functions
_CLIENT_REGISTRY = {
    "mistral": _get_mistral_client,
    "ollama": _get_ollama_client,
    "vllm": _get_vllm_client,
    "openai": _get_openai_client,
    "gemini": _get_gemini_client,
    "watsonx": _get_watsonx_client,
}


def get_client(provider: str) -> Type[BaseLlmClient]:
    """
    Get LLM client class for the specified provider.

    Uses lazy imports, so client packages are only loaded when actually used.

    Args:
        provider: Provider name (mistral, ollama, vllm, openai, gemini, watsonx)

    Returns:
        The client class for the provider

    Raises:
        ValueError: If provider is not recognized
        ImportError: If provider package is not installed
    """
    if provider not in _CLIENT_REGISTRY:
        available = ", ".join(_CLIENT_REGISTRY.keys())
        raise ValueError(f"Unknown provider '{provider}'. Available providers: {available}")

    # Call the lazy import function
    return _CLIENT_REGISTRY[provider]()
