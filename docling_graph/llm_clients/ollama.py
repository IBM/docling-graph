"""
Ollama (local LLM) client implementation.
Based on https://ollama.com/blog/structured-outputs
"""

import logging
from typing import Any, Dict

from ..exceptions import ClientError, ConfigurationError
from .base import BaseLlmClient

logger = logging.getLogger(__name__)

_ollama: Any | None = None
try:
    import ollama as ollama_module

    _ollama = ollama_module
except ImportError:
    logger.warning(
        "ollama package not found. Please run `pip install ollama` to use Ollama client."
    )
    _ollama = None

ollama: Any = _ollama


class OllamaClient(BaseLlmClient):
    """Ollama (local LLM) implementation using template method pattern."""

    def _provider_id(self) -> str:
        """Return provider ID for config lookup."""
        return "ollama"

    def _setup_client(self, **kwargs: Any) -> None:
        """Initialize Ollama-specific client."""
        if ollama is None:
            raise ConfigurationError(
                "ollama package not installed",
                details={"package": "ollama", "install": "pip install ollama"},
            )

        try:
            base_url = self.connection.base_url
            if base_url:
                self.client = ollama.Client(host=base_url)
            else:
                self.client = ollama.Client()

            logger.info(f"Checking Ollama connection and model '{self.model}'...")
            self.client.show(self.model)
            logger.info(f"Ollama client initialized with model: {self.model}")
        except Exception as e:
            raise ConfigurationError(
                f"Ollama connection failed: {e}",
                details={
                    "model": self.model,
                    "base_url": self.connection.base_url,
                    "error": str(e),
                    "instructions": [
                        "1. Ensure Ollama is running: ollama serve",
                        f"2. Model is available: ollama pull {self.model}",
                    ],
                },
            ) from e

    def _call_api(
        self, messages: list[Dict[str, str]], **params: Any
    ) -> tuple[str, Dict[str, Any]]:
        """
        Call Ollama API.

        Args:
            messages: List of message dicts with 'role' and 'content'
            **params: Additional parameters (schema_json, etc.)

        Returns:
            Tuple of (raw_response, metadata) - Ollama doesn't provide finish_reason

        Raises:
            ClientError: If API call fails
        """
        try:
            gen = self.generation
            max_tokens = gen.max_tokens or self._max_output_tokens

            options: dict[str, Any] = {
                "temperature": gen.temperature,
                "num_predict": max_tokens,
            }
            if gen.top_p is not None:
                options["top_p"] = gen.top_p
            if gen.top_k is not None:
                options["top_k"] = gen.top_k
            if gen.repetition_penalty is not None:
                options["repeat_penalty"] = gen.repetition_penalty

            response = self.client.chat(
                model=self.model,
                messages=messages,
                format="json",
                options=options,
            )

            raw_json = response["message"]["content"]

            if not raw_json:
                raise ClientError("Ollama returned empty content", details={"model": self.model})

            # Ollama doesn't provide finish_reason
            metadata = {
                "model": self.model,
            }

            return str(raw_json), metadata

        except Exception as e:
            if isinstance(e, ClientError):
                raise
            raise ClientError(
                f"Ollama API call failed: {type(e).__name__}",
                details={"model": self.model, "error": str(e)},
            ) from e
