"""
Mistral API client implementation.
Based on https://docs.mistral.ai/api/endpoint/chat
"""

import logging
from typing import Any, Dict, cast

from ..exceptions import ClientError, ConfigurationError
from .base import BaseLlmClient

logger = logging.getLogger(__name__)

_Mistral: Any | None = None
try:
    from mistralai import Mistral as Mistral_module

    _Mistral = Mistral_module
except ImportError:
    logger.warning(
        "mistralai package not found. Please run `pip install mistralai` to use Mistral client."
    )
    _Mistral = None

Mistral: Any = _Mistral


class MistralClient(BaseLlmClient):
    """Mistral API implementation using template method pattern."""

    def _provider_id(self) -> str:
        """Return provider ID for config lookup."""
        return "mistral"

    def _setup_client(self, **kwargs: Any) -> None:
        """Initialize Mistral-specific client."""
        if Mistral is None:
            raise ConfigurationError(
                "mistralai package not installed",
                details={"package": "mistralai", "install": "pip install mistralai"},
            )

        api_key = self.connection.api_key.get_secret_value() if self.connection.api_key else None
        if not api_key:
            raise ConfigurationError(
                "Mistral API key missing",
                details={"provider": "mistral"},
            )
        self.client = Mistral(api_key=api_key)

        logger.info(f"Mistral client initialized for model: {self.model}")

    def _call_api(
        self, messages: list[Dict[str, str]], **params: Any
    ) -> tuple[str, Dict[str, Any]]:
        """
        Call Mistral API.

        Args:
            messages: List of message dicts with 'role' and 'content'
            **params: Additional parameters (schema_json, etc.)

        Returns:
            Tuple of (raw_response, metadata) where metadata contains finish_reason

        Raises:
            ClientError: If API call fails
        """
        try:
            gen = self.generation
            max_tokens = gen.max_tokens or self._max_output_tokens
            # Note: Mistral SDK doesn't support timeout parameter in chat.complete()
            # Timeout should be handled at HTTP client level if needed

            params: dict[str, Any] = {
                "model": self.model,
                "messages": cast(Any, messages),
                "response_format": {"type": "json_object"},
                "temperature": gen.temperature,
                "max_tokens": max_tokens,
            }
            if gen.top_p is not None:
                params["top_p"] = gen.top_p

            response = self.client.chat.complete(**params)

            response_content = response.choices[0].message.content
            finish_reason = response.choices[0].finish_reason

            if not response_content:
                raise ClientError("Mistral returned empty content", details={"model": self.model})

            if isinstance(response_content, str):
                content = response_content
            else:
                parts: list[str] = []
                for chunk in response_content:
                    text = getattr(chunk, "text", None)
                    if isinstance(text, str):
                        parts.append(text)
                content = "".join(parts)

            # Return response and metadata
            metadata = {
                "finish_reason": finish_reason,
                "model": self.model,
            }

            return content, metadata

        except Exception as e:
            if isinstance(e, ClientError):
                raise
            raise ClientError(
                f"Mistral API call failed: {type(e).__name__}",
                details={"model": self.model, "error": str(e)},
            ) from e
