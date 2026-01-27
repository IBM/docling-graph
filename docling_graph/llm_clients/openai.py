"""
OpenAI API client implementation (refactored).

Reduced from 124 lines to ~70 lines by using the new base class
and ResponseHandler.
"""

import logging
from typing import TYPE_CHECKING, Any, Dict

from rich import print as rich_print

from ..exceptions import ClientError, ConfigurationError
from .base import BaseLlmClient

logger = logging.getLogger(__name__)

# Lazy import for optional dependency
_OpenAI: Any | None = None

try:
    from openai import OpenAI as OpenAI_module

    _OpenAI = OpenAI_module
except ImportError:
    logger.warning("openai package not found. Install with: pip install 'docling-graph[openai]'")
    _OpenAI = None

OpenAI: Any = _OpenAI

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageParam


class OpenAIClient(BaseLlmClient):
    """OpenAI API client - refactored version."""

    def _provider_id(self) -> str:
        """Return provider ID for configuration."""
        return "openai"

    def _setup_client(self, **kwargs: Any) -> None:
        """
        Initialize OpenAI client.

        Raises:
            ConfigurationError: If API key is missing or package not installed
        """
        if _OpenAI is None:
            raise ConfigurationError(
                "OpenAI client requires 'openai' package",
                details={
                    "install_command": "pip install 'docling-graph[openai]'",
                    "alternative": "pip install openai",
                },
            )

        api_key = self.connection.api_key.get_secret_value() if self.connection.api_key else None
        if not api_key:
            raise ConfigurationError(
                "OpenAI API key missing",
                details={"provider": "openai"},
            )

        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if self.connection.base_url:
            client_kwargs["base_url"] = self.connection.base_url
        if self.connection.organization:
            client_kwargs["organization"] = self.connection.organization

        self.client = OpenAI(**client_kwargs)

        rich_print(f"[blue][OpenAI][/blue] Initialized for model: [cyan]{self.model}[/cyan]")

    def _call_api(
        self, messages: list[Dict[str, str]], **params: Any
    ) -> tuple[str, Dict[str, Any]]:
        """
        Call OpenAI API.

        Args:
            messages: List of message dicts
            **params: Additional parameters

        Returns:
            Tuple of (raw_response, metadata) where metadata contains finish_reason

        Raises:
            ClientError: If API call fails
        """
        # Convert to OpenAI message format
        if TYPE_CHECKING:
            api_messages: list[ChatCompletionMessageParam] = messages  # type: ignore
        else:
            api_messages = messages

        try:
            gen = self.generation
            max_tokens = gen.max_tokens or self._max_output_tokens
            timeout_seconds = self.timeout

            request_params: dict[str, Any] = {
                "model": self.model,
                "messages": api_messages,
                "response_format": {"type": "json_object"},
                "temperature": gen.temperature,
                "max_tokens": max_tokens,
                "timeout": timeout_seconds,
            }
            if gen.top_p is not None:
                request_params["top_p"] = gen.top_p
            if gen.frequency_penalty is not None:
                request_params["frequency_penalty"] = gen.frequency_penalty
            if gen.presence_penalty is not None:
                request_params["presence_penalty"] = gen.presence_penalty
            if gen.seed is not None:
                request_params["seed"] = gen.seed
            if gen.stop is not None:
                request_params["stop"] = gen.stop

            response = self.client.chat.completions.create(**request_params)

            content = response.choices[0].message.content
            finish_reason = response.choices[0].finish_reason

            if not content:
                raise ClientError("OpenAI returned empty content", details={"model": self.model})

            # Return response and metadata
            metadata = {
                "finish_reason": finish_reason,
                "model": self.model,
            }

            return str(content), metadata

        except Exception as e:
            raise ClientError(
                f"OpenAI API call failed: {type(e).__name__}",
                details={"model": self.model, "error": str(e)},
                cause=e,
            ) from e
