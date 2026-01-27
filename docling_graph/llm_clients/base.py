"""
Enhanced base class for all LLM clients with template method pattern.

This refactored base class eliminates code duplication by providing:
- Shared JSON response handling via ResponseHandler
- Common message preparation logic
- Unified error handling
- Consistent configuration loading

Each client only needs to implement provider-specific API calls.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict

from .config import EffectiveModelConfig
from .response_handler import ResponseHandler

logger = logging.getLogger(__name__)


class BaseLlmClient(ABC):
    """
    Enhanced base class for all LLM clients.

    Uses template method pattern to eliminate duplication while allowing
    provider-specific customization where needed.

    Subclasses must implement:
    - _setup_client(): Provider-specific initialization
    - _call_api(): Provider-specific API call
    - _provider_id(): Return provider identifier for config lookup
    """

    def __init__(self, model_config: EffectiveModelConfig, **kwargs: Any) -> None:
        """
        Initialize LLM client.

        Args:
            model_config: Fully-resolved model configuration
            **kwargs: Provider-specific parameters
        """
        self._config = model_config
        self.model = model_config.provider_model
        self.model_id = model_config.model_id
        self._context_limit = model_config.context_limit
        self._max_output_tokens = model_config.max_output_tokens
        self._generation = model_config.generation
        self._reliability = model_config.reliability
        self._connection = model_config.connection

        # Provider-specific setup
        self._setup_client(**kwargs)

        logger.info(f"{self.__class__.__name__} initialized for model: {self.model}")

    @abstractmethod
    def _setup_client(self, **kwargs: Any) -> None:
        """
        Provider-specific client initialization.

        This method should:
        - Load API credentials
        - Initialize provider SDK
        - Validate configuration

        Args:
            **kwargs: Provider-specific parameters

        Raises:
            ConfigurationError: If setup fails
        """

    @abstractmethod
    def _call_api(
        self, messages: list[Dict[str, str]], **params: Any
    ) -> tuple[str, Dict[str, Any]]:
        """
        Provider-specific API call.

        This method should call the provider's API and return both the raw
        response and metadata about the generation (e.g., finish_reason).

        Args:
            messages: List of message dicts with 'role' and 'content'
            **params: Additional parameters (e.g., schema_json)

        Returns:
            Tuple of (raw_response, metadata) where metadata contains:
            - finish_reason: Why generation stopped (if available)
            - usage: Token usage info (if available)
            - other provider-specific data

        Raises:
            ClientError: If API call fails
        """

    @abstractmethod
    def _provider_id(self) -> str:
        """
        Return provider identifier for configuration lookup.

        Returns:
            Provider ID (e.g., "openai", "mistral", "watsonx")
        """

    def get_json_response(
        self, prompt: str | dict[str, str], schema_json: str
    ) -> Dict[str, Any] | list[Any]:
        """
        Execute LLM call and return parsed JSON response.

        This method is the same for all clients - it handles:
        - Message preparation
        - API call
        - Truncation detection
        - Response parsing and validation

        Args:
            prompt: Either a string or dict with 'system' and 'user' keys
            schema_json: Pydantic schema as JSON string

        Returns:
            Parsed and validated JSON (dictionary or list)

        Raises:
            ClientError: If API call or parsing fails
        """
        # Prepare messages
        messages = self._prepare_messages(prompt)

        # Call provider API (returns response + metadata)
        raw_response, metadata = self._call_api(messages, schema_json=schema_json)

        # Check for truncation
        truncated = self._check_truncation(metadata)

        # Parse using shared handler with truncation awareness
        return ResponseHandler.parse_json_response(
            raw_response,
            self.__class__.__name__,
            aggressive_clean=self._needs_aggressive_cleaning(),
            truncated=truncated,
            max_tokens=self.max_tokens,
        )

    def _prepare_messages(self, prompt: str | dict) -> list[Dict[str, str]]:
        """
        Convert prompt to standardized message format.

        Args:
            prompt: String or dict with 'system' and 'user' keys

        Returns:
            List of message dictionaries
        """
        if isinstance(prompt, dict):
            messages = []
            if prompt.get("system"):
                messages.append({"role": "system", "content": prompt["system"]})
            if "user" in prompt:
                messages.append({"role": "user", "content": prompt["user"]})
            return messages
        else:
            return [{"role": "user", "content": prompt}]

    def _needs_aggressive_cleaning(self) -> bool:
        """
        Override this to enable aggressive response cleaning.

        Returns:
            True if provider needs extra cleaning (e.g., WatsonX)
        """
        return False

    def _check_truncation(self, metadata: Dict[str, Any]) -> bool:
        """
        Check if response was truncated due to max_tokens limit.

        Args:
            metadata: Response metadata from _call_api()

        Returns:
            True if response was truncated, False otherwise
        """
        # Check for finish_reason (OpenAI-compatible APIs)
        finish_reason = metadata.get("finish_reason")
        if finish_reason:
            return bool(finish_reason == "length")

        # For providers without finish_reason, use heuristics
        # (Conservative: don't assume truncation without evidence)
        return False

    @property
    def provider(self) -> str:
        """Return the provider identifier for this client."""
        return self._config.provider_id

    @property
    def context_limit(self) -> int:
        """Return the context window size in tokens."""
        return self._context_limit

    @property
    def max_tokens(self) -> int:
        """Return the maximum tokens to generate."""
        return self._generation.max_tokens or self._max_output_tokens

    @property
    def timeout(self) -> int:
        """Return the request timeout in seconds."""
        return self._reliability.timeout_s

    @property
    def generation(self) -> Any:
        """Return resolved generation settings."""
        return self._generation

    @property
    def reliability(self) -> Any:
        """Return resolved reliability settings."""
        return self._reliability

    @property
    def connection(self) -> Any:
        """Return resolved connection settings."""
        return self._connection

    @property
    def model_config(self) -> EffectiveModelConfig:
        """Return the resolved model configuration."""
        return self._config
