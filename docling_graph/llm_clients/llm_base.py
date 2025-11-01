from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseLlmClient(ABC):
    """
    Abstract base class for all LLM clients (Mistral, Ollama, OpenAI, etc.).
    Defines a common interface for the ManyToOneExtractor.
    """

    @abstractmethod
    def __init__(self, model: str, **kwargs: Any) -> None:
        """
        Initialize the client.
        All client implementations must accept at least a 'model' argument.
        """
        ...

    @abstractmethod
    def get_json_response(self, prompt: str | dict[str, str], schema_json: str) -> Dict[str, Any]:
        """
        Executes the LLM call with the given prompt and schema.

        Args:
            prompt (str | dict[str, str]): The full prompt to send to the model (legacy string or structured dict).
            schema_json (str): The Pydantic schema (for models that support it).

        Returns:
            Dict[str, Any]: The parsed JSON dictionary from the LLM.
        """

    @property
    @abstractmethod
    def context_limit(self) -> int:
        """
        Returns the effective context limit (in tokens) for the model.
        This should be a conservative number, leaving room for prompts.
        """
