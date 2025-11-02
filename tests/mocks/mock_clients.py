"""
Mock LLM clients for testing.
"""

from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock


class MockOllamaClient(MagicMock):
    """Mock Ollama client."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.generate = MagicMock(
            return_value={
                "response": "Mock response from Ollama",
                "model": "llama:7b",
            }
        )
        self.list_models = MagicMock(return_value={"models": [{"name": "llama:7b"}]})


class MockMistralClient(MagicMock):
    """Mock Mistral API client."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.complete = MagicMock(
            return_value=MagicMock(
                choices=[MagicMock(message=MagicMock(content="Mock response from Mistral"))]
            )
        )


class MockOpenAIClient(MagicMock):
    """Mock OpenAI client."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.chat.completions.create = MagicMock(
            return_value=MagicMock(
                choices=[MagicMock(message=MagicMock(content="Mock response from OpenAI"))]
            )
        )
