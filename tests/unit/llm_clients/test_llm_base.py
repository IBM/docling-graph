from abc import ABC, abstractmethod
from typing import Any, Dict

import pytest

from docling_graph.llm_clients.base import BaseLlmClient


def test_base_llm_client_is_abc():
    """Tests that BaseLlmClient is an Abstract Base Class."""
    assert issubclass(BaseLlmClient, ABC)


def test_concrete_implementation_requires_methods():
    """Tests that instantiating a subclass fails if abstract methods are not implemented."""

    # This class is incomplete
    class IncompleteClient(BaseLlmClient):
        pass

    with pytest.raises(TypeError, match=r"Can't instantiate abstract class .* abstract methods"):
        IncompleteClient()

    # This class implements all required abstract methods
    class CompleteClient(BaseLlmClient):
        def __init__(self, model: str, **kwargs: Any) -> None:
            self.model = model
            self._context_limit = 8000

        def get_json_response(
            self, prompt: str | dict[str, str], schema_json: str
        ) -> Dict[str, Any]:
            return {"success": True}

        @property
        def context_limit(self) -> int:
            return self._context_limit

    # This should not raise an error
    client = CompleteClient(model="test-model")
    assert client.model == "test-model"
    assert client.context_limit == 8000
    assert client.get_json_response("prompt", "{}") == {"success": True}
