"""
Unit tests for LlmClientBase class.
"""

import pytest

from docling_graph.llm_clients.llm_base import BaseLlmClient


class ConcreteClient(BaseLlmClient):
    """Concrete implementation for testing."""
    
    def __init__(self):
        self._context_limit = 4096
    
    def get_json_response(self, prompt, schema_json):
        return {}
    
    @property
    def context_limit(self):
        return self._context_limit


class TestBaseLlmClientAbstract:
    """Tests for BaseLlmClient abstract class."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that BaseLlmClient cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseLlmClient()

    def test_concrete_implementation_works(self):
        """Test that concrete implementation can be instantiated."""
        client = ConcreteClient()
        assert client is not None

    def test_context_limit_property(self):
        """Test context_limit property."""
        client = ConcreteClient()
        assert client.context_limit == 4096

    def test_get_json_response_method(self):
        """Test get_json_response method."""
        client = ConcreteClient()
        result = client.get_json_response("test prompt", "{}")
        assert isinstance(result, dict)
