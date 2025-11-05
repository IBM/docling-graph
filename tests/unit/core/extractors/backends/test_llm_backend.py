import json
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, ValidationError

from docling_graph.core.extractors.backends.llm_backend import LlmBackend
from docling_graph.llm_clients.base import BaseLlmClient


# A simple Pydantic model for testing
class MockTemplate(BaseModel):
    name: str
    age: int


# Fixture for a mock LLM client
@pytest.fixture
def mock_llm_client():
    client = MagicMock(spec=BaseLlmClient)
    client.__class__.__name__ = "MockLlmClient"
    return client


# Fixture for the LlmBackend
@pytest.fixture
def llm_backend(mock_llm_client):
    return LlmBackend(llm_client=mock_llm_client)


def test_init(llm_backend, mock_llm_client):
    """Test that the backend initializes with the client."""
    assert llm_backend.client == mock_llm_client


@patch("docling_graph.core.extractors.backends.llm_backend.get_extraction_prompt")
def test_extract_from_markdown_success(mock_get_prompt, llm_backend, mock_llm_client):
    """Test successful extraction and validation."""
    markdown = "This is a test."
    context = "test context"
    expected_json = {"name": "Test", "age": 30}
    schema_json = json.dumps(MockTemplate.model_json_schema(), indent=2)

    # Configure mock client
    mock_llm_client.get_json_response.return_value = expected_json
    mock_get_prompt.return_value = {"system": "sys", "user": "user"}

    # Run extraction
    result = llm_backend.extract_from_markdown(
        markdown=markdown, template=MockTemplate, context=context
    )

    # Assertions
    assert isinstance(result, MockTemplate)
    assert result.name == "Test"
    assert result.age == 30
    mock_get_prompt.assert_called_with(
        markdown_content=markdown, schema_json=schema_json, is_partial=False
    )
    mock_llm_client.get_json_response.assert_called_with(
        prompt={"system": "sys", "user": "user"}, schema_json=schema_json
    )


def test_extract_from_markdown_empty_input(llm_backend):
    """Test that empty or whitespace-only markdown returns None."""
    result_empty = llm_backend.extract_from_markdown(markdown="", template=MockTemplate)
    result_whitespace = llm_backend.extract_from_markdown(markdown="   \n ", template=MockTemplate)

    assert result_empty is None
    assert result_whitespace is None


def test_extract_from_markdown_no_json_returned(llm_backend, mock_llm_client):
    """Test when the LLM client returns no valid JSON (e.g., None)."""
    mock_llm_client.get_json_response.return_value = None

    result = llm_backend.extract_from_markdown(markdown="Some content", template=MockTemplate)
    assert result is None


@patch("docling_graph.core.extractors.backends.llm_backend.rich_print")
def test_extract_from_markdown_validation_error(mock_rich_print, llm_backend, mock_llm_client):
    """Test when the LLM returns JSON that fails Pydantic validation."""
    # 'age' is missing, which is a required field
    invalid_json = {"name": "Test Only"}
    mock_llm_client.get_json_response.return_value = invalid_json

    result = llm_backend.extract_from_markdown(markdown="Some content", template=MockTemplate)

    # Should fail validation and return None
    assert result is None
    # Check that a validation error was printed
    mock_rich_print.assert_any_call(
        "[blue][LlmBackend][/blue] [yellow]Validation Error for document:[/yellow]"
    )


@patch("docling_graph.core.extractors.backends.llm_backend.get_consolidation_prompt")
def test_consolidate_success(mock_get_prompt, llm_backend, mock_llm_client):
    """Test successful consolidation."""
    raw_models = [MockTemplate(name="Test", age=30)]
    programmatic_model = MockTemplate(name="Test", age=30)
    expected_json = {"name": "Consolidated", "age": 31}

    mock_llm_client.get_json_response.return_value = expected_json
    mock_get_prompt.return_value = "consolidation_prompt"

    result = llm_backend.consolidate_from_pydantic_models(
        raw_models=raw_models,
        programmatic_model=programmatic_model,
        template=MockTemplate,
    )

    assert isinstance(result, MockTemplate)
    assert result.name == "Consolidated"
    assert result.age == 31


@patch("docling_graph.core.extractors.backends.llm_backend.rich_print")
def test_consolidate_validation_error(mock_rich_print, llm_backend, mock_llm_client):
    """Test consolidation with a Pydantic validation error."""
    raw_models = [MockTemplate(name="Test", age=30)]
    programmatic_model = MockTemplate(name="Test", age=30)
    # 'age' is missing
    invalid_json = {"name": "Consolidated"}

    mock_llm_client.get_json_response.return_value = invalid_json

    result = llm_backend.consolidate_from_pydantic_models(
        raw_models=raw_models,
        programmatic_model=programmatic_model,
        template=MockTemplate,
    )

    assert result is None
    mock_rich_print.assert_any_call(
        "[blue][LlmBackend][/blue] [yellow]Validation Error during consolidation:[/yellow]"
    )


@patch("gc.collect")
def test_cleanup(mock_gc_collect, mock_llm_client):
    """Test that cleanup removes the client and calls garbage collection."""
    # Add a mock cleanup method to the client
    mock_llm_client.cleanup = MagicMock()

    backend = LlmBackend(llm_client=mock_llm_client)
    assert hasattr(backend, "client")

    backend.cleanup()

    # Check client's cleanup was called
    mock_llm_client.cleanup.assert_called_once()
    # Check client attribute was deleted
    assert not hasattr(backend, "client")
    # Check gc was called
    mock_gc_collect.assert_called_once()
