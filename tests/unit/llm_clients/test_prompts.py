import json
from typing import List

import pytest
from pydantic import BaseModel, Field

from docling_graph.llm_clients.prompts import (
    _CONSOLIDATION_PROMPT,
    _SYSTEM_PROMPT_COMPLETE,
    _SYSTEM_PROMPT_PARTIAL,
    _USER_PROMPT_TEMPLATE,
    get_consolidation_prompt,
    get_extraction_prompt,
)

# --- Test get_extraction_prompt ---


def test_get_extraction_prompt_partial():
    """Tests that the partial prompt (for chunks) is generated correctly."""
    markdown = "This is page 1."
    schema = '{"title": "Test"}'

    prompt_dict = get_extraction_prompt(markdown, schema, is_partial=True)

    assert prompt_dict["system"] == _SYSTEM_PROMPT_PARTIAL

    expected_user_prompt = _USER_PROMPT_TEMPLATE.format(
        document_type="document page",
        document_type_lower="document page",
        delimiter="DOCUMENT PAGE",
        markdown_content=markdown,
        schema_json=schema,
    )
    assert prompt_dict["user"] == expected_user_prompt
    assert "document page" in prompt_dict["user"]
    assert "DOCUMENT PAGE" in prompt_dict["user"]


def test_get_extraction_prompt_complete():
    """Tests that the complete prompt (for full docs) is generated correctly."""
    markdown = "This is the full document."
    schema = '{"title": "Test"}'

    prompt_dict = get_extraction_prompt(markdown, schema, is_partial=False)

    assert prompt_dict["system"] == _SYSTEM_PROMPT_COMPLETE

    expected_user_prompt = _USER_PROMPT_TEMPLATE.format(
        document_type="complete document",
        document_type_lower="complete document",
        delimiter="COMPLETE DOCUMENT",
        markdown_content=markdown,
        schema_json=schema,
    )
    assert prompt_dict["user"] == expected_user_prompt
    assert "complete document" in prompt_dict["user"]
    assert "COMPLETE DOCUMENT" in prompt_dict["user"]


# --- Test get_consolidation_prompt ---


class SimpleModel(BaseModel):
    name: str
    value: int


class ComplexModel(BaseModel):
    items: List[SimpleModel] = Field(default_factory=list)


@pytest.fixture
def sample_models():
    """Provides a list of Pydantic models for consolidation testing."""
    m1 = ComplexModel(items=[SimpleModel(name="A", value=1)])
    m2 = ComplexModel(items=[SimpleModel(name="B", value=2)])
    return [m1, m2]


@pytest.fixture
def sample_programmatic_model():
    """Provides a merged Pydantic model for consolidation testing."""
    return ComplexModel(items=[SimpleModel(name="A", value=1), SimpleModel(name="B", value=2)])


@pytest.fixture
def sample_schema_json():
    """Provides the JSON schema string for the test model."""
    # Pydantic v2 model_json_schema() does not take 'indent'
    return json.dumps(ComplexModel.model_json_schema(), indent=2)


def test_get_consolidation_prompt_with_programmatic(
    sample_models, sample_programmatic_model, sample_schema_json
):
    """Tests the consolidation prompt when a programmatic draft is available."""
    schema_json = sample_schema_json

    prompt = get_consolidation_prompt(
        schema_json=schema_json,
        raw_models=sample_models,
        programmatic_model=sample_programmatic_model,
    )

    raw_jsons_expected = "\n\n---\n\n".join(m.model_dump_json(indent=2) for m in sample_models)
    programmatic_json_expected = sample_programmatic_model.model_dump_json(indent=2)

    expected_prompt = _CONSOLIDATION_PROMPT.format(
        schema_json=schema_json,
        raw_jsons=raw_jsons_expected,
        programmatic_json=programmatic_json_expected,
    )

    assert prompt == expected_prompt
    assert schema_json in prompt
    assert raw_jsons_expected in prompt
    assert programmatic_json_expected in prompt


def test_get_consolidation_prompt_no_programmatic(sample_models, sample_schema_json):
    """Tests the consolidation prompt when no programmatic draft is provided."""
    schema_json = sample_schema_json

    prompt = get_consolidation_prompt(
        schema_json=schema_json, raw_models=sample_models, programmatic_model=None
    )

    raw_jsons_expected = "\n\n---\n\n".join(m.model_dump_json(indent=2) for m in sample_models)
    programmatic_json_expected = "No programmatic merge available."

    expected_prompt = _CONSOLIDATION_PROMPT.format(
        schema_json=schema_json,
        raw_jsons=raw_jsons_expected,
        programmatic_json=programmatic_json_expected,
    )

    assert prompt == expected_prompt
    assert schema_json in prompt
    assert raw_jsons_expected in prompt
    assert programmatic_json_expected in prompt


def test_get_consolidation_prompt_empty_raw(sample_schema_json):
    """Tests the consolidation prompt when the list of raw models is empty."""
    schema_json = sample_schema_json

    prompt = get_consolidation_prompt(
        schema_json=schema_json, raw_models=[], programmatic_model=None
    )

    raw_jsons_expected = ""
    programmatic_json_expected = "No programmatic merge available."

    expected_prompt = _CONSOLIDATION_PROMPT.format(
        schema_json=schema_json,
        raw_jsons=raw_jsons_expected,
        programmatic_json=programmatic_json_expected,
    )

    assert prompt == expected_prompt
    assert raw_jsons_expected in prompt
    assert programmatic_json_expected in prompt
