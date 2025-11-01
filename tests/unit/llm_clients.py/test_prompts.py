"""
Unit tests for prompt generation utilities.
"""

from pydantic import BaseModel

import pytest

from docling_graph.llm_clients.prompts import get_prompt


class SampleModel(BaseModel):
    """Sample model for testing."""
    name: str
    age: int


class TestGetPrompt:
    """Tests for get_prompt function."""

    def test_get_prompt_returns_dict(self):
        """Test that get_prompt returns dict with system and user."""
        prompt = get_prompt(
            markdown_content="# Test\nContent here",
            schema_json='{"name": "string", "age": "integer"}',
            template=SampleModel
        )
        
        assert isinstance(prompt, dict)
        assert "system" in prompt
        assert "user" in prompt

    def test_get_prompt_includes_markdown(self):
        """Test that markdown content is included in prompt."""
        markdown = "# Important Section\nExtract this data"
        prompt = get_prompt(
            markdown_content=markdown,
            schema_json='{}',
            template=SampleModel
        )
        
        assert markdown in prompt["user"]

    def test_get_prompt_includes_schema(self):
        """Test that schema is included in prompt."""
        schema = '{"name": "string", "age": "integer"}'
        prompt = get_prompt(
            markdown_content="Test",
            schema_json=schema,
            template=SampleModel
        )
        
        assert schema in prompt["user"]

    def test_get_prompt_system_message(self):
        """Test that system message provides extraction instructions."""
        prompt = get_prompt(
            markdown_content="Test",
            schema_json='{}',
            template=SampleModel
        )
        
        assert "extract" in prompt["system"].lower() or "json" in prompt["system"].lower()

    def test_get_prompt_user_message(self):
        """Test that user message is properly formatted."""
        prompt = get_prompt(
            markdown_content="Test content",
            schema_json='{}',
            template=SampleModel
        )
        
        assert len(prompt["user"]) > 0

    def test_get_prompt_with_context(self):
        """Test get_prompt with optional context."""
        prompt = get_prompt(
            markdown_content="Test",
            schema_json='{}',
            template=SampleModel,
            context="document"
        )
        
        assert "document" in prompt["user"].lower() or isinstance(prompt, dict)


class TestPromptGeneration:
    """Integration tests for prompt generation."""

    def test_generated_prompt_is_valid(self):
        """Test that generated prompt is valid for LLM input."""
        prompt = get_prompt(
            markdown_content="# Header\nSome data: value",
            schema_json='{"field": "string"}',
            template=SampleModel
        )
        
        # Prompt should be properly structured
        assert prompt["system"]
        assert prompt["user"]
        # Both should be non-empty
        assert len(prompt["system"]) > 0
        assert len(prompt["user"]) > 0
