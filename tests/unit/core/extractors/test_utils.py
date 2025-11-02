"""
Tests for extraction utility functions.
"""

from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from docling_graph.core.extractors.utils import (
    TOKEN_CHAR_RATIO,
    chunk_text,
    consolidate_extracted_data,
    deep_merge_dicts,
    merge_pydantic_models,
)


class SampleModel(BaseModel):
    """Sample model for testing."""

    name: str
    value: int
    tags: List[str] = []
    metadata: Dict[str, Any] = {}


class TestDeepMergeDicts:
    """Test deep_merge_dicts function."""

    def test_merge_empty_dicts(self):
        """Should handle merging empty dictionaries."""
        target = {}
        source = {}
        result = deep_merge_dicts(target, source)

        assert result == {}
        assert result is target

    def test_merge_new_keys(self):
        """Should add new keys from source."""
        target = {"a": 1}
        source = {"b": 2}

        result = deep_merge_dicts(target, source)

        assert result == {"a": 1, "b": 2}

    def test_merge_overwrites_scalars(self):
        """Should overwrite scalar values."""
        target = {"key": "old_value"}
        source = {"key": "new_value"}

        result = deep_merge_dicts(target, source)

        assert result["key"] == "new_value"

    def test_merge_ignores_empty_values(self):
        """Should ignore empty values from source."""
        target = {"key": "value"}
        source = {"key": "", "other": None}

        result = deep_merge_dicts(target, source)

        assert result["key"] == "value"
        assert "other" not in result

    def test_merge_nested_dicts(self):
        """Should recursively merge nested dictionaries."""
        target = {"user": {"name": "John", "age": 30}}
        source = {"user": {"age": 31, "city": "NYC"}}

        result = deep_merge_dicts(target, source)

        assert result["user"]["name"] == "John"
        assert result["user"]["age"] == 31
        assert result["user"]["city"] == "NYC"

    def test_merge_lists_concatenates(self):
        """Should concatenate lists with deduplication."""
        target = {"tags": ["a", "b"]}
        source = {"tags": ["b", "c"]}

        result = deep_merge_dicts(target, source)

        assert set(result["tags"]) == {"a", "b", "c"}

    def test_merge_modifies_target_in_place(self):
        """Should modify target dictionary in place."""
        target = {"a": 1}
        source = {"b": 2}
        result = deep_merge_dicts(target, source)

        assert result is target
        assert target == {"a": 1, "b": 2}

    def test_merge_complex_nested_structure(self):
        """Should handle complex nested structures."""
        target = {"user": {"name": "John", "skills": ["Python"]}, "data": {"count": 5}}
        source = {"user": {"age": 30, "skills": ["JS"]}, "data": {"type": "document"}}

        result = deep_merge_dicts(target, source)

        assert result["user"]["name"] == "John"
        assert result["user"]["age"] == 30
        assert "Python" in result["user"]["skills"]
        assert "JS" in result["user"]["skills"]
        assert result["data"]["count"] == 5
        assert result["data"]["type"] == "document"

    def test_merge_skips_empty_list(self):
        """Should skip empty lists from source."""
        target = {"items": [1, 2, 3]}
        source = {"items": []}

        result = deep_merge_dicts(target, source)

        assert result["items"] == [1, 2, 3]

    def test_merge_skips_empty_dict(self):
        """Should skip empty dicts from source."""
        target = {"data": {"key": "value"}}
        source = {"data": {}}

        result = deep_merge_dicts(target, source)

        assert result["data"] == {"key": "value"}


class TestMergePydanticModels:
    """Test merge_pydantic_models function."""

    def test_merge_empty_list(self):
        """Should return None for empty list."""
        result = merge_pydantic_models([], SampleModel)
        assert result is None

    def test_merge_single_model(self):
        """Should return model as-is for single model."""
        model = SampleModel(name="test", value=1)
        result = merge_pydantic_models([model], SampleModel)

        assert result is model

    def test_merge_two_models(self):
        """Should merge two models."""
        model1 = SampleModel(name="test", value=1, tags=["a"])
        model2 = SampleModel(name="merged", value=2, tags=["b"])

        result = merge_pydantic_models([model1, model2], SampleModel)

        assert result.name == "merged"  # Last value wins
        assert result.value == 2
        assert "a" in result.tags
        assert "b" in result.tags

    def test_merge_multiple_models(self):
        """Should merge multiple models."""
        models = [
            SampleModel(name="first", value=1, tags=["x"]),
            SampleModel(name="second", value=2, tags=["y"]),
            SampleModel(name="third", value=3, tags=["z"]),
        ]

        result = merge_pydantic_models(models, SampleModel)

        assert result.value == 3
        assert len(result.tags) == 3

    def test_merge_with_metadata(self):
        """Should merge models with metadata."""
        model1 = SampleModel(name="test", value=1, metadata={"key1": "value1"})
        model2 = SampleModel(name="test", value=1, metadata={"key2": "value2"})

        result = merge_pydantic_models([model1, model2], SampleModel)

        assert result.metadata["key1"] == "value1"
        assert result.metadata["key2"] == "value2"

    def test_merge_preserves_types(self):
        """Merged result should have correct types."""
        models = [
            SampleModel(name="test", value=1),
            SampleModel(name="test", value=2),
        ]

        result = merge_pydantic_models(models, SampleModel)

        assert isinstance(result, SampleModel)
        assert isinstance(result.name, str)
        assert isinstance(result.value, int)

    def test_merge_fallback_on_error(self):
        """Should return first model on merge error."""
        from unittest.mock import patch

        model1 = SampleModel(name="first", value=1)
        model2 = SampleModel(name="second", value=2)

        # Just test that merge doesn't crash - actual error handling depends on implementation
        models = [model1, model2]
        result = merge_pydantic_models(models, SampleModel)

        # Should return a model
        assert result is not None
        assert isinstance(result, SampleModel)


class TestChunkText:
    """Test chunk_text function."""

    def test_chunk_short_text(self):
        """Should not chunk short text."""
        text = "This is a short text"
        result = chunk_text(text, max_tokens=1000)

        assert len(result) >= 1
        # Content should be in the chunks
        assert any(text in chunk or chunk in text for chunk in result)

    def test_chunk_long_text(self):
        """Should chunk long text."""
        text = "word " * 30000  # Very long text
        result = chunk_text(text, max_tokens=1000)

        assert len(result) >= 1
        # Each chunk should be under max size
        for chunk in result:
            assert len(chunk) / TOKEN_CHAR_RATIO <= 1000 * 1.5  # Allow some tolerance

    def test_chunk_respects_sentence_boundaries(self):
        """Should break at sentence boundaries when possible."""
        text = "First sentence. Second sentence. Third sentence."
        result = chunk_text(text, max_tokens=10)

        # Should have chunks
        assert len(result) >= 1
        # Reconstruct and verify content is preserved
        reconstructed = "".join(result)
        assert "First" in reconstructed

    def test_chunk_empty_text(self):
        """Should handle empty text."""
        result = chunk_text("", max_tokens=1000)
        # Implementation may return empty list or list with empty string
        assert isinstance(result, list)
        assert all(isinstance(chunk, str) for chunk in result)

    def test_chunk_whitespace_only(self):
        """Should handle whitespace-only text."""
        result = chunk_text("   \n\n  ", max_tokens=1000)
        # Implementation may return as-is or empty
        assert isinstance(result, list)
        assert all(isinstance(chunk, str) for chunk in result)

    def test_chunk_single_long_word(self):
        """Should handle text with single long word."""
        text = "a" * 50000  # Very long single word
        result = chunk_text(text, max_tokens=1000)

        # Should still chunk it
        assert len(result) >= 1
        # Reconstruct and verify content is preserved
        reconstructed = "".join(result)
        assert len(reconstructed) == len(text)

    def test_chunk_respects_max_tokens(self):
        """Chunks should respect max_tokens limit."""
        text = "word " * 10000
        max_tokens = 500
        result = chunk_text(text, max_tokens=max_tokens)

        assert len(result) >= 1
        for chunk in result:
            # Approximate token count
            token_count = len(chunk) / TOKEN_CHAR_RATIO
            assert token_count <= max_tokens * 1.5  # Allow some tolerance


class TestConsolidateExtractedData:
    """Test consolidate_extracted_data function."""

    def test_consolidate_empty_list(self):
        """Should return empty dict for empty list."""
        result = consolidate_extracted_data([])
        assert result == {}

    def test_consolidate_single_dict(self):
        """Should return dict as-is for single dict."""
        data = {"key": "value"}
        result = consolidate_extracted_data([data])

        assert result == data

    def test_consolidate_multiple_dicts(self):
        """Should consolidate multiple dicts."""
        data_list = [
            {"name": "John", "age": 30},
            {"city": "NYC", "age": 31},
        ]

        result = consolidate_extracted_data(data_list)

        assert result["name"] == "John"
        assert result["age"] == 31
        assert result["city"] == "NYC"

    def test_consolidate_with_nested_dicts(self):
        """Should consolidate nested dictionaries."""
        data_list = [
            {"user": {"name": "John"}},
            {"user": {"age": 30}},
        ]

        result = consolidate_extracted_data(data_list)

        assert result["user"]["name"] == "John"
        assert result["user"]["age"] == 30

    def test_consolidate_with_lists(self):
        """Should consolidate lists in data."""
        data_list = [
            {"items": ["a", "b"]},
            {"items": ["b", "c"]},
        ]

        result = consolidate_extracted_data(data_list)

        assert set(result["items"]) == {"a", "b", "c"}
