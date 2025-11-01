"""
Unit tests for extractor utility functions.
"""

import pytest
from pydantic import BaseModel, Field

from docling_graph.core.extractors.utils import (
    chunk_text,
    consolidate_extracted_data,
    deep_merge_dicts,
    merge_pydantic_models,
    TOKEN_CHAR_RATIO,
)


# Keep existing tests and add these new ones:

class TestDeepMergeDicts:
    """Tests for deep_merge_dicts function."""

    def test_merge_simple_dicts(self):
        """Test merging two simple dictionaries."""
        target = {"a": 1, "b": 2}
        source = {"c": 3}
        
        result = deep_merge_dicts(target, source)
        
        assert result == {"a": 1, "b": 2, "c": 3}
        assert result is target  # Modified in place

    def test_merge_nested_dicts(self):
        """Test merging nested dictionaries."""
        target = {"user": {"name": "Alice", "age": 25}}
        source = {"user": {"age": 26, "email": "alice@example.com"}}
        
        result = deep_merge_dicts(target, source)
        
        assert result["user"]["name"] == "Alice"
        assert result["user"]["age"] == 26
        assert result["user"]["email"] == "alice@example.com"

    def test_merge_lists(self):
        """Test merging lists with deduplication."""
        target = {"tags": ["python", "testing"]}
        source = {"tags": ["testing", "pytest"]}
        
        result = deep_merge_dicts(target, source)
        
        assert "python" in result["tags"]
        assert "pytest" in result["tags"]
        assert result["tags"].count("testing") == 1  # Deduplicated

    def test_skip_empty_values(self):
        """Test that empty values are skipped."""
        target = {"a": 1}
        source = {"b": None, "c": "", "d": [], "e": {}}
        
        result = deep_merge_dicts(target, source)
        
        assert result == {"a": 1}  # No empty values added

    def test_overwrite_scalar_values(self):
        """Test that scalar values are overwritten."""
        target = {"count": 5, "name": "old"}
        source = {"count": 10}
        
        result = deep_merge_dicts(target, source)
        
        assert result["count"] == 10
        assert result["name"] == "old"


class TestChunkText:
    """Tests for chunk_text function."""

    def test_text_within_limit(self):
        """Test text that doesn't need chunking."""
        text = "Short text"
        chunks = chunk_text(text, max_tokens=1000)
        
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_text_exceeding_limit(self):
        """Test text that needs chunking."""
        text = "A" * 10000  # Long text
        chunks = chunk_text(text, max_tokens=100)
        
        assert len(chunks) > 1
        total_length = sum(len(chunk) for chunk in chunks)
        assert total_length <= len(text)

    def test_chunk_on_sentence_boundary(self):
        """Test that chunks break at sentence boundaries."""
        text = "First sentence. " * 1000
        chunks = chunk_text(text, max_tokens=100)
        
        # Most chunks should end with sentence delimiter
        for chunk in chunks[:-1]:
            assert chunk.strip().endswith(".")

    def test_empty_text(self):
        """Test chunking empty text."""
        chunks = chunk_text("", max_tokens=1000)
        
        assert len(chunks) == 0

    def test_token_char_ratio_constant(self):
        """Test TOKEN_CHAR_RATIO constant value."""
        assert TOKEN_CHAR_RATIO == 3.5


class TestConsolidateExtractedData:
    """Tests for consolidate_extracted_data function."""

    def test_consolidate_empty_list(self):
        """Test consolidating empty list."""
        result = consolidate_extracted_data([])
        assert result == {}

    def test_consolidate_single_dict(self):
        """Test consolidating single dictionary."""
        data = [{"name": "Alice", "age": 25}]
        result = consolidate_extracted_data(data)
        
        assert result == {"name": "Alice", "age": 25}

    def test_consolidate_multiple_dicts(self):
        """Test consolidating multiple dictionaries."""
        data = [
            {"name": "Alice", "age": 25},
            {"age": 26, "email": "alice@example.com"},
            {"city": "Paris"}
        ]
        result = consolidate_extracted_data(data)
        
        assert result["name"] == "Alice"
        assert result["age"] == 26
        assert result["email"] == "alice@example.com"
        assert result["city"] == "Paris"

    def test_consolidate_nested_data(self):
        """Test consolidating nested dictionaries."""
        data = [
            {"user": {"name": "Alice"}},
            {"user": {"age": 25}}
        ]
        result = consolidate_extracted_data(data)
        
        assert result["user"]["name"] == "Alice"
        assert result["user"]["age"] == 25


# Keep all your existing tests for merge_pydantic_models
class TestMergePydanticModels:
    """Tests for merge_pydantic_models function."""
    # ... (keep your existing tests)
