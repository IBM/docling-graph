"""
Tests for string formatting utilities.
"""

import pytest

from docling_graph.core.utils.formatting import (
    format_property_key,
    format_property_value,
    truncate_string,
)


class TestFormatPropertyValue:
    """Test property value formatting."""

    def test_format_property_value_string(self):
        """Should format string values."""
        result = format_property_value("test string")
        assert result == "test string"

    def test_format_property_value_list(self):
        """Should format lists as Python notation."""
        result = format_property_value([1, 2, 3])
        assert result == "[1, 2, 3]"

    def test_format_property_value_dict(self):
        """Should format dicts as string."""
        result = format_property_value({"key": "value"})
        assert "key" in result
        assert "value" in result

    def test_format_property_value_truncation(self):
        """Should truncate long strings."""
        long_string = "a" * 100
        result = format_property_value(long_string, max_length=50)

        assert len(result) <= 50
        assert result.endswith("...")

    def test_format_property_value_no_truncation_needed(self):
        """Should not truncate short strings."""
        short_string = "short"
        result = format_property_value(short_string)

        assert result == short_string

    def test_format_property_value_number(self):
        """Should format numbers."""
        assert format_property_value(42) == "42"
        assert format_property_value(3.14) == "3.14"

    def test_format_property_value_none(self):
        """Should format None."""
        result = format_property_value(None)
        assert result == "None"

    def test_format_property_value_empty_list(self):
        """Should format empty lists."""
        result = format_property_value([])
        assert result == "[]"

    def test_format_property_value_nested_list(self):
        """Should format nested lists."""
        result = format_property_value([1, [2, 3], 4])
        assert "[2, 3]" in result


class TestFormatPropertyKey:
    """Test property key formatting."""

    def test_format_property_key_snake_case(self):
        """Should convert snake_case to Title Case."""
        result = format_property_key("snake_case_key")
        assert result == "Snake Case Key"

    def test_format_property_key_camel_case(self):
        """Should convert camelCase to Title Case."""
        result = format_property_key("camelCaseKey")
        assert result == "Camel Case Key"

    def test_format_property_key_single_word(self):
        """Should handle single words."""
        result = format_property_key("name")
        assert result == "Name"

    def test_format_property_key_already_capitalized(self):
        """Should handle already-capitalized keys."""
        result = format_property_key("FirstName")
        assert "First" in result

    def test_format_property_key_with_numbers(self):
        """Should handle keys with numbers."""
        result = format_property_key("test_key_123")
        assert "123" in result

    def test_format_property_key_multiple_underscores(self):
        """Should handle multiple underscores."""
        result = format_property_key("key__with__underscores")
        assert "Key" in result

    def test_format_property_key_empty_string(self):
        """Should handle empty strings."""
        result = format_property_key("")
        assert isinstance(result, str)

    def test_format_property_key_all_caps(self):
        """Should handle all caps."""
        result = format_property_key("ID")
        assert isinstance(result, str)


class TestTruncateString:
    """Test string truncation."""

    def test_truncate_string_no_truncation_needed(self):
        """Should return original string if within limit."""
        text = "short text"
        result = truncate_string(text, max_length=20)
        assert result == text

    def test_truncate_string_with_truncation(self):
        """Should truncate long strings."""
        text = "this is a long string"
        result = truncate_string(text, max_length=10)

        assert len(result) == 10
        assert result.endswith("...")

    def test_truncate_string_custom_suffix(self):
        """Should use custom suffix."""
        text = "this is a long string"
        result = truncate_string(text, max_length=10, suffix=">>")

        assert len(result) == 10
        assert result.endswith(">>")

    def test_truncate_string_exact_length_with_suffix(self):
        """Should handle when text exactly matches max_length - suffix."""
        text = "abc"
        result = truncate_string(text, max_length=6, suffix="...")
        # Text is shorter than max_length, so no truncation needed
        assert result == "abc"

    def test_truncate_string_requires_truncation(self):
        """Should truncate when text exceeds max_length - suffix."""
        text = "this_is_long"
        result = truncate_string(text, max_length=6, suffix="...")

        assert len(result) == 6
        assert result.endswith("...")

    def test_truncate_string_suffix_too_long_raises_error(self):
        """Should raise ValueError if suffix is too long."""
        with pytest.raises(ValueError):
            truncate_string("text", max_length=2, suffix="...")

    def test_truncate_string_empty_string(self):
        """Should handle empty strings."""
        result = truncate_string("", max_length=10)
        assert result == ""

    def test_truncate_string_longer_suffix(self):
        """Should work with longer custom suffix."""
        text = "this is a long string"
        result = truncate_string(text, max_length=12, suffix=">>")

        assert len(result) == 12
        assert result.endswith(">>")

    def test_truncate_string_single_char_suffix(self):
        """Should work with single character suffix."""
        text = "abcdefghij"
        result = truncate_string(text, max_length=5, suffix="~")

        assert len(result) == 5
        assert result.endswith("~")
