"""
Unit tests for string formatting utilities.
"""

import pytest

from docling_graph.core.utils.formatting import (
    format_property_key,
    format_property_value,
    truncate_string,
)


class TestFormatPropertyKey:
    """Tests for format_property_key function."""

    def test_snake_case_conversion(self):
        """Test conversion of snake_case to Title Case."""
        result = format_property_key("property_name")
        assert result == "Property Name"

    def test_snake_case_multiple_words(self):
        """Test snake_case with multiple words."""
        result = format_property_key("first_last_name")
        assert result == "First Last Name"

    def test_camel_case_conversion(self):
        """Test conversion of camelCase to Title Case."""
        result = format_property_key("propertyName")
        assert result == "Property Name"

    def test_camel_case_multiple_words(self):
        """Test camelCase with multiple words."""
        result = format_property_key("firstLastName")
        assert result == "First Last Name"

    def test_single_word_unchanged(self):
        """Test single word is capitalized."""
        result = format_property_key("name")
        assert result == "Name"

    def test_all_caps_abbreviation(self):
        """Test all caps abbreviation."""
        result = format_property_key("ID")
        assert result == "I D" or result == "ID"

    def test_mixed_separators(self):
        """Test string with underscores."""
        result = format_property_key("user_id_number")
        assert result == "User Id Number"

    def test_empty_string(self):
        """Test empty string."""
        result = format_property_key("")
        assert result == ""

    def test_numbers_in_key(self):
        """Test key with numbers."""
        result = format_property_key("property_1_name")
        assert "1" in result


class TestFormatPropertyValue:
    """Tests for format_property_value function."""

    def test_format_string(self):
        """Test formatting simple string."""
        result = format_property_value("hello")
        assert result == "hello"

    def test_format_number(self):
        """Test formatting number."""
        result = format_property_value(42)
        assert result == "42"

    def test_format_list(self):
        """Test formatting list."""
        result = format_property_value(["a", "b", "c"])
        assert "[" in result and "]" in result

    def test_format_dict(self):
        """Test formatting dictionary."""
        result = format_property_value({"key": "value"})
        assert isinstance(result, str)
        assert len(result) > 0

    def test_truncate_long_string(self):
        """Test that long strings are truncated."""
        long_string = "a" * 100
        result = format_property_value(long_string, max_length=20)
        
        assert len(result) <= 20
        assert result.endswith("...")

    def test_max_length_parameter(self):
        """Test custom max_length parameter."""
        long_string = "a" * 50
        result = format_property_value(long_string, max_length=10)
        
        assert len(result) <= 10

    def test_format_none(self):
        """Test formatting None value."""
        result = format_property_value(None)
        assert result == "None"

    def test_format_boolean(self):
        """Test formatting boolean."""
        assert format_property_value(True) == "True"
        assert format_property_value(False) == "False"

    def test_format_float(self):
        """Test formatting float."""
        result = format_property_value(3.14159)
        assert "3.14" in result


class TestTruncateString:
    """Tests for truncate_string function."""

    def test_truncate_long_string(self):
        """Test truncating long string."""
        text = "Hello World This Is A Long String"
        result = truncate_string(text, max_length=15)
        
        assert len(result) <= 15
        assert result.endswith("...")

    def test_short_string_unchanged(self):
        """Test short string is not truncated."""
        text = "Hello"
        result = truncate_string(text, max_length=20)
        
        assert result == "Hello"

    def test_exact_length_string(self):
        """Test string with exact max_length."""
        text = "Hello"  # 5 chars
        result = truncate_string(text, max_length=5)
        
        assert result == "Hello"

    def test_custom_suffix(self):
        """Test custom suffix parameter."""
        text = "Hello World This Is Long"
        result = truncate_string(text, max_length=15, suffix=">>")
        
        assert result.endswith(">>")
        assert len(result) <= 15

    def test_default_suffix(self):
        """Test default suffix is used."""
        text = "a" * 100
        result = truncate_string(text, max_length=20)
        
        assert result.endswith("...")

    def test_max_length_less_than_suffix_raises_error(self):
        """Test error when max_length is less than suffix."""
        with pytest.raises(ValueError):
            truncate_string("hello", max_length=2, suffix="...")

    def test_max_length_equals_suffix_raises_error(self):
        """Test error when max_length equals suffix length."""
        with pytest.raises(ValueError):
            truncate_string("hello", max_length=3, suffix="...")

    def test_truncate_unicode_string(self):
        """Test truncating unicode string."""
        text = "François México 北京 " * 5
        result = truncate_string(text, max_length=30)
        
        assert len(result) <= 30
        assert result.endswith("...")

    def test_truncate_empty_string(self):
        """Test truncating empty string."""
        result = truncate_string("", max_length=10)
        assert result == ""

    def test_truncate_with_newlines(self):
        """Test truncating string with newlines."""
        text = "Line 1\nLine 2\nLine 3\nLine 4"
        result = truncate_string(text, max_length=15)
        
        assert len(result) <= 15
        assert result.endswith("...")
