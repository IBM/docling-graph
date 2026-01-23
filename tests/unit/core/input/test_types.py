"""Unit tests for input type detection."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from docling_graph.core.input.types import InputType, InputTypeDetector
from docling_graph.exceptions import ConfigurationError


class TestInputType:
    """Test InputType enum."""

    def test_input_type_values(self):
        """Test that all expected input types are defined."""
        expected_types = {
            "PDF",
            "IMAGE",
            "TEXT",
            "TEXT_FILE",
            "MARKDOWN",
            "URL",
            "DOCLING_DOCUMENT",
        }
        actual_types = {t.name for t in InputType}
        assert actual_types == expected_types


class TestInputTypeDetector:
    """Test InputTypeDetector class."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    # ==================== PDF Detection ====================

    def test_detect_pdf_from_extension(self, temp_dir):
        """Test PDF detection from file extension."""
        pdf_file = temp_dir / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake pdf")

        result = InputTypeDetector.detect(str(pdf_file), mode="api")
        assert result == InputType.PDF

    def test_detect_pdf_case_insensitive(self, temp_dir):
        """Test PDF detection is case-insensitive."""
        pdf_file = temp_dir / "test.PDF"
        pdf_file.write_bytes(b"fake pdf")

        result = InputTypeDetector.detect(str(pdf_file), mode="api")
        assert result == InputType.PDF

    # ==================== Image Detection ====================

    def test_detect_png_image(self, temp_dir):
        """Test PNG image detection."""
        img_file = temp_dir / "test.png"
        img_file.write_bytes(b"fake png")

        result = InputTypeDetector.detect(str(img_file), mode="api")
        assert result == InputType.IMAGE

    def test_detect_jpg_image(self, temp_dir):
        """Test JPG image detection."""
        img_file = temp_dir / "test.jpg"
        img_file.write_bytes(b"fake jpg")

        result = InputTypeDetector.detect(str(img_file), mode="api")
        assert result == InputType.IMAGE

    def test_detect_jpeg_image(self, temp_dir):
        """Test JPEG image detection."""
        img_file = temp_dir / "test.jpeg"
        img_file.write_bytes(b"fake jpeg")

        result = InputTypeDetector.detect(str(img_file), mode="api")
        assert result == InputType.IMAGE

    # ==================== Text File Detection ====================

    def test_detect_txt_file(self, temp_dir):
        """Test .txt file detection."""
        txt_file = temp_dir / "test.txt"
        txt_file.write_text("sample text")

        result = InputTypeDetector.detect(str(txt_file), mode="api")
        assert result == InputType.TEXT_FILE

    # ==================== Markdown Detection ====================

    def test_detect_markdown_file(self, temp_dir):
        """Test .md file detection."""
        md_file = temp_dir / "test.md"
        md_file.write_text("# Markdown")

        result = InputTypeDetector.detect(str(md_file), mode="api")
        assert result == InputType.MARKDOWN

    # ==================== URL Detection ====================

    def test_detect_https_url(self):
        """Test HTTPS URL detection."""
        url = "https://example.com/document.pdf"
        result = InputTypeDetector.detect(url, mode="api")
        assert result == InputType.URL

    def test_detect_http_url(self):
        """Test HTTP URL detection."""
        url = "http://example.com/file.txt"
        result = InputTypeDetector.detect(url, mode="api")
        assert result == InputType.URL

    def test_detect_url_with_query_params(self):
        """Test URL detection with query parameters."""
        url = "https://example.com/doc.pdf?version=1&format=pdf"
        result = InputTypeDetector.detect(url, mode="api")
        assert result == InputType.URL

    # ==================== DoclingDocument Detection ====================

    def test_detect_docling_document_from_content(self, temp_dir):
        """Test DoclingDocument detection from JSON content."""
        import json

        doc_data = {"schema_name": "DoclingDocument", "version": "1.0.0", "name": "test"}
        doc_file = temp_dir / "doc.json"
        doc_file.write_text(json.dumps(doc_data))

        result = InputTypeDetector.detect(str(doc_file), mode="api")
        assert result == InputType.DOCLING_DOCUMENT

    def test_detect_regular_json_not_docling(self, temp_dir):
        """Test that regular JSON is not detected as DoclingDocument."""
        import json

        regular_json = {"data": "value", "other": "field"}
        json_file = temp_dir / "regular.json"
        json_file.write_text(json.dumps(regular_json))

        result = InputTypeDetector.detect(str(json_file), mode="api")
        # Should fall back to TEXT_FILE since .json extension
        # is not in the standard list
        assert result == InputType.TEXT

    # ==================== Plain Text Detection ====================

    def test_detect_plain_text_api_mode(self):
        """Test plain text detection in API mode."""
        text = "This is plain text content"
        result = InputTypeDetector.detect(text, mode="api")
        assert result == InputType.TEXT

    def test_detect_plain_text_multiline_api_mode(self):
        """Test multiline plain text detection in API mode."""
        text = "Line 1\nLine 2\nLine 3"
        result = InputTypeDetector.detect(text, mode="api")
        assert result == InputType.TEXT

    # ==================== CLI Mode Restrictions ====================

    def test_cli_mode_rejects_plain_text(self):
        """Test that CLI mode rejects plain text input."""
        text = "Plain text content"
        with pytest.raises(
            ConfigurationError, match="Plain text input is only supported via Python API"
        ):
            InputTypeDetector.detect(text, mode="cli")

    def test_cli_mode_requires_file_existence(self, temp_dir):
        """Test that CLI mode requires files to exist."""
        nonexistent = temp_dir / "nonexistent.txt"

        with pytest.raises(ConfigurationError, match="not found"):
            InputTypeDetector.detect(str(nonexistent), mode="cli")

    def test_cli_mode_accepts_existing_files(self, temp_dir):
        """Test that CLI mode accepts existing files."""
        txt_file = temp_dir / "test.txt"
        txt_file.write_text("content")

        result = InputTypeDetector.detect(str(txt_file), mode="cli")
        assert result == InputType.TEXT_FILE

    def test_cli_mode_accepts_urls(self):
        """Test that CLI mode accepts URLs."""
        url = "https://example.com/doc.pdf"
        result = InputTypeDetector.detect(url, mode="cli")
        assert result == InputType.URL

    # ==================== Edge Cases ====================

    def test_detect_empty_string_api_mode(self):
        """Test detection of empty string in API mode."""
        # Empty string should be detected as TEXT but will fail validation later
        result = InputTypeDetector.detect("", mode="api")
        assert result == InputType.TEXT

    def test_detect_whitespace_only_api_mode(self):
        """Test detection of whitespace-only string in API mode."""
        result = InputTypeDetector.detect("   \n\t  ", mode="api")
        assert result == InputType.TEXT

    def test_detect_path_like_string_api_mode(self):
        """Test that non-existent path-like strings are treated as text in API mode."""
        # In API mode, if it looks like a path but doesn't exist, treat as text
        fake_path = "/nonexistent/path/to/file.txt"
        result = InputTypeDetector.detect(fake_path, mode="api")
        # Should be TEXT since file doesn't exist
        assert result == InputType.TEXT

    def test_detect_with_pathlib_path(self, temp_dir):
        """Test detection with pathlib.Path object."""
        txt_file = temp_dir / "test.txt"
        txt_file.write_text("content")

        # Pass Path object instead of string
        result = InputTypeDetector.detect(txt_file, mode="api")
        assert result == InputType.TEXT_FILE

    # ==================== Mode Parameter ====================

    def test_default_mode_is_api(self):
        """Test that default mode is 'api'."""
        text = "Plain text"
        # Should not raise since default is API mode
        result = InputTypeDetector.detect(text)
        assert result == InputType.TEXT

    def test_invalid_mode_raises_error(self):
        """Test that invalid mode raises error."""
        with pytest.raises(ValueError, match="mode must be"):
            InputTypeDetector.detect("test", mode="invalid")


class TestInputTypeDetectorHelpers:
    """Test helper methods of InputTypeDetector."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_is_url_helper(self):
        """Test _is_url helper method."""
        assert InputTypeDetector._is_url("https://example.com")
        assert InputTypeDetector._is_url("http://example.com")
        assert not InputTypeDetector._is_url("/local/path")
        assert not InputTypeDetector._is_url("plain text")

    def test_detect_from_file_helper(self, temp_dir):
        """Test _detect_from_file helper method."""
        # PDF
        pdf_file = temp_dir / "test.pdf"
        pdf_file.write_bytes(b"fake")
        assert InputTypeDetector._detect_from_file(pdf_file) == InputType.PDF

        # Image
        img_file = temp_dir / "test.png"
        img_file.write_bytes(b"fake")
        assert InputTypeDetector._detect_from_file(img_file) == InputType.IMAGE

        # Text
        txt_file = temp_dir / "test.txt"
        txt_file.write_text("text")
        assert InputTypeDetector._detect_from_file(txt_file) == InputType.TEXT_FILE

        # Markdown
        md_file = temp_dir / "test.md"
        md_file.write_text("# MD")
        assert InputTypeDetector._detect_from_file(md_file) == InputType.MARKDOWN

    def test_is_docling_document_helper(self, temp_dir):
        """Test _is_docling_document helper method."""
        import json

        # Valid DoclingDocument
        valid_doc = {"schema_name": "DoclingDocument", "version": "1.0.0"}
        valid_file = temp_dir / "valid.json"
        valid_file.write_text(json.dumps(valid_doc))
        assert InputTypeDetector._is_docling_document(valid_file) is True

        # Invalid JSON
        invalid_file = temp_dir / "invalid.json"
        invalid_file.write_text("not json {")
        assert InputTypeDetector._is_docling_document(invalid_file) is False

        # Regular JSON
        regular_file = temp_dir / "regular.json"
        regular_file.write_text(json.dumps({"data": "value"}))
        assert InputTypeDetector._is_docling_document(regular_file) is False

        # Non-JSON file
        txt_file = temp_dir / "test.txt"
        txt_file.write_text("text")
        assert InputTypeDetector._is_docling_document(txt_file) is False
