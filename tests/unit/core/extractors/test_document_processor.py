"""
Unit tests for DocumentProcessor.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest

from docling_graph.core.extractors.document_processor import DocumentProcessor


class TestDocumentProcessorInitialization:
    """Tests for DocumentProcessor initialization."""

    @patch("docling_graph.core.extractors.document_processor.DocumentConverter")
    def test_init_with_ocr_config(self, mock_converter):
        """Test initialization with OCR configuration."""
        processor = DocumentProcessor(docling_config="ocr")
        
        assert processor.docling_config == "ocr"
        mock_converter.assert_called_once()

    @patch("docling_graph.core.extractors.document_processor.DocumentConverter")
    def test_init_with_vision_config(self, mock_converter):
        """Test initialization with vision configuration."""
        processor = DocumentProcessor(docling_config="vision")
        
        assert processor.docling_config == "vision"
        mock_converter.assert_called_once()

    @patch("docling_graph.core.extractors.document_processor.DocumentConverter")
    def test_default_config_is_ocr(self, mock_converter):
        """Test that default configuration is OCR."""
        processor = DocumentProcessor()
        
        assert processor.docling_config == "ocr"


class TestDocumentProcessorConversion:
    """Tests for document conversion methods."""

    @patch("docling_graph.core.extractors.document_processor.DocumentConverter")
    def test_convert_to_markdown(self, mock_converter):
        """Test converting document to markdown."""
        mock_result = Mock()
        mock_document = Mock()
        mock_document.num_pages.return_value = 3
        mock_result.document = mock_document
        mock_converter_instance = mock_converter.return_value
        mock_converter_instance.convert.return_value = mock_result

        processor = DocumentProcessor()
        document = processor.convert_to_markdown("test.pdf")

        assert document == mock_document
        mock_converter_instance.convert.assert_called_once_with("test.pdf")

    @patch("docling_graph.core.extractors.document_processor.DocumentConverter")
    def test_extract_page_markdowns(self, mock_converter):
        """Test extracting markdown for each page."""
        mock_document = Mock()
        mock_document.pages = {0: Mock(), 1: Mock(), 2: Mock()}
        mock_document.export_to_markdown.side_effect = ["Page 1", "Page 2", "Page 3"]

        processor = DocumentProcessor()
        markdowns = processor.extract_page_markdowns(mock_document)

        assert len(markdowns) == 3
        assert markdowns[0] == "Page 1"
        assert markdowns[1] == "Page 2"
        assert markdowns[2] == "Page 3"
        assert mock_document.export_to_markdown.call_count == 3

    @patch("docling_graph.core.extractors.document_processor.DocumentConverter")
    def test_extract_full_markdown(self, mock_converter):
        """Test extracting full document markdown."""
        mock_document = Mock()
        mock_document.export_to_markdown.return_value = "Full document content"

        processor = DocumentProcessor()
        markdown = processor.extract_full_markdown(mock_document)

        assert markdown == "Full document content"
        mock_document.export_to_markdown.assert_called_once_with()

    @patch("docling_graph.core.extractors.document_processor.DocumentConverter")
    def test_process_document(self, mock_converter):
        """Test high-level process_document method."""
        mock_result = Mock()
        mock_document = Mock()
        mock_document.num_pages.return_value = 2
        mock_document.pages = {0: Mock(), 1: Mock()}
        mock_document.export_to_markdown.side_effect = ["Page 1", "Page 2"]
        mock_result.document = mock_document
        mock_converter_instance = mock_converter.return_value
        mock_converter_instance.convert.return_value = mock_result

        processor = DocumentProcessor()
        markdowns = processor.process_document("test.pdf")

        assert len(markdowns) == 2
        assert markdowns[0] == "Page 1"
        assert markdowns[1] == "Page 2"


class TestDocumentProcessorCleanup:
    """Tests for cleanup functionality."""

    @patch("docling_graph.core.extractors.document_processor.gc")
    @patch("docling_graph.core.extractors.document_processor.DocumentConverter")
    def test_cleanup_success(self, mock_converter, mock_gc):
        """Test successful cleanup."""
        processor = DocumentProcessor()
        processor.cleanup()

        mock_gc.collect.assert_called_once()

    @patch("docling_graph.core.extractors.document_processor.gc")
    @patch("docling_graph.core.extractors.document_processor.DocumentConverter")
    def test_cleanup_with_error(self, mock_converter, mock_gc):
        """Test cleanup handles errors gracefully."""
        mock_gc.collect.side_effect = RuntimeError("GC error")

        processor = DocumentProcessor()
        # Should not raise, just print warning
        processor.cleanup()


class TestDocumentProcessorPageHandling:
    """Tests for handling different page numbering schemes."""

    @patch("docling_graph.core.extractors.document_processor.DocumentConverter")
    def test_pages_starting_at_zero(self, mock_converter):
        """Test handling pages indexed from 0."""
        mock_document = Mock()
        mock_document.pages = {0: Mock(), 1: Mock()}
        mock_document.export_to_markdown.side_effect = ["Page 0", "Page 1"]

        processor = DocumentProcessor()
        markdowns = processor.extract_page_markdowns(mock_document)

        assert len(markdowns) == 2
        assert markdowns[0] == "Page 0"
        assert markdowns[1] == "Page 1"

    @patch("docling_graph.core.extractors.document_processor.DocumentConverter")
    def test_pages_starting_at_one(self, mock_converter):
        """Test handling pages indexed from 1."""
        mock_document = Mock()
        mock_document.pages = {1: Mock(), 2: Mock(), 3: Mock()}
        mock_document.export_to_markdown.side_effect = ["Page 1", "Page 2", "Page 3"]

        processor = DocumentProcessor()
        markdowns = processor.extract_page_markdowns(mock_document)

        assert len(markdowns) == 3

    @patch("docling_graph.core.extractors.document_processor.DocumentConverter")
    def test_empty_document(self, mock_converter):
        """Test handling document with no pages."""
        mock_document = Mock()
        mock_document.pages = {}

        processor = DocumentProcessor()
        markdowns = processor.extract_page_markdowns(mock_document)

        assert len(markdowns) == 0
