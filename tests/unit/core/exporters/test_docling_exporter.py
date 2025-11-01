"""
Unit tests for DoclingExporter.
"""

from pathlib import Path
from unittest.mock import Mock, MagicMock

import pytest

from docling_graph.core.exporters.docling_exporter import DoclingExporter


class TestDoclingExporterInitialization:
    """Tests for DoclingExporter initialization."""

    def test_init_with_default_output_dir(self):
        """Test initialization with default output directory."""
        exporter = DoclingExporter()
        
        assert exporter.output_dir == Path("outputs")

    def test_init_with_custom_output_dir(self, temp_dir):
        """Test initialization with custom output directory."""
        exporter = DoclingExporter(output_dir=temp_dir)
        
        assert exporter.output_dir == temp_dir


class TestDoclingExporterExportDocument:
    """Tests for export_document method."""

    def test_export_document_creates_directory(self, temp_dir):
        """Test that export creates output directory."""
        exporter = DoclingExporter(output_dir=temp_dir / "new_dir")
        mock_document = Mock()
        mock_document.export_to_dict.return_value = {"content": "test"}
        mock_document.export_to_markdown.return_value = "# Test"
        
        assert not (temp_dir / "new_dir").exists()
        exporter.export_document(
            document=mock_document,
            base_name="test",
            include_json=True,
            include_markdown=False
        )
        assert (temp_dir / "new_dir").exists()

    def test_export_json_only(self, temp_dir):
        """Test exporting only JSON."""
        exporter = DoclingExporter(output_dir=temp_dir)
        mock_document = Mock()
        mock_document.export_to_dict.return_value = {"content": "test"}
        
        result = exporter.export_document(
            document=mock_document,
            base_name="test",
            include_json=True,
            include_markdown=False
        )
        
        assert (temp_dir / "test.json").exists()
        assert "json_path" in result
        assert "markdown_path" not in result

    def test_export_markdown_only(self, temp_dir):
        """Test exporting only markdown."""
        exporter = DoclingExporter(output_dir=temp_dir)
        mock_document = Mock()
        mock_document.export_to_markdown.return_value = "# Test Document"
        
        result = exporter.export_document(
            document=mock_document,
            base_name="test",
            include_json=False,
            include_markdown=True
        )
        
        assert (temp_dir / "test.md").exists()
        assert "markdown_path" in result
        assert "json_path" not in result

    def test_export_both_formats(self, temp_dir):
        """Test exporting both JSON and markdown."""
        exporter = DoclingExporter(output_dir=temp_dir)
        mock_document = Mock()
        mock_document.export_to_dict.return_value = {"content": "test"}
        mock_document.export_to_markdown.return_value = "# Test"
        
        result = exporter.export_document(
            document=mock_document,
            base_name="test",
            include_json=True,
            include_markdown=True
        )
        
        assert (temp_dir / "test.json").exists()
        assert (temp_dir / "test.md").exists()
        assert "json_path" in result
        assert "markdown_path" in result

    def test_export_per_page_markdown(self, temp_dir):
        """Test exporting per-page markdown."""
        exporter = DoclingExporter(output_dir=temp_dir)
        mock_document = Mock()
        mock_document.pages = {0: Mock(), 1: Mock()}
        mock_document.export_to_markdown.side_effect = ["# Page 1", "# Page 2"]
        
        result = exporter.export_document(
            document=mock_document,
            base_name="test",
            include_json=False,
            include_markdown=True,
            per_page=True
        )
        
        assert (temp_dir / "test_page_0.md").exists()
        assert (temp_dir / "test_page_1.md").exists()
        assert "per_page_markdown_paths" in result
        assert len(result["per_page_markdown_paths"]) == 2

    def test_export_no_formats_raises_error(self, temp_dir):
        """Test that requesting no exports raises error."""
        exporter = DoclingExporter(output_dir=temp_dir)
        mock_document = Mock()
        
        with pytest.raises(ValueError, match="At least one format"):
            exporter.export_document(
                document=mock_document,
                base_name="test",
                include_json=False,
                include_markdown=False
            )


class TestDoclingExporterFileNaming:
    """Tests for file naming."""

    def test_json_filename_format(self, temp_dir):
        """Test JSON filename format."""
        exporter = DoclingExporter(output_dir=temp_dir)
        mock_document = Mock()
        mock_document.export_to_dict.return_value = {}
        
        exporter.export_document(
            document=mock_document,
            base_name="my_document",
            include_json=True,
            include_markdown=False
        )
        
        assert (temp_dir / "my_document.json").exists()

    def test_markdown_filename_format(self, temp_dir):
        """Test markdown filename format."""
        exporter = DoclingExporter(output_dir=temp_dir)
        mock_document = Mock()
        mock_document.export_to_markdown.return_value = "# Test"
        
        exporter.export_document(
            document=mock_document,
            base_name="my_document",
            include_json=False,
            include_markdown=True
        )
        
        assert (temp_dir / "my_document.md").exists()

    def test_per_page_filename_format(self, temp_dir):
        """Test per-page markdown filename format."""
        exporter = DoclingExporter(output_dir=temp_dir)
        mock_document = Mock()
        mock_document.pages = {0: Mock(), 1: Mock()}
        mock_document.export_to_markdown.side_effect = ["Page 1", "Page 2"]
        
        exporter.export_document(
            document=mock_document,
            base_name="doc",
            include_json=False,
            include_markdown=True,
            per_page=True
        )
        
        assert (temp_dir / "doc_page_0.md").exists()
        assert (temp_dir / "doc_page_1.md").exists()


class TestDoclingExporterErrorHandling:
    """Tests for error handling."""

    def test_handles_document_export_error(self, temp_dir):
        """Test handling of document export errors."""
        exporter = DoclingExporter(output_dir=temp_dir)
        mock_document = Mock()
        mock_document.export_to_dict.side_effect = RuntimeError("Export failed")
        
        with pytest.raises(RuntimeError, match="Export failed"):
            exporter.export_document(
                document=mock_document,
                base_name="test",
                include_json=True,
                include_markdown=False
            )

    def test_handles_markdown_export_error(self, temp_dir):
        """Test handling of markdown export errors."""
        exporter = DoclingExporter(output_dir=temp_dir)
        mock_document = Mock()
        mock_document.export_to_markdown.side_effect = RuntimeError("Markdown export failed")
        
        with pytest.raises(RuntimeError):
            exporter.export_document(
                document=mock_document,
                base_name="test",
                include_json=False,
                include_markdown=True
            )


class TestDoclingExporterReturnValues:
    """Tests for return value structure."""

    def test_return_value_has_paths(self, temp_dir):
        """Test that return value contains file paths."""
        exporter = DoclingExporter(output_dir=temp_dir)
        mock_document = Mock()
        mock_document.export_to_dict.return_value = {}
        mock_document.export_to_markdown.return_value = "# Test"
        
        result = exporter.export_document(
            document=mock_document,
            base_name="test",
            include_json=True,
            include_markdown=True
        )
        
        assert isinstance(result, dict)
        assert isinstance(result["json_path"], str)
        assert isinstance(result["markdown_path"], str)

    def test_return_value_per_page_is_list(self, temp_dir):
        """Test that per-page paths are returned as list."""
        exporter = DoclingExporter(output_dir=temp_dir)
        mock_document = Mock()
        mock_document.pages = {0: Mock()}
        mock_document.export_to_markdown.return_value = "# Page"
        
        result = exporter.export_document(
            document=mock_document,
            base_name="test",
            include_json=False,
            include_markdown=True,
            per_page=True
        )
        
        assert isinstance(result["per_page_markdown_paths"], list)
