"""
Unit tests for ReportGenerator.
"""

from pathlib import Path

import networkx as nx
import pytest

from docling_graph.core.visualizers.report_generator import ReportGenerator


class TestReportGeneratorInitialization:
    """Tests for ReportGenerator initialization."""

    def test_init_creates_instance(self):
        """Test that initializer creates instance."""
        generator = ReportGenerator()
        
        assert generator is not None


class TestReportGeneratorVisualize:
    """Tests for visualize method."""

    def test_visualize_creates_markdown_file(self, simple_graph, temp_dir):
        """Test that visualize creates markdown file."""
        output_file = temp_dir / "report"
        generator = ReportGenerator()
        
        generator.visualize(simple_graph, output_file)
        
        # Should add .md extension
        assert (temp_dir / "report.md").exists()

    def test_visualize_with_md_extension(self, simple_graph, temp_dir):
        """Test visualize with .md extension already present."""
        output_file = temp_dir / "report.md"
        generator = ReportGenerator()
        
        generator.visualize(simple_graph, output_file)
        
        assert output_file.exists()

    def test_visualize_empty_graph_raises_error(self, temp_dir):
        """Test that visualizing empty graph raises ValueError."""
        empty_graph = nx.DiGraph()
        output_file = temp_dir / "report.md"
        generator = ReportGenerator()
        
        with pytest.raises(ValueError, match="Cannot generate report for empty graph"):
            generator.visualize(empty_graph, output_file)

    def test_report_contains_markdown_headers(self, simple_graph, temp_dir):
        """Test that report contains markdown headers."""
        output_file = temp_dir / "report.md"
        generator = ReportGenerator()
        
        generator.visualize(simple_graph, output_file)
        
        content = output_file.read_text()
        assert "#" in content  # Markdown headers

    def test_report_contains_statistics(self, simple_graph, temp_dir):
        """Test that report contains graph statistics."""
        output_file = temp_dir / "report.md"
        generator = ReportGenerator()
        
        generator.visualize(simple_graph, output_file)
        
        content = output_file.read_text()
        assert "node" in content.lower()
        assert "edge" in content.lower()

    def test_report_includes_samples_by_default(self, simple_graph, temp_dir):
        """Test that report includes sample nodes/edges by default."""
        output_file = temp_dir / "report.md"
        generator = ReportGenerator()
        
        generator.visualize(simple_graph, output_file, include_samples=True)
        
        content = output_file.read_text()
        assert len(content) > 100  # Should have substantial content

    def test_report_without_samples(self, simple_graph, temp_dir):
        """Test report generation without samples."""
        output_file = temp_dir / "report.md"
        generator = ReportGenerator()
        
        generator.visualize(simple_graph, output_file, include_samples=False)
        
        content = output_file.read_text()
        # Should still have content but less detailed
        assert "#" in content


class TestReportGeneratorStatistics:
    """Tests for statistics in reports."""

    def test_report_shows_node_count(self, simple_graph, temp_dir):
        """Test that report shows node count."""
        output_file = temp_dir / "report.md"
        generator = ReportGenerator()
        
        generator.visualize(simple_graph, output_file)
        
        content = output_file.read_text()
        node_count_str = str(simple_graph.number_of_nodes())
        assert node_count_str in content

    def test_report_shows_edge_count(self, simple_graph, temp_dir):
        """Test that report shows edge count."""
        output_file = temp_dir / "report.md"
        generator = ReportGenerator()
        
        generator.visualize(simple_graph, output_file)
        
        content = output_file.read_text()
        edge_count_str = str(simple_graph.number_of_edges())
        assert edge_count_str in content

    def test_report_shows_source_model_count(self, simple_graph, temp_dir):
        """Test that report shows source model count."""
        output_file = temp_dir / "report.md"
        generator = ReportGenerator()
        
        generator.visualize(simple_graph, output_file, source_model_count=5)
        
        content = output_file.read_text()
        assert "5" in content


class TestReportGeneratorValidation:
    """Tests for validate_graph method."""

    def test_validate_graph_with_nodes(self, simple_graph):
        """Test validation of graph with nodes."""
        generator = ReportGenerator()
        assert generator.validate_graph(simple_graph) is True

    def test_validate_empty_graph(self):
        """Test validation of empty graph."""
        empty_graph = nx.DiGraph()
        generator = ReportGenerator()
        
        assert generator.validate_graph(empty_graph) is False


class TestReportGeneratorFormatting:
    """Tests for report formatting."""

    def test_report_is_valid_markdown(self, simple_graph, temp_dir):
        """Test that generated report is valid markdown."""
        output_file = temp_dir / "report.md"
        generator = ReportGenerator()
        
        generator.visualize(simple_graph, output_file)
        
        content = output_file.read_text()
        # Check for markdown elements
        assert "#" in content  # Headers
        lines = content.split("\n")
        assert len(lines) > 5  # Multiple lines

    def test_report_readable_format(self, simple_graph, temp_dir):
        """Test that report is in readable format."""
        output_file = temp_dir / "report.md"
        generator = ReportGenerator()
        
        generator.visualize(simple_graph, output_file)
        
        content = output_file.read_text()
        # Should not be empty and should be readable
        assert len(content) > 50
        assert content.isprintable() or "\n" in content
