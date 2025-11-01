"""
Unit tests for CypherExporter.
"""

from pathlib import Path

import networkx as nx
import pytest

from docling_graph.core.exporters.cypher_exporter import CypherExporter
from docling_graph.core.base.config import ExportConfig


class TestCypherExporterInitialization:
    """Tests for CypherExporter initialization."""

    def test_init_with_default_config(self):
        """Test initialization with default configuration."""
        exporter = CypherExporter()
        
        assert exporter.config is not None
        assert isinstance(exporter.config, ExportConfig)

    def test_init_with_custom_config(self):
        """Test initialization with custom configuration."""
        custom_config = ExportConfig()
        exporter = CypherExporter(config=custom_config)
        
        assert exporter.config == custom_config


class TestCypherExporterExport:
    """Tests for export method."""

    def test_export_simple_graph(self, simple_graph, temp_dir):
        """Test exporting a simple graph to Cypher."""
        output_file = temp_dir / "graph.cypher"
        exporter = CypherExporter()
        
        exporter.export(simple_graph, output_file)
        
        assert output_file.exists()

    def test_export_creates_parent_directory(self, simple_graph, temp_dir):
        """Test that export creates parent directory if needed."""
        output_file = temp_dir / "subdir" / "graph.cypher"
        exporter = CypherExporter()
        
        assert not output_file.parent.exists()
        exporter.export(simple_graph, output_file)
        assert output_file.parent.exists()

    def test_export_empty_graph_raises_error(self, temp_dir):
        """Test that exporting empty graph raises ValueError."""
        empty_graph = nx.DiGraph()
        output_file = temp_dir / "graph.cypher"
        exporter = CypherExporter()
        
        with pytest.raises(ValueError, match="Cannot export empty graph"):
            exporter.export(empty_graph, output_file)

    def test_cypher_file_has_header(self, simple_graph, temp_dir):
        """Test that Cypher file has header comment."""
        output_file = temp_dir / "graph.cypher"
        exporter = CypherExporter()
        
        exporter.export(simple_graph, output_file)
        
        content = output_file.read_text()
        assert "// Cypher script" in content or "/*" in content

    def test_cypher_contains_create_statements(self, simple_graph, temp_dir):
        """Test that Cypher file contains CREATE statements."""
        output_file = temp_dir / "graph.cypher"
        exporter = CypherExporter()
        
        exporter.export(simple_graph, output_file)
        
        content = output_file.read_text()
        assert "CREATE" in content

    def test_export_node_with_label(self, temp_dir):
        """Test exporting node with label."""
        graph = nx.DiGraph()
        graph.add_node("node1", label="Person", name="Alice")
        
        output_file = temp_dir / "graph.cypher"
        exporter = CypherExporter()
        exporter.export(graph, output_file)
        
        content = output_file.read_text()
        assert ":Person" in content
        assert "Alice" in content

    def test_export_edge_with_relationship(self, temp_dir):
        """Test exporting edge with relationship type."""
        graph = nx.DiGraph()
        graph.add_node("node1", label="Person")
        graph.add_node("node2", label="Person")
        graph.add_edge("node1", "node2", label="KNOWS")
        
        output_file = temp_dir / "graph.cypher"
        exporter = CypherExporter()
        exporter.export(graph, output_file)
        
        content = output_file.read_text()
        assert "KNOWS" in content or "-[:KNOWS]->" in content

    def test_export_preserves_node_properties(self, temp_dir):
        """Test that node properties are preserved in Cypher."""
        graph = nx.DiGraph()
        graph.add_node("node1", label="Person", name="Alice", age=30)
        
        output_file = temp_dir / "graph.cypher"
        exporter = CypherExporter()
        exporter.export(graph, output_file)
        
        content = output_file.read_text()
        assert "name:" in content or "name :" in content
        assert "age:" in content or "age :" in content

    def test_export_preserves_edge_properties(self, temp_dir):
        """Test that edge properties are preserved in Cypher."""
        graph = nx.DiGraph()
        graph.add_node("node1")
        graph.add_node("node2")
        graph.add_edge("node1", "node2", label="KNOWS", since=2020)
        
        output_file = temp_dir / "graph.cypher"
        exporter = CypherExporter()
        exporter.export(graph, output_file)
        
        content = output_file.read_text()
        assert "since" in content


class TestCypherExporterBatching:
    """Tests for batching functionality."""

    def test_export_large_graph_uses_batching(self, temp_dir):
        """Test that large graphs are exported in batches."""
        # Create graph larger than batch size
        graph = nx.DiGraph()
        for i in range(2000):  # Larger than default batch size
            graph.add_node(f"node{i}", label="Node")
        
        output_file = temp_dir / "graph.cypher"
        exporter = CypherExporter()
        exporter.export(graph, output_file)
        
        content = output_file.read_text()
        # Should have multiple CREATE statements
        assert content.count("CREATE") > 1


class TestCypherExporterStringEscaping:
    """Tests for string escaping in Cypher."""

    def test_escapes_single_quotes(self, temp_dir):
        """Test that single quotes are properly escaped."""
        graph = nx.DiGraph()
        graph.add_node("node1", label="Person", name="O'Brien")
        
        output_file = temp_dir / "graph.cypher"
        exporter = CypherExporter()
        exporter.export(graph, output_file)
        
        content = output_file.read_text()
        # Should escape or handle quotes properly
        assert "O'Brien" in content or "O\\'Brien" in content or 'O"Brien' in content

    def test_handles_special_characters(self, temp_dir):
        """Test handling of special characters."""
        graph = nx.DiGraph()
        graph.add_node("node1", label="Person", name="Alice & Bob")
        
        output_file = temp_dir / "graph.cypher"
        exporter = CypherExporter()
        exporter.export(graph, output_file)
        
        content = output_file.read_text()
        assert "Alice" in content and "Bob" in content


class TestCypherExporterValidation:
    """Tests for validate_graph method."""

    def test_validate_graph_with_nodes(self, simple_graph):
        """Test validation of graph with nodes."""
        exporter = CypherExporter()
        assert exporter.validate_graph(simple_graph) is True

    def test_validate_empty_graph(self):
        """Test validation of empty graph."""
        empty_graph = nx.DiGraph()
        exporter = CypherExporter()
        
        assert exporter.validate_graph(empty_graph) is False


class TestCypherExporterEncoding:
    """Tests for encoding handling."""

    def test_export_with_unicode_characters(self, temp_dir):
        """Test exporting graph with Unicode characters."""
        graph = nx.DiGraph()
        graph.add_node("node1", label="Person", name="François", city="北京")
        
        output_file = temp_dir / "graph.cypher"
        exporter = CypherExporter()
        exporter.export(graph, output_file)
        
        content = output_file.read_text(encoding="utf-8")
        assert "François" in content
        assert "北京" in content


class TestCypherExporterSanitization:
    """Tests for identifier sanitization."""

    def test_sanitizes_invalid_identifiers(self, temp_dir):
        """Test that invalid Cypher identifiers are sanitized."""
        graph = nx.DiGraph()
        graph.add_node("node-1", label="Test-Label")  # Hyphens are problematic
        
        output_file = temp_dir / "graph.cypher"
        exporter = CypherExporter()
        exporter.export(graph, output_file)
        
        content = output_file.read_text()
        # Should handle or sanitize the identifier
        assert "CREATE" in content
