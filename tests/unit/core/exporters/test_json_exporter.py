"""
Unit tests for JSONExporter.
"""

import json
from pathlib import Path

import networkx as nx
import pytest

from docling_graph.core.exporters.json_exporter import JSONExporter
from docling_graph.core.base.config import ExportConfig


class TestJSONExporterInitialization:
    """Tests for JSONExporter initialization."""

    def test_init_with_default_config(self):
        """Test initialization with default configuration."""
        exporter = JSONExporter()
        
        assert exporter.config is not None
        assert isinstance(exporter.config, ExportConfig)

    def test_init_with_custom_config(self):
        """Test initialization with custom configuration."""
        custom_config = ExportConfig()
        exporter = JSONExporter(config=custom_config)
        
        assert exporter.config == custom_config


class TestJSONExporterExport:
    """Tests for export method."""

    def test_export_simple_graph(self, simple_graph, temp_dir):
        """Test exporting a simple graph to JSON."""
        output_file = temp_dir / "graph.json"
        exporter = JSONExporter()
        
        exporter.export(simple_graph, output_file)
        
        assert output_file.exists()

    def test_export_creates_parent_directory(self, simple_graph, temp_dir):
        """Test that export creates parent directory if needed."""
        output_file = temp_dir / "subdir" / "graph.json"
        exporter = JSONExporter()
        
        assert not output_file.parent.exists()
        exporter.export(simple_graph, output_file)
        assert output_file.parent.exists()

    def test_export_empty_graph_raises_error(self, temp_dir):
        """Test that exporting empty graph raises ValueError."""
        empty_graph = nx.DiGraph()
        output_file = temp_dir / "graph.json"
        exporter = JSONExporter()
        
        with pytest.raises(ValueError, match="Cannot export empty graph"):
            exporter.export(empty_graph, output_file)

    def test_exported_json_structure(self, simple_graph, temp_dir):
        """Test that exported JSON has correct structure."""
        output_file = temp_dir / "graph.json"
        exporter = JSONExporter()
        
        exporter.export(simple_graph, output_file)
        
        with open(output_file, encoding="utf-8") as f:
            data = json.load(f)
        
        assert "nodes" in data
        assert "edges" in data
        assert "metadata" in data
        assert isinstance(data["nodes"], list)
        assert isinstance(data["edges"], list)

    def test_exported_nodes_have_id(self, simple_graph, temp_dir):
        """Test that exported nodes have 'id' field."""
        output_file = temp_dir / "graph.json"
        exporter = JSONExporter()
        
        exporter.export(simple_graph, output_file)
        
        with open(output_file, encoding="utf-8") as f:
            data = json.load(f)
        
        for node in data["nodes"]:
            assert "id" in node

    def test_exported_edges_have_source_target(self, simple_graph, temp_dir):
        """Test that exported edges have 'source' and 'target' fields."""
        output_file = temp_dir / "graph.json"
        exporter = JSONExporter()
        
        exporter.export(simple_graph, output_file)
        
        with open(output_file, encoding="utf-8") as f:
            data = json.load(f)
        
        for edge in data["edges"]:
            assert "source" in edge
            assert "target" in edge

    def test_export_preserves_node_properties(self, temp_dir):
        """Test that node properties are preserved in export."""
        graph = nx.DiGraph()
        graph.add_node("node1", label="Person", name="Alice", age=30)
        
        output_file = temp_dir / "graph.json"
        exporter = JSONExporter()
        exporter.export(graph, output_file)
        
        with open(output_file, encoding="utf-8") as f:
            data = json.load(f)
        
        node = data["nodes"][0]
        assert node["id"] == "node1"
        assert node["label"] == "Person"
        assert node["name"] == "Alice"
        assert node["age"] == 30

    def test_export_preserves_edge_properties(self, temp_dir):
        """Test that edge properties are preserved in export."""
        graph = nx.DiGraph()
        graph.add_node("node1")
        graph.add_node("node2")
        graph.add_edge("node1", "node2", label="knows", weight=0.8)
        
        output_file = temp_dir / "graph.json"
        exporter = JSONExporter()
        exporter.export(graph, output_file)
        
        with open(output_file, encoding="utf-8") as f:
            data = json.load(f)
        
        edge = data["edges"][0]
        assert edge["source"] == "node1"
        assert edge["target"] == "node2"
        assert edge["label"] == "knows"
        assert edge["weight"] == 0.8


class TestJSONExporterMetadata:
    """Tests for metadata in exported JSON."""

    def test_metadata_has_counts(self, simple_graph, temp_dir):
        """Test that metadata includes node and edge counts."""
        output_file = temp_dir / "graph.json"
        exporter = JSONExporter()
        
        exporter.export(simple_graph, output_file)
        
        with open(output_file, encoding="utf-8") as f:
            data = json.load(f)
        
        assert "node_count" in data["metadata"]
        assert "edge_count" in data["metadata"]
        assert data["metadata"]["node_count"] == simple_graph.number_of_nodes()
        assert data["metadata"]["edge_count"] == simple_graph.number_of_edges()


class TestJSONExporterFormatting:
    """Tests for JSON formatting options."""

    def test_json_is_formatted_with_indent(self, simple_graph, temp_dir):
        """Test that exported JSON is formatted with indentation."""
        output_file = temp_dir / "graph.json"
        exporter = JSONExporter()
        
        exporter.export(simple_graph, output_file)
        
        content = output_file.read_text()
        # Check for indentation (newlines and spaces)
        assert "\n" in content
        assert "  " in content or "\t" in content

    def test_unicode_characters_preserved(self, temp_dir):
        """Test that Unicode characters are properly preserved."""
        graph = nx.DiGraph()
        graph.add_node("node1", name="François", city="北京")
        
        output_file = temp_dir / "graph.json"
        exporter = JSONExporter()
        exporter.export(graph, output_file)
        
        with open(output_file, encoding="utf-8") as f:
            data = json.load(f)
        
        node = data["nodes"][0]
        assert node["name"] == "François"
        assert node["city"] == "北京"


class TestJSONExporterValidation:
    """Tests for validate_graph method."""

    def test_validate_graph_with_nodes(self, simple_graph):
        """Test validation of graph with nodes."""
        exporter = JSONExporter()
        assert exporter.validate_graph(simple_graph) is True

    def test_validate_empty_graph(self):
        """Test validation of empty graph."""
        empty_graph = nx.DiGraph()
        exporter = JSONExporter()
        
        assert exporter.validate_graph(empty_graph) is False
