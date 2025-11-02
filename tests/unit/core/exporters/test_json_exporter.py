"""
Tests for JSON exporter.
"""

import json
from pathlib import Path

import networkx as nx
import pytest

from docling_graph.core.base.config import ExportConfig
from docling_graph.core.exporters.json_exporter import JSONExporter


@pytest.fixture
def sample_graph():
    """Create a sample graph."""
    graph = nx.DiGraph()
    graph.add_node("n1", label="Person", name="John")
    graph.add_node("n2", label="Company", name="ACME")
    graph.add_edge("n1", "n2", label="works_for", strength=0.9)
    return graph


@pytest.fixture
def empty_graph():
    """Create an empty graph."""
    return nx.DiGraph()


class TestJSONExporterInitialization:
    """Test JSONExporter initialization."""

    def test_initialization_default(self):
        """Should initialize with default config."""
        exporter = JSONExporter()
        assert exporter.config is not None

    def test_initialization_custom_config(self):
        """Should accept custom config."""
        config = ExportConfig()
        exporter = JSONExporter(config=config)
        assert exporter.config is config


class TestJSONExporterValidation:
    """Test graph validation."""

    def test_validate_graph_with_nodes(self, sample_graph):
        """Should return True for non-empty graph."""
        exporter = JSONExporter()
        assert exporter.validate_graph(sample_graph) is True

    def test_validate_graph_empty(self, empty_graph):
        """Should return False for empty graph."""
        exporter = JSONExporter()
        assert exporter.validate_graph(empty_graph) is False


class TestJSONExporterGraphToDict:
    """Test graph to dictionary conversion."""

    def test_graph_to_dict_structure(self, sample_graph):
        """Should convert graph to dict with correct structure."""
        result = JSONExporter._graph_to_dict(sample_graph)

        assert "nodes" in result
        assert "edges" in result
        assert "metadata" in result

    def test_graph_to_dict_nodes_list(self, sample_graph):
        """Nodes should be list."""
        result = JSONExporter._graph_to_dict(sample_graph)

        assert isinstance(result["nodes"], list)
        assert len(result["nodes"]) == 2

    def test_graph_to_dict_edges_list(self, sample_graph):
        """Edges should be list."""
        result = JSONExporter._graph_to_dict(sample_graph)

        assert isinstance(result["edges"], list)
        assert len(result["edges"]) == 1

    def test_graph_to_dict_node_attributes(self, sample_graph):
        """Nodes should include attributes."""
        result = JSONExporter._graph_to_dict(sample_graph)

        node = result["nodes"][0]
        assert "id" in node
        assert "label" in node

    def test_graph_to_dict_edge_attributes(self, sample_graph):
        """Edges should include attributes."""
        result = JSONExporter._graph_to_dict(sample_graph)

        edge = result["edges"][0]
        assert "source" in edge
        assert "target" in edge
        assert "label" in edge

    def test_graph_to_dict_metadata(self, sample_graph):
        """Metadata should contain node and edge counts."""
        result = JSONExporter._graph_to_dict(sample_graph)

        assert result["metadata"]["node_count"] == 2
        assert result["metadata"]["edge_count"] == 1


class TestJSONExporterExport:
    """Test JSON export functionality."""

    def test_export_creates_file(self, sample_graph, tmp_path):
        """Should create JSON file."""
        exporter = JSONExporter()
        output_file = tmp_path / "graph.json"

        exporter.export(sample_graph, output_file)

        assert output_file.exists()
        assert output_file.suffix == ".json"

    def test_export_empty_graph_raises_error(self, empty_graph, tmp_path):
        """Should raise error for empty graph."""
        exporter = JSONExporter()

        with pytest.raises(ValueError):
            exporter.export(empty_graph, tmp_path / "output.json")

    def test_export_creates_parent_directories(self, sample_graph, tmp_path):
        """Should create parent directories if needed."""
        exporter = JSONExporter()
        output_file = tmp_path / "nested" / "deep" / "graph.json"

        exporter.export(sample_graph, output_file)

        assert output_file.exists()

    def test_export_creates_valid_json(self, sample_graph, tmp_path):
        """Exported file should be valid JSON."""
        exporter = JSONExporter()
        output_file = tmp_path / "graph.json"

        exporter.export(sample_graph, output_file)

        with open(output_file) as f:
            data = json.load(f)

        assert "nodes" in data
        assert "edges" in data

    def test_export_preserves_node_data(self, sample_graph, tmp_path):
        """Export should preserve node attributes."""
        exporter = JSONExporter()
        output_file = tmp_path / "graph.json"

        exporter.export(sample_graph, output_file)

        with open(output_file) as f:
            data = json.load(f)

        nodes = data["nodes"]
        assert len(nodes) == 2
        assert any(n["name"] == "John" for n in nodes)

    def test_export_preserves_edge_data(self, sample_graph, tmp_path):
        """Export should preserve edge attributes."""
        exporter = JSONExporter()
        output_file = tmp_path / "graph.json"

        exporter.export(sample_graph, output_file)

        with open(output_file) as f:
            data = json.load(f)

        edges = data["edges"]
        assert len(edges) == 1
        assert edges[0]["strength"] == 0.9

    def test_export_uses_configured_encoding(self, sample_graph, tmp_path):
        """Should use configured encoding."""
        config = ExportConfig()
        exporter = JSONExporter(config=config)
        output_file = tmp_path / "graph.json"

        exporter.export(sample_graph, output_file)

        # Verify encoding by reading file
        with open(output_file, encoding=config.JSON_ENCODING) as f:
            data = json.load(f)
        assert data is not None

    def test_export_uses_configured_indent(self, sample_graph, tmp_path):
        """Should use configured indentation."""
        config = ExportConfig()
        exporter = JSONExporter(config=config)
        output_file = tmp_path / "graph.json"

        exporter.export(sample_graph, output_file)

        content = output_file.read_text()
        # Indented JSON should have newlines and spaces
        assert "\n" in content
        assert "  " in content
