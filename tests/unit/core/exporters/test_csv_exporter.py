"""
Unit tests for CSVExporter.
"""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import networkx as nx
import pandas as pd
import pytest

from docling_graph.core.exporters.csv_exporter import CSVExporter
from docling_graph.core.base.config import ExportConfig


class TestCSVExporterInitialization:
    """Tests for CSVExporter initialization."""

    def test_init_with_default_config(self):
        """Test initialization with default configuration."""
        exporter = CSVExporter()
        
        assert exporter.config is not None
        assert isinstance(exporter.config, ExportConfig)
        assert exporter.config.CSV_ENCODING == "utf-8"

    def test_init_with_custom_config(self):
        """Test initialization with custom configuration."""
        custom_config = ExportConfig()
        exporter = CSVExporter(config=custom_config)
        
        assert exporter.config == custom_config


class TestCSVExporterExport:
    """Tests for export method."""

    def test_export_simple_graph(self, simple_graph, temp_dir):
        """Test exporting a simple graph to CSV."""
        exporter = CSVExporter()
        exporter.export(simple_graph, temp_dir)
        
        nodes_file = temp_dir / "nodes.csv"
        edges_file = temp_dir / "edges.csv"
        
        assert nodes_file.exists()
        assert edges_file.exists()

    def test_export_creates_directory(self, simple_graph, temp_dir):
        """Test that export creates output directory if it doesn't exist."""
        output_dir = temp_dir / "new_dir"
        exporter = CSVExporter()
        
        assert not output_dir.exists()
        exporter.export(simple_graph, output_dir)
        assert output_dir.exists()

    def test_export_empty_graph_raises_error(self, temp_dir):
        """Test that exporting empty graph raises ValueError."""
        empty_graph = nx.DiGraph()
        exporter = CSVExporter()
        
        with pytest.raises(ValueError, match="Cannot export empty graph"):
            exporter.export(empty_graph, temp_dir)

    def test_nodes_csv_structure(self, simple_graph, temp_dir):
        """Test that nodes CSV has correct structure."""
        exporter = CSVExporter()
        exporter.export(simple_graph, temp_dir)
        
        nodes_df = pd.read_csv(temp_dir / "nodes.csv")
        
        assert "id" in nodes_df.columns
        assert len(nodes_df) == simple_graph.number_of_nodes()

    def test_edges_csv_structure(self, simple_graph, temp_dir):
        """Test that edges CSV has correct structure."""
        exporter = CSVExporter()
        exporter.export(simple_graph, temp_dir)
        
        edges_df = pd.read_csv(temp_dir / "edges.csv")
        
        assert "source" in edges_df.columns
        assert "target" in edges_df.columns
        assert len(edges_df) == simple_graph.number_of_edges()

    def test_export_preserves_node_properties(self, temp_dir):
        """Test that node properties are preserved in export."""
        graph = nx.DiGraph()
        graph.add_node("node1", label="Person", name="Alice", age=30)
        
        exporter = CSVExporter()
        exporter.export(graph, temp_dir)
        
        nodes_df = pd.read_csv(temp_dir / "nodes.csv")
        node_row = nodes_df[nodes_df["id"] == "node1"].iloc[0]
        
        assert node_row["label"] == "Person"
        assert node_row["name"] == "Alice"
        assert node_row["age"] == 30

    def test_export_preserves_edge_properties(self, temp_dir):
        """Test that edge properties are preserved in export."""
        graph = nx.DiGraph()
        graph.add_node("node1")
        graph.add_node("node2")
        graph.add_edge("node1", "node2", label="knows", since=2020)
        
        exporter = CSVExporter()
        exporter.export(graph, temp_dir)
        
        edges_df = pd.read_csv(temp_dir / "edges.csv")
        edge_row = edges_df.iloc[0]
        
        assert edge_row["source"] == "node1"
        assert edge_row["target"] == "node2"
        assert edge_row["label"] == "knows"
        assert edge_row["since"] == 2020


class TestCSVExporterValidation:
    """Tests for validate_graph method."""

    def test_validate_graph_with_nodes(self, simple_graph):
        """Test validation of graph with nodes."""
        exporter = CSVExporter()
        assert exporter.validate_graph(simple_graph) is True

    def test_validate_empty_graph(self):
        """Test validation of empty graph."""
        empty_graph = nx.DiGraph()
        exporter = CSVExporter()
        
        assert exporter.validate_graph(empty_graph) is False


class TestCSVExporterEncoding:
    """Tests for encoding handling."""

    def test_export_with_unicode_characters(self, temp_dir):
        """Test exporting graph with Unicode characters."""
        graph = nx.DiGraph()
        graph.add_node("node1", label="Personne", name="François", city="París")
        
        exporter = CSVExporter()
        exporter.export(graph, temp_dir)
        
        nodes_df = pd.read_csv(temp_dir / "nodes.csv")
        node_row = nodes_df.iloc[0]
        
        assert node_row["name"] == "François"
        assert node_row["city"] == "París"

    def test_export_respects_encoding_config(self, simple_graph, temp_dir):
        """Test that export respects encoding configuration."""
        config = ExportConfig()
        exporter = CSVExporter(config=config)
        
        exporter.export(simple_graph, temp_dir)
        
        # Verify files can be read with specified encoding
        nodes_df = pd.read_csv(temp_dir / "nodes.csv", encoding=config.CSV_ENCODING)
        assert len(nodes_df) > 0


class TestCSVExporterFilenames:
    """Tests for filename configuration."""

    def test_uses_configured_filenames(self, simple_graph, temp_dir):
        """Test that configured filenames are used."""
        config = ExportConfig()
        exporter = CSVExporter(config=config)
        
        exporter.export(simple_graph, temp_dir)
        
        assert (temp_dir / config.CSV_NODE_FILENAME).exists()
        assert (temp_dir / config.CSV_EDGE_FILENAME).exists()
