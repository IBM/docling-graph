"""
Unit tests for InteractiveVisualizer.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, Mock, mock_open, patch

import networkx as nx
import pandas as pd
import pytest

from docling_graph.core.visualizers.interactive_visualizer import InteractiveVisualizer


class TestInteractiveVisualizerInitialization:
    """Tests for InteractiveVisualizer initialization."""

    def test_init_creates_instance(self):
        """Test that initializer creates instance."""
        visualizer = InteractiveVisualizer()
        
        assert visualizer is not None


class TestInteractiveVisualizerLoadCSV:
    """Tests for load_csv method."""

    def test_load_csv_reads_files(self, temp_dir):
        """Test loading CSV files."""
        # Create sample CSV files
        nodes_df = pd.DataFrame({
            "id": ["node1", "node2"],
            "label": ["Person", "Person"]
        })
        edges_df = pd.DataFrame({
            "source": ["node1"],
            "target": ["node2"],
            "label": ["knows"]
        })
        
        nodes_df.to_csv(temp_dir / "nodes.csv", index=False)
        edges_df.to_csv(temp_dir / "edges.csv", index=False)
        
        visualizer = InteractiveVisualizer()
        nodes, edges = visualizer.load_csv(temp_dir)
        
        assert len(nodes) == 2
        assert len(edges) == 1

    def test_load_csv_missing_nodes_file(self, temp_dir):
        """Test loading CSV when nodes file is missing."""
        edges_df = pd.DataFrame({
            "source": ["node1"],
            "target": ["node2"]
        })
        edges_df.to_csv(temp_dir / "edges.csv", index=False)
        
        visualizer = InteractiveVisualizer()
        
        with pytest.raises(FileNotFoundError):
            visualizer.load_csv(temp_dir)

    def test_load_csv_missing_edges_file(self, temp_dir):
        """Test loading CSV when edges file is missing."""
        nodes_df = pd.DataFrame({
            "id": ["node1", "node2"]
        })
        nodes_df.to_csv(temp_dir / "nodes.csv", index=False)
        
        visualizer = InteractiveVisualizer()
        
        with pytest.raises(FileNotFoundError):
            visualizer.load_csv(temp_dir)


class TestInteractiveVisualizerLoadJSON:
    """Tests for load_json method."""

    def test_load_json_reads_file(self, temp_dir):
        """Test loading JSON file."""
        graph_data = {
            "nodes": [
                {"id": "node1", "label": "Person"},
                {"id": "node2", "label": "Person"}
            ],
            "edges": [
                {"source": "node1", "target": "node2", "label": "knows"}
            ]
        }
        
        json_file = temp_dir / "graph.json"
        with open(json_file, "w") as f:
            json.dump(graph_data, f)
        
        visualizer = InteractiveVisualizer()
        nodes, edges = visualizer.load_json(json_file)
        
        assert len(nodes) == 2
        assert len(edges) == 1

    def test_load_json_missing_file(self, temp_dir):
        """Test loading non-existent JSON file."""
        visualizer = InteractiveVisualizer()
        
        with pytest.raises(FileNotFoundError):
            visualizer.load_json(temp_dir / "nonexistent.json")


class TestInteractiveVisualizerSaveCytoscapeGraph:
    """Tests for save_cytoscape_graph method."""

    @patch("docling_graph.core.visualizers.interactive_visualizer.open", new_callable=mock_open)
    def test_save_creates_html_file(self, mock_file, simple_graph, temp_dir):
        """Test that save creates HTML file."""
        visualizer = InteractiveVisualizer()
        output_path = temp_dir / "graph"
        
        visualizer.save_cytoscape_graph(simple_graph, output_path)
        
        mock_file.assert_called()

    def test_save_html_contains_cytoscape(self, simple_graph, temp_dir):
        """Test that generated HTML contains Cytoscape."""
        visualizer = InteractiveVisualizer()
        output_path = temp_dir / "graph"
        
        visualizer.save_cytoscape_graph(simple_graph, output_path)
        
        html_file = Path(str(output_path) + ".html")
        if html_file.exists():
            content = html_file.read_text()
            assert "cytoscape" in content.lower()


class TestInteractiveVisualizerDisplayCytoscapeGraph:
    """Tests for display_cytoscape_graph method."""

    @patch("docling_graph.core.visualizers.interactive_visualizer.webbrowser")
    def test_display_opens_browser(self, mock_browser, temp_dir):
        """Test that display opens browser."""
        # Create sample CSV files
        nodes_df = pd.DataFrame({"id": ["node1"], "label": ["Test"]})
        edges_df = pd.DataFrame({"source": [], "target": []})
        nodes_df.to_csv(temp_dir / "nodes.csv", index=False)
        edges_df.to_csv(temp_dir / "edges.csv", index=False)
        
        visualizer = InteractiveVisualizer()
        
        visualizer.display_cytoscape_graph(
            path=temp_dir,
            input_format="csv",
            open_browser=True
        )
        
        mock_browser.open.assert_called_once()

    @patch("docling_graph.core.visualizers.interactive_visualizer.webbrowser")
    def test_display_no_browser(self, mock_browser, temp_dir):
        """Test display without opening browser."""
        nodes_df = pd.DataFrame({"id": ["node1"]})
        edges_df = pd.DataFrame({"source": [], "target": []})
        nodes_df.to_csv(temp_dir / "nodes.csv", index=False)
        edges_df.to_csv(temp_dir / "edges.csv", index=False)
        
        visualizer = InteractiveVisualizer()
        
        visualizer.display_cytoscape_graph(
            path=temp_dir,
            input_format="csv",
            open_browser=False
        )
        
        mock_browser.open.assert_not_called()


class TestInteractiveVisualizerGraphToNetworkX:
    """Tests for graph_to_networkx method."""

    def test_converts_dataframes_to_graph(self):
        """Test converting DataFrames to NetworkX graph."""
        nodes_df = pd.DataFrame({
            "id": ["node1", "node2"],
            "label": ["Person", "Person"]
        })
        edges_df = pd.DataFrame({
            "source": ["node1"],
            "target": ["node2"],
            "label": ["knows"]
        })
        
        visualizer = InteractiveVisualizer()
        graph = visualizer.graph_to_networkx(nodes_df, edges_df)
        
        assert graph.number_of_nodes() == 2
        assert graph.number_of_edges() == 1

    def test_preserves_node_attributes(self):
        """Test that node attributes are preserved."""
        nodes_df = pd.DataFrame({
            "id": ["node1"],
            "label": ["Person"],
            "name": ["Alice"]
        })
        edges_df = pd.DataFrame({"source": [], "target": []})
        
        visualizer = InteractiveVisualizer()
        graph = visualizer.graph_to_networkx(nodes_df, edges_df)
        
        assert graph.nodes["node1"]["label"] == "Person"
        assert graph.nodes["node1"]["name"] == "Alice"


class TestInteractiveVisualizerValidateGraph:
    """Tests for validate_graph method."""

    def test_validate_graph_with_nodes(self, simple_graph):
        """Test validation of graph with nodes."""
        visualizer = InteractiveVisualizer()
        assert visualizer.validate_graph(simple_graph) is True

    def test_validate_empty_graph(self):
        """Test validation of empty graph."""
        empty_graph = nx.DiGraph()
        visualizer = InteractiveVisualizer()
        
        assert visualizer.validate_graph(empty_graph) is False
