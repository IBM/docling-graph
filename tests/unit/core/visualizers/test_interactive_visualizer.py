"""
Tests for InteractiveVisualizer class.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import networkx as nx
import pandas as pd
import pytest

from docling_graph.core.visualizers.interactive_visualizer import InteractiveVisualizer


@pytest.fixture
def visualizer():
    """Create InteractiveVisualizer instance."""
    return InteractiveVisualizer()


@pytest.fixture
def sample_nodes_csv(tmp_path):
    """Create sample nodes CSV file."""
    csv_path = tmp_path / "nodes.csv"
    csv_path.write_text("id,label,type\nnode_1,Invoice,entity\nnode_2,Amount,entity\n")
    return csv_path


@pytest.fixture
def sample_edges_csv(tmp_path):
    """Create sample edges CSV file."""
    csv_path = tmp_path / "edges.csv"
    csv_path.write_text("source,target,label\nnode_1,node_2,contains\n")
    return csv_path


@pytest.fixture
def sample_json_file(tmp_path):
    """Create sample JSON graph file."""
    import json

    json_path = tmp_path / "graph.json"
    data = {
        "nodes": [{"id": "node_1", "label": "Invoice"}, {"id": "node_2", "label": "Amount"}],
        "edges": [{"source": "node_1", "target": "node_2", "label": "contains"}],
    }
    json_path.write_text(json.dumps(data))
    return json_path


class TestLoadCSV:
    """Test CSV loading."""

    def test_load_csv_success(self, tmp_path):
        """Should load CSV files successfully."""
        nodes_file = tmp_path / "nodes.csv"
        nodes_file.write_text("id,label\nnode_1,Test\n")

        edges_file = tmp_path / "edges.csv"
        edges_file.write_text("source,target,label\nnode_1,node_2,rel\n")

        visualizer = InteractiveVisualizer()
        nodes_df, edges_df = visualizer.load_csv(tmp_path)

        assert isinstance(nodes_df, pd.DataFrame)
        assert isinstance(edges_df, pd.DataFrame)
        assert len(nodes_df) > 0
        assert len(edges_df) > 0

    def test_load_csv_missing_nodes_raises_error(self, tmp_path):
        """Should raise error if nodes.csv missing."""
        edges_file = tmp_path / "edges.csv"
        edges_file.write_text("source,target\nnode_1,node_2\n")

        visualizer = InteractiveVisualizer()

        with pytest.raises(FileNotFoundError):
            visualizer.load_csv(tmp_path)

    def test_load_csv_missing_edges_raises_error(self, tmp_path):
        """Should raise error if edges.csv missing."""
        nodes_file = tmp_path / "nodes.csv"
        nodes_file.write_text("id,label\nnode_1,Test\n")

        visualizer = InteractiveVisualizer()

        with pytest.raises(FileNotFoundError):
            visualizer.load_csv(tmp_path)


class TestLoadJSON:
    """Test JSON loading."""

    def test_load_json_success(self, sample_json_file):
        """Should load JSON file successfully."""
        visualizer = InteractiveVisualizer()
        nodes_df, edges_df = visualizer.load_json(sample_json_file)

        assert isinstance(nodes_df, pd.DataFrame)
        assert isinstance(edges_df, pd.DataFrame)
        assert len(nodes_df) == 2
        assert len(edges_df) == 1

    def test_load_json_file_not_found(self):
        """Should raise error for missing file."""
        visualizer = InteractiveVisualizer()

        with pytest.raises(FileNotFoundError):
            visualizer.load_json(Path("nonexistent.json"))


class TestPrepareDataForCytoscape:
    """Test data preparation for Cytoscape."""

    def test_prepare_data_basic(self, visualizer):
        """Should prepare basic data for Cytoscape."""
        nodes_df = pd.DataFrame({"id": ["node_1", "node_2"], "label": ["Person", "Company"]})
        edges_df = pd.DataFrame(
            {"source": ["node_1"], "target": ["node_2"], "label": ["works_for"]}
        )

        result = visualizer.prepare_data_for_cytoscape(nodes_df, edges_df)

        assert "nodes" in result
        assert "edges" in result
        assert "meta" in result
        assert len(result["nodes"]) == 2
        assert len(result["edges"]) == 1

    def test_prepare_data_validates_edges(self, visualizer):
        """Should require source and target in edges."""
        nodes_df = pd.DataFrame({"id": ["n1", "n2"]})
        edges_df = pd.DataFrame(
            {
                "source": ["n1"],
                "label": ["rel"],  # Missing target
            }
        )

        with pytest.raises(ValueError):
            visualizer.prepare_data_for_cytoscape(nodes_df, edges_df)

    def test_prepare_data_handles_missing_values(self, visualizer):
        """Should handle NaN and None values."""
        nodes_df = pd.DataFrame({"id": ["node_1"], "label": [None], "value": [float("nan")]})
        edges_df = pd.DataFrame({"source": ["node_1"], "target": ["node_2"], "label": ["rel"]})

        result = visualizer.prepare_data_for_cytoscape(nodes_df, edges_df)

        assert result is not None
        assert len(result["nodes"]) > 0


class TestSerializeValue:
    """Test value serialization."""

    def test_serialize_none(self, visualizer):
        """Should serialize None."""
        result = visualizer._serialize_value(None)
        assert result is None

    def test_serialize_list(self, visualizer):
        """Should serialize lists."""
        result = visualizer._serialize_value([1, 2, 3])
        assert result == [1, 2, 3]

    def test_serialize_dict(self, visualizer):
        """Should serialize dicts."""
        d = {"key": "value"}
        result = visualizer._serialize_value(d)
        assert result == d

    def test_serialize_string(self, visualizer):
        """Should serialize strings."""
        result = visualizer._serialize_value("test")
        assert result == "test"


class TestIsValidValue:
    """Test value validation."""

    def test_is_valid_value_none(self, visualizer):
        """Should return False for None."""
        assert visualizer._is_valid_value(None) is False

    def test_is_valid_value_empty_list(self, visualizer):
        """Should return False for empty list."""
        assert visualizer._is_valid_value([]) is False

    def test_is_valid_value_filled_list(self, visualizer):
        """Should return True for non-empty list."""
        assert visualizer._is_valid_value([1, 2, 3]) is True

    def test_is_valid_value_empty_string(self, visualizer):
        """Should return False for empty string."""
        assert visualizer._is_valid_value("") is False

    def test_is_valid_value_filled_string(self, visualizer):
        """Should return True for non-empty string."""
        assert visualizer._is_valid_value("test") is True


class TestDisplayCytoscapeGraph:
    """Test Cytoscape graph display."""

    def test_display_csv_format(self, tmp_path):
        """Should display CSV format graph."""
        # Create test files
        nodes_file = tmp_path / "nodes.csv"
        nodes_file.write_text("id,label\nnode_1,Test\n")
        edges_file = tmp_path / "edges.csv"
        edges_file.write_text("source,target,label\nnode_1,node_2,rel\n")

        visualizer = InteractiveVisualizer()

        with patch("webbrowser.open"):
            output = visualizer.display_cytoscape_graph(
                tmp_path, input_format="csv", open_browser=False
            )

        assert output.exists()
        assert output.suffix == ".html"

    def test_display_json_format(self, sample_json_file, tmp_path):
        """Should display JSON format graph."""
        visualizer = InteractiveVisualizer()
        output_file = tmp_path / "output.html"

        with patch("webbrowser.open"):
            output = visualizer.display_cytoscape_graph(
                sample_json_file, input_format="json", output_path=output_file, open_browser=False
            )

        assert output.exists()

    def test_display_invalid_format_raises_error(self, tmp_path):
        """Should raise error for invalid format."""
        visualizer = InteractiveVisualizer()

        with pytest.raises(ValueError):
            visualizer.display_cytoscape_graph(tmp_path, input_format="invalid")
