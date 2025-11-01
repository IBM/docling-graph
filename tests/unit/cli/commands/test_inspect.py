"""
Unit tests for inspect command module.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import typer

from docling_graph.cli.commands.inspect import inspect_command


class TestInspectCommand:
    """Tests for inspect_command function."""

    @patch("docling_graph.cli.commands.inspect.InteractiveVisualizer")
    @patch("docling_graph.cli.commands.inspect.rich_print")
    def test_inspect_command_csv_format(self, mock_print, mock_visualizer, temp_dir):
        """Test inspect command with CSV format."""
        # Create dummy CSV files
        nodes_csv = temp_dir / "nodes.csv"
        edges_csv = temp_dir / "edges.csv"
        nodes_csv.write_text("id,label\n1,Node1")
        edges_csv.write_text("source,target,label\n1,2,connects")

        mock_viz_instance = MagicMock()
        mock_visualizer.return_value = mock_viz_instance

        inspect_command(path=temp_dir, input_format="csv")

        # Verify visualizer was created and called
        mock_visualizer.assert_called_once()
        mock_viz_instance.display_cytoscape_graph.assert_called_once()

        # Check arguments
        call_args = mock_viz_instance.display_cytoscape_graph.call_args
        assert call_args[1]["path"] == temp_dir
        assert call_args[1]["input_format"] == "csv"
        assert call_args[1]["open_browser"] is True

    @patch("docling_graph.cli.commands.inspect.InteractiveVisualizer")
    @patch("docling_graph.cli.commands.inspect.rich_print")
    def test_inspect_command_json_format(self, mock_print, mock_visualizer, temp_dir):
        """Test inspect command with JSON format."""
        # Create dummy JSON file
        json_file = temp_dir / "graph.json"
        json_file.write_text('{"nodes": [], "edges": []}')

        mock_viz_instance = MagicMock()
        mock_visualizer.return_value = mock_viz_instance

        inspect_command(path=json_file, input_format="json")

        call_args = mock_viz_instance.display_cytoscape_graph.call_args
        assert call_args[1]["path"] == json_file
        assert call_args[1]["input_format"] == "json"

    @patch("docling_graph.cli.commands.inspect.rich_print")
    def test_inspect_command_invalid_format(self, mock_print, temp_dir):
        """Test inspect command with invalid format."""
        with pytest.raises(typer.Exit) as exc_info:
            inspect_command(path=temp_dir, input_format="invalid")

        assert exc_info.value.exit_code == 1

    @patch("docling_graph.cli.commands.inspect.rich_print")
    def test_inspect_command_csv_missing_nodes(self, mock_print, temp_dir):
        """Test inspect command when nodes.csv is missing."""
        # Only create edges.csv
        edges_csv = temp_dir / "edges.csv"
        edges_csv.write_text("source,target,label\n1,2,connects")

        with pytest.raises(typer.Exit) as exc_info:
            inspect_command(path=temp_dir, input_format="csv")

        assert exc_info.value.exit_code == 1

    @patch("docling_graph.cli.commands.inspect.rich_print")
    def test_inspect_command_csv_missing_edges(self, mock_print, temp_dir):
        """Test inspect command when edges.csv is missing."""
        # Only create nodes.csv
        nodes_csv = temp_dir / "nodes.csv"
        nodes_csv.write_text("id,label\n1,Node1")

        with pytest.raises(typer.Exit) as exc_info:
            inspect_command(path=temp_dir, input_format="csv")

        assert exc_info.value.exit_code == 1

    @patch("docling_graph.cli.commands.inspect.rich_print")
    def test_inspect_command_csv_path_not_directory(self, mock_print, temp_dir):
        """Test inspect command when CSV path is not a directory."""
        file_path = temp_dir / "not_a_dir.txt"
        file_path.write_text("content")

        with pytest.raises(typer.Exit) as exc_info:
            inspect_command(path=file_path, input_format="csv")

        assert exc_info.value.exit_code == 1

    @patch("docling_graph.cli.commands.inspect.rich_print")
    def test_inspect_command_json_invalid_path(self, mock_print, temp_dir):
        """Test inspect command when JSON path is invalid."""
        txt_file = temp_dir / "not_json.txt"
        txt_file.write_text("content")

        with pytest.raises(typer.Exit) as exc_info:
            inspect_command(path=txt_file, input_format="json")

        assert exc_info.value.exit_code == 1

    @patch("docling_graph.cli.commands.inspect.InteractiveVisualizer")
    @patch("docling_graph.cli.commands.inspect.rich_print")
    def test_inspect_command_with_custom_output(self, mock_print, mock_visualizer, temp_dir):
        """Test inspect command with custom output path."""
        nodes_csv = temp_dir / "nodes.csv"
        edges_csv = temp_dir / "edges.csv"
        nodes_csv.write_text("id,label\n1,Node1")
        edges_csv.write_text("source,target,label\n1,2,connects")

        output_file = temp_dir / "visualization.html"

        mock_viz_instance = MagicMock()
        mock_visualizer.return_value = mock_viz_instance

        inspect_command(path=temp_dir, input_format="csv", output=output_file)

        call_args = mock_viz_instance.display_cytoscape_graph.call_args
        assert call_args[1]["output_path"] == output_file

    @patch("docling_graph.cli.commands.inspect.InteractiveVisualizer")
    @patch("docling_graph.cli.commands.inspect.rich_print")
    def test_inspect_command_no_open_browser(self, mock_print, mock_visualizer, temp_dir):
        """Test inspect command without opening browser."""
        nodes_csv = temp_dir / "nodes.csv"
        edges_csv = temp_dir / "edges.csv"
        nodes_csv.write_text("id,label\n1,Node1")
        edges_csv.write_text("source,target,label\n1,2,connects")

        mock_viz_instance = MagicMock()
        mock_visualizer.return_value = mock_viz_instance

        inspect_command(path=temp_dir, input_format="csv", open_browser=False)

        call_args = mock_viz_instance.display_cytoscape_graph.call_args
        assert call_args[1]["open_browser"] is False

    @patch("docling_graph.cli.commands.inspect.InteractiveVisualizer")
    @patch("docling_graph.cli.commands.inspect.rich_print")
    def test_inspect_command_visualizer_error(self, mock_print, mock_visualizer, temp_dir):
        """Test inspect command when visualizer raises error."""
        nodes_csv = temp_dir / "nodes.csv"
        edges_csv = temp_dir / "edges.csv"
        nodes_csv.write_text("id,label\n1,Node1")
        edges_csv.write_text("source,target,label\n1,2,connects")

        mock_viz_instance = MagicMock()
        mock_viz_instance.display_cytoscape_graph.side_effect = Exception("Visualization failed")
        mock_visualizer.return_value = mock_viz_instance

        with pytest.raises(typer.Exit) as exc_info:
            inspect_command(path=temp_dir, input_format="csv")

        assert exc_info.value.exit_code == 1
