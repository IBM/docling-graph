"""
Inspect command - visualizes graph data in browser.
"""

from pathlib import Path
from typing import Optional

import typer
from rich import print as rich_print
from typing_extensions import Annotated

from ...core.visualizers.interactive_visualizer import InteractiveVisualizer


def inspect_command(
    path: Annotated[
        Path,
        typer.Argument(
            help="Path to graph data. For CSV: directory with nodes.csv and edges.csv. For JSON: path to .json file.",
            exists=True,
        ),
    ],
    input_format: Annotated[
        str, typer.Option("--format", "-f", help="Import format: 'csv' or 'json'.")
    ] = "csv",
    output: Annotated[
        Path | None,
        typer.Option(
            "--output", "-o", help="Output HTML file path. If not specified, uses temporary file."
        ),
    ] = None,
    open_browser: Annotated[
        bool, typer.Option("--open/--no-open", help="Automatically open browser.")
    ] = True,
) -> None:
    """
    Visualize graph data in the browser.

    This command creates an interactive HTML visualization that opens
    in your default web browser. The HTML file is self-contained and
    can be shared or saved for later viewing.

    Examples:
        # Visualize CSV format (default) - opens in browser
        docling-graph inspect ./output_dir

        # Visualize JSON format
        docling-graph inspect graph.json --format json

        # Save to specific location
        docling-graph inspect ./output_dir --output graph_viz.html

        # Create HTML without opening browser
        docling-graph inspect ./output_dir --no-open --output viz.html
    """

    # Validate format
    input_format = input_format.lower()
    if input_format not in ["csv", "json"]:
        rich_print(
            f"[bold red]Error:[/bold red] Format must be 'csv' or 'json', got '{input_format}'"
        )
        raise typer.Exit(code=1)

    # Validate path based on format
    if input_format == "csv":
        if not path.is_dir():
            rich_print(
                "[bold red]Error:[/bold red] For CSV format, path must be a directory containing nodes.csv and edges.csv"
            )
            raise typer.Exit(code=1)

        nodes_path = path / "nodes.csv"
        edges_path = path / "edges.csv"

        if not nodes_path.exists():
            rich_print(f"[bold red]Error:[/bold red] nodes.csv not found in {path}")
            raise typer.Exit(code=1)

        if not edges_path.exists():
            rich_print(f"[bold red]Error:[/bold red] edges.csv not found in {path}")
            raise typer.Exit(code=1)

    elif input_format == "json":
        if not path.is_file() or path.suffix != ".json":
            rich_print("[bold red]Error:[/bold red] For JSON format, path must be a .json file")
            raise typer.Exit(code=1)

    rich_print("--- [blue]Starting Docling-Graph Inspection[/blue] ---")
    rich_print("\n[bold]Interactive Visualization[/bold]")
    rich_print(f"  Input: [cyan]{path}[/cyan]")
    rich_print(f"  Format: [cyan]{input_format}[/cyan]")
    if output:
        rich_print(f"  Output: [cyan]{output}[/cyan]")
    else:
        rich_print("  Output: [cyan]temporary file[/cyan]")

    try:
        # Create visualizer
        visualizer = InteractiveVisualizer()

        # Load and visualize
        rich_print("\nLoading graph data...")
        visualizer.display_cytoscape_graph(
            path=path, input_format=input_format, output_path=output, open_browser=open_browser
        )

        rich_print("--- [blue]Docling-Graph Inspection Finished Successfully[/blue] ---")

        if not open_browser:
            rich_print(
                "\n[blue]Tip:[/blue] Open the HTML file in your browser to view the visualization"
            )

    except Exception as e:
        rich_print(f"[bold red]Error:[/bold red] {type(e).__name__}: {e}")
        raise typer.Exit(code=1) from e
