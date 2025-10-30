"""
CosmoGraph visualizer for interactive graph visualization in the browser.
"""

from typing import Optional, Literal, Tuple
from pathlib import Path

import webbrowser
import tempfile
import json
import os

from rich import print
import networkx as nx
import pandas as pd

# Dependencies
try:
    from cosmograph import cosmo
    COSMOGRAPH_AVAILABLE = True
except ImportError:
    COSMOGRAPH_AVAILABLE = False

try:
    # Use embed_data to build a custom full-viewport HTML
    from ipywidgets.embed import embed_data
    IPYWIDGETS_AVAILABLE = True
except ImportError:
    IPYWIDGETS_AVAILABLE = False


class CosmoGraphVisualizer:
    """Visualize graphs using CosmoGraph in the browser."""

    def __init__(self) -> None:
        """Initialize CosmoGraph visualizer."""
        if not COSMOGRAPH_AVAILABLE:
            raise ImportError(
                "CosmoGraph is not installed. Install it with: pip install cosmograph"
            )
        if not IPYWIDGETS_AVAILABLE:
            raise ImportError(
                "ipywidgets is not installed. Install it with: pip install ipywidgets"
            )

    def load_csv(self, path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load graph data from CSV files.

        Args:
            path: Directory containing nodes.csv and edges.csv
        Returns:
            Tuple of (nodes_df, edges_df)
        """
        nodes_path = path / "nodes.csv"
        edges_path = path / "edges.csv"

        print(f"Loading nodes from {nodes_path}...")
        nodes_df = pd.read_csv(nodes_path)
        print(f"[green][GraphInspector] Loaded {len(nodes_df)} nodes[/green]")

        print(f"Loading edges from {edges_path}...")
        edges_df = pd.read_csv(edges_path)
        print(f"[green][GraphInspector] Loaded {len(edges_df)} edges[/green]")

        return nodes_df, edges_df

    def load_json(self, path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load graph data from a JSON file.

        JSON structure:
        {
          "nodes": [ { "id": ..., "label": ..., ... }, ... ],
          "edges": [ { "source": ..., "target": ..., ... }, ... ]
        }

        Args:
            path: Path to JSON file
        Returns:
            Tuple of (nodes_df, edges_df)
        """
        print(f"Loading graph from {path}...")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        nodes = data.get("nodes", [])
        edges = data.get("edges", [])
        nodes_df = pd.DataFrame(nodes)
        edges_df = pd.DataFrame(edges)

        print(f"[green][GraphInspector] Loaded {len(nodes_df)} nodes and {len(edges_df)} edges[/green]")
        return nodes_df, edges_df

    def prepare_data_for_cosmograph(
        self,
        nodes_df: pd.DataFrame,
        edges_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare dataframes for CosmoGraph visualization.
        
        Sorts columns so populated ones appear first in the node panel.
        """
        prepared_nodes = nodes_df.copy()
        
        # Ensure 'id' exists
        if "id" not in prepared_nodes.columns:
            if prepared_nodes.index.name is not None:
                prepared_nodes["id"] = prepared_nodes.index
            else:
                prepared_nodes["id"] = range(len(prepared_nodes))
        
        # Sort columns: essential first, then by non-null count (descending)
        essential_cols = ['id', 'idx', 'label', 'type', 'name']
        existing_essential = [c for c in essential_cols if c in prepared_nodes.columns]
        
        other_cols = [c for c in prepared_nodes.columns if c not in existing_essential]
        if other_cols:
            non_null_counts = prepared_nodes[other_cols].notna().sum()
            sorted_other_cols = non_null_counts.sort_values(ascending=False).index.tolist()
        else:
            sorted_other_cols = []
        
        # Reorder columns
        new_column_order = existing_essential + sorted_other_cols
        prepared_nodes = prepared_nodes[new_column_order]
        
        # Prepare edges
        prepared_edges = edges_df.copy()
        
        if "source" not in prepared_edges.columns or "target" not in prepared_edges.columns:
            raise ValueError("Edges dataframe must have 'source' and 'target' columns")
        
        return prepared_nodes, prepared_edges

    def display_cosmo_graph(
        self,
        path: Path,
        format: Literal["csv", "json"] = "csv",
        output_path: Optional[Path] = None,
        open_browser: bool = True,
    ) -> Path:
        """Load graph data from file and visualize with CosmoGraph in the browser."""
        
        # Load data
        if format == "csv":
            nodes_df, edges_df = self.load_csv(path)
        elif format == "json":
            nodes_df, edges_df = self.load_json(path)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return self._prepare_and_visualize(nodes_df, edges_df, output_path, open_browser)

    def save_cosmo_graph(
        self,
        graph: nx.DiGraph,
        output_path: Path,
        open_browser: bool = False,
        **kwargs
    ) -> Path:
        """Visualize a NetworkX graph using CosmoGraph."""
        
        # Convert NetworkX graph to DataFrames
        nodes_data = [{"id": n, **attrs} for n, attrs in graph.nodes(data=True)]
        edges_data = [{"source": s, "target": t, **attrs} for s, t, attrs in graph.edges(data=True)]
        
        nodes_df = pd.DataFrame(nodes_data)
        edges_df = pd.DataFrame(edges_data)
        
        return self._prepare_and_visualize(nodes_df, edges_df, output_path, open_browser)

    def _prepare_and_visualize(
        self,
        nodes_df: pd.DataFrame,
        edges_df: pd.DataFrame,
        output_path: Optional[Path],
        open_browser: bool
    ) -> Path:
        """Common logic to prepare data, create CosmoGraph widget, and export HTML."""
        
        # Prepare data
        prepared_nodes, prepared_edges = self.prepare_data_for_cosmograph(nodes_df, edges_df)
        
        # Display statistics
        print("[bold]Graph Statistics:[/bold]")
        print(f"  Nodes: [cyan]{len(prepared_nodes)}[/cyan]")
        print(f"  Edges: [cyan]{len(prepared_edges)}[/cyan]")
        
        # Include all columns in point metadata except id and name
        node_columns = [c for c in prepared_nodes.columns if c not in ['id']]
        
        print("Creating CosmoGraph visualization...")
        widget = cosmo(
            points=prepared_nodes,
            links=prepared_edges,
            point_id_by='id',
            link_source_by='source',
            link_target_by='target',
            point_label_by='id',
            simulation_repulsion=2,
            simulation_friction=1,
            point_include_columns=node_columns,
            fit_view_delay=2000
        )
        
        # Handle default output path
        if output_path is None:
            output_path = Path("outputs/temp_cosmo_graph.html")
        elif not str(output_path).endswith(".html"):
            output_path = Path(str(output_path) + ".html")
        
        return self._export_and_open(widget, output_path, open_browser)


    def _export_and_open(self, widget, output_path: Optional[Path], open_browser: bool) -> Path:
        """
        Write the full-viewport HTML and optionally open it in the default browser.
        """
        if output_path is None:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".html", delete=False, prefix="cosmograph_"
            ) as f:
                output_html = Path(f.name)
        else:
            output_html = Path(output_path)
            output_html.parent.mkdir(parents=True, exist_ok=True)

        print("Exporting to HTML...")
        self._write_fullpage_widget_html(widget, output_html, title="CosmoGraph Visualization")
        print(f"[green][GraphInspector][/green] HTML file created: {output_html}")

        if open_browser:
            print("Opening in browser...")
            webbrowser.open("file://" + os.path.abspath(str(output_html)))
            print("[blue][GraphInspector][/blue] Visualization opened in browser!")
        else:
            print(f"[cyan]To view, open: {output_html}[/cyan]")

        return output_html

    def _write_fullpage_widget_html(self, widget, path: Path, title: str = "CosmoGraph Visualization") -> None:
        """
        Export a single widget into a full-viewport HTML page with a modern font stack.
        """
        # IMPORTANT: include non-default traits (e.g., anywidget _esm/_css) to avoid null class errors
        data = embed_data(views=[widget], drop_defaults=False)
        manager_state = json.dumps(data["manager_state"])
        widget_view = json.dumps(data["view_specs"][0])

        # Pin the anywidget UMD build so the HTML manager can load the runtime in static contexts
        ANYWIDGET_VERSION = "0.9.13"

        html = f"""<!doctype html>
            <html lang="en">
            <head>
            <meta charset="utf-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1" />
            <title>{title}</title>
            <style>
                html, body {{
                height: 100%;
                margin: 0;
                padding: 0;
                }}
                body {{
                /* Modern, legible system font stack */
                font-family: Inter, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Noto Sans", "Liberation Sans", sans-serif;
                line-height: 1.4;
                background: #111;
                color: #eaeaea;
                }}
                /* Full-viewport container for the widget */
                #root {{
                position: fixed;
                inset: 0;
                display: grid;
                overflow: hidden;
                }}
                /* Make the widget fill the container */
                #widget-host {{
                width: 100%;
                height: 100%;
                }}
                .wrapper {{
                height: 100vh !important;
                }}
            </style>

            <!-- RequireJS for the ipywidgets HTML manager -->
            <script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.4/require.min.js"
                    integrity="sha256-Ae2Vz/4ePdIu6ZyI/5ZGsYnb+m0JlOmKPjt6XZ9JJkA="
                    crossorigin="anonymous"></script>

            <!-- Map the anywidget runtime to a UMD build on a CDN for static embedding -->
            <script>
                (function() {{
                if (window.require && window.require.config) {{
                    window.require.config({{
                    paths: {{
                        anywidget: "https://unpkg.com/anywidget@{ANYWIDGET_VERSION}/dist/index.umd"
                    }}
                    }});
                }}
                }})();
            </script>

            <!-- IPywidgets bundle for embedding (RequireJS embedder for custom widgets) -->
            <script
                data-jupyter-widgets-cdn="https://unpkg.com/"
                data-jupyter-widgets-cdn-only
                src="https://cdn.jsdelivr.net/npm/@jupyter-widgets/html-manager@*/dist/embed-amd.js"
                crossorigin="anonymous">
            </script>

            <!-- Widget manager state -->
            <script type="application/vnd.jupyter.widget-state+json">
            {manager_state}
            </script>
            </head>
            <body>
            <div id="root">
                <div id="widget-host">
                <!-- This script tag will be replaced by the widget's DOM -->
                <script type="application/vnd.jupyter.widget-view+json">
                {widget_view}
                </script>
                </div>
            </div>
            </body>
            </html>
            """
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)