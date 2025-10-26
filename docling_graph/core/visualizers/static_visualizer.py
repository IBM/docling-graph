"""Static graph visualizer using Matplotlib."""

from typing import Optional, Literal
import matplotlib.pyplot as plt
from pathlib import Path
import networkx as nx

from ..utils.formatting import format_property_key, format_property_value
from ..common.config import VisualizationConfig


class StaticVisualizer:
    """Create static graph visualizations with Matplotlib."""

    def __init__(self, config: Optional[VisualizationConfig] = None):
        """Initialize static visualizer.

        Args:
            config: Visualization configuration. Uses defaults if None.
        """
        self.config = config or VisualizationConfig()

    def visualize(
        self,
        graph: nx.DiGraph,
        output_path: Path,
        format: Literal["png", "svg", "pdf"] = "png",
        show_properties: bool = True,
        max_properties: int = 5
    ) -> None:
        """Create static visualization matching original graph_visualizer.py output.

        Args:
            graph: NetworkX directed graph to visualize.
            output_path: Base path for output file (without extension).
            format: Output format ('png', 'svg', or 'pdf').
            show_properties: Whether to show property boxes next to nodes.
            max_properties: Maximum number of properties to display per node.

        Raises:
            ValueError: If graph is empty or format is invalid.
        """
        if not self.validate_graph(graph):
            print("Graph is empty. Nothing to draw.")
            return

        if format not in self.config.STATIC_FORMATS:
            raise ValueError(
                f"Invalid format '{format}'. "
                f"Must be one of: {', '.join(self.config.STATIC_FORMATS)}"
            )

        # Ensure extension is added (matches original: Path(filename).with_suffix(f".{format}"))
        output_file = Path(str(output_path)).with_suffix(f".{format}")
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with plt.rc_context():
            # Create figure with better proportions (matches original)
            fig, ax = plt.subplots(figsize=(24, 18), facecolor='white')

            # Choose layout based on graph size (matches original)
            if graph.number_of_nodes() < 20:
                try:
                    pos = nx.kamada_kawai_layout(graph)
                except:
                    pos = nx.spring_layout(graph, seed=42, k=2, iterations=100)
            else:
                pos = nx.spring_layout(graph, seed=42, k=1.5, iterations=150)

            # Enhanced node styling (matches original)
            nx.draw_networkx_nodes(
                graph, pos,
                node_size=self.config.STATIC_NODE_SIZE,
                node_color=self.config.STATIC_NODE_COLOR,
                edgecolors=self.config.STATIC_NODE_EDGE_COLOR,
                linewidths=self.config.STATIC_NODE_EDGE_WIDTH,
                alpha=0.9,
                ax=ax
            )

            # Enhanced edge styling (matches original)
            nx.draw_networkx_edges(
                graph, pos,
                edge_color=self.config.STATIC_EDGE_COLOR,
                arrows=True,
                arrowstyle=self.config.STATIC_ARROW_STYLE,
                arrowsize=self.config.STATIC_ARROW_SIZE,
                width=self.config.STATIC_EDGE_WIDTH,
                connectionstyle='arc3,rad=0.15',
                alpha=0.7,
                ax=ax
            )

            # Edge labels with background (matches original)
            edge_labels = nx.get_edge_attributes(graph, 'label')
            nx.draw_networkx_edge_labels(
                graph, pos,
                edge_labels=edge_labels,
                font_color=self.config.STATIC_EDGE_LABEL_COLOR,
                font_size=self.config.STATIC_EDGE_LABEL_FONT_SIZE,
                font_weight=self.config.STATIC_EDGE_LABEL_FONT_WEIGHT,
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.8),
                ax=ax
            )

            # Node labels with properties (matches original)
            for node, (x, y) in pos.items():
                data = graph.nodes[node]
                label = data.get('label', str(node))

                # Main label
                ax.text(
                    x, y, label,
                    ha='center', va='center',
                    fontsize=self.config.STATIC_NODE_LABEL_FONT_SIZE,
                    fontweight=self.config.STATIC_NODE_LABEL_FONT_WEIGHT,
                    color=self.config.STATIC_NODE_LABEL_COLOR,
                    zorder=10
                )

                # Properties box with improved formatting (matches original)
                if show_properties:
                    # Exclude internal keys and filter meaningful properties
                    important_props = {k: v for k, v in data.items()
                                     if k not in ['label', 'pos', 'id'] and v is not None and str(v).strip()}

                    if important_props:
                        # Limit to most important properties
                        props_items = list(important_props.items())[:max_properties]

                        # Format each property with clear labels
                        props_lines = []
                        for k, v in props_items:
                            formatted_key = format_property_key(k)
                            formatted_value = format_property_value(v, max_length=40)
                            props_lines.append(f"{formatted_key}:")
                            props_lines.append(f"  {formatted_value}")

                        props_text = '\n'.join(props_lines)
                        ax.text(
                            x + 0.04, y, props_text,
                            ha='left', va='center',
                            fontsize=8,
                            family='monospace',
                            bbox=dict(boxstyle='round,pad=0.2', fc='#FFF8DC',
                                    ec='#2C3E50', alpha=0.95, linewidth=1.5),
                            zorder=5
                        )

            ax.set_title(
                "Document Knowledge Graph",
                fontsize=24, fontweight='bold',
                pad=30, color='#2C3E50'
            )

            ax.axis('off')
            plt.tight_layout()

            # Save with appropriate DPI based on format (matches original)
            dpi = 300 if format == 'png' else None
            plt.savefig(
                output_file,
                format=format.upper(),
                dpi=dpi,
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none'
            )

            plt.close(fig)

    def validate_graph(self, graph: nx.DiGraph) -> bool:
        """Validate that graph is not empty."""
        return graph.number_of_nodes() > 0
