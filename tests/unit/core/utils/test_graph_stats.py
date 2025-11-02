"""
Tests for graph statistics utilities.
"""

import networkx as nx
import pytest

from docling_graph.core.utils.graph_stats import (
    calculate_graph_stats,
    get_edge_type_distribution,
    get_node_type_distribution,
)


@pytest.fixture
def sample_graph():
    """Create a sample directed graph for testing."""
    graph = nx.DiGraph()

    # Add nodes with labels
    graph.add_node("node_1", label="Person")
    graph.add_node("node_2", label="Company")
    graph.add_node("node_3", label="Person")
    graph.add_node("node_4", label="Location")

    # Add edges with labels
    graph.add_edge("node_1", "node_2", label="works_for")
    graph.add_edge("node_2", "node_4", label="located_in")
    graph.add_edge("node_1", "node_3", label="knows")

    return graph


@pytest.fixture
def empty_graph():
    """Create an empty graph."""
    return nx.DiGraph()


class TestCalculateGraphStats:
    """Test graph statistics calculation."""

    def test_calculate_graph_stats_basic(self, sample_graph):
        """Should calculate basic graph statistics."""
        stats = calculate_graph_stats(sample_graph, source_model_count=1)

        assert stats.node_count == 4
        assert stats.edge_count == 3
        assert stats.source_models == 1

    def test_calculate_graph_stats_node_types(self, sample_graph):
        """Should calculate node type distribution."""
        stats = calculate_graph_stats(sample_graph, source_model_count=1)

        assert stats.node_types["Person"] == 2
        assert stats.node_types["Company"] == 1
        assert stats.node_types["Location"] == 1

    def test_calculate_graph_stats_edge_types(self, sample_graph):
        """Should calculate edge type distribution."""
        stats = calculate_graph_stats(sample_graph, source_model_count=1)

        assert stats.edge_types["works_for"] == 1
        assert stats.edge_types["located_in"] == 1
        assert stats.edge_types["knows"] == 1

    def test_calculate_graph_stats_average_degree(self, sample_graph):
        """Should calculate average degree."""
        stats = calculate_graph_stats(sample_graph, source_model_count=1)

        # Average degree = (2 * edges) / nodes = (2 * 3) / 4 = 1.5
        assert stats.average_degree == 1.5

    def test_calculate_graph_stats_empty_graph(self, empty_graph):
        """Should handle empty graph."""
        stats = calculate_graph_stats(empty_graph, source_model_count=0)

        assert stats.node_count == 0
        assert stats.edge_count == 0
        assert stats.average_degree == 0.0

    def test_calculate_graph_stats_source_models(self, sample_graph):
        """Should store source model count."""
        stats = calculate_graph_stats(sample_graph, source_model_count=5)
        assert stats.source_models == 5


class TestGetNodeTypeDistribution:
    """Test node type distribution calculation."""

    def test_get_node_type_distribution(self, sample_graph):
        """Should calculate node type distribution."""
        distribution = get_node_type_distribution(sample_graph)

        assert distribution["Person"] == 2
        assert distribution["Company"] == 1
        assert distribution["Location"] == 1

    def test_get_node_type_distribution_empty_graph(self, empty_graph):
        """Should return empty dict for empty graph."""
        distribution = get_node_type_distribution(empty_graph)
        assert distribution == {}

    def test_get_node_type_distribution_unknown_label(self):
        """Should count nodes without label as Unknown."""
        graph = nx.DiGraph()
        graph.add_node("node_1")  # No label
        graph.add_node("node_2", label="Known")

        distribution = get_node_type_distribution(graph)

        assert distribution["Unknown"] == 1
        assert distribution["Known"] == 1

    def test_get_node_type_distribution_single_type(self):
        """Should handle graphs with single node type."""
        graph = nx.DiGraph()
        graph.add_node("n1", label="Type")
        graph.add_node("n2", label="Type")

        distribution = get_node_type_distribution(graph)

        assert len(distribution) == 1
        assert distribution["Type"] == 2


class TestGetEdgeTypeDistribution:
    """Test edge type distribution calculation."""

    def test_get_edge_type_distribution(self, sample_graph):
        """Should calculate edge type distribution."""
        distribution = get_edge_type_distribution(sample_graph)

        assert distribution["works_for"] == 1
        assert distribution["located_in"] == 1
        assert distribution["knows"] == 1

    def test_get_edge_type_distribution_empty_graph(self, empty_graph):
        """Should return empty dict for empty graph."""
        distribution = get_edge_type_distribution(empty_graph)
        assert distribution == {}

    def test_get_edge_type_distribution_repeated_types(self):
        """Should count repeated edge types."""
        graph = nx.DiGraph()
        graph.add_edge("n1", "n2", label="related")
        graph.add_edge("n2", "n3", label="related")
        graph.add_edge("n3", "n1", label="related")

        distribution = get_edge_type_distribution(graph)

        assert distribution["related"] == 3

    def test_get_edge_type_distribution_unknown_label(self):
        """Should count edges without label as Unknown."""
        graph = nx.DiGraph()
        graph.add_edge("n1", "n2")  # No label
        graph.add_edge("n2", "n3", label="known")

        distribution = get_edge_type_distribution(graph)

        assert distribution["Unknown"] == 1
        assert distribution["known"] == 1
