"""
Unit tests for graph statistics utilities.
"""

import pytest
import networkx as nx

from docling_graph.core.utils.graph_stats import (
    calculate_graph_stats,
    get_node_type_distribution,
    get_edge_type_distribution,
)
from docling_graph.core.base.models import GraphMetadata


class TestCalculateGraphStats:
    """Tests for calculate_graph_stats function."""

    def test_stats_for_simple_graph(self, simple_graph):
        """Test statistics calculation for simple graph."""
        metadata = calculate_graph_stats(simple_graph, source_model_count=1)
        
        assert isinstance(metadata, GraphMetadata)
        assert metadata.node_count == simple_graph.number_of_nodes()
        assert metadata.edge_count == simple_graph.number_of_edges()
        assert metadata.source_models == 1

    def test_average_degree_calculation(self, simple_graph):
        """Test average degree calculation."""
        metadata = calculate_graph_stats(simple_graph, source_model_count=1)
        
        # Average degree = (2 * edge_count) / node_count
        expected_avg_degree = (2 * simple_graph.number_of_edges()) / simple_graph.number_of_nodes()
        assert metadata.average_degree == pytest.approx(expected_avg_degree)

    def test_stats_for_empty_graph(self):
        """Test statistics for empty graph."""
        empty_graph = nx.DiGraph()
        metadata = calculate_graph_stats(empty_graph, source_model_count=1)
        
        assert metadata.node_count == 0
        assert metadata.edge_count == 0
        assert metadata.average_degree == 0.0
        assert metadata.source_models == 1

    def test_stats_for_single_node_graph(self):
        """Test statistics for graph with single node."""
        graph = nx.DiGraph()
        graph.add_node("node1", label="Person")
        
        metadata = calculate_graph_stats(graph, source_model_count=1)
        
        assert metadata.node_count == 1
        assert metadata.edge_count == 0
        assert metadata.average_degree == 0.0

    def test_stats_with_multiple_source_models(self, simple_graph):
        """Test source_models parameter."""
        metadata = calculate_graph_stats(simple_graph, source_model_count=5)
        
        assert metadata.source_models == 5

    def test_node_types_distribution(self, simple_graph):
        """Test node types are included in metadata."""
        metadata = calculate_graph_stats(simple_graph, source_model_count=1)
        
        assert isinstance(metadata.node_types, dict)
        assert len(metadata.node_types) > 0

    def test_edge_types_distribution(self, simple_graph):
        """Test edge types are included in metadata."""
        metadata = calculate_graph_stats(simple_graph, source_model_count=1)
        
        assert isinstance(metadata.edge_types, dict)

    def test_stats_for_complete_graph(self):
        """Test statistics for complete graph (all nodes connected)."""
        graph = nx.complete_graph(3, create_using=nx.DiGraph)
        
        metadata = calculate_graph_stats(graph, source_model_count=1)
        
        # In complete directed graph: 3 nodes, 6 edges (3*2)
        assert metadata.node_count == 3
        assert metadata.edge_count == 6


class TestGetNodeTypeDistribution:
    """Tests for get_node_type_distribution function."""

    def test_single_node_type(self):
        """Test distribution with single node type."""
        graph = nx.DiGraph()
        graph.add_node("node1", label="Person")
        graph.add_node("node2", label="Person")
        
        dist = get_node_type_distribution(graph)
        
        assert dist == {"Person": 2}

    def test_multiple_node_types(self):
        """Test distribution with multiple node types."""
        graph = nx.DiGraph()
        graph.add_node("node1", label="Person")
        graph.add_node("node2", label="Person")
        graph.add_node("node3", label="Organization")
        
        dist = get_node_type_distribution(graph)
        
        assert dist["Person"] == 2
        assert dist["Organization"] == 1

    def test_missing_label_defaults_to_unknown(self):
        """Test nodes without label default to Unknown."""
        graph = nx.DiGraph()
        graph.add_node("node1", label="Person")
        graph.add_node("node2")  # No label
        
        dist = get_node_type_distribution(graph)
        
        assert dist["Person"] == 1
        assert dist["Unknown"] == 1

    def test_empty_graph_distribution(self):
        """Test distribution for empty graph."""
        graph = nx.DiGraph()
        dist = get_node_type_distribution(graph)
        
        assert dist == {}

    def test_distribution_counts_are_accurate(self):
        """Test that counts are accurate."""
        graph = nx.DiGraph()
        for i in range(10):
            graph.add_node(f"person{i}", label="Person")
        for i in range(5):
            graph.add_node(f"org{i}", label="Organization")
        
        dist = get_node_type_distribution(graph)
        
        assert dist["Person"] == 10
        assert dist["Organization"] == 5
        assert sum(dist.values()) == 15


class TestGetEdgeTypeDistribution:
    """Tests for get_edge_type_distribution function."""

    def test_single_edge_type(self):
        """Test distribution with single edge type."""
        graph = nx.DiGraph()
        graph.add_node("node1")
        graph.add_node("node2")
        graph.add_edge("node1", "node2", label="knows")
        graph.add_edge("node2", "node1", label="knows")
        
        dist = get_edge_type_distribution(graph)
        
        assert dist == {"knows": 2}

    def test_multiple_edge_types(self):
        """Test distribution with multiple edge types."""
        graph = nx.DiGraph()
        graph.add_node("node1")
        graph.add_node("node2")
        graph.add_node("node3")
        graph.add_edge("node1", "node2", label="knows")
        graph.add_edge("node2", "node3", label="works_for")
        
        dist = get_edge_type_distribution(graph)
        
        assert dist["knows"] == 1
        assert dist["works_for"] == 1

    def test_missing_label_defaults_to_unknown(self):
        """Test edges without label default to Unknown."""
        graph = nx.DiGraph()
        graph.add_node("node1")
        graph.add_node("node2")
        graph.add_edge("node1", "node2", label="knows")
        graph.add_edge("node2", "node1")  # No label
        
        dist = get_edge_type_distribution(graph)
        
        assert dist["knows"] == 1
        assert dist["Unknown"] == 1

    def test_empty_graph_distribution(self):
        """Test distribution for empty graph."""
        graph = nx.DiGraph()
        dist = get_edge_type_distribution(graph)
        
        assert dist == {}

    def test_distribution_counts_are_accurate(self):
        """Test that counts are accurate."""
        graph = nx.DiGraph()
        for i in range(10):
            graph.add_node(f"person{i}")
        
        for i in range(8):
            graph.add_edge(f"person{i}", f"person{i+1}", label="knows")
        for i in range(2):
            graph.add_edge(f"person{i}", f"person{i+1}", label="works_for")
        
        dist = get_edge_type_distribution(graph)
        
        assert dist["knows"] == 8
        assert dist["works_for"] == 2
