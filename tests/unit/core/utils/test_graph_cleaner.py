import networkx as nx
import pytest

from docling_graph.core.utils.graph_cleaner import (
    GraphCleaner,
    validate_graph_structure,
)


@pytest.fixture
def cleaner():
    # Disable verbose printing during tests
    return GraphCleaner(verbose=False)


@pytest.fixture
def dirty_graph() -> nx.DiGraph:
    """Returns a graph with duplicates, phantoms, and orphans."""
    G = nx.DiGraph()
    # Add nodes
    G.add_node("node-1", name="Alice")
    G.add_node("node-2", name="Acme")
    G.add_node("node-3", name="Bob")
    # Add a semantic duplicate node
    G.add_node("node-4", name="Alice")
    # Add a phantom node (only metadata)
    G.add_node("phantom-1", id="phantom-1", label="Person")

    # Add edges
    G.add_edge("node-1", "node-2", label="WORKS_AT")
    # Add a duplicate edge
    G.add_edge("node-1", "node-2", label="WORKS_AT")
    # Add edge from the semantic duplicate
    G.add_edge("node-4", "node-2", label="WORKS_AT")
    # Add edge to the phantom node
    G.add_edge("node-3", "phantom-1", label="KNOWS")
    # Add an orphaned edge (node-99 doesn't exist)
    G.add_edge("node-1", "node-99", label="ORPHAN")

    return G


def test_clean_graph(cleaner: GraphCleaner, dirty_graph: nx.DiGraph):
    """Test the full clean_graph method."""
    assert dirty_graph.number_of_nodes() == 6
    assert dirty_graph.number_of_edges() == 4

    # Run the cleanup
    cleaned_graph = cleaner.clean_graph(dirty_graph)

    # Check nodes:
    # "node-1" (canonical)
    # "node-2"
    # "node-3"
    # "node-4" (merged into "node-1")
    # "phantom-1" (removed)
    assert cleaned_graph.number_of_nodes() == 3
    assert "node-1" in cleaned_graph
    assert "node-2" in cleaned_graph
    assert "node-3" in cleaned_graph
    assert "node-4" not in cleaned_graph
    assert "phantom-1" not in cleaned_graph

    # Check edges:
    # 1. ("node-1", "node-2") - original
    # 2. ("node-1", "node-2") - duplicate, removed
    # 3. ("node-4", "node-2") - redirected to ("node-1", "node-2"), removed as duplicate
    # 4. ("node-3", "phantom-1") - removed (phantom node deleted)
    # 5. ("node-1", "node-99") - removed (orphaned)
    assert cleaned_graph.number_of_edges() == 1
    assert cleaned_graph.has_edge("node-1", "node-2")


def test_validate_graph_structure_valid():
    """Test validation on a clean graph."""
    G = nx.DiGraph()
    G.add_node("A", name="Node A")
    G.add_node("B", name="Node B")
    G.add_edge("A", "B", label="CONNECTS")

    assert validate_graph_structure(G, raise_on_error=True) is True


def test_validate_graph_structure_orphan_edge():
    """Test validation failure for an auto-created empty node."""
    G = nx.DiGraph()
    G.add_node("A", name="Node A")
    G.add_edge("A", "B", label="CONNECTS")  # networkx auto-creates node "B" with no data

    with pytest.raises(ValueError, match="Empty node: B"):
        validate_graph_structure(G, raise_on_error=True)


def test_validate_graph_structure_empty_node():
    """Test validation failure for an empty node."""
    G = nx.DiGraph()
    G.add_node("A", name="Node A")
    G.add_node("B", id="B", label="Test")  # Empty node

    with pytest.raises(ValueError, match="Empty node: B"):
        validate_graph_structure(G, raise_on_error=True)
