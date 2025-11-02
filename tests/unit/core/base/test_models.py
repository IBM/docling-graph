"""
Tests for graph model classes.
"""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from docling_graph.core.base.models import Edge, GraphMetadata


class TestEdgeModel:
    """Test Edge model."""

    def test_edge_initialization(self):
        """Should create edge with required fields."""
        edge = Edge(source="node1", target="node2", label="connected_to")

        assert edge.source == "node1"
        assert edge.target == "node2"
        assert edge.label == "connected_to"
        assert edge.properties == {}

    def test_edge_with_properties(self):
        """Should accept additional properties."""
        props = {"weight": 0.9, "confidence": 0.95}
        edge = Edge(source="node1", target="node2", label="related", properties=props)

        assert edge.properties == props
        assert edge.properties["weight"] == 0.9

    def test_edge_is_frozen(self):
        """Edge should be immutable."""
        edge = Edge(source="n1", target="n2", label="rel")

        with pytest.raises(ValidationError):
            edge.source = "different"

    def test_edge_missing_required_fields_raises_error(self):
        """Should require source, target, and label."""
        with pytest.raises(ValidationError):
            Edge(source="n1", target="n2")  # Missing label

    def test_edge_properties_defaults_to_empty_dict(self):
        """Properties should default to empty dict."""
        edge = Edge(source="n1", target="n2", label="rel")
        assert isinstance(edge.properties, dict)
        assert len(edge.properties) == 0

    def test_edge_with_complex_properties(self):
        """Should handle complex property values."""
        props = {
            "nested": {"key": "value"},
            "list": [1, 2, 3],
            "number": 42,
        }
        edge = Edge(source="n1", target="n2", label="rel", properties=props)

        assert edge.properties["nested"]["key"] == "value"
        assert edge.properties["list"] == [1, 2, 3]


class TestGraphMetadataModel:
    """Test GraphMetadata model."""

    def test_graph_metadata_initialization(self):
        """Should create metadata with required fields."""
        metadata = GraphMetadata(node_count=10, edge_count=15, source_models=2)

        assert metadata.node_count == 10
        assert metadata.edge_count == 15
        assert metadata.source_models == 2
        assert isinstance(metadata.created_at, datetime)

    def test_graph_metadata_with_type_distributions(self):
        """Should accept node and edge type distributions."""
        node_types = {"Person": 5, "Company": 3, "Location": 2}
        edge_types = {"works_for": 4, "located_in": 6}

        metadata = GraphMetadata(
            node_count=10,
            edge_count=10,
            node_types=node_types,
            edge_types=edge_types,
            source_models=1,
        )

        assert metadata.node_types == node_types
        assert metadata.edge_types == edge_types

    def test_graph_metadata_average_degree_optional(self):
        """Average degree should be optional."""
        metadata = GraphMetadata(node_count=10, edge_count=15, source_models=1)

        assert metadata.average_degree is None

    def test_graph_metadata_with_average_degree(self):
        """Should accept average degree value."""
        metadata = GraphMetadata(node_count=10, edge_count=15, source_models=1, average_degree=1.5)

        assert metadata.average_degree == 1.5

    def test_graph_metadata_created_at_timestamp(self):
        """Should have creation timestamp."""
        metadata = GraphMetadata(node_count=10, edge_count=15, source_models=1)

        assert isinstance(metadata.created_at, datetime)
        # Timestamp should be recent
        time_diff = datetime.now(timezone.utc) - metadata.created_at
        assert time_diff.total_seconds() < 1  # Less than 1 second ago

    def test_graph_metadata_is_frozen(self):
        """Metadata should be immutable."""
        metadata = GraphMetadata(node_count=10, edge_count=15, source_models=1)

        with pytest.raises(ValidationError):
            metadata.node_count = 20

    def test_graph_metadata_default_collections(self):
        """Type distributions should default to empty."""
        metadata = GraphMetadata(node_count=0, edge_count=0, source_models=0)

        assert metadata.node_types == {}
        assert metadata.edge_types == {}

    def test_graph_metadata_missing_required_fields_raises_error(self):
        """Should require basic fields."""
        with pytest.raises(ValidationError):
            GraphMetadata(node_count=10)  # Missing edge_count and source_models
