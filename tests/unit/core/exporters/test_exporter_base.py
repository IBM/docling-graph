"""
Unit tests for exporter base protocol.
"""

import networkx as nx
import pytest

from docling_graph.core.exporters.exporter_base import GraphExporterProtocol


class TestGraphExporterProtocol:
    """Tests for GraphExporterProtocol."""

    def test_protocol_is_runtime_checkable(self):
        """Test that protocol can be checked at runtime."""
        # Create a class that implements the protocol
        class MockExporter:
            def export(self, graph, output_path):
                pass
            
            def validate_graph(self, graph):
                return True
        
        exporter = MockExporter()
        assert isinstance(exporter, GraphExporterProtocol)

    def test_protocol_requires_export_method(self):
        """Test that protocol requires export method."""
        # Class without export method
        class IncompleteExporter:
            def validate_graph(self, graph):
                return True
        
        exporter = IncompleteExporter()
        assert not isinstance(exporter, GraphExporterProtocol)

    def test_protocol_requires_validate_method(self):
        """Test that protocol requires validate_graph method."""
        # Class without validate_graph method
        class IncompleteExporter:
            def export(self, graph, output_path):
                pass
        
        exporter = IncompleteExporter()
        assert not isinstance(exporter, GraphExporterProtocol)

    def test_protocol_with_both_methods(self):
        """Test that class with both methods implements protocol."""
        class CompleteExporter:
            def export(self, graph, output_path):
                pass
            
            def validate_graph(self, graph):
                return True
        
        exporter = CompleteExporter()
        assert isinstance(exporter, GraphExporterProtocol)
