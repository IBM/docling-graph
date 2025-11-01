"""
Unit tests for visualizer base protocol.
"""

import networkx as nx
import pytest
from pathlib import Path

from docling_graph.core.visualizers.visualizer_base import GraphVisualizerProtocol


class TestGraphVisualizerProtocol:
    """Tests for GraphVisualizerProtocol."""

    def test_protocol_is_runtime_checkable(self):
        """Test that protocol can be checked at runtime."""
        # Create a class that implements the protocol
        class MockVisualizer:
            def visualize(self, graph, output_path, **kwargs):
                pass
            
            def validate_graph(self, graph):
                return True
        
        visualizer = MockVisualizer()
        assert isinstance(visualizer, GraphVisualizerProtocol)

    def test_protocol_requires_visualize_method(self):
        """Test that protocol requires visualize method."""
        # Class without visualize method
        class IncompleteVisualizer:
            def validate_graph(self, graph):
                return True
        
        visualizer = IncompleteVisualizer()
        assert not isinstance(visualizer, GraphVisualizerProtocol)

    def test_protocol_requires_validate_method(self):
        """Test that protocol requires validate_graph method."""
        # Class without validate_graph method
        class IncompleteVisualizer:
            def visualize(self, graph, output_path):
                pass
        
        visualizer = IncompleteVisualizer()
        assert not isinstance(visualizer, GraphVisualizerProtocol)

    def test_protocol_with_both_methods(self):
        """Test that class with both methods implements protocol."""
        class CompleteVisualizer:
            def visualize(self, graph, output_path, **kwargs):
                pass
            
            def validate_graph(self, graph):
                return True
        
        visualizer = CompleteVisualizer()
        assert isinstance(visualizer, GraphVisualizerProtocol)

    def test_protocol_accepts_kwargs(self):
        """Test that visualize method can accept kwargs."""
        class FlexibleVisualizer:
            def visualize(self, graph, output_path, **kwargs):
                self.kwargs = kwargs
            
            def validate_graph(self, graph):
                return True
        
        visualizer = FlexibleVisualizer()
        graph = nx.DiGraph()
        graph.add_node("test")
        
        visualizer.visualize(graph, Path("test.html"), option1="value1", option2="value2")
        
        assert visualizer.kwargs["option1"] == "value1"
        assert visualizer.kwargs["option2"] == "value2"
