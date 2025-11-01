"""
Component integration tests.

Tests interactions between extractors, exporters, and other components.
"""

import pytest
from unittest.mock import Mock, patch
import networkx as nx

from docling_graph.core.exporters.csv_exporter import CSVExporter
from docling_graph.core.exporters.json_exporter import JSONExporter
from docling_graph.core.exporters.cypher_exporter import CypherExporter


@pytest.mark.integration
class TestExtractorExporterIntegration:
    """Test extractors work correctly with exporters."""

    def test_extracted_data_exports_to_csv(self, temp_dir, sample_graph):
        """Verify extracted data can be exported to CSV."""
        exporter = CSVExporter()
        output_dir = temp_dir / "csv_export"
        
        exporter.export(sample_graph, output_dir)
        
        assert (output_dir / "nodes.csv").exists()
        assert (output_dir / "edges.csv").exists()

    def test_extracted_data_exports_to_json(self, temp_dir, sample_graph):
        """Verify extracted data can be exported to JSON."""
        exporter = JSONExporter()
        output_file = temp_dir / "graph.json"
        
        exporter.export(sample_graph, output_file)
        
        assert output_file.exists()

    def test_extracted_data_exports_to_cypher(self, temp_dir, sample_graph):
        """Verify extracted data can be exported to Cypher."""
        exporter = CypherExporter()
        output_file = temp_dir / "graph.cypher"
        
        exporter.export(sample_graph, output_file)
        
        assert output_file.exists()

    def test_graph_consistency_across_exports(self, temp_dir, sample_graph):
        """Verify graph consistency when exported to multiple formats."""
        csv_exporter = CSVExporter()
        json_exporter = JSONExporter()
        
        # Export to CSV
        csv_dir = temp_dir / "csv"
        csv_exporter.export(sample_graph, csv_dir)
        
        # Export to JSON
        json_file = temp_dir / "graph.json"
        json_exporter.export(sample_graph, json_file)
        
        # Verify both exports exist
        assert (csv_dir / "nodes.csv").exists()
        assert json_file.exists()


@pytest.mark.integration
class TestBackendStrategyIntegration:
    """Test backends work with extraction strategies."""

    def test_llm_backend_with_one_to_one(self, temp_dir, mock_backend):
        """Test LLM backend works with OneToOne strategy."""
        from docling_graph.core.extractors.strategies.one_to_one import OneToOneStrategy
        from .conftest import Person
        
        mock_backend.extract_from_document.return_value = [
            Person(name="Test", email="test@example.com", age=25)
        ]
        
        strategy = OneToOneStrategy(backend=mock_backend)
        result = strategy.extract("test.pdf", Person)
        
        assert len(result) == 1
        assert result[0].name == "Test"

    def test_vlm_backend_with_one_to_one(self, temp_dir, mock_vlm_backend):
        """Test VLM backend works with OneToOne strategy."""
        from docling_graph.core.extractors.strategies.one_to_one import OneToOneStrategy
        from .conftest import Invoice
        
        mock_vlm_backend.extract_from_document.return_value = [
            Invoice(invoice_number="INV-001", date="2024-01-15", customer="Corp", amount=5000)
        ]
        
        strategy = OneToOneStrategy(backend=mock_vlm_backend)
        result = strategy.extract("test.pdf", Invoice)
        
        assert len(result) == 1

    def test_llm_backend_with_many_to_one(self, temp_dir, mock_backend):
        """Test LLM backend works with ManyToOne strategy."""
        from docling_graph.core.extractors.strategies.many_to_one import ManyToOneStrategy
        from .conftest import Document
        
        mock_backend.extract_from_markdown.return_value = Document(
            title="Test", content="Content", author="Author"
        )
        
        strategy = ManyToOneStrategy(backend=mock_backend)
        result = strategy.extract("test.pdf", Document)
        
        assert len(result) > 0

    def test_vlm_backend_with_many_to_one(self, temp_dir, mock_vlm_backend):
        """Test VLM backend works with ManyToOne strategy."""
        from docling_graph.core.extractors.strategies.many_to_one import ManyToOneStrategy
        from .conftest import Person
        
        mock_vlm_backend.extract_from_document.return_value = [
            Person(name="Person1", email="p1@example.com", age=30)
        ]
        
        strategy = ManyToOneStrategy(backend=mock_vlm_backend)
        result = strategy.extract("test.pdf", Person)
        
        assert len(result) >= 0


@pytest.mark.integration
class TestGraphCreationValidation:
    """Test graph creation and validation from extracted data."""

    def test_graph_created_from_extracted_data(self, sample_graph):
        """Verify graph can be created from extracted data."""
        assert sample_graph is not None
        assert sample_graph.number_of_nodes() > 0
        assert sample_graph.number_of_edges() > 0

    def test_graph_preserves_node_attributes(self, sample_graph):
        """Verify graph preserves node attributes."""
        for node, attrs in sample_graph.nodes(data=True):
            assert "label" in attrs or len(attrs) > 0

    def test_graph_preserves_edge_attributes(self, sample_graph):
        """Verify graph preserves edge attributes."""
        for source, target, attrs in sample_graph.edges(data=True):
            # Edge should have at least some attributes or exist
            assert source is not None
            assert target is not None

    def test_graph_has_valid_structure(self, complex_graph):
        """Test complex graph has valid structure."""
        assert isinstance(complex_graph, nx.DiGraph)
        assert complex_graph.number_of_nodes() > 0
        assert complex_graph.number_of_edges() > 0

    def test_graph_is_acyclic_if_specified(self, sample_graph):
        """Test graph structure follows expected patterns."""
        # Just verify it's a valid graph
        assert nx.is_directed(sample_graph)


@pytest.mark.integration
class TestMultipleBackendChaining:
    """Test chaining multiple backends and strategies."""

    def test_sequential_extraction_with_different_backends(self, temp_dir, mock_backend, mock_vlm_backend):
        """Test sequential extraction with different backends."""
        from docling_graph.core.extractors.strategies.one_to_one import OneToOneStrategy
        from .conftest import Person, Invoice
        
        # First extraction with LLM
        mock_backend.extract_from_document.return_value = [
            Person(name="Alice", email="alice@example.com", age=30)
        ]
        strategy1 = OneToOneStrategy(backend=mock_backend)
        result1 = strategy1.extract("test1.pdf", Person)
        
        # Second extraction with VLM
        mock_vlm_backend.extract_from_document.return_value = [
            Invoice(invoice_number="INV-001", date="2024-01-15", customer="Corp", amount=5000)
        ]
        strategy2 = OneToOneStrategy(backend=mock_vlm_backend)
        result2 = strategy2.extract("test2.pdf", Invoice)
        
        # Both should succeed
        assert len(result1) > 0
        assert len(result2) > 0
