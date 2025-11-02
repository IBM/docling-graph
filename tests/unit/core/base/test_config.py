"""
Tests for graph configuration classes.
"""

import pytest

from docling_graph.core.base.config import ExportConfig, GraphConfig


class TestGraphConfig:
    """Test GraphConfig class."""

    def test_graph_config_initialization(self):
        """Should initialize with default values."""
        config = GraphConfig()
        assert config.NODE_ID_HASH_LENGTH == 12
        assert config.MAX_STRING_LENGTH == 1000
        assert config.TRUNCATE_SUFFIX == "..."
        assert config.add_reverse_edges is False
        assert config.validate_graph is True

    def test_graph_config_is_frozen(self):
        """Should be immutable (frozen)."""
        config = GraphConfig()
        with pytest.raises(AttributeError):
            config.add_reverse_edges = True

    def test_graph_config_with_custom_values(self):
        """Should accept custom configuration values."""
        config = GraphConfig(add_reverse_edges=True, validate_graph=False)
        assert config.add_reverse_edges is True
        assert config.validate_graph is False

    def test_graph_config_constants_are_final(self):
        """Should have immutable constants."""
        config = GraphConfig()
        assert config.NODE_ID_HASH_LENGTH == 12
        assert isinstance(config.TRUNCATE_SUFFIX, str)
        assert isinstance(config.MAX_STRING_LENGTH, int)

    def test_graph_config_node_id_hash_length_reasonable(self):
        """Hash length should be reasonable for Blake2b."""
        config = GraphConfig()
        assert 6 <= config.NODE_ID_HASH_LENGTH <= 32

    def test_graph_config_max_string_length_positive(self):
        """Max string length should be positive."""
        config = GraphConfig()
        assert config.MAX_STRING_LENGTH > 0


class TestExportConfig:
    """Test ExportConfig class."""

    def test_export_config_csv_settings(self):
        """Should have CSV export settings."""
        config = ExportConfig()
        assert config.CSV_ENCODING == "utf-8"
        assert config.CSV_NODE_FILENAME == "nodes.csv"
        assert config.CSV_EDGE_FILENAME == "edges.csv"

    def test_export_config_json_settings(self):
        """Should have JSON export settings."""
        config = ExportConfig()
        assert config.JSON_ENCODING == "utf-8"
        assert config.JSON_INDENT == 2
        assert config.JSON_FILENAME == "graph.json"

    def test_export_config_cypher_settings(self):
        """Should have Cypher export settings."""
        config = ExportConfig()
        assert config.CYPHER_ENCODING == "utf-8"
        assert config.CYPHER_FILENAME == "graph.cypher"
        assert config.CYPHER_BATCH_SIZE == 1000

    def test_export_config_general_settings(self):
        """Should have general export settings."""
        config = ExportConfig()
        assert config.ENSURE_ASCII is False

    def test_export_config_is_frozen(self):
        """Should be immutable (frozen)."""
        config = ExportConfig()
        with pytest.raises(AttributeError):
            config.CSV_NODE_FILENAME = "different.csv"

    def test_export_config_batch_size_positive(self):
        """Batch size should be positive."""
        config = ExportConfig()
        assert config.CYPHER_BATCH_SIZE > 0

    def test_export_config_json_indent_positive(self):
        """JSON indent should be positive."""
        config = ExportConfig()
        assert config.JSON_INDENT > 0
