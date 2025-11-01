"""
Unit tests for GraphConfig and ExportConfig.
"""

import pytest

from docling_graph.core.base.config import ExportConfig, GraphConfig


class TestGraphConfig:
    """Tests for GraphConfig dataclass."""

    def test_graph_config_defaults(self):
        """Test GraphConfig with default values."""
        config = GraphConfig()
        
        assert config.NODE_ID_HASH_LENGTH == 12
        assert config.MAX_STRING_LENGTH == 1000
        assert config.TRUNCATE_SUFFIX == "..."
        assert config.add_reverse_edges is False
        assert config.validate_graph is True

    def test_graph_config_custom_values(self):
        """Test GraphConfig with custom values."""
        config = GraphConfig(add_reverse_edges=True, validate_graph=False)
        
        assert config.add_reverse_edges is True
        assert config.validate_graph is False
        # Constants should remain unchanged
        assert config.NODE_ID_HASH_LENGTH == 12

    def test_graph_config_immutable(self):
        """Test that GraphConfig is frozen (immutable)."""
        config = GraphConfig()
        
        with pytest.raises(Exception):  # FrozenInstanceError
            config.add_reverse_edges = True

    def test_graph_config_constants_are_final(self):
        """Test that constants are defined with Final type."""
        config = GraphConfig()
        
        # These should be accessible but not modifiable
        assert isinstance(config.NODE_ID_HASH_LENGTH, int)
        assert isinstance(config.MAX_STRING_LENGTH, int)
        assert isinstance(config.TRUNCATE_SUFFIX, str)


class TestExportConfig:
    """Tests for ExportConfig dataclass."""

    def test_export_config_csv_defaults(self):
        """Test ExportConfig CSV defaults."""
        config = ExportConfig()
        
        assert config.CSV_ENCODING == "utf-8"
        assert config.CSV_NODE_FILENAME == "nodes.csv"
        assert config.CSV_EDGE_FILENAME == "edges.csv"

    def test_export_config_cypher_defaults(self):
        """Test ExportConfig Cypher defaults."""
        config = ExportConfig()
        
        assert config.CYPHER_ENCODING == "utf-8"
        assert config.CYPHER_FILENAME == "graph.cypher"
        assert config.CYPHER_BATCH_SIZE == 1000

    def test_export_config_json_defaults(self):
        """Test ExportConfig JSON defaults."""
        config = ExportConfig()
        
        assert config.JSON_ENCODING == "utf-8"
        assert config.JSON_INDENT == 2
        assert config.JSON_FILENAME == "graph.json"
        assert config.ENSURE_ASCII is False

    def test_export_config_immutable(self):
        """Test that ExportConfig is frozen."""
        config = ExportConfig()
        
        with pytest.raises(Exception):  # FrozenInstanceError
            config.CSV_ENCODING = "latin1"

    def test_export_config_filenames_consistency(self):
        """Test that export config filenames have correct extensions."""
        config = ExportConfig()
        
        assert config.CSV_NODE_FILENAME.endswith(".csv")
        assert config.CSV_EDGE_FILENAME.endswith(".csv")
        assert config.CYPHER_FILENAME.endswith(".cypher")
        assert config.JSON_FILENAME.endswith(".json")


class TestConfigInteraction:
    """Tests for interactions between config classes."""

    def test_both_configs_can_coexist(self):
        """Test that both config classes can be instantiated together."""
        graph_config = GraphConfig()
        export_config = ExportConfig()
        
        assert graph_config is not None
        assert export_config is not None
        assert graph_config.NODE_ID_HASH_LENGTH == 12
        assert export_config.CSV_ENCODING == "utf-8"

    def test_configs_are_independent(self):
        """Test that configs are independent from each other."""
        config1 = GraphConfig(add_reverse_edges=True)
        config2 = GraphConfig(add_reverse_edges=False)
        
        # They should maintain independent state
        assert config1.add_reverse_edges is True
        assert config2.add_reverse_edges is False
