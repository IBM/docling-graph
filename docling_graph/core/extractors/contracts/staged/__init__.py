"""
Staged extraction contract (3-pass node catalog strategy).

Provides catalog build, ID discovery, fill, and edge assembly for staged extraction.
"""

from .benchmark import (
    QualityDelta,
    RunMetrics,
    compare_metrics,
    load_run_metrics,
    summarize_comparison,
)
from .catalog import (
    EdgeSpec,
    NodeCatalog,
    NodeSpec,
    build_discovery_schema,
    build_node_catalog,
    flat_nodes_to_path_lists,
    get_discovery_prompt,
    get_model_for_path,
    validate_id_pass_skeleton_response,
    write_catalog_artifact,
    write_id_pass_artifact,
)
from .orchestrator import CatalogOrchestrator, CatalogOrchestratorConfig

__all__ = [
    "CatalogOrchestrator",
    "CatalogOrchestratorConfig",
    "EdgeSpec",
    "NodeCatalog",
    "NodeSpec",
    "QualityDelta",
    "RunMetrics",
    "build_discovery_schema",
    "build_node_catalog",
    "compare_metrics",
    "flat_nodes_to_path_lists",
    "get_discovery_prompt",
    "get_model_for_path",
    "load_run_metrics",
    "summarize_comparison",
    "validate_id_pass_skeleton_response",
    "write_catalog_artifact",
    "write_id_pass_artifact",
]
