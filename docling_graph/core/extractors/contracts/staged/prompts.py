"""
Prompts for the catalog staged strategy (discovery, fill).
Delegates to catalog.py for discovery; orchestrator uses catalog directly.
"""

from __future__ import annotations

from typing import Any

from .catalog import (
    NodeCatalog,
    build_discovery_schema as _build_discovery_schema,
    get_discovery_prompt as _get_discovery_prompt,
    validate_id_pass_skeleton_response as _validate_id_pass_skeleton_response,
)


def build_discovery_schema(catalog: NodeCatalog, allowed_paths: list[str] | None = None) -> str:
    """Delegates to catalog.build_discovery_schema."""
    return _build_discovery_schema(catalog, allowed_paths)


def get_discovery_prompt(
    markdown_content: str,
    catalog: NodeCatalog,
    primary_paths: list[str] | None = None,
    allowed_paths: list[str] | None = None,
) -> dict[str, str]:
    """Delegates to catalog.get_discovery_prompt."""
    return _get_discovery_prompt(markdown_content, catalog, primary_paths, allowed_paths)


def validate_id_pass_skeleton_response(
    data: dict[str, Any],
    catalog: NodeCatalog,
    allowed_paths_override: set[str] | None = None,
) -> tuple[bool, list[str], list[dict[str, Any]], dict[str, int]]:
    """Delegates to catalog.validate_id_pass_skeleton_response."""
    return _validate_id_pass_skeleton_response(data, catalog, allowed_paths_override)
