"""
Prompt templates for LLM document extraction.

This module provides optimized prompts for structured data extraction
from document markdown using LLMs with model-aware adaptive prompting.

Design goals:
- Domain-agnostic (works across domains).
- Chunk-friendly (partial extraction).
- Relationship-friendly (avoid default empty arrays that kill edges).
"""

from __future__ import annotations

from typing import TypedDict

from pydantic import BaseModel

from .config import ModelCapability, ModelConfigLike


class PromptDict(TypedDict):
    """Type definition for prompt dictionaries."""

    system: str
    user: str


# ---------------------------------------------------------------------------
# Instruction variants for different model capabilities
# ---------------------------------------------------------------------------

_EXTRACTION_INSTRUCTIONS_SIMPLE = (
    "1. Read the provided document text carefully.\n"
    "2. Extract information matching the schema.\n"
    "3. Return valid JSON only.\n"
    "4. Omit fields with no data.\n"
)

# IMPORTANT: do NOT default to "" / [] / {} for missing values.
# This is critical for chunk extraction + graph edges:
# - If a relationship is implied, produce a minimal reference instead of [].
_EXTRACTION_INSTRUCTIONS_STANDARD = (
    "1. Read the provided document text carefully.\n"
    "2. Extract ALL information that matches the provided schema AND is supported by the text.\n"
    "3. Return ONLY valid JSON that matches the schema (no extra keys).\n"
    "4. If a field is not evidenced in the provided text chunk, OMIT the field (preferred).\n"
    "5. Do NOT use empty strings \"\" for missing text; omit the field or use null for optional scalars.\n"
    "6. For arrays/objects: ONLY output [] or {} when the text explicitly indicates "
    "\"none / not applicable / no items\". Otherwise omit the field.\n"
    "7. For relationship-like arrays (lists of nested objects): if the relationship is stated "
    "but details are incomplete, output a minimal reference object using identifier fields "
    "(e.g., {\"name\": \"...\"} or {\"id\": \"...\"}) rather than outputting an empty list.\n"
    "8. Keep identifiers consistent across references (same entity => same identifier value).\n"
)

_EXTRACTION_INSTRUCTIONS_ADVANCED = (
    "1. Carefully analyze the provided document text.\n"
    "2. Extract ALL information that matches the provided schema AND is supported by the text.\n"
    "3. Return ONLY valid JSON that strictly adheres to the schema (no extra keys).\n"
    "4. Never invent facts. If not evidenced, omit the field (preferred) or use null.\n"
    "5. Do NOT use empty strings \"\" for missing text; omit the field or use null.\n"
    "6. For arrays/objects: ONLY output [] or {} when explicitly indicated as empty/none. Otherwise omit.\n"
    "7. Preserve relationships from the source. If a relationship is mentioned, represent it in the JSON.\n"
    "8. For relationship arrays: prefer minimal ID-only references over missing/empty arrays.\n"
    "9. Maintain data consistency across related fields and references (same entity => same identifier).\n"
)

# Legacy constant (kept for backward compatibility)
_EXTRACTION_INSTRUCTIONS = _EXTRACTION_INSTRUCTIONS_STANDARD


_SYSTEM_PROMPT_PARTIAL = (
    "You are an expert data extraction assistant. Your task is to extract "
    "structured information from document pages.\n\n"
    f"Instructions:\n{_EXTRACTION_INSTRUCTIONS}\n"
    "Note: It's okay if the page only contains partial information.\n\n"
    "Important: Your response MUST be valid JSON that can be parsed."
)

_SYSTEM_PROMPT_COMPLETE = (
    "You are an expert data extraction assistant. Your task is to extract "
    "structured information from complete documents.\n\n"
    f"Instructions:\n{_EXTRACTION_INSTRUCTIONS}\n"
    "Be thorough: This is the complete document; try to extract all information.\n\n"
    "Important: Your response MUST be valid JSON that can be parsed."
)


_USER_PROMPT_TEMPLATE = (
    "Extract information from this {document_type}:\n\n"
    "=== {delimiter} ===\n"
    "{markdown_content}\n"
    "=== END {delimiter} ===\n\n"
    "=== TARGET SCHEMA ===\n"
    "{schema_json}\n"
    "=== END SCHEMA ===\n\n"
    "Return ONLY a JSON object that follows the target schema."
)


_USER_PROMPT_CONTEXT_TEMPLATE = (
    "Here is a list of EXISTING ENTITIES (compact registry):\n"
    "{registry_content}\n\n"
    "Extract information from this {document_type}:\n\n"
    "=== {delimiter} ===\n"
    "{markdown_content}\n"
    "=== END {delimiter} ===\n\n"
    "=== TARGET SCHEMA ===\n"
    "{schema_json}\n"
    "=== END SCHEMA ===\n\n"
    "=== DELTA SCHEMA ===\n"
    "{delta_schema_json}\n"
    "=== END DELTA SCHEMA ===\n\n"
    "Return ONLY a JSON object that matches the DELTA SCHEMA."
)


_CONSOLIDATION_PROMPT = """You are a data consolidation expert. Your task is to merge multiple \
partial JSON objects from a document into one single, accurate, and complete JSON object that \
strictly adheres to the provided schema.

You will be given three pieces of information:
1. **SCHEMA**: A JSON schema that the final object MUST validate against.
2. **RAW_JSONS**: A list of partial JSON objects extracted from different document chunks.
3. **DRAFT_JSON**: A JSON object created by a programmatic (non-LLM) merge ("first draft").

Your job is to act as a final reviewer. Use the DRAFT_JSON as a starting point, but critically \
evaluate it against the RAW_JSONS to fix any errors and ensure all data is captured.

**Your Instructions:**
1. **Merge & Deduplicate**: Merge entities intelligently. If the same entity (same identifier fields like \
name/id) appears multiple times, represent it only ONCE in the final output.
2. **Preserve Relationships**: Do NOT remove relationship items that are minimal references (ID-only objects). \
They represent valid links and may be enriched by other chunks.
3. **Remove Phantoms**: Remove phantom/empty objects (no identifier + no meaningful fields) unless fully \
specified in RAW_JSONS.
4. **Ensure Completeness**: Ensure all non-duplicate valid data from all RAW_JSONS is present.
5. **Validate Schema**: The final JSON MUST strictly follow the SCHEMA.

Output only the final, consolidated JSON object. No extra text.

**SCHEMA:**
{schema_json}

**RAW_JSONS:**
{raw_jsons}

**DRAFT_JSON:**
{programmatic_json}

**FINAL CONSOLIDATED JSON:**
"""


# ---------------------------------------------------------------------------
# Methods for formatting and serving the prompts
# ---------------------------------------------------------------------------

def get_extraction_prompt(
    markdown_content: str,
    schema_json: str,
    is_partial: bool = False,
    model_config: ModelConfigLike | None = None,
) -> dict[str, str]:
    """Generate system and user prompts for LLM extraction with model-aware adaptation."""
    if model_config:
        if model_config.capability == ModelCapability.SIMPLE:
            instructions = _EXTRACTION_INSTRUCTIONS_SIMPLE
        elif model_config.capability == ModelCapability.ADVANCED:
            instructions = _EXTRACTION_INSTRUCTIONS_ADVANCED
        else:
            instructions = _EXTRACTION_INSTRUCTIONS_STANDARD
    else:
        instructions = _EXTRACTION_INSTRUCTIONS_STANDARD

    if is_partial:
        system_prompt = (
            "You are an expert data extraction assistant. "
            "Extract structured information from document pages.\n\n"
            f"Instructions:\n{instructions}\n"
            "Note: This is a partial page; incomplete data is expected.\n\n"
            "Important: Your response MUST be valid JSON."
        )
    else:
        system_prompt = (
            "You are an expert data extraction assistant. "
            "Extract structured information from complete documents.\n\n"
            f"Instructions:\n{instructions}\n"
            "Be thorough: Extract all available information.\n\n"
            "Important: Your response MUST be valid JSON."
        )

    document_type = "document page" if is_partial else "complete document"
    delimiter = "DOCUMENT PAGE" if is_partial else "COMPLETE DOCUMENT"

    user_prompt = _USER_PROMPT_TEMPLATE.format(
        document_type=document_type,
        delimiter=delimiter,
        markdown_content=markdown_content,
        schema_json=schema_json,
    )

    return {"system": system_prompt, "user": user_prompt}


def get_context_aware_prompt(
    markdown_content: str,
    schema_json: str,
    registry_content: str,
    delta_schema_json: str,
    is_partial: bool = True,
    model_config: ModelConfigLike | None = None,
) -> dict[str, str]:
    """Generate system and user prompts for context-aware delta extraction."""
    if model_config:
        if model_config.capability == ModelCapability.SIMPLE:
            instructions = _EXTRACTION_INSTRUCTIONS_SIMPLE
        elif model_config.capability == ModelCapability.ADVANCED:
            instructions = _EXTRACTION_INSTRUCTIONS_ADVANCED
        else:
            instructions = _EXTRACTION_INSTRUCTIONS_STANDARD
    else:
        instructions = _EXTRACTION_INSTRUCTIONS_STANDARD

    system_prompt = (
        "You are an expert data extraction assistant. "
        "Extract structured information from document chunks.\n\n"
        f"Instructions:\n{instructions}\n"
        "Context rules:\n"
        "1. You MUST reuse IDs from the registry for existing entities.\n"
        "2. If an entity is new, create a new ID and include minimal evidence from the current chunk.\n"
        "3. Evidence must be a short quote or pointer from the current chunk.\n"
        "4. Use deletes[] for removals; do not use null to delete data.\n"
        "5. For relationships: if a link is stated but attributes are missing, emit an ID-only reference.\n"
        "6. Return ONLY a JSON object matching the DELTA SCHEMA.\n\n"
        "Important: Your response MUST be valid JSON."
    )

    document_type = "document page" if is_partial else "complete document"
    delimiter = "DOCUMENT PAGE" if is_partial else "COMPLETE DOCUMENT"

    user_prompt = _USER_PROMPT_CONTEXT_TEMPLATE.format(
        document_type=document_type,
        delimiter=delimiter,
        markdown_content=markdown_content,
        schema_json=schema_json,
        delta_schema_json=delta_schema_json,
        registry_content=registry_content,
    )

    return {"system": system_prompt, "user": user_prompt}


def get_consolidation_prompt(
    schema_json: str,
    raw_models: list,
    programmatic_model: BaseModel | None = None,
    model_config: ModelConfigLike | None = None,
) -> str | list[str]:
    """Generate the prompt(s) for LLM-based consolidation with model-aware adaptation.

    Args:
        schema_json: The Pydantic model schema.
        raw_models: List of Pydantic models from each extraction batch.
        programmatic_model: Result of the programmatic merge (optional).
        model_config: Optional model configuration for adaptive prompting.

    Returns:
        - str: Single prompt for simple/standard models
        - list[str]: Multiple prompts for advanced models (Chain of Density)
    """
    raw_jsons = "\n\n---\n\n".join(m.model_dump_json(indent=2) for m in raw_models)

    # Simple models: basic merge only
    if model_config and model_config.capability == ModelCapability.SIMPLE:
        return f"""Merge these JSON objects into one, removing duplicates.

SCHEMA:
{schema_json}

OBJECTS TO MERGE:
{raw_jsons}

Output the merged JSON only."""

    # Advanced models: Chain of Density (multi-turn)
    if model_config and model_config.capability == ModelCapability.ADVANCED:
        stage1 = f"""Merge these JSON objects, removing duplicates.

SCHEMA:
{schema_json}

OBJECTS:
{raw_jsons}

Output merged JSON only."""

        # IMPORTANT: keep placeholders consistent with your caller.
        # Use only {stage1_result} as an injection placeholder.
        stage2_template = f"""Review and refine this merged JSON.

SCHEMA:
{schema_json}

MERGED:
{{stage1_result}}

ORIGINALS:
{raw_jsons}

Fix any missing data or errors. Do not drop minimal ID-only relationship items.
Output final JSON only."""

        return [stage1, stage2_template]

    # Standard models: single-pass with draft (default)
    programmatic_json = (
        programmatic_model.model_dump_json(indent=2)
        if programmatic_model
        else "No draft available."
    )

    return _CONSOLIDATION_PROMPT.format(
        schema_json=schema_json,
        raw_jsons=raw_jsons,
        programmatic_json=programmatic_json,
    )
