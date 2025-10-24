"""
Utility functions for document extraction.
"""

import copy
from typing import Dict, Any, List

# Heuristic for token calculation (chars / 3.5 is a rough proxy)
# You can replace this with a proper tokenizer like tiktoken if you want more accuracy.
TOKEN_CHAR_RATIO = 3.5


def deep_merge_dicts(target: Dict[str, Any], source: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merges a 'source' dict into a 'target' dict.

    Merge behavior:
    - New keys from source are added to target
    - Lists are concatenated with smart deduplication
    - Nested dicts are recursively merged
    - Scalars are overwritten only if source has non-empty value
    - None, empty strings, empty lists, and empty dicts are ignored

    Args:
        target (Dict[str, Any]): The dictionary to merge into (modified in place)
        source (Dict[str, Any]): The dictionary to merge from

    Returns:
        Dict[str, Any]: The merged dictionary (same as target)
    """
    for key, source_value in source.items():
        # Skip empty values
        if source_value is None or source_value == "" or source_value == [] or source_value == {}:
            continue

        if key not in target:
            # New key, just add it
            target[key] = copy.deepcopy(source_value)
        else:
            # Key exists, merge intelligently
            target_value = target[key]

            if isinstance(target_value, list) and isinstance(source_value, list):
                # --- Merge Lists ---
                target[key] = merge_lists(target_value, source_value)

            elif isinstance(target_value, dict) and isinstance(source_value, dict):
                # --- Recurse for Dictionaries ---
                deep_merge_dicts(target_value, source_value)

            else:
                # --- Overwrite Scalar/Other ---
                # Only overwrite if target is empty or None
                if target_value is None or target_value == "" or target_value == [] or target_value == {}:
                    target[key] = source_value
                # Otherwise keep target value (first non-empty wins)

    return target


def merge_lists(list1: List[Any], list2: List[Any]) -> List[Any]:
    """
    Merge two lists with intelligent deduplication.

    For simple types (str, int, float, bool): uses set-based deduplication
    For dicts: merges dicts with same 'id' field, or appends unique dicts
    For other types: simple concatenation without duplicates

    Args:
        list1 (List[Any]): First list
        list2 (List[Any]): Second list to merge in

    Returns:
        List[Any]: Merged list
    """
    if not list2:
        return list1
    if not list1:
        return copy.deepcopy(list2)

    # Check if all items are simple hashable types
    try:
        # Try set-based deduplication for simple types
        existing_items = set(list1)
        result = list1.copy()
        for item in list2:
            if item not in existing_items:
                result.append(item)
                existing_items.add(item)
        return result
    except TypeError:
        # Contains unhashable types (like dicts), need smarter merging
        pass

    # Handle list of dicts with potential 'id' field
    if all(isinstance(item, dict) for item in list1 + list2):
        return merge_dict_lists(list1, list2)

    # Fallback: simple append without smart deduplication
    result = list1.copy()
    for item in list2:
        if item not in result:  # Simple equality check
            result.append(item)
    return result


def merge_dict_lists(list1: List[Dict], list2: List[Dict]) -> List[Dict]:
    """
    Merge two lists of dictionaries.

    If dicts have an 'id' field, merge dicts with matching ids.
    Otherwise, append unique dicts.

    Args:
        list1 (List[Dict]): First list of dicts
        list2 (List[Dict]): Second list of dicts

    Returns:
        List[Dict]: Merged list of dicts
    """
    # Check if dicts use 'id' for identification
    has_ids = any('id' in d for d in list1 + list2)

    if has_ids:
        # Merge by id
        merged_by_id = {}

        # Add all from list1
        for item in list1:
            item_id = item.get('id')
            if item_id:
                merged_by_id[item_id] = copy.deepcopy(item)
            else:
                # No id, just include as-is (will be appended at end)
                pass

        # Merge from list2
        items_without_id = []
        for item in list2:
            item_id = item.get('id')
            if item_id:
                if item_id in merged_by_id:
                    # Merge this dict with existing one
                    deep_merge_dicts(merged_by_id[item_id], item)
                else:
                    merged_by_id[item_id] = copy.deepcopy(item)
            else:
                items_without_id.append(item)

        # Reconstruct list
        result = list(merged_by_id.values())

        # Add items without ids (from both lists)
        for item in list1:
            if 'id' not in item or not item.get('id'):
                result.append(item)
        result.extend(items_without_id)

        return result
    else:
        # No ids, just append unique dicts
        result = list1.copy()
        for item in list2:
            if item not in result:
                result.append(copy.deepcopy(item))
        return result


def merge_pydantic_models(models: List[Any], template_class: type) -> Any:
    """
    Merge multiple Pydantic models into a single consolidated model.

    Args:
        models (List[Any]): List of Pydantic model instances to merge
        template_class (type): The Pydantic model class

    Returns:
        Any: A single merged Pydantic model instance
    """
    if not models:
        return None

    if len(models) == 1:
        return models[0]

    # Convert all models to dicts
    merged_dict = {}
    for model in models:
        model_dict = model.model_dump()
        deep_merge_dicts(merged_dict, model_dict)

    # Convert back to Pydantic model
    return template_class(**merged_dict)
