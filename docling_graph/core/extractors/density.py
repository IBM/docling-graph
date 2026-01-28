"""
Schema-based density estimation for output token budgeting.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Type, Union, cast, get_args, get_origin

from pydantic import BaseModel

VERBATIM_KEYWORDS = ("provenance", "raw_text", "excerpt", "quote")
COMPLEX_LIST_PENALTY = 0.3
PRIMITIVE_LIST_STR_PENALTY = 0.2
PRIMITIVE_LIST_ENUM_PENALTY = 0.1
VERBATIM_PENALTY = 0.5
BASE_DENSITY = 1.0
MIN_DENSITY = 1.2
MAX_DENSITY = 3.0


def _is_basemodel_type(tp: Any) -> bool:
    return isinstance(tp, type) and issubclass(tp, BaseModel)


def _is_enum_type(tp: Any) -> bool:
    return isinstance(tp, type) and issubclass(tp, Enum)


def calculate_dynamic_density(
    template: Type[BaseModel],
    max_depth: int = 5,
) -> float:
    visited: set[type] = set()
    density = BASE_DENSITY

    def walk(model: Type[BaseModel], depth: int) -> None:
        nonlocal density
        if depth > max_depth:
            return
        if model in visited:
            return
        visited.add(model)

        for field_name, field in model.model_fields.items():
            lowered = field_name.lower()
            if any(keyword in lowered for keyword in VERBATIM_KEYWORDS):
                density += VERBATIM_PENALTY

            annotation = field.annotation
            origin = get_origin(annotation)
            args = get_args(annotation)

            if origin is list:
                if not args:
                    continue
                item_type = args[0]
                if _is_basemodel_type(item_type):
                    density += COMPLEX_LIST_PENALTY
                    walk(item_type, depth + 1)
                elif _is_enum_type(item_type):
                    density += PRIMITIVE_LIST_ENUM_PENALTY
                elif item_type is str:
                    density += PRIMITIVE_LIST_STR_PENALTY
                continue

            if origin is None:
                if _is_basemodel_type(annotation):
                    walk(cast(Type[BaseModel], annotation), depth + 1)
                continue

            if origin is tuple or origin is set:
                for arg in args:
                    if _is_basemodel_type(arg):
                        walk(arg, depth + 1)

            if origin is None:
                continue

            if origin is type(None):
                continue

            if origin is Union:
                for arg in args:
                    arg_origin = get_origin(arg)
                    arg_args = get_args(arg)
                    if _is_basemodel_type(arg):
                        walk(arg, depth + 1)
                        continue
                    if arg_origin is list and arg_args:
                        item_type = arg_args[0]
                        if _is_basemodel_type(item_type):
                            density += COMPLEX_LIST_PENALTY
                            walk(item_type, depth + 1)
                        elif _is_enum_type(item_type):
                            density += PRIMITIVE_LIST_ENUM_PENALTY
                        elif item_type is str:
                            density += PRIMITIVE_LIST_STR_PENALTY

    walk(template, 1)
    density = max(MIN_DENSITY, min(MAX_DENSITY, density))
    return density
