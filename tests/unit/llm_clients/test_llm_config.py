import pytest
from pydantic import ValidationError

from docling_graph.llm_clients.config import (
    LlmRegistry,
    LlmRuntimeOverrides,
    resolve_effective_model_config,
)


def test_missing_required_model_fields_fail_fast():
    data = {
        "providers": {"openai": {"requires_api_key": False}},
        "models": {
            "gpt-4o": {
                # missing provider + context_limit
                "max_output_tokens": 4096
            }
        },
    }
    with pytest.raises(ValidationError):
        LlmRegistry.model_validate(data)


def test_unknown_keys_fail_validation():
    data = {
        "providers": {"openai": {"requires_api_key": False, "unknown": True}},
        "models": {
            "gpt-4o": {
                "provider": "openai",
                "context_limit": 128000,
                "max_output_tokens": 4096,
                "extra_field": "nope",
            }
        },
    }
    with pytest.raises(ValidationError):
        LlmRegistry.model_validate(data)


def test_defaults_apply_in_effective_config():
    data = {
        "providers": {"openai": {"requires_api_key": False}},
        "models": {
            "gpt-4o": {
                "provider": "openai",
                "context_limit": 128000,
                "max_output_tokens": 4096,
            }
        },
    }
    registry = LlmRegistry.model_validate(data)
    effective = resolve_effective_model_config("openai", "gpt-4o", registry=registry)

    assert effective.generation.temperature == 0.1
    assert effective.generation.max_tokens == 4096
    assert effective.reliability.timeout_s == 300


def test_runtime_overrides_take_precedence():
    data = {
        "providers": {"openai": {"requires_api_key": False}},
        "models": {
            "gpt-4o": {
                "provider": "openai",
                "context_limit": 128000,
                "max_output_tokens": 4096,
            }
        },
    }
    registry = LlmRegistry.model_validate(data)
    overrides = LlmRuntimeOverrides(
        generation={"temperature": 0.3, "max_tokens": 1024},
        reliability={"timeout_s": 10},
        connection={"base_url": "https://proxy.example.com"},
    )

    effective = resolve_effective_model_config(
        "openai", "gpt-4o", registry=registry, overrides=overrides
    )

    assert effective.generation.temperature == 0.3
    assert effective.generation.max_tokens == 1024
    assert effective.reliability.timeout_s == 10
    assert effective.connection.base_url == "https://proxy.example.com"
