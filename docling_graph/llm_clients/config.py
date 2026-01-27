"""
Typed LLM configuration registry and resolver.

This module defines a strict, validated schema for:
- Provider connection/auth settings
- Routing (provider/model selection)
- Generation defaults
- Reliability defaults

It also resolves an "effective" model config by merging:
provider defaults -> model overrides -> runtime overrides.
"""

from __future__ import annotations

import logging
import os
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator
from typing_extensions import Self

from ..exceptions import ConfigurationError

logger = logging.getLogger(__name__)

# Load .env for provider credentials
load_dotenv()


class ModelCapability(Enum):
    """Model capability tiers for adaptive extraction strategies."""

    SIMPLE = "simple"
    STANDARD = "standard"
    ADVANCED = "advanced"


class GenerationDefaults(BaseModel):
    """Default generation parameters."""

    model_config = ConfigDict(extra="forbid")

    temperature: float = 0.1
    max_tokens: int | None = None
    top_p: float | None = None
    top_k: int | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    stop: list[str] | None = None
    seed: int | None = None
    min_tokens: int | None = None
    repetition_penalty: float | None = None
    decoding_method: str | None = None


class GenerationOverrides(BaseModel):
    """Overrides for generation parameters (None means 'no override')."""

    model_config = ConfigDict(extra="forbid")

    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    top_k: int | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    stop: list[str] | None = None
    seed: int | None = None
    min_tokens: int | None = None
    repetition_penalty: float | None = None
    decoding_method: str | None = None


class BackoffDefaults(BaseModel):
    """Retry backoff configuration."""

    model_config = ConfigDict(extra="forbid")

    initial_s: float = 1.0
    max_s: float = 30.0
    multiplier: float = 2.0
    jitter: float = 0.1


class BackoffOverrides(BaseModel):
    """Override retry backoff configuration."""

    model_config = ConfigDict(extra="forbid")

    initial_s: float | None = None
    max_s: float | None = None
    multiplier: float | None = None
    jitter: float | None = None


class ReliabilityDefaults(BaseModel):
    """Default reliability parameters."""

    model_config = ConfigDict(extra="forbid")

    timeout_s: int = 300
    max_retries: int = 2
    backoff: BackoffDefaults = Field(default_factory=BackoffDefaults)


class ReliabilityOverrides(BaseModel):
    """Overrides for reliability parameters."""

    model_config = ConfigDict(extra="forbid")

    timeout_s: int | None = None
    max_retries: int | None = None
    backoff: BackoffOverrides | None = None


class ProviderConnection(BaseModel):
    """Provider connection and auth settings."""

    model_config = ConfigDict(extra="forbid")

    api_key: SecretStr | None = None
    api_key_env: str | None = None
    base_url: str | None = None
    base_url_env: str | None = None
    organization: str | None = None
    project_id: str | None = None
    project_id_env: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)


class ConnectionOverrides(BaseModel):
    """Runtime connection overrides."""

    model_config = ConfigDict(extra="forbid")

    api_key: SecretStr | None = None
    base_url: str | None = None
    organization: str | None = None
    project_id: str | None = None
    headers: dict[str, str] | None = None


class ProviderDefinition(BaseModel):
    """Provider configuration with defaults."""

    model_config = ConfigDict(extra="forbid")

    requires_api_key: bool = False
    requires_project_id: bool = False
    connection: ProviderConnection = Field(default_factory=ProviderConnection)
    generation_defaults: GenerationDefaults = Field(default_factory=GenerationDefaults)
    reliability_defaults: ReliabilityDefaults = Field(default_factory=ReliabilityDefaults)
    tokenizer: str = "sentence-transformers/all-MiniLM-L6-v2"
    content_ratio: float = 0.8
    merge_threshold: float = 0.85
    rate_limit_rpm: int | None = None
    supports_batching: bool = True

    @field_validator("content_ratio", "merge_threshold")
    @classmethod
    def _validate_ratio(cls, value: float) -> float:
        if value <= 0 or value > 1:
            raise ValueError("Must be between 0 (exclusive) and 1 (inclusive).")
        return value


class ModelDefinition(BaseModel):
    """Definition for a single model."""

    model_config = ConfigDict(extra="forbid")

    provider: str
    model: str | None = None
    context_limit: int
    max_output_tokens: int = 4096
    capability: ModelCapability = ModelCapability.STANDARD
    description: str = ""
    notes: str = ""
    generation: GenerationOverrides = Field(default_factory=GenerationOverrides)
    reliability: ReliabilityOverrides = Field(default_factory=ReliabilityOverrides)

    @field_validator("context_limit", "max_output_tokens")
    @classmethod
    def _validate_limits(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("Must be positive.")
        return value


class LlmRegistry(BaseModel):
    """Registry of provider and model definitions."""

    model_config = ConfigDict(extra="forbid")

    providers: dict[str, ProviderDefinition]
    models: dict[str, ModelDefinition]

    @model_validator(mode="after")
    def _normalize_and_validate(self) -> Self:
        normalized_providers: dict[str, ProviderDefinition] = {}
        for provider_id, provider in self.providers.items():
            normalized_providers[provider_id.lower()] = provider
        self.providers = normalized_providers

        for model_id, model in self.models.items():
            model.provider = model.provider.lower()
            if model.model is None:
                model.model = model_id
            if model.provider not in self.providers:
                raise ValueError(
                    f"Model '{model_id}' references unknown provider '{model.provider}'."
                )
        return self

    def get_provider(self, provider_id: str) -> ProviderDefinition | None:
        return self.providers.get(provider_id.lower())

    def get_model(self, model_id: str) -> ModelDefinition | None:
        return self.models.get(model_id)


class ResolvedConnection(BaseModel):
    """Connection values after env + override resolution."""

    model_config = ConfigDict(extra="forbid")

    api_key: SecretStr | None = None
    base_url: str | None = None
    organization: str | None = None
    project_id: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)


class EffectiveModelConfig(BaseModel):
    """Fully-resolved runtime configuration for a model."""

    model_config = ConfigDict(extra="forbid")

    model_id: str
    provider_id: str
    provider_model: str
    context_limit: int
    max_output_tokens: int
    capability: ModelCapability
    generation: GenerationDefaults
    reliability: ReliabilityDefaults
    connection: ResolvedConnection
    tokenizer: str
    content_ratio: float
    merge_threshold: float
    rate_limit_rpm: int | None
    supports_batching: bool

    @property
    def supports_chain_of_density(self) -> bool:
        return self.capability == ModelCapability.ADVANCED

    @property
    def requires_strict_schema(self) -> bool:
        return self.capability == ModelCapability.SIMPLE


class LlmRuntimeOverrides(BaseModel):
    """Runtime overrides for a resolved model configuration."""

    model_config = ConfigDict(extra="forbid")

    generation: GenerationOverrides = Field(default_factory=GenerationOverrides)
    reliability: ReliabilityOverrides = Field(default_factory=ReliabilityOverrides)
    connection: ConnectionOverrides = Field(default_factory=ConnectionOverrides)


_registry: LlmRegistry | None = None


def _merge_generation(
    base: GenerationDefaults, model_overrides: GenerationOverrides, runtime: GenerationOverrides
) -> GenerationDefaults:
    data = base.model_dump()
    data.update(model_overrides.model_dump(exclude_none=True))
    data.update(runtime.model_dump(exclude_none=True))
    return GenerationDefaults(**data)


def _merge_reliability(
    base: ReliabilityDefaults, model_overrides: ReliabilityOverrides, runtime: ReliabilityOverrides
) -> ReliabilityDefaults:
    data = base.model_dump()
    data.update(model_overrides.model_dump(exclude_none=True))
    data.update(runtime.model_dump(exclude_none=True))

    if model_overrides.backoff or runtime.backoff:
        backoff_data = base.backoff.model_dump()
        if model_overrides.backoff:
            backoff_data.update(model_overrides.backoff.model_dump(exclude_none=True))
        if runtime.backoff:
            backoff_data.update(runtime.backoff.model_dump(exclude_none=True))
        data["backoff"] = BackoffDefaults(**backoff_data)

    return ReliabilityDefaults(**data)


def _resolve_connection(
    provider: ProviderDefinition, overrides: ConnectionOverrides | None, provider_id: str
) -> ResolvedConnection:
    connection = provider.connection

    def _env_value(key: str | None) -> str | None:
        if not key:
            return None
        value = os.getenv(key)
        return value or None

    api_key = connection.api_key or (
        SecretStr(_env_value(connection.api_key_env))
        if _env_value(connection.api_key_env)
        else None
    )
    base_url = connection.base_url or _env_value(connection.base_url_env)
    organization = connection.organization
    project_id = connection.project_id or _env_value(connection.project_id_env)
    headers = dict(connection.headers)

    if overrides:
        if overrides.api_key is not None:
            api_key = overrides.api_key
        if overrides.base_url is not None:
            base_url = overrides.base_url
        if overrides.organization is not None:
            organization = overrides.organization
        if overrides.project_id is not None:
            project_id = overrides.project_id
        if overrides.headers:
            headers.update(overrides.headers)

    if provider.requires_api_key and not api_key:
        raise ConfigurationError(
            "Missing required API key for provider",
            details={"provider": provider_id, "env_hint": connection.api_key_env},
        )
    if provider.requires_project_id and not project_id:
        raise ConfigurationError(
            "Missing required project ID for provider",
            details={"provider": provider_id, "env_hint": connection.project_id_env},
        )

    return ResolvedConnection(
        api_key=api_key,
        base_url=base_url,
        organization=organization,
        project_id=project_id,
        headers=headers,
    )


def load_registry_from_path(config_path: Path) -> LlmRegistry:
    if not config_path.exists():
        raise FileNotFoundError(f"LLM registry file not found: {config_path}")
    with open(config_path) as f:
        data = yaml.safe_load(f) or {}
    return LlmRegistry.model_validate(data)


def set_registry(registry: LlmRegistry) -> None:
    global _registry
    _registry = registry


def get_registry() -> LlmRegistry:
    global _registry
    if _registry is None:
        default_path = Path(__file__).parent / "models.yaml"
        _registry = load_registry_from_path(default_path)
    return _registry


def resolve_effective_model_config(
    provider_id: str,
    model_id: str,
    overrides: LlmRuntimeOverrides | None = None,
    registry: LlmRegistry | None = None,
) -> EffectiveModelConfig:
    registry = registry or get_registry()
    provider = registry.get_provider(provider_id)
    if not provider:
        raise ConfigurationError(
            "Unknown LLM provider",
            details={"provider": provider_id, "available": sorted(registry.providers.keys())},
        )
    model = registry.get_model(model_id)
    if not model:
        raise ConfigurationError(
            "Unknown LLM model",
            details={"model": model_id, "available": sorted(registry.models.keys())},
        )
    if model.provider != provider_id.lower():
        raise ConfigurationError(
            "Model provider mismatch",
            details={"model": model_id, "model_provider": model.provider, "provider": provider_id},
        )

    if overrides is None:
        overrides = LlmRuntimeOverrides()
    elif isinstance(overrides, dict):
        overrides = LlmRuntimeOverrides(**overrides)

    generation = _merge_generation(
        provider.generation_defaults, model.generation, overrides.generation
    )
    reliability = _merge_reliability(
        provider.reliability_defaults, model.reliability, overrides.reliability
    )
    connection = _resolve_connection(provider, overrides.connection, provider_id)

    if generation.max_tokens is None:
        generation = generation.model_copy(update={"max_tokens": model.max_output_tokens})
    if generation.max_tokens > model.max_output_tokens:
        raise ConfigurationError(
            "max_tokens exceeds model limit",
            details={
                "model": model_id,
                "max_tokens": generation.max_tokens,
                "model_max_output_tokens": model.max_output_tokens,
            },
        )

    return EffectiveModelConfig(
        model_id=model_id,
        provider_id=provider_id.lower(),
        provider_model=model.model or model_id,
        context_limit=model.context_limit,
        max_output_tokens=model.max_output_tokens,
        capability=model.capability,
        generation=generation,
        reliability=reliability,
        connection=connection,
        tokenizer=provider.tokenizer,
        content_ratio=provider.content_ratio,
        merge_threshold=provider.merge_threshold,
        rate_limit_rpm=provider.rate_limit_rpm,
        supports_batching=provider.supports_batching,
    )


def get_provider_config(provider_id: str) -> ProviderDefinition | None:
    return get_registry().get_provider(provider_id)


def get_model_config(provider_id: str, model_id: str) -> ModelDefinition | None:
    model = get_registry().get_model(model_id)
    if model and model.provider == provider_id.lower():
        return model
    return None


def get_tokenizer_for_provider(provider_id: str) -> str:
    provider = get_registry().get_provider(provider_id)
    if provider:
        return provider.tokenizer
    return "sentence-transformers/all-MiniLM-L6-v2"


def get_recommended_chunk_size(provider: str, model: str, schema_size: int = 0) -> int:
    registry = get_registry()
    model_config = registry.get_model(model)
    provider_config = registry.get_provider(provider)
    if not model_config or not provider_config:
        return 5120

    if schema_size > 10000:
        output_ratio = 0.8
    elif schema_size > 5000:
        output_ratio = 0.5
    elif schema_size > 0:
        output_ratio = 0.3
    else:
        output_ratio = 0.4

    system_prompt_tokens = 500
    safety_buffer = 0.8

    max_safe_chunk = int(model_config.max_output_tokens / output_ratio * safety_buffer)
    max_by_context = int((model_config.context_limit - system_prompt_tokens) * 0.7)
    chunk_size = min(max_safe_chunk, max_by_context)
    return max(1024, chunk_size)


def detect_model_capability(
    context_limit: int, model_name: str = "", max_output_tokens: int | None = None
) -> ModelCapability:
    name_lower = model_name.lower()

    if any(size in name_lower for size in ["1b", "350m", "500m", "2b", "3b"]):
        logger.info(
            f"Detected SIMPLE capability from model name: {model_name} (small parameter count)"
        )
        return ModelCapability.SIMPLE

    if any(size in name_lower for size in ["70b", "65b", "405b"]):
        logger.info(
            f"Detected ADVANCED capability from model name: {model_name} (large parameter count)"
        )
        return ModelCapability.ADVANCED

    if max_output_tokens is not None:
        if max_output_tokens <= 2048:
            logger.info(
                f"Detected SIMPLE capability from max_output_tokens: {max_output_tokens} "
                "(limited output capacity)"
            )
            return ModelCapability.SIMPLE
        if max_output_tokens <= 4096:
            logger.info(f"Detected STANDARD capability from max_output_tokens: {max_output_tokens}")
            return ModelCapability.STANDARD
        logger.info(
            f"Detected ADVANCED capability from max_output_tokens: {max_output_tokens} "
            "(high output capacity)"
        )
        return ModelCapability.ADVANCED

    if context_limit <= 4096:
        logger.warning(
            f"Detected SIMPLE capability from context_limit: {context_limit} "
            "(fallback heuristic, may be inaccurate)"
        )
        return ModelCapability.SIMPLE
    if context_limit <= 32768:
        logger.warning(
            f"Detected STANDARD capability from context_limit: {context_limit} "
            "(fallback heuristic, may be inaccurate)"
        )
        return ModelCapability.STANDARD
    logger.warning(
        f"Detected ADVANCED capability from context_limit: {context_limit} "
        "(fallback heuristic, may be inaccurate for small models with large contexts)"
    )
    return ModelCapability.ADVANCED
