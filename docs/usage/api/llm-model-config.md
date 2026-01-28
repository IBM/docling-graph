# LLM Model Configuration

This guide explains how to define models, override settings, and inspect the
resolved (effective) LLM configuration at runtime.

## Add a New Model Definition

Edit the registry YAML (default: `docling_graph/llm_clients/models.yaml`) or
provide your own via `llm_registry_path`.

Minimal model entry (guardrails + LiteLLM routing):

```yaml
models:
  my-model-id:
    provider: openai
    # Optional guardrails (fallback to LiteLLM metadata if omitted)
    context_limit: 128000
    max_output_tokens: 4096
    # Optional: override LiteLLM model string if it differs
    # litellm_model: openai/gpt-4o
```

Provider entries define auth, defaults, and transport:

```yaml
providers:
  openai:
    requires_api_key: true
    connection:
      api_key_env: OPENAI_API_KEY
    generation_defaults:
      temperature: 0.1
    reliability_defaults:
      timeout_s: 300
```

## Override via Python (API)

You can override generation, reliability, or connection settings at runtime:

```python
from docling_graph import PipelineConfig

config = PipelineConfig(
    source="doc.pdf",
    template="templates.BillingDocument",
    backend="llm",
    inference="remote",
    model_override="gpt-4o",
    provider_override="openai",
    llm_overrides={
        "generation": {"temperature": 0.2, "max_tokens": 2048},
        "reliability": {"timeout_s": 120, "max_retries": 1},
    },
)
```

## Override via Config File

In `config.yaml`, use the same `llm_overrides` shape:

```yaml
models:
  llm:
    remote:
      provider: openai
      model: gpt-4o

llm_overrides:
  generation:
    temperature: 0.2
    max_tokens: 2048
  reliability:
    timeout_s: 120
```

To use a custom registry:

```yaml
llm_registry_path: "/path/to/custom_llm_registry.yaml"
```

## Override via CLI

Common overrides:

```bash
docling-graph convert doc.pdf --template templates.BillingDocument \
  --provider openai --model gpt-4o \
  --llm-temperature 0.2 --llm-max-tokens 2048 --llm-timeout 120
```

## View the Resolved Config

CLI:

```bash
docling-graph convert doc.pdf --template templates.BillingDocument --show-llm-config
```

Python:

```python
from docling_graph.llm_clients.config import resolve_effective_model_config

effective = resolve_effective_model_config("openai", "gpt-4o")
print(effective.model_dump())
```
