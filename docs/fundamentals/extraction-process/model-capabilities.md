# Model Capabilities

## Overview

Docling-Graph automatically classifies models into **three capability tiers** to optimize extraction quality and performance. This intelligent system adapts prompts, consolidation strategies, and processing approaches based on model size and capabilities.

**What You'll Learn:**
- Model capability tier system
- Automatic model detection
- Adaptive behavior per tier
- Performance implications
- Supported models by tier

---

## Model Capability Tiers

### Tier Classification

| Tier | Model Size | Characteristics | Use Cases |
|------|-----------|-----------------|-----------|
| **SIMPLE** | 1B-7B params | Fast, basic understanding | Simple forms, invoices, quick extraction |
| **STANDARD** | 7B-13B params | Balanced speed/accuracy | General documents, contracts |
| **ADVANCED** | 13B+ params | High accuracy, complex reasoning | Rheology researchs, legal documents, complex analysis |

---

## Automatic Detection

Models are automatically classified based on their name and known characteristics:

```python
from docling_graph.llm_clients.config import detect_model_capability, ModelCapability

# Automatic detection examples
capability = detect_model_capability("llama-3.1-8b")
print(capability)  # ModelCapability.STANDARD

capability = detect_model_capability("gpt-4-turbo")
print(capability)  # ModelCapability.ADVANCED

capability = detect_model_capability("granite-4.0-1b")
print(capability)  # ModelCapability.SIMPLE
```

### Detection Logic

The system uses a **3-tier priority system** for accurate capability detection:

**Priority 1: Model Name Pattern Matching** (Most Reliable)
- Detects parameter count from model name (e.g., "1b", "8b", "70b")
- Small models (1B-3B) → SIMPLE
- Large models (70B+) → ADVANCED

**Priority 2: Max Output Tokens** (Better Proxy)
- Uses `max_new_tokens` as capability indicator
- ≤2048 tokens → SIMPLE (limited output capacity)
- ≤4096 tokens → STANDARD
- \>4096 tokens → ADVANCED

**Priority 3: Context Window** (Fallback)

- Uses context limit only when other signals unavailable
- Includes warnings when using this heuristic

!!! note "Context Window Caveat"
    Modern small models can have large contexts (e.g., Granite 1B with 128K)

**Registry Lookup:**
- For known models in `models.yaml`, explicit capability is always used
- Fallback detection only applies to unregistered models

---

## Adaptive Behavior

The extraction system automatically adapts based on model capability:

### SIMPLE Models (1B-7B)

**Optimized for Speed**

- ✅ **Minimal Instructions**: Focused, concise prompts
- ✅ **Basic Consolidation**: Simple programmatic merging
- ✅ **Fast Processing**: Optimized for throughput
- ✅ **Lower Memory**: Efficient resource usage

**Best For:**
- Simple forms and invoices
- Structured data extraction
- High-volume processing
- Resource-constrained environments

**Example Models:**
- `ibm-granite/granite-4.0-1b`
- `meta-llama/Llama-3.2-1B`
- `numind/NuExtract-2.0-2B`

```python
from docling_graph import run_pipeline, PipelineConfig

config = PipelineConfig(
    source="invoice.pdf",
    template="templates.BillingDocument",
    backend="llm",
    inference="local",
    model_override="ibm-granite/granite-4.0-1b"  # SIMPLE tier
)
run_pipeline(config)
```

---

### STANDARD Models (7B-13B)

**Balanced Performance**

- ✅ **Balanced Instructions**: Moderate detail level
- ✅ **Standard Consolidation**: Programmatic + optional LLM
- ✅ **Good Accuracy**: Reliable for most documents
- ✅ **Reasonable Speed**: Good throughput

**Best For:**
- General business documents
- Multi-page contracts
- Standard extraction tasks
- Production workloads

**Example Models:**
- `meta-llama/Llama-3.1-8B`
- `mistralai/Mistral-7B-v0.1`
- `numind/NuExtract-2.0-8B`

```python
config = PipelineConfig(
    source="contract.pdf",
    template="templates.Contract",
    backend="llm",
    inference="local",
    model_override="meta-llama/Llama-3.1-8B"  # STANDARD tier
)
run_pipeline(config)
```

---

### ADVANCED Models (13B+)

**Maximum Accuracy**

- ✅ **Detailed Instructions**: Comprehensive prompts
- ✅ **Chain of Density**: Multi-turn consolidation
- ✅ **High Accuracy**: Best extraction quality
- ✅ **Complex Reasoning**: Handles nuanced content

**Best For:**
- Rheology researchs
- Legal documents
- Complex technical content
- High-accuracy requirements

**Example Models:**
- `gpt-4-turbo` (OpenAI)
- `claude-3.5-sonnet` (Anthropic)
- `gemini-2.5-flash` (Google)
- `mistral-large-latest` (Mistral)

```python
config = PipelineConfig(
    source="research_paper.pdf",
    template="templates.ScholarlyRheologyPaper",
    backend="llm",
    inference="remote",
    provider_override="openai",
    model_override="gpt-4-turbo",  # ADVANCED tier
    llm_consolidation=True  # Enable Chain of Density
)
run_pipeline(config)
```

---

## Chain of Density Consolidation

**ADVANCED models** automatically use Chain of Density consolidation when `llm_consolidation=True`:

### Three-Step Refinement

1. **Initial Extraction**: Extract from raw document chunks
2. **Refinement**: Merge with programmatic consolidation
3. **Final Polish**: LLM refines for consistency and completeness

```python
# Chain of Density is automatic for ADVANCED models
config = PipelineConfig(
    source="complex_document.pdf",
    template="templates.ComplexTemplate",
    backend="llm",
    inference="remote",
    model_override="gpt-4-turbo",  # ADVANCED tier
    llm_consolidation=True,  # Enables Chain of Density
    processing_mode="many-to-one"
)
run_pipeline(config)
```

### When to Use Chain of Density

✅ **Use When:**
- Document has conflicting information
- Need highest accuracy
- Complex narrative content
- Using ADVANCED tier models

❌ **Skip When:**
- Simple structured data
- Speed is critical
- Using SIMPLE/STANDARD models
- Budget constraints

---

## Supported Models

Model capability tiers are derived from context limits and output limits at runtime.
The project registry only keeps guardrails (context/output limits) and provider
defaults; LiteLLM provides model metadata when available.

To see the current model list, review `docling_graph/llm_clients/models.yaml`.

---

## Performance Implications

### Speed vs Accuracy Trade-off

| Tier | Speed | Accuracy | Memory | Cost |
|------|-------|----------|--------|------|
| **SIMPLE** | ⚡⚡⚡ Very Fast | 🟡 Moderate | 2-4 GB | $ Low |
| **STANDARD** | ⚡⚡ Fast | 🟢 Good | 8-16 GB | $$ Medium |
| **ADVANCED** | ⚡ Moderate | 💎 Excellent | 16-32 GB | $$$ High |

### Benchmark Results

**Document**: 10-page contract

| Model Tier | Time | Accuracy | Tokens Used |
|-----------|------|----------|-------------|
| SIMPLE (1B) | 15s | 85% | 2,500 |
| STANDARD (8B) | 45s | 92% | 3,200 |
| ADVANCED (GPT-4) | 90s | 97% | 4,100 |

---

## Best Practices

### 👍 Match Tier to Task Complexity

```python
# ✅ Good - Simple task, simple model
config = PipelineConfig(
    source="invoice.pdf",
    template="templates.BillingDocument",
    model_override="granite-4.0-1b"  # SIMPLE tier
)

# ✅ Good - Complex task, advanced model
config = PipelineConfig(
    source="research_paper.pdf",
    template="templates.ScholarlyRheologyPaper",
    model_override="gpt-4-turbo"  # ADVANCED tier
)

# ❌ Avoid - Overkill
config = PipelineConfig(
    source="simple_form.pdf",
    template="templates.SimpleForm",
    model_override="gpt-4-turbo"  # Unnecessary
)
```

### 👍 Start Small, Scale Up

```python
# Start with SIMPLE tier
config = PipelineConfig(
    source="document.pdf",
    template="templates.MyTemplate",
    model_override="granite-4.0-1b"
)
result = run_pipeline(config)

# If accuracy insufficient, upgrade to STANDARD
if accuracy_check(result) < 0.90:
    config.model_override = "llama-3.1-8b"
    result = run_pipeline(config)

# If still insufficient, upgrade to ADVANCED
if accuracy_check(result) < 0.95:
    config.model_override = "gpt-4-turbo"
    config.llm_consolidation = True
    result = run_pipeline(config)
```

### 👍 Use Chain of Density Wisely

```python
# ✅ Good - Complex document with ADVANCED model
config = PipelineConfig(
    source="complex_legal.pdf",
    template="templates.LegalDocument",
    model_override="gpt-4-turbo",  # ADVANCED
    llm_consolidation=True  # Enable Chain of Density
)

# ❌ Avoid - Chain of Density with SIMPLE model
config = PipelineConfig(
    source="invoice.pdf",
    template="templates.BillingDocument",
    model_override="granite-4.0-1b",  # SIMPLE
    llm_consolidation=True  # Won't use Chain of Density
)
```

---

## Troubleshooting

### 🐛 Model Not Detected Correctly

**Solution:**
```python
# Override automatic detection
from docling_graph.llm_clients.config import ModelCapability

# Force specific capability
config = PipelineConfig(
    source="document.pdf",
    template="templates.MyTemplate",
    model_override="custom-model-7b",
    # Note: Capability override not directly exposed
    # Model will be detected based on name patterns
)
```

### 🐛 Poor Accuracy with SIMPLE Model

**Solution:**
```python
# Upgrade to STANDARD or ADVANCED tier
config = PipelineConfig(
    source="document.pdf",
    template="templates.MyTemplate",
    model_override="llama-3.1-8b"  # STANDARD tier
)
```

### 🐛 Slow Processing with ADVANCED Model

**Solution:**
```python
# Try STANDARD tier first
config = PipelineConfig(
    source="document.pdf",
    template="templates.MyTemplate",
    model_override="mistral-small-latest"  # STANDARD tier
)
```

---

## Next Steps

Now that you understand model capabilities:

1. **[Extraction Backends →](extraction-backends.md)** - Learn about LLM and VLM backends
2. **[Model Configuration →](../pipeline-configuration/model-configuration.md)** - Configure model settings
3. **[Performance Tuning →](../../usage/advanced/performance-tuning.md)** - Optimize for your use case