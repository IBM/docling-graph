# LLM Clients API

!!! note
    LiteLLM is the only client path for LLM calls.

## Overview

**Module:** `docling_graph.llm_clients`

All LLM calls go through `BaseLlmClient.get_json_response()` via the LiteLLM-backed
client. This preserves the existing extraction/consolidation pipeline while
standardizing provider differences through LiteLLM.

---

## LiteLLMClient (Default)

`LiteLLMClient` wraps `litellm.completion()` and uses OpenAI-style parameters with
`drop_params=True` to avoid provider-specific branching.

### Example

```python
from docling_graph.llm_clients import get_client
from docling_graph.llm_clients.config import resolve_effective_model_config

effective = resolve_effective_model_config(
    "mistral",
    "mistral-large-latest",
    overrides={"generation": {"max_tokens": 4096}},
)
client_class = get_client("mistral")
client = client_class(model_config=effective)

result = client.get_json_response(
    prompt={"system": "Extract data", "user": "Alice is a manager"},
    schema_json="{}",
)
```

### JSON Mode

JSON/Structured Outputs are requested by default via `response_format`, with
`ResponseHandler` providing a fallback if the model output is not strictly JSON.

---

## Custom LLM Clients

You can supply a custom LLM client (for bespoke API calls, chat templates, etc.)
as long as it implements `LLMClientProtocol` and provides `get_json_response()`.

```python
from docling_graph import run_pipeline
from docling_graph.protocols import LLMClientProtocol

class MyCustomClient(LLMClientProtocol):
    def get_json_response(self, prompt, schema_json):
        return {"custom": "response"}

config = {
    "source": "doc.pdf",
    "template": "templates.BillingDocument",
    "backend": "llm",
    "inference": "remote",
    "llm_client": MyCustomClient(),
}
run_pipeline(config)
```

---

## See Also

- **[API Keys Setup](../fundamentals/installation/api-keys.md)** - Configure API keys including WatsonX
- **[Model Configuration](../fundamentals/pipeline-configuration/model-configuration.md)** - Model setup
- **[Remote Inference](../fundamentals/pipeline-configuration/backend-selection.md)** - Backend selection