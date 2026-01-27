# LLM Clients API

!!! note
    Clients are instantiated with resolved configs.
    Prefer resolving via `resolve_effective_model_config(provider, model)` and
    passing `model_config` to client constructors.


## Overview

LLM client implementations for various providers.

**Module:** `docling_graph.llm_clients`

All clients implement `LLMClientProtocol`.

---

## Base Client

### BaseLLMClient

Base class for LLM clients with configurable generation limits and timeouts.

```python
class BaseLLMClient(LLMClientProtocol):
    """Base LLM client implementation."""
    
    def __init__(
        self,
        model: str,
        max_tokens: int | None = None,
        timeout: int | None = None,
        **kwargs
    ):
        """
        Initialize LLM client.
        
        Args:
            model: Model identifier
            max_tokens: Maximum tokens to generate (overrides config, default: 8192)
            timeout: Request timeout in seconds (overrides config, default: 300-600)
            **kwargs: Provider-specific parameters
        """
    
    @property
    def context_limit(self) -> int:
        """Return effective context limit in tokens."""
        raise NotImplementedError
    
    @property
    def max_tokens(self) -> int:
        """Return maximum tokens to generate."""
        return self._max_tokens or 8192
    
    @property
    def timeout(self) -> int:
        """Return request timeout in seconds."""
        return self._timeout or 300
    
    def get_json_response(
        self,
        prompt: str | Mapping[str, str],
        schema_json: str
    ) -> Dict[str, Any]:
        """Execute LLM call and return parsed JSON."""
        raise NotImplementedError
```

#### Configuration

All clients use resolved configs from the registry:

- **Generation defaults** (e.g., `max_tokens`, `temperature`)
- **Reliability defaults** (e.g., `timeout_s`, `max_retries`)

Use `llm_overrides` in `PipelineConfig` or resolve explicitly:

```python
from docling_graph.llm_clients import VllmClient
from docling_graph.llm_clients.config import resolve_effective_model_config

effective = resolve_effective_model_config(
    "vllm",
    "qwen/Qwen2-7B",
    overrides={
        "generation": {"max_tokens": 4096},
        "reliability": {"timeout_s": 300},
    },
)

client = VllmClient(model_config=effective)
```

---

## Local Clients

### OllamaClient

Client for Ollama local inference.

```python
class OllamaClient(BaseLLMClient):
    """Ollama LLM client."""
    
    def __init__(
        self,
        model: str = "llama-3.1-8b",
        base_url: str = "http://localhost:11434"
    ):
        """Initialize Ollama client."""
        self.model = model
        self.base_url = base_url
    
    @property
    def context_limit(self) -> int:
        """Return context limit."""
        return 8000  # Conservative
```

**Example:**

```python
from docling_graph.llm_clients import OllamaClient
from docling_graph.llm_clients.config import resolve_effective_model_config

effective = resolve_effective_model_config(
    "ollama",
    "llama-3.1-8b",
    overrides={"connection": {"base_url": "http://localhost:11434"}},
)
client = OllamaClient(model_config=effective)

response = client.get_json_response(
    prompt="Extract data from: ...",
    schema_json=schema
)
```

---

### VLLMClient

Client for vLLM server with generation limits and timeout protection.

```python
class VLLMClient(BaseLLMClient):
    """vLLM server client."""
    
    def __init__(
        self,
        model: str = "ibm-granite/granite-4.0-1b",
        base_url: str = "http://localhost:8000/v1",
        max_tokens: int | None = None,
        timeout: int | None = None
    ):
        """
        Initialize vLLM client.
        
        Args:
            model: Model identifier
            base_url: vLLM server URL
            max_tokens: Maximum tokens to generate (default: 8192)
            timeout: Request timeout in seconds (default: 600)
        """
        self.model = model
        self.base_url = base_url
    
    @property
    def context_limit(self) -> int:
        """Return context limit."""
        return 8000
```

**Example:**

```python
from docling_graph.llm_clients import VllmClient
from docling_graph.llm_clients.config import resolve_effective_model_config

# Basic usage (uses registry defaults)
effective = resolve_effective_model_config(
    "vllm",
    "ibm-granite/granite-4.0-1b",
    overrides={"connection": {"base_url": "http://localhost:8000/v1"}},
)
client = VllmClient(model_config=effective)

# Custom limits to prevent hanging
effective = resolve_effective_model_config(
    "vllm",
    "qwen/Qwen2-7B",
    overrides={
        "connection": {"base_url": "http://localhost:8000/v1"},
        "generation": {"max_tokens": 4096},
        "reliability": {"timeout_s": 300},
    },
)
client = VllmClient(model_config=effective)
```

!!! warning "Timeout Protection"
    vLLM client now includes timeout protection to prevent indefinite hangs. If a request exceeds the timeout (default: 10 minutes), it will raise a `ClientError`. This is especially important when processing documents that don't match your template schema.

---

## Remote Clients

### MistralClient

Client for Mistral AI API.

```python
class MistralClient(BaseLLMClient):
    """Mistral AI client."""
    
    def __init__(
        self,
        model: str = "mistral-small-latest",
        api_key: str | None = None
    ):
        """
        Initialize Mistral client.
        
        Args:
            model: Model name
            api_key: API key (or set MISTRAL_API_KEY env var)
        """
        self.model = model
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
    
    @property
    def context_limit(self) -> int:
        """Return context limit."""
        return 32000
```

**Example:**

```python
from docling_graph.llm_clients import MistralClient
from docling_graph.llm_clients.config import resolve_effective_model_config

effective = resolve_effective_model_config("mistral", "mistral-small-latest")
client = MistralClient(model_config=effective)
```

---

### OpenAIClient

Client for OpenAI API.

```python
class OpenAIClient(BaseLLMClient):
    """OpenAI client."""
    
    def __init__(
        self,
        model: str = "gpt-4-turbo",
        api_key: str | None = None
    ):
        """
        Initialize OpenAI client.
        
        Args:
            model: Model name
            api_key: API key (or set OPENAI_API_KEY env var)
        """
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
    
    @property
    def context_limit(self) -> int:
        """Return context limit."""
        return 128000  # GPT-4 Turbo
```

**Example:**

```python
from docling_graph.llm_clients import OpenAIClient
from docling_graph.llm_clients.config import resolve_effective_model_config

effective = resolve_effective_model_config("openai", "gpt-4o")
client = OpenAIClient(model_config=effective)
```

---

### GeminiClient

Client for Google Gemini API.

```python
class GeminiClient(BaseLLMClient):
    """Google Gemini client."""
    
    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        api_key: str | None = None
    ):
        """
        Initialize Gemini client.
        
        Args:
            model: Model name
            api_key: API key (or set GEMINI_API_KEY env var)
        """
        self.model = model
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
    
    @property
    def context_limit(self) -> int:
        """Return context limit."""
        return 1000000  # Gemini 2.5 Flash
```

**Example:**

```python
from docling_graph.llm_clients import GeminiClient
from docling_graph.llm_clients.config import resolve_effective_model_config

effective = resolve_effective_model_config("gemini", "gemini-2.5-flash")
client = GeminiClient(model_config=effective)
```

---

### WatsonxClient

Client for IBM watsonx.ai API.

```python
class WatsonxClient(BaseLLMClient):
    """IBM watsonx.ai client."""
    
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        project_id: str | None = None,
        url: str | None = None
    ):
        """
        Initialize watsonx client.
        
        Args:
            model: Model name
            api_key: API key (or set WATSONX_API_KEY)
            project_id: Project ID (or set WATSONX_PROJECT_ID)
            url: Service URL (or set WATSONX_URL)
        """
        self.model = model
        self.api_key = api_key or os.getenv("WATSONX_API_KEY")
        self.project_id = project_id or os.getenv("WATSONX_PROJECT_ID")
        self.url = url or os.getenv("WATSONX_URL")
```

**Example:**

```python
from docling_graph.llm_clients import WatsonxClient
from docling_graph.llm_clients.config import resolve_effective_model_config

effective = resolve_effective_model_config("watsonx", "ibm/granite-13b-chat-v2")
client = WatsonxClient(model_config=effective)
```

---

## Client Configuration

### Model Configuration

Models are configured in `models.yaml` with generation limits and timeouts:

```yaml
providers:
  mistral:
    tokenizer: "mistralai/Mistral-7B-Instruct-v0.2"
    default_max_tokens: 8192      # Default response limit
    timeout_seconds: 300          # 5 minute timeout
    models:
      mistral-small-latest:
        context_limit: 32000
        max_new_tokens: 4096
        max_tokens: 8192          # Optional model-specific override
        timeout: 300              # Optional model-specific timeout
    
  vllm:
    tokenizer: "sentence-transformers/all-MiniLM-L6-v2"
    default_max_tokens: 8192      # Prevents infinite generation
    timeout_seconds: 600          # 10 minute timeout for local inference
    models:
      qwen/Qwen2-7B:
        context_limit: 128000
        max_new_tokens: 4096
```

### Configuration Hierarchy

Configuration is resolved in this order (highest priority first):

1. **Runtime overrides**: `llm_overrides` or `resolve_effective_model_config(..., overrides=...)`
2. **Model overrides**: `models` registry entry
3. **Provider defaults**: `providers` registry entry
4. **Built-in defaults**: schema defaults

**Example:**

```python
from docling_graph.llm_clients import VllmClient
from docling_graph.llm_clients.config import resolve_effective_model_config

effective = resolve_effective_model_config(
    "vllm",
    "qwen/Qwen2-7B",
    overrides={
        "generation": {"max_tokens": 4096},
        "reliability": {"timeout_s": 300},
    },
)

client = VllmClient(model_config=effective)
```

### Timeout Defaults by Provider

| Provider | Default Timeout | Reason |
|----------|----------------|--------|
| OpenAI, Mistral, Gemini, Anthropic | 300s (5 min) | Fast API responses |
| vLLM, Ollama, WatsonX | 600s (10 min) | Local/slower inference |

---

## API Key Management

### Environment Variables

Set API keys via environment variables:

```bash
# Mistral
export MISTRAL_API_KEY="your_key"

# OpenAI
export OPENAI_API_KEY="your_key"

# Gemini
export GEMINI_API_KEY="your_key"

# watsonx
export WATSONX_API_KEY="your_key"
export WATSONX_PROJECT_ID="your_project"
export WATSONX_URL="https://us-south.ml.cloud.ibm.com"
```

### .env File

Or use a `.env` file:

```bash
# .env
MISTRAL_API_KEY=your_key
OPENAI_API_KEY=your_key
GEMINI_API_KEY=your_key
```

Load with:

```python
from dotenv import load_dotenv
load_dotenv()
```

---

## Usage with Pipeline

Clients are automatically selected based on configuration:

```python
from docling_graph import run_pipeline, PipelineConfig

# Uses MistralClient automatically
config = PipelineConfig(
    source="document.pdf",
    template="templates.MyTemplate",
    backend="llm",
    inference="remote",
    provider_override="mistral",
    model_override="mistral-small-latest"
)
run_pipeline(config)
```

---

## Custom Clients

Create custom clients by implementing `LLMClientProtocol`:

```python
from docling_graph.protocols import LLMClientProtocol
from typing import Dict, Any, Mapping

class MyCustomClient(LLMClientProtocol):
    """Custom LLM client."""
    
    @property
    def context_limit(self) -> int:
        return 8000
    
    def get_json_response(
        self,
        prompt: str | Mapping[str, str],
        schema_json: str
    ) -> Dict[str, Any]:
        # Your implementation
        pass
```

---

## Error Handling

All clients raise `ClientError` on failures, including timeouts:

```python
from docling_graph.llm_clients import VllmClient
from docling_graph.llm_clients.config import resolve_effective_model_config
from docling_graph.exceptions import ClientError

effective = resolve_effective_model_config(
    "vllm",
    "qwen/Qwen2-7B",
    overrides={"reliability": {"timeout_s": 300}},
)
client = VllmClient(model_config=effective)

try:
    response = client.get_json_response(prompt, schema)
except ClientError as e:
    print(f"Error: {e.message}")
    print(f"Details: {e.details}")
    
    # Check if it was a timeout
    if "timeout" in str(e).lower():
        print("Request exceeded timeout limit")
        print(f"Timeout was: {client.timeout}s")
```

### Common Error Scenarios

**Timeout Error:**
```python
ClientError: vLLM request timeout after 600s
Details: {
    'model': 'qwen/Qwen2-7B',
    'timeout': 600,
    'max_tokens': 8192
}
```

**Infinite Generation (Fixed):**

Before the fix, vLLM could generate indefinitely when content didn't match the template. Now it's limited by `max_tokens`:

```python
# Old behavior: Could hang for hours
# New behavior: Stops at 8192 tokens (or custom limit)
from docling_graph.llm_clients import VllmClient
from docling_graph.llm_clients.config import resolve_effective_model_config

effective = resolve_effective_model_config(
    "vllm",
    "qwen/Qwen2-7B",
    overrides={"generation": {"max_tokens": 4096}},
)
client = VllmClient(model_config=effective)
```

### Troubleshooting

**Problem: Request times out**

- Increase timeout: `resolve_effective_model_config(..., overrides={"reliability": {"timeout_s": 1200}})`
- Reduce max_tokens: `resolve_effective_model_config(..., overrides={"generation": {"max_tokens": 4096}})`
- Check if content matches template schema

**Problem: Response truncated**

- Increase max_tokens: `resolve_effective_model_config(..., overrides={"generation": {"max_tokens": 16384}})`
- Simplify template to require less output
- Use chunking for large documents

---

## Related APIs

- **[Protocols](protocols.md)** - LLMClientProtocol
- **[Exceptions](exceptions.md)** - ClientError
- **[Model Configuration](../fundamentals/pipeline-configuration/model-configuration.md)** - Configure models

---

## Recent Changes

### Registry-Based Resolution

LLM settings now come from the registry plus overrides:

1. **Runtime overrides** (`llm_overrides` or resolver overrides)
2. **Model defaults** (model entry)
3. **Provider defaults** (provider entry)

**Example:**

```python
from docling_graph.llm_clients import VllmClient
from docling_graph.llm_clients.config import resolve_effective_model_config

effective = resolve_effective_model_config(
    "vllm",
    "qwen/Qwen2-7B",
    overrides={
        "generation": {"max_tokens": 8192},
        "reliability": {"timeout_s": 600},
    },
)

client = VllmClient(model_config=effective)
```

---

## See Also

- **[API Keys Setup](../fundamentals/installation/api-keys.md)** - Configure API keys including WatsonX
- **[Model Configuration](../fundamentals/pipeline-configuration/model-configuration.md)** - Model setup
- **[Remote Inference](../fundamentals/pipeline-configuration/backend-selection.md)** - Backend selection