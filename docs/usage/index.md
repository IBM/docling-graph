# Usage

Welcome to the **Usage** section! This section covers practical guides for using Docling Graph through different interfaces and advanced techniques.

---

## What You'll Learn

This section provides comprehensive guides for working with Docling Graph:

1. **[CLI Reference](cli/index.md)** - Command-line interface for quick document processing
2. **[Python API](api/index.md)** - Programmatic usage and integration into your applications
3. **[Examples](examples/index.md)** - Working code examples and real-world templates
4. **[Advanced Topics](advanced/index.md)** - Performance tuning, custom backends, and error handling

---

## Quick Links

<div class="grid cards" markdown>

- **[CLI Reference →](cli/index.md)**

    Use the command-line interface for document processing

- **[Python API →](api/index.md)**

    Integrate Docling Graph into your Python applications

- **[Examples →](examples/index.md)**

    Explore working examples and template gallery

- **[Advanced Topics →](advanced/index.md)**

    Optimize performance and implement custom solutions

</div>

---

## Choose Your Interface

### Command-Line Interface (CLI)

Perfect for quick processing and scripting:

```bash
uv run docling-graph convert document.pdf \
    --template "templates.Invoice" \
    --output-dir "outputs/invoice"
```

**[→ Learn More About CLI](cli/index.md)**

### Python API

Ideal for programmatic integration:

```python
from docling_graph import PipelineConfig

config = PipelineConfig(
    source="document.pdf",
    template="templates.Invoice",
    backend="llm",
    inference="remote"
)
config.run()
```

**[→ Learn More About Python API](api/index.md)**

---

## Learning Path

We recommend this approach:

1. **Start with Examples** to see working code
2. **Choose Your Interface** (CLI or Python API)
3. **Explore Advanced Topics** for optimization

---

## Next Steps

- **New to Docling Graph?** Start with [Examples](examples/index.md)
- **Prefer CLI?** Check out [CLI Reference](cli/index.md)
- **Building an application?** See [Python API](api/index.md)
- **Need optimization?** Explore [Advanced Topics](advanced/index.md)