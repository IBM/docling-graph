# Introduction to Docling Graph

Welcome to Docling Graph! This section introduces the core concepts of knowledge graph extraction from documents.

## What is Docling Graph?

Docling Graph transforms unstructured documents into validated **knowledge graphs** with precise semantic relationships. Unlike traditional approaches that convert text to embeddings (losing exact relationships), Docling Graph preserves explicit entity connections.

**Key Advantage**: Answer questions like "who issued what to whom" with exact relationships, not approximate embeddings.

## Why Knowledge Graphs?

Knowledge graphs are essential for domains requiring exact entity connections:

- **Chemistry**: Track compounds, reactions, and measurements
- **Finance/Legal**: Map instruments, obligations, and dependencies
- **Research**: Connect authors, methodologies, and results
- **Healthcare**: Relate patients, treatments, and outcomes

## How It Works

Docling Graph follows a clear pipeline:

1. **[Installation](../fundamentals/installation/index.md)** - Set up environment
2. **[Schema Definition](../fundamentals/schema-definition/index.md)** - Create Pydantic templates
3. **[Pipeline Configuration](../fundamentals/pipeline-configuration/index.md)** - Configure backend and mode
4. **[Extraction Process](../fundamentals/extraction-process/index.md)** - Run extraction
5. **[Graph Management](../fundamentals/graph-management/index.md)** - Export and visualize

## Quick Example

```python
from docling_graph import run_pipeline, PipelineConfig
from pydantic import BaseModel, Field

class Person(BaseModel):
    model_config = {'is_entity': True, 'graph_id_fields': ['name']}
    name: str = Field(description="Person's full name")
    email: str = Field(description="Email address")

config = PipelineConfig(
    source="document.pdf",
    template=Person,
    backend="llm",
    inference="remote"
)

context = run_pipeline(config)
print(f"Extracted {context.knowledge_graph.number_of_nodes()} nodes")
```

## Learn More

- **[Key Concepts](key-concepts.md)** - Entities, components, nodes, edges
- **[Use Cases](use-cases.md)** - Domain-specific examples
- **[Architecture](architecture.md)** - System design
- **[Quick Start](quickstart.md)** - 5-minute tutorial

---

**Ready to start?** Begin with [installation](../fundamentals/installation/index.md) or try the [quick start](quickstart.md).