# Debug Mode

## Overview

Debug mode provides visibility into the pipeline's intermediate stages for debugging, performance analysis, and quality assurance. When enabled, in-memory trace data is captured and optionally persisted to disk.

**What's Captured:**
- **In-memory `TraceData`**: When `debug=True`, `context.trace_data` is populated with pages, chunks, extractions, intermediate graph summaries, and consolidation info (see [PipelineContext](../../reference/pipeline.md) and the `TraceData` structure below).
- **`debug/trace_data.json`**: When debug is on and output is written to disk, a JSON snapshot of trace data is saved under the run's `debug/` directory. Long text fields are truncated to keep the file small.

---

## Quick Start

### Enable Debug Mode

Debug mode is controlled by a single flag.

**CLI:**
```bash
# Enable debug mode
uv run docling-graph convert document.pdf \
    --template "templates.BillingDocument" \
    --debug
```

**API:**
```python
from docling_graph import run_pipeline, PipelineConfig

# Enable debug mode
config = PipelineConfig(
    source="document.pdf",
    template="templates.BillingDocument",
    debug=True
)

context = run_pipeline(config)
```

---

## Output Structure

When debug mode is enabled and the pipeline writes to disk, trace data is saved under `outputs/{document}_{timestamp}/debug/`:

```
outputs/invoice_pdf_20260206_094500/
├── metadata.json                    # Pipeline metadata
├── docling/                         # Docling conversion output
│   ├── document.json
│   └── document.md
├── docling_graph/                   # Graph outputs
│   ├── graph.json
│   ├── nodes.csv
│   ├── edges.csv
│   └── ...
└── debug/                           # Debug output (when debug=True)
    └── trace_data.json              # Snapshot of pages, chunks, extractions, intermediate graphs, consolidation
```

---

## TraceData Structure

`TraceData` (and its JSON export) contains the following sections.

### Pages

Per-page text and metadata from document processing:

```json
{
  "pages": [
    {
      "page_number": 1,
      "text_content": "Full or truncated page text...",
      "metadata": {}
    }
  ]
}
```

**Use cases:** Verify page-level input to extraction, inspect OCR/vision output.

### Chunks

When chunking is used, each chunk's metadata and text:

```json
{
  "chunks": [
    {
      "chunk_id": 0,
      "page_numbers": [1, 2],
      "text_content": "Chunk text...",
      "token_count": 1500,
      "metadata": {}
    }
  ]
}
```

**Use cases:** Verify chunking strategy, token counts, and page coverage.

### Extractions

One entry per extraction (per page or per chunk):

```json
{
  "extractions": [
    {
      "extraction_id": 0,
      "source_type": "chunk",
      "source_id": 0,
      "parsed_model": { "invoice_number": "INV-001", "total_amount": 1250.50 },
      "extraction_time": 2.34,
      "error": null,
      "metadata": {}
    }
  ]
}
```

**Use cases:** Debug extraction failures (check `error`), analyze timing, inspect parsed models.

### Intermediate Graphs

Summaries of per-source graphs before consolidation (e.g. in many-to-one mode):

```json
{
  "intermediate_graphs": [
    {
      "graph_id": 0,
      "source_type": "chunk",
      "source_id": 0,
      "node_count": 6,
      "edge_count": 4
    }
  ]
}
```

**Use cases:** Verify graph generation per page/chunk, debug consolidation input.

### Consolidation

When graph consolidation is used:

```json
{
  "consolidation": {
    "strategy": "programmatic",
    "input_graph_ids": [0, 1, 2],
    "merge_conflicts": null
  }
}
```

**Use cases:** Understand merge strategy and conflict handling.

---

## Common Debugging Patterns

### Pattern 1: Find Extraction Errors

```python
# From API – use in-memory trace_data
context = run_pipeline(PipelineConfig(source="doc.pdf", template="templates.MyTemplate", debug=True))

if context.trace_data:
    for e in context.trace_data.extractions:
        if e.error:
            print(f"Extraction {e.extraction_id} ({e.source_type} {e.source_id}): {e.error}")
```

```bash
# From disk – use trace_data.json
cat outputs/doc_pdf_*/debug/trace_data.json | jq '.extractions[] | select(.error != null) | {extraction_id, source_type, source_id, error}'
```

### Pattern 2: Inspect Page/Chunk Coverage

```bash
# Page count
cat outputs/doc_pdf_*/debug/trace_data.json | jq '.pages | length'

# Chunk count and token usage
cat outputs/doc_pdf_*/debug/trace_data.json | jq '.chunks[] | {chunk_id, page_numbers, token_count}'
```

### Pattern 3: Analyze Extraction Times

```python
import json
from pathlib import Path

with open(Path("outputs/doc_pdf_123/debug/trace_data.json")) as f:
    data = json.load(f)

times = [e["extraction_time"] for e in data["extractions"]]
print(f"Extractions: {len(times)}, total time: {sum(times):.2f}s, avg: {sum(times)/len(times):.2f}s")
```

### Pattern 4: Verify Intermediate Graphs

```bash
# Summarize intermediate graphs
cat outputs/doc_pdf_*/debug/trace_data.json | jq '.intermediate_graphs'
```

---

## Programmatic Analysis

### Load trace_data.json in Python

```python
import json
from pathlib import Path

debug_dir = Path("outputs/invoice_pdf_20260206_094500/debug")
with open(debug_dir / "trace_data.json") as f:
    trace = json.load(f)

print(f"Pages: {len(trace['pages'])}")
print(f"Chunks: {len(trace.get('chunks') or [])}")
print(f"Extractions: {len(trace['extractions'])}")
print(f"Intermediate graphs: {len(trace['intermediate_graphs'])}")
```

### Use In-Memory trace_data (API)

```python
from docling_graph import run_pipeline, PipelineConfig

config = PipelineConfig(
    source="document.pdf",
    template="templates.BillingDocument",
    debug=True
)
context = run_pipeline(config)

if context.trace_data:
    for e in context.trace_data.extractions:
        if e.parsed_model:
            print(e.parsed_model.model_dump())
```

---

## Best Practices

### 1. Enable Debug During Development

```python
config = PipelineConfig(
    source="document.pdf",
    template="templates.BillingDocument",
    debug=True
)
```

### 2. Disable Debug in Production

```python
import os

config = PipelineConfig(
    source="document.pdf",
    template="templates.BillingDocument",
    debug=os.getenv("DEBUG", "false").lower() == "true"
)
```

### 3. Use Trace Data in Tests

```python
def test_extraction_succeeds():
    config = PipelineConfig(
        source="test_document.pdf",
        template="templates.BillingDocument",
        debug=True
    )
    context = run_pipeline(config)

    assert context.trace_data is not None
    errors = [e for e in context.trace_data.extractions if e.error]
    assert len(errors) == 0, f"Extraction errors: {errors}"
```

---

## Troubleshooting

### No trace_data / trace_data.json

**Problem:** `context.trace_data` is `None` or `debug/trace_data.json` is missing.

**Solution:** Ensure `debug=True` in `PipelineConfig` (or `--debug` in CLI). The JSON file is only written when output is written to disk (e.g. CLI or when the pipeline is configured to dump results).

### Extraction Errors in trace_data

**Problem:** Some entries in `trace_data.extractions` have non-null `error`.

**Solution:** Inspect `error` and `source_type`/`source_id` to locate the failing page or chunk. Check input text in `trace_data.pages` or `trace_data.chunks` and template/field definitions.

### Large trace_data.json

**Problem:** File is very large.

**Solution:** Text in `trace_data.json` is truncated (see `trace_data_to_jsonable(..., max_text_len=2000)`). For full text, use the in-memory `context.trace_data` during the run or add custom export logic.

---

## Performance Considerations

- **Memory:** Debug mode keeps full trace data in memory (pages, chunks, extraction results). For very large documents, this can be significant.
- **Disk:** Only one debug file is written (`trace_data.json`); size depends on document size and truncation. Clean up old debug directories when no longer needed.

---

## Next Steps

- **[CLI Documentation](../../usage/cli/convert-command.md)** – CLI usage with `--debug`
- **[Pipeline reference](../../reference/pipeline.md)** – PipelineContext and stages
- **[Configuration API](../../reference/config.md)** – PipelineConfig options
