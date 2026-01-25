# Problem Statement: Docling-Graph

## The Core Challenge

**Unstructured documents contain valuable knowledge that remains locked in formats designed for human reading, not machine understanding.** Traditional document processing approaches face critical limitations:

1. **Loss of Semantic Relationships**: Converting documents to text vectors or embeddings loses the precise relationships between entities (e.g., "who issued what to whom" or "which chemical reacts with which compound")

2. **Domain Complexity**: Technical domains like chemistry, finance, legal, and research require exact understanding of entity connections and dependencies, not approximate similarity matching

3. **Lack of Explainability**: Vector-based approaches cannot explain *why* two pieces of information are related or trace the reasoning path through document relationships

4. **Validation Gaps**: Extracted information often lacks structure validation, leading to inconsistent or incorrect data that propagates through downstream systems

5. **Input Format Barriers**: Processing pipelines are often rigid, requiring specific input formats and manual preprocessing steps that slow down workflows

## What Docling-Graph Solves

Docling-graph addresses these challenges by providing an **intelligent, adaptive pipeline** that transforms diverse document sources into validated knowledge graphs with precise semantic relationships.

### The Solution Architecture

```
Multiple Input Formats → Smart Processing → Validated Extraction → Knowledge Graph → Queryable Relationships
```

### Key Capabilities

#### 1. **Universal Input Processing**
- Accepts **multiple input formats** with automatic detection and routing:
  - PDF documents and images (full OCR/VLM pipeline)
  - Text and Markdown files (direct extraction)
  - URLs (automatic download and processing)
  - DoclingDocument JSON (skip to graph conversion)
  - Plain text strings (programmatic API)
- Smart routing skips unnecessary processing stages based on input type

#### 2. **Structured Extraction with Validation**
- Converts documents into **validated Pydantic objects** using either:
  - Local Vision-Language Models (VLM) via Docling
  - Large Language Models (LLM) via multiple providers (local or cloud)
- **Adaptive prompting** based on model capabilities (SIMPLE/STANDARD/ADVANCED tiers)
- **Intelligent chunking** with provider-specific batching and real tokenizers
- Ensures data quality through schema validation before graph construction

#### 3. **Semantic Knowledge Graph Construction**
- Transforms validated objects into **directed knowledge graphs** with:
  - Explicit entity nodes (people, organizations, documents)
  - Precise relationship edges (ISSUED_BY, CONTAINS_LINE, LOCATED_AT)
  - Rich metadata and stable node identifiers
- Preserves semantic meaning and relationships that vectors lose

#### 4. **Explainable Reasoning**
- Enables traversal of exact relationships between entities
- Supports queries like "Find all invoices issued by Organization X to Person Y"
- Provides audit trails showing how information connects
- **Trace data capture** for debugging with pages, chunks, and intermediate schemas

#### 5. **Flexible Integration**
- **API-first design**: Returns data directly in memory (no disk writes by default)
- Exports to multiple Neo4j-compatible formats (CSV, Cypher, JSON)
- Generates interactive HTML visualizations and detailed Markdown reports
- Supports both page-wise and document-level processing strategies
- Optional disk export with `dump_to_disk` flag for CLI workflows

## Why This Matters

For **complex technical domains**, understanding exact relationships is critical:

- **Chemistry**: Which compounds react with which catalysts under what conditions?
- **Finance**: Which instruments depend on which underlying assets?
- **Legal**: Which clauses reference which parties and obligations?
- **Research**: Which experiments produced which results using which methods?

Traditional text search or vector similarity cannot answer these questions with the precision required for production systems, regulatory compliance, or scientific accuracy.

## The Value Proposition

Docling-graph provides an **intelligent, production-ready pipeline** that:

1. **Accepts** diverse input formats (PDF, images, text, Markdown, URLs, JSON, plain strings)
2. **Routes** intelligently through optimized processing stages based on input type
3. **Extracts** using customizable Pydantic templates with adaptive prompting and intelligent batching
4. **Validates** data against schemas before graph construction
5. **Converts** to NetworkX directed graphs with rich metadata and stable node IDs
6. **Exports** to multiple formats (Neo4j-ready CSV/Cypher, JSON) or returns data directly via API
7. **Visualizes** through interactive HTML and detailed Markdown reports
8. **Debugs** with comprehensive trace data capturing pages, chunks, and intermediate results

This enables organizations to build **explainable AI systems** that understand document content through precise semantic relationships rather than approximate statistical patterns, with the flexibility to integrate into existing workflows through both CLI and programmatic APIs.

## Use Cases

- **Document Intelligence**: Extract structured data from invoices, contracts, insurance policies, and ID cards across multiple formats (PDF, images, scanned documents)
- **Research Analysis**: Build knowledge graphs from scientific papers (PDFs or URLs) to understand experimental relationships and methodologies
- **Compliance & Audit**: Trace exact relationships between entities for regulatory requirements with full audit trails via trace data
- **Knowledge Management**: Create queryable knowledge bases from technical documentation in any format (Markdown, text, PDF)
- **Web Content Processing**: Extract structured information from web pages and online documents via URL processing
- **Programmatic Integration**: Embed document-to-graph conversion directly into applications using the Python API
- **Data Integration**: Connect information across multiple documents and formats through shared entities

## Technical Approach

Unlike traditional approaches that:
- Convert text to vectors and lose relationship precision
- Require manual relationship extraction and validation
- Lack explainability in entity connections
- Are limited to specific input formats
- Require manual performance tuning

Docling-graph:
- **Accepts diverse inputs**: Automatically detects and processes PDFs, images, text, Markdown, URLs, JSON, and plain strings
- **Adapts intelligently**: Uses model capability detection to optimize prompts and batching strategies
- **Validates rigorously**: Uses Pydantic schemas to guide LLM/VLM extraction with validation
- **Constructs precisely**: Automatically builds graphs from validated objects with explicit relationships
- **Traces completely**: Provides full traceability via trace data capturing pages, chunks, and intermediate results
- **Integrates flexibly**: Supports multiple extraction backends, export formats, and both CLI/API workflows
- **Optimizes automatically**: Provider-specific batching with real tokenizers and improved GPU memory management