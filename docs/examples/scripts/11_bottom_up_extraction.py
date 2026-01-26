"""
Bottom-Up Extraction Example for Small LLMs.

This example demonstrates how to use the bottom-up extraction strategy,
which is optimized for small LLMs (≈1B–3B parameters) by:
1. Breaking documents into focused slots (table rows, figures, text blocks)
2. Extracting entities progressively (leaves → root)
3. Using caching, rate limiting, and comprehensive metrics
"""

from pathlib import Path

from pydantic import BaseModel, Field

from docling_graph import PipelineConfig


# Define a hierarchical schema for materials research
class Material(BaseModel):
    """A material with properties."""
    name: str = Field(description="Material name")
    composition: str = Field(description="Chemical composition")
    properties: list[str] = Field(default_factory=list, description="Material properties")
    surrogatekey: str = ""
    sourcelocation: dict = Field(default_factory=dict)


class Experiment(BaseModel):
    """An experiment involving materials."""
    experiment_id: str = Field(description="Experiment identifier")
    materials: list[str] = Field(default_factory=list, description="Material IDs used")
    conditions: str = Field(description="Experimental conditions")
    results: str = Field(description="Experimental results")
    surrogatekey: str = ""
    sourcelocation: dict = Field(default_factory=dict)


class ResearchDocument(BaseModel):
    """Root model for research document."""
    title: str = Field(description="Document title")
    materials: list[Material] = Field(default_factory=list)
    experiments: list[Experiment] = Field(default_factory=list)


def main() -> None:
    """Run bottom-up extraction example."""

    # Configure pipeline for bottom-up extraction
    config = PipelineConfig(
        source="path/to/research_paper.pdf",
        template=ResearchDocument,

        # Core settings
        backend="llm",
        inference="local",  # or "remote" for cloud APIs
        processing_mode="bottom-up",  # Enable bottom-up strategy

        # Model selection (small LLM recommended)
        model_override="ibm-granite/granite-4.0-1b",  # 1B parameter model
        provider_override="vllm",

        # Bottom-up specific settings
        min_text_length=20,           # Minimum text block length
        enable_caching=True,          # Cache extracted entities per slot
        enable_rate_limiting=True,    # Respect provider rate limits
        max_retries=3,                # Retry failed API calls
        enable_metrics=True,          # Collect comprehensive metrics

        # Output settings
        output_dir="outputs/bottom_up_example",
        export_format="csv",
        dump_to_disk=True,
        include_trace=True,
    )

    print("=" * 80)
    print("BOTTOM-UP EXTRACTION EXAMPLE")
    print("=" * 80)
    print(f"Source: {config.source}")
    print(f"Model: {config.model_override}")
    print(f"Processing Mode: {config.processing_mode}")
    print(f"Caching: {config.enable_caching}")
    print(f"Rate Limiting: {config.enable_rate_limiting}")
    print(f"Metrics: {config.enable_metrics}")
    print("=" * 80)
    print()

    # Run pipeline
    print("Starting extraction...")
    config.run()

    print()
    print("=" * 80)
    print("EXTRACTION COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {config.output_dir}")
    print()
    print("The pipeline will have printed:")
    print("  • Slot enumeration summary (tables, figures, text blocks)")
    print("  • Hierarchy analysis (extraction levels)")
    print("  • Per-stage progress (batches, cache hits)")
    print("  • Comprehensive metrics summary")
    print()
    print("Check the metrics summary for:")
    print("  • Cache hit rate (% of slots served from cache)")
    print("  • API success rate (% of successful calls)")
    print("  • Entity assignment rate (% of entities assigned to root model)")
    print("  • Token consumption (input + output)")
    print("  • Timing breakdown (conversion, extraction, assembly)")


def example_with_remote_api() -> None:
    """Example using remote API (e.g., Mistral, OpenAI)."""

    config = PipelineConfig(
        source="path/to/document.pdf",
        template=ResearchDocument,

        backend="llm",
        inference="remote",
        processing_mode="bottom-up",

        # Use a small remote model
        model_override="mistral-small-latest",
        provider_override="mistral",

        # Bottom-up settings
        enable_caching=True,
        enable_rate_limiting=True,  # Important for API rate limits!
        max_retries=3,
        enable_metrics=True,

        output_dir="outputs/remote_bottom_up",
    )

    config.run()


def example_with_custom_settings() -> None:
    """Example with custom bottom-up settings."""

    config = PipelineConfig(
        source="path/to/large_document.pdf",
        template=ResearchDocument,

        backend="llm",
        inference="local",
        processing_mode="bottom-up",

        model_override="ibm-granite/granite-4.0-3b",
        provider_override="vllm",

        # Custom settings for large documents
        min_text_length=50,           # Longer text blocks
        enable_caching=True,          # Essential for large docs
        enable_rate_limiting=False,   # No limit for local inference
        max_retries=5,                # More retries for stability
        enable_metrics=True,

        output_dir="outputs/custom_bottom_up",
    )

    config.run()


if __name__ == "__main__":
    # Run the main example
    main()

    # Uncomment to try other examples:
    # example_with_remote_api()
    # example_with_custom_settings()

# Made with Bob
