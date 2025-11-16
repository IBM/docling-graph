"""
Example: Using IBM WatsonX for document extraction with docling-graph.

This example demonstrates how to use IBM WatsonX's Granite models
for extracting structured data from documents.

Prerequisites:
    1. Install WatsonX support: uv sync --extra watsonx
    2. Set environment variables:
       - WATSONX_API_KEY: Your IBM Cloud API key
       - WATSONX_PROJECT_ID: Your WatsonX project ID
       - WATSONX_URL: (Optional) Custom endpoint URL
"""

import sys
from pathlib import Path

# Add the workspace root to Python path to enable imports
workspace_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(workspace_root))

from docling_graph import PipelineConfig, run_pipeline
from examples.templates.battery_research import Research

# Example 1: Using WatsonX with default Granite model
config_default = PipelineConfig(
    source="examples/data/battery_research/bauer2014.pdf",
    template=Research,
    backend="llm",
    inference="remote",
    processing_mode="many-to-one",
    provider_override="watsonx",
    model_override="ibm-granite/granite-4.0-h-small",  # Default Granite model
    output_dir="outputs/watsonx_example_default",
)

# Example 2: Using WatsonX with Llama model
config_llama = PipelineConfig(
    source="examples/data/battery_research/bauer2014.pdf",
    template=Research,
    backend="llm",
    inference="remote",
    processing_mode="many-to-one",
    provider_override="watsonx",
    model_override="meta-llama/llama-3-70b-instruct",  # Llama model on WatsonX
    output_dir="outputs/watsonx_example_llama",
)

# Example 3: Using WatsonX with Mixtral model
config_mixtral = PipelineConfig(
    source="examples/data/battery_research/bauer2014.pdf",
    template=Research,
    backend="llm",
    inference="remote",
    processing_mode="many-to-one",
    provider_override="watsonx",
    model_override="mistralai/mixtral-8x7b-instruct-v01",  # Mixtral on WatsonX
    output_dir="outputs/watsonx_example_mixtral",
)


def main() -> None:
    """Run the WatsonX extraction examples."""
    print("=" * 80)
    print("IBM WatsonX Document Extraction Examples")
    print("=" * 80)

    # Run Example 1: Default Granite model
    print("\n[1/3] Running extraction with Granite 4.0 H Small...")
    try:
        run_pipeline(config_default)
        print(f"✓ Success! Output saved to: {config_default.output_dir}")
    except Exception as e:
        print(f"✗ Error: {e}")

    # Run Example 2: Llama model
    print("\n[2/3] Running extraction with Llama 3 70B...")
    try:
        run_pipeline(config_llama)
        print(f"✓ Success! Output saved to: {config_llama.output_dir}")
    except Exception as e:
        print(f"✗ Error: {e}")

    # Run Example 3: Mixtral model
    print("\n[3/3] Running extraction with Mixtral 8x7B...")
    try:
        run_pipeline(config_mixtral)
        print(f"✓ Success! Output saved to: {config_mixtral.output_dir}")
    except Exception as e:
        print(f"✗ Error: {e}")

    print("\n" + "=" * 80)
    print("All examples completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()

# Made with Bob, Checked by Guilhaume
