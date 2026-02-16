from docling_graph.core.extractors.contracts.delta.prompts import get_delta_batch_prompt


def test_prompt_mentions_cross_batch_identifier_stability() -> None:
    prompt = get_delta_batch_prompt(
        batch_markdown="--- CHUNK 1 ---\ntext",
        schema_semantic_guide="guide",
        path_catalog_block="catalog",
        batch_index=0,
        total_batches=2,
    )
    assert "across the entire document" in prompt["system"]
    assert "exact catalog paths" in prompt["system"]
    assert "identity" in prompt["system"]
    assert "ids" in prompt["system"]
    assert (
        "List-entity" in prompt["user"]
        or "list-entity" in prompt["user"]
        or "offres[]" in prompt["user"]
    )
    assert "parent" in prompt["user"]
    assert "path" in prompt["user"]
    assert "total_amount" not in prompt["user"]
