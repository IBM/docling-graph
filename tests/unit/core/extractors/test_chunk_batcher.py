from docling_graph.core.extractors.chunk_batcher import ChunkBatch, ChunkBatcher


class DummyTokenizer:
    def count_tokens(self, text: str) -> int:
        return len(text.split())


def test_chunk_batch_properties():
    batch = ChunkBatch(
        batch_id=0,
        chunks=["chunk1", "chunk2"],
        total_tokens=100,
        prompt_tokens=140,
        chunk_indices=[0, 1],
    )

    assert batch.chunk_count == 2
    assert batch.batch_id == 0
    assert batch.total_tokens == 100
    assert batch.prompt_tokens == 140
    combined = batch.combined_text
    assert "[Chunk 1/2]" in combined
    assert "[Chunk 2/2]" in combined
    assert "---CHUNK BOUNDARY---" in combined


def test_batcher_budget_math():
    tokenizer = DummyTokenizer()
    batcher = ChunkBatcher(
        context_limit=2000,
        schema_json='{"title": "Schema"}',
        tokenizer=tokenizer,
        reserved_output_tokens=200,
        safety_margin_tokens=100,
    )

    assert batcher.prompt_budget == 1700
    assert batcher.available_tokens == batcher.prompt_budget - batcher.static_overhead_tokens


def test_batch_chunks_respects_prompt_budget():
    tokenizer = DummyTokenizer()
    batcher = ChunkBatcher(
        context_limit=1200,
        schema_json='{"title": "Schema"}',
        tokenizer=tokenizer,
        reserved_output_tokens=200,
        safety_margin_tokens=100,
        merge_threshold=0.5,
    )

    chunks = ["word " * 50, "word " * 60, "word " * 70]
    batches = batcher.batch_chunks(chunks)

    assert batches
    assert all(batch.prompt_tokens <= batcher.prompt_budget for batch in batches)
