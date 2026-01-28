from unittest.mock import MagicMock, Mock, patch

import pytest

from docling_graph.core.extractors.document_chunker import DocumentChunker


@pytest.fixture
def mock_hf_tokenizer():
    """Mock HuggingFace tokenizer."""
    tok = MagicMock()
    tok.count_tokens = MagicMock(return_value=10)
    return tok


@pytest.fixture
def mock_hf_tokenizer_wrapper():
    """Mock HuggingFaceTokenizer wrapper."""
    wrapper = MagicMock()
    wrapper.count_tokens = MagicMock(return_value=10)
    wrapper.max_tokens = 1024
    return wrapper


@patch("docling_graph.core.extractors.document_chunker.AutoTokenizer")
@patch("docling_graph.core.extractors.document_chunker.HuggingFaceTokenizer")
@patch("docling_graph.core.extractors.document_chunker.HybridChunker")
def test_chunker_init_with_tokenizer_name(
    mock_hybrid_chunker, mock_hf_tokenizer_class, mock_auto_tokenizer
):
    """Test initialization with a specific tokenizer name."""
    mock_tokenizer_instance = MagicMock()
    mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

    mock_wrapper = MagicMock()
    mock_hf_tokenizer_class.return_value = mock_wrapper

    config = {
        "tokenizer_name": "test-model",
        "max_tokens": 1024,
        "merge_peers": False,
    }

    chunker = DocumentChunker(**config)

    mock_auto_tokenizer.from_pretrained.assert_called_with("test-model")
    mock_hybrid_chunker.assert_called_with(
        tokenizer=mock_wrapper,
        merge_peers=False,
    )

    assert chunker.tokenizer_name == "test-model"
    assert chunker.max_tokens == 1024


@patch("docling_graph.core.extractors.document_chunker.get_tokenizer_for_provider")
@patch("docling_graph.core.extractors.document_chunker.AutoTokenizer")
@patch("docling_graph.core.extractors.document_chunker.HuggingFaceTokenizer")
@patch("docling_graph.core.extractors.document_chunker.HybridChunker")
def test_chunker_init_with_provider(
    mock_hybrid_chunker, mock_hf_tokenizer_class, mock_auto_tokenizer, mock_get_tokenizer
):
    """Test initialization using a provider shortcut."""
    expected_name = "mistralai/Mistral-7B-Instruct-v0.2"
    mock_get_tokenizer.return_value = expected_name

    mock_tokenizer_instance = MagicMock()
    mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

    mock_wrapper = MagicMock()
    mock_hf_tokenizer_class.return_value = mock_wrapper

    config = {"provider": "mistral", "max_tokens": 2048}
    chunker = DocumentChunker(**config)

    mock_get_tokenizer.assert_called_with("mistral")
    assert chunker.tokenizer_name == expected_name


@patch("docling_graph.core.extractors.document_chunker.AutoTokenizer")
@patch("docling_graph.core.extractors.document_chunker.HuggingFaceTokenizer")
@patch("docling_graph.core.extractors.document_chunker.HybridChunker")
def test_chunk_document(mock_hybrid_chunker_class, mock_hf_tokenizer_class, mock_auto_tokenizer):
    """Test the chunk_document method."""
    mock_tokenizer_instance = MagicMock()
    mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

    mock_wrapper = MagicMock()
    mock_hf_tokenizer_class.return_value = mock_wrapper

    # Mock the chunker instance
    mock_chunker_instance = MagicMock()
    mock_hybrid_chunker_class.return_value = mock_chunker_instance

    # Mock chunk iterator
    mock_chunk1 = MagicMock()
    mock_chunk2 = MagicMock()
    mock_chunker_instance.chunk.return_value = iter([mock_chunk1, mock_chunk2])
    mock_chunker_instance.contextualize.side_effect = ["enriched_chunk1", "enriched_chunk2"]

    chunker = DocumentChunker(max_tokens=1024)
    mock_doc = MagicMock()
    chunks = chunker.chunk_document(mock_doc)

    assert chunks == ["enriched_chunk1", "enriched_chunk2"]
    mock_chunker_instance.chunk.assert_called_with(dl_doc=mock_doc)
    assert mock_chunker_instance.contextualize.call_count == 2


@patch("docling_graph.core.extractors.document_chunker.AutoTokenizer")
@patch("docling_graph.core.extractors.document_chunker.HuggingFaceTokenizer")
@patch("docling_graph.core.extractors.document_chunker.HybridChunker")
def test_chunk_document_with_stats(
    mock_hybrid_chunker_class, mock_hf_tokenizer_class, mock_auto_tokenizer
):
    """Test the stats-enabled chunking method."""
    mock_tokenizer_instance = MagicMock()
    mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

    mock_wrapper = MagicMock()
    mock_wrapper.count_tokens.side_effect = [150, 250]
    mock_hf_tokenizer_class.return_value = mock_wrapper

    mock_chunker_instance = MagicMock()
    mock_hybrid_chunker_class.return_value = mock_chunker_instance

    mock_chunk1 = MagicMock()
    mock_chunk2 = MagicMock()
    mock_chunker_instance.chunk.return_value = iter([mock_chunk1, mock_chunk2])
    mock_chunker_instance.contextualize.side_effect = ["chunk1_text", "chunk2_text"]

    chunker = DocumentChunker(max_tokens=1024)
    mock_doc = MagicMock()

    chunks, stats = chunker.chunk_document_with_stats(mock_doc)

    assert chunks == ["chunk1_text", "chunk2_text"]
    assert stats["total_chunks"] == 2
    assert stats["total_tokens"] == 400  # 150 + 250
    assert stats["avg_tokens"] == 200.0
    assert stats["max_tokens_in_chunk"] == 250


@patch("docling_graph.core.extractors.document_chunker.AutoTokenizer")
@patch("docling_graph.core.extractors.document_chunker.HuggingFaceTokenizer")
@patch("docling_graph.core.extractors.document_chunker.HybridChunker")
def test_get_config_summary(
    mock_hybrid_chunker_class, mock_hf_tokenizer_class, mock_auto_tokenizer
):
    """Test the configuration summary."""
    mock_tokenizer_instance = MagicMock()
    mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

    mock_wrapper = MagicMock()
    mock_wrapper.__class__.__name__ = "HuggingFaceTokenizer"
    mock_hf_tokenizer_class.return_value = mock_wrapper

    mock_chunker_instance = MagicMock()
    mock_hybrid_chunker_class.return_value = mock_chunker_instance

    chunker = DocumentChunker(tokenizer_name="test-model", max_tokens=1234, merge_peers=True)

    summary = chunker.get_config_summary()

    assert summary["tokenizer_name"] == "test-model"
    assert summary["max_tokens"] == 1234
    assert summary["merge_peers"] is True


# ============================================================================
# Fix 6: Dynamic Chunker Configuration Tests
# ============================================================================


class TestDynamicChunkerConfiguration:
    """Tests for Fix 6: update_schema_config() method."""

    def test_update_schema_config_recalculates(self):
        """Recompute max_tokens using prompt overhead math."""
        with (
            patch("docling_graph.core.extractors.document_chunker.AutoTokenizer"),
            patch("docling_graph.core.extractors.document_chunker.HuggingFaceTokenizer"),
            patch("docling_graph.core.extractors.document_chunker.HybridChunker") as mock_hybrid,
            patch(
                "docling_graph.core.extractors.document_chunker.DocumentChunker._prompt_overhead_tokens",
                return_value=300,
            ),
        ):
            mock_chunker_instance = Mock()
            mock_chunker_instance.max_tokens = 8000
            mock_hybrid.return_value = mock_chunker_instance

            doc_chunker = DocumentChunker(max_tokens=8000)
            doc_chunker.original_max_tokens = 8000
            doc_chunker.context_limit = 8000

            doc_chunker.update_schema_config('{"title": "TestSchema"}')

            expected_tokens = 8000 - 300 - 2048 - 100
            assert doc_chunker.max_tokens == expected_tokens

    def test_update_schema_config_without_context_limit(self):
        """No context limit: keep existing max_tokens."""
        with (
            patch("docling_graph.core.extractors.document_chunker.AutoTokenizer"),
            patch("docling_graph.core.extractors.document_chunker.HuggingFaceTokenizer"),
            patch("docling_graph.core.extractors.document_chunker.HybridChunker"),
        ):
            doc_chunker = DocumentChunker(max_tokens=1000)
            doc_chunker.context_limit = None

            doc_chunker.update_schema_config('{"title": "NoContext"}')
            assert doc_chunker.max_tokens == 1000

    def test_update_schema_config_preserves_original(self):
        """Original max_tokens should remain unchanged after updates."""
        with (
            patch("docling_graph.core.extractors.document_chunker.AutoTokenizer"),
            patch("docling_graph.core.extractors.document_chunker.HuggingFaceTokenizer"),
            patch("docling_graph.core.extractors.document_chunker.HybridChunker") as mock_hybrid,
            patch(
                "docling_graph.core.extractors.document_chunker.DocumentChunker._prompt_overhead_tokens",
                return_value=200,
            ),
        ):
            mock_chunker_instance = Mock()
            mock_chunker_instance.max_tokens = 5000
            mock_hybrid.return_value = mock_chunker_instance

            doc_chunker = DocumentChunker(max_tokens=5000)
            doc_chunker.original_max_tokens = 5000
            doc_chunker.context_limit = 5000

            doc_chunker.update_schema_config('{"title": "A"}')
            doc_chunker.update_schema_config('{"title": "B"}')

            assert doc_chunker.original_max_tokens == 5000
