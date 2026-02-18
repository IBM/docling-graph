"""
Unit tests for ManyToOneStrategy.

Tests the many-to-one strategy with direct full-document extraction:
- Direct extraction (single LLM call)
- VLM backend support
"""

from typing import List
from unittest.mock import MagicMock, Mock, patch

import pytest
from pydantic import BaseModel

from docling_graph.core.extractors.strategies.many_to_one import ManyToOneStrategy
from docling_graph.protocols import ExtractionBackendProtocol, TextExtractionBackendProtocol


class MockTemplate(BaseModel):
    """Simple test template."""

    name: str
    value: int = 0


@pytest.fixture
def mock_llm_backend():
    """Create a mock LLM backend."""
    backend = MagicMock(spec=TextExtractionBackendProtocol)
    backend.client = MagicMock()
    backend.__class__.__name__ = "MockLlmBackend"

    def mock_extract(markdown, template, context, is_partial) -> MockTemplate | None:
        if "fail" in markdown:
            return None
        return template(name=context, value=len(markdown))

    backend.extract_from_markdown.side_effect = mock_extract

    return backend


@pytest.fixture
def mock_vlm_backend():
    """Create a mock VLM backend."""
    backend = MagicMock(spec=ExtractionBackendProtocol)
    backend.__class__.__name__ = "MockVlmBackend"

    def mock_extract(source, template) -> List[MockTemplate]:
        if "single" in source:
            return [template(name="Page 1", value=10)]
        if "multi" in source:
            return [
                template(name="Page 1", value=10),
                template(name="Page 2", value=20),
            ]
        return []

    backend.extract_from_document.side_effect = mock_extract

    return backend


@pytest.fixture(autouse=True)
def patch_deps():
    """Patch common dependencies."""
    with (
        patch("docling_graph.core.extractors.strategies.many_to_one.DocumentProcessor") as mock_dp,
        patch(
            "docling_graph.core.extractors.strategies.many_to_one.merge_pydantic_models"
        ) as mock_merge,
        patch("docling_graph.core.extractors.strategies.many_to_one.is_llm_backend") as mock_is_llm,
        patch("docling_graph.core.extractors.strategies.many_to_one.is_vlm_backend") as mock_is_vlm,
    ):
        mock_doc_processor = mock_dp.return_value
        mock_doc_processor.convert_to_docling_doc.return_value = "MockDoc"
        mock_doc_processor.extract_full_markdown.return_value = "full_doc_md"

        mock_merge.return_value = MockTemplate(name="Merged", value=123)

        mock_is_llm.return_value = False
        mock_is_vlm.return_value = False

        yield mock_dp, mock_merge, mock_is_llm, mock_is_vlm


class TestInitialization:
    """Test strategy initialization."""

    def test_init_with_llm_backend(self, mock_llm_backend, patch_deps):
        """Test initialization with LLM backend."""
        _, _, mock_is_llm, _ = patch_deps
        mock_is_llm.return_value = True

        strategy = ManyToOneStrategy(backend=mock_llm_backend)

        assert strategy.backend == mock_llm_backend

    def test_init_with_docling_config(self, mock_llm_backend, patch_deps):
        """Test initialization with custom docling config."""
        _, _, mock_is_llm, _ = patch_deps
        mock_is_llm.return_value = True

        strategy = ManyToOneStrategy(
            backend=mock_llm_backend,
            docling_config="vision",
        )

        assert strategy.doc_processor is not None

    def test_delta_initializes_chunker_with_default_tokens(self, mock_llm_backend, patch_deps):
        """Delta mode should always pass non-empty chunker_config."""
        mock_dp, _, mock_is_llm, _ = patch_deps
        mock_is_llm.return_value = True

        ManyToOneStrategy(
            backend=mock_llm_backend,
            extraction_contract="delta",
            use_chunking=True,
            chunk_max_tokens=None,
        )

        kwargs = mock_dp.call_args.kwargs
        assert kwargs["chunker_config"] == {"chunk_max_tokens": 512}

    def test_delta_initializes_chunker_with_custom_tokens(self, mock_llm_backend, patch_deps):
        """Delta mode should propagate configured chunk_max_tokens."""
        mock_dp, _, mock_is_llm, _ = patch_deps
        mock_is_llm.return_value = True

        ManyToOneStrategy(
            backend=mock_llm_backend,
            extraction_contract="delta",
            use_chunking=True,
            chunk_max_tokens=1024,
        )

        kwargs = mock_dp.call_args.kwargs
        assert kwargs["chunker_config"] == {"chunk_max_tokens": 1024}

    def test_delta_requires_chunking_enabled(self, mock_llm_backend, patch_deps):
        """Delta mode should reject disabled chunking."""
        _, _, mock_is_llm, _ = patch_deps
        mock_is_llm.return_value = True

        with pytest.raises(ValueError, match="requires use_chunking=True"):
            ManyToOneStrategy(
                backend=mock_llm_backend,
                extraction_contract="delta",
                use_chunking=False,
            )


class TestVLMExtraction:
    """Test VLM backend extraction."""

    def test_extract_single_page(self, mock_vlm_backend, patch_deps):
        """Test VLM extraction for single-page document."""
        _, mock_merge, _, mock_is_vlm = patch_deps
        mock_is_vlm.return_value = True

        strategy = ManyToOneStrategy(backend=mock_vlm_backend)
        results, _document = strategy.extract("single_page_doc.pdf", MockTemplate)

        assert len(results) == 1
        assert results[0].name == "Page 1"
        mock_merge.assert_not_called()

    def test_extract_multi_page(self, mock_vlm_backend, patch_deps):
        """Test VLM extraction and merge for multi-page document."""
        _, mock_merge, _, mock_is_vlm = patch_deps
        mock_is_vlm.return_value = True

        strategy = ManyToOneStrategy(backend=mock_vlm_backend)
        results, _document = strategy.extract("multi_page_doc.pdf", MockTemplate)

        assert len(results) == 1
        assert results[0].name == "Merged"
        mock_merge.assert_called_once()

    def test_merge_failure_returns_all_pages(self, mock_vlm_backend, patch_deps):
        """Test that VLM merge failure returns all page models (zero data loss)."""
        _, mock_merge, _, mock_is_vlm = patch_deps
        mock_is_vlm.return_value = True

        mock_merge.return_value = None

        strategy = ManyToOneStrategy(backend=mock_vlm_backend)
        results, _ = strategy.extract("multi_page_doc.pdf", MockTemplate)

        assert len(results) == 2
        assert results[0].name == "Page 1"
        assert results[1].name == "Page 2"


class TestDirectExtraction:
    """Test direct extraction (single LLM call)."""

    def test_direct_full_document_extraction(self, mock_llm_backend, patch_deps):
        """Test direct full-document extraction."""
        mock_dp, _, mock_is_llm, _ = patch_deps
        mock_is_llm.return_value = True

        mock_doc_processor = mock_dp.return_value
        mock_doc_processor.extract_full_markdown.return_value = "test content"

        strategy = ManyToOneStrategy(backend=mock_llm_backend)
        results, _ = strategy.extract("test.pdf", MockTemplate)

        assert mock_llm_backend.extract_from_markdown.called
        assert len(results) >= 0

    def test_direct_failure_returns_empty(self, mock_llm_backend, patch_deps):
        """Test that direct extraction returns empty list on failure."""
        mock_dp, _, mock_is_llm, _ = patch_deps
        mock_is_llm.return_value = True

        mock_doc_processor = mock_dp.return_value
        mock_doc_processor.extract_full_markdown.return_value = "fail"

        strategy = ManyToOneStrategy(backend=mock_llm_backend)
        results, _ = strategy.extract("test.pdf", MockTemplate)

        assert len(results) == 0

    def test_delta_falls_back_to_direct_when_no_model(self, mock_llm_backend, patch_deps):
        """Delta mode should retry once with direct extraction when delta returns no model."""
        mock_dp, _, mock_is_llm, _ = patch_deps
        mock_is_llm.return_value = True
        mock_doc_processor = mock_dp.return_value
        mock_doc_processor.extract_full_markdown.return_value = "invoice markdown"
        mock_doc_processor.extract_chunks_with_metadata.return_value = (
            ["chunk1"],
            [{"chunk_id": 0, "token_count": 15, "page_numbers": [1]}],
        )

        mock_llm_backend.extraction_contract = "delta"
        mock_llm_backend.extract_from_chunk_batches = Mock(return_value=None)
        mock_llm_backend.extract_from_markdown = Mock(
            return_value=MockTemplate(name="fallback", value=123)
        )

        strategy = ManyToOneStrategy(backend=mock_llm_backend, extraction_contract="delta")
        results, _ = strategy.extract("test.pdf", MockTemplate)

        assert len(results) == 1
        assert results[0].name == "fallback"
        mock_llm_backend.extract_from_chunk_batches.assert_called_once()
        mock_llm_backend.extract_from_markdown.assert_called_once()

    def test_delta_fallback_emits_trace_event(self, mock_llm_backend, patch_deps):
        """Delta fallback should emit explicit trace diagnostics."""
        mock_dp, _, mock_is_llm, _ = patch_deps
        mock_is_llm.return_value = True
        mock_doc_processor = mock_dp.return_value
        mock_doc_processor.extract_full_markdown.return_value = "invoice markdown"
        mock_doc_processor.extract_chunks_with_metadata.return_value = (
            ["chunk1"],
            [{"chunk_id": 0, "token_count": 15, "page_numbers": [1]}],
        )

        mock_llm_backend.extraction_contract = "delta"
        mock_llm_backend.extract_from_chunk_batches = Mock(return_value=None)
        mock_llm_backend.extract_from_markdown = Mock(
            return_value=MockTemplate(name="fallback", value=123)
        )

        strategy = ManyToOneStrategy(backend=mock_llm_backend, extraction_contract="delta")
        strategy.trace_data = MagicMock()
        strategy.trace_data.latest_payload.return_value = {
            "quality_gate": {"ok": False, "reasons": ["missing_root_instance"]},
            "merge_stats": {"parent_lookup_miss": 2},
            "normalizer_stats": {"unknown_path_dropped": 1},
        }

        results, _ = strategy.extract("test.pdf", MockTemplate)

        assert len(results) == 1
        emit_calls = strategy.trace_data.emit.call_args_list
        assert any(call.args[0] == "delta_failed_then_direct_fallback" for call in emit_calls)

    def test_extract_unknown_backend_returns_empty_and_none(self, mock_llm_backend, patch_deps):
        """Backend that is neither LLM nor VLM: TypeError is caught, returns [], None (106-117)."""
        _, _, mock_is_llm, mock_is_vlm = patch_deps
        mock_is_llm.return_value = False
        mock_is_vlm.return_value = False

        strategy = ManyToOneStrategy(backend=mock_llm_backend)
        results, doc = strategy.extract("test.pdf", MockTemplate)

        assert results == []
        assert doc is None

    def test_vlm_exception_returns_empty_and_none(self, mock_vlm_backend, patch_deps):
        """VLM extract_from_document raises -> returns [], None and logger.error (148-153)."""
        _, _, _, mock_is_vlm = patch_deps
        mock_is_vlm.return_value = True
        mock_vlm_backend.extract_from_document.side_effect = RuntimeError("VLM failed")

        strategy = ManyToOneStrategy(backend=mock_vlm_backend)
        results, doc = strategy.extract("test.pdf", MockTemplate)

        assert results == []
        assert doc is None

    def test_vlm_merge_returns_none_returns_all_page_models(self, mock_vlm_backend, patch_deps):
        """VLM merge_pydantic_models returns None -> return all page models (144-146)."""
        _, mock_merge, _, mock_is_vlm = patch_deps
        mock_is_vlm.return_value = True
        page1 = MockTemplate(name="P1", value=1)
        page2 = MockTemplate(name="P2", value=2)
        mock_vlm_backend.extract_from_document.side_effect = None
        mock_vlm_backend.extract_from_document.return_value = [page1, page2]
        mock_merge.return_value = None

        strategy = ManyToOneStrategy(backend=mock_vlm_backend)
        results, _ = strategy.extract("multi.pdf", MockTemplate)

        assert len(results) == 2
        assert results[0].name == "P1" and results[1].name == "P2"

    @patch("docling_graph.core.extractors.strategies.many_to_one.extract_delta_from_text")
    def test_extract_direct_mode_from_text_delta_path_emits_trace(
        self, mock_extract_delta, mock_llm_backend, patch_deps
    ):
        """_extract_direct_mode_from_text with delta contract and trace_data emits events (218-264)."""
        _mock_dp, _, mock_is_llm, _ = patch_deps
        mock_is_llm.return_value = True
        mock_llm_backend.extract_from_chunk_batches = Mock(return_value=None)
        mock_llm_backend.extract_from_markdown = Mock(return_value=MockTemplate(name="T", value=1))
        mock_extract_delta.return_value = (MockTemplate(name="Delta", value=2), 1.0)

        strategy = ManyToOneStrategy(
            backend=mock_llm_backend,
            extraction_contract="delta",
        )
        strategy.trace_data = MagicMock()

        # Call _extract_direct_mode_from_text directly (entry used by pipeline for text input)
        results, _ = strategy._extract_with_llm_from_text(
            mock_llm_backend, "markdown text", MockTemplate
        )

        assert len(results) == 1
        assert results[0].name == "Delta"
        emit_calls = [c[0][0] for c in strategy.trace_data.emit.call_args_list]
        assert (
            "page_markdown_extracted" in emit_calls or "docling_conversion_completed" in emit_calls
        )
        assert "extraction_completed" in emit_calls

    @patch("docling_graph.core.extractors.strategies.many_to_one.extract_delta_from_text")
    def test_extract_direct_mode_from_text_exception_emits_extraction_failed(
        self, mock_extract_delta, mock_llm_backend, patch_deps
    ):
        """_extract_direct_mode_from_text exception -> extraction_failed emit and [], None (274-290)."""
        _mock_dp, _, mock_is_llm, _ = patch_deps
        mock_is_llm.return_value = True
        mock_llm_backend.extract_from_chunk_batches = Mock()
        mock_extract_delta.side_effect = RuntimeError("delta failed")

        strategy = ManyToOneStrategy(
            backend=mock_llm_backend,
            extraction_contract="delta",
        )
        strategy.trace_data = MagicMock()

        results, _ = strategy._extract_with_llm_from_text(mock_llm_backend, "text", MockTemplate)

        assert results == []
        emit_calls = [c[0][0] for c in strategy.trace_data.emit.call_args_list]
        assert "extraction_failed" in emit_calls

    def test_extract_direct_mode_no_model_returns_empty_list_and_document(
        self, mock_llm_backend, patch_deps
    ):
        """Direct path: extract_from_markdown returns None -> [], document (462-464)."""
        mock_dp, _, mock_is_llm, _ = patch_deps
        mock_is_llm.return_value = True
        mock_doc = MagicMock()
        mock_dp.return_value.convert_to_docling_doc.return_value = mock_doc
        mock_dp.return_value.extract_full_markdown.return_value = "full md"
        mock_llm_backend.extract_from_markdown.side_effect = None
        mock_llm_backend.extract_from_markdown.return_value = None

        strategy = ManyToOneStrategy(backend=mock_llm_backend)
        results, doc = strategy.extract("test.pdf", MockTemplate)

        assert results == []
        assert doc is mock_doc

    @patch("docling_graph.core.extractors.strategies.many_to_one.extract_delta_from_document")
    def test_delta_fallback_returned_no_model_emits_trace(
        self, mock_extract_delta_doc, mock_llm_backend, patch_deps
    ):
        """Delta returns None, direct fallback returns None -> emit direct_fallback_returned_no_model (374-386)."""
        mock_dp, _, mock_is_llm, _ = patch_deps
        mock_is_llm.return_value = True
        mock_dp.return_value.convert_to_docling_doc.return_value = MagicMock()
        mock_dp.return_value.extract_full_markdown.return_value = "md"
        mock_extract_delta_doc.return_value = (None, 0.0)
        mock_llm_backend.extraction_contract = "delta"
        mock_llm_backend.extract_from_chunk_batches = Mock()
        mock_llm_backend.extract_from_markdown = Mock(return_value=None)

        strategy = ManyToOneStrategy(backend=mock_llm_backend, extraction_contract="delta")
        strategy.trace_data = MagicMock()
        strategy.trace_data.latest_payload.return_value = {}

        results, _ = strategy.extract("test.pdf", MockTemplate)

        assert results == []
        emit_calls = strategy.trace_data.emit.call_args_list
        reasons = []
        for c in emit_calls:
            if len(c[0]) >= 3 and isinstance(c[0][2], dict) and "reason" in c[0][2]:
                reasons.append(c[0][2]["reason"])
        assert any(
            c[0][2].get("reason") == "direct_fallback_returned_no_model"
            for c in emit_calls
            if len(c[0]) > 2 and isinstance(c[0][2], dict)
        ), f"Expected emit with reason direct_fallback_returned_no_model, got reasons={reasons}"

    @patch("docling_graph.core.extractors.strategies.many_to_one.extract_delta_from_document")
    def test_extract_direct_fallback_model_switches_contract_then_restores(
        self, mock_extract_delta_doc, mock_llm_backend, patch_deps
    ):
        """_extract_direct_fallback_model sets backend.extraction_contract to direct then restores (394-418)."""
        mock_dp, _, mock_is_llm, _ = patch_deps
        mock_is_llm.return_value = True
        mock_dp.return_value.extract_full_markdown.return_value = "full md"
        mock_extract_delta_doc.return_value = (None, 0.0)
        mock_llm_backend.extraction_contract = "delta"
        mock_llm_backend.extract_from_markdown = Mock(
            return_value=MockTemplate(name="Fallback", value=99)
        )

        strategy = ManyToOneStrategy(backend=mock_llm_backend, extraction_contract="delta")
        results, _ = strategy.extract("test.pdf", MockTemplate)

        assert len(results) == 1
        assert results[0].name == "Fallback"
        assert mock_llm_backend.extraction_contract == "delta"
        call_kwargs = mock_llm_backend.extract_from_markdown.call_args[1]
        assert call_kwargs.get("is_partial") is False
