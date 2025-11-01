"""
Unit tests for protocol definitions and type checking utilities.
"""

from unittest.mock import Mock

import pytest

from docling_graph.protocols import (
    ExtractionBackendProtocol,
    TextExtractionBackendProtocol,
    ExtractorProtocol,
    LLMClientProtocol,
    DocumentProcessorProtocol,
    is_vlm_backend,
    is_llm_backend,
    get_backend_type,
)


class TestExtractionBackendProtocol:
    """Tests for ExtractionBackendProtocol."""

    def test_protocol_with_required_methods(self):
        """Test class implementing protocol."""
        class VLMBackend:
            def extract_from_document(self, source, template):
                return []
            
            def cleanup(self):
                pass
        
        backend = VLMBackend()
        assert isinstance(backend, ExtractionBackendProtocol)

    def test_protocol_missing_extract_method(self):
        """Test protocol requires extract_from_document."""
        class IncompleteBackend:
            def cleanup(self):
                pass
        
        backend = IncompleteBackend()
        assert not isinstance(backend, ExtractionBackendProtocol)


class TestTextExtractionBackendProtocol:
    """Tests for TextExtractionBackendProtocol."""

    def test_protocol_with_required_methods(self):
        """Test class implementing protocol."""
        class LLMBackend:
            client = Mock()
            
            def extract_from_markdown(self, markdown, template, context="document"):
                return None
            
            def cleanup(self):
                pass
        
        backend = LLMBackend()
        assert isinstance(backend, TextExtractionBackendProtocol)

    def test_protocol_missing_extract_method(self):
        """Test protocol requires extract_from_markdown."""
        class IncompleteBackend:
            client = Mock()
            
            def cleanup(self):
                pass
        
        backend = IncompleteBackend()
        assert not isinstance(backend, TextExtractionBackendProtocol)


class TestLLMClientProtocol:
    """Tests for LLMClientProtocol."""

    def test_protocol_with_required_methods(self):
        """Test class implementing protocol."""
        class OllamaClient:
            @property
            def context_limit(self):
                return 4096
            
            def get_json_response(self, prompt, schema_json):
                return {}
        
        client = OllamaClient()
        assert isinstance(client, LLMClientProtocol)

    def test_protocol_missing_context_limit(self):
        """Test protocol requires context_limit."""
        class IncompleteClient:
            def get_json_response(self, prompt, schema_json):
                return {}
        
        client = IncompleteClient()
        assert not isinstance(client, LLMClientProtocol)


class TestExtractorProtocol:
    """Tests for ExtractorProtocol."""

    def test_protocol_with_required_methods(self):
        """Test class implementing protocol."""
        class OneToOneStrategy:
            backend = Mock()
            
            def extract(self, source, template):
                return []
        
        extractor = OneToOneStrategy()
        assert isinstance(extractor, ExtractorProtocol)


class TestDocumentProcessorProtocol:
    """Tests for DocumentProcessorProtocol."""

    def test_protocol_with_required_methods(self):
        """Test class implementing protocol."""
        class DocProcessor:
            def convert_to_markdown(self, source):
                return Mock()
            
            def extract_full_markdown(self, document):
                return "# Content"
            
            def extract_page_markdowns(self, document):
                return ["# Page 1"]
        
        processor = DocProcessor()
        assert isinstance(processor, DocumentProcessorProtocol)


class TestIsVLMBackend:
    """Tests for is_vlm_backend type guard."""

    def test_vlm_backend_detected(self):
        """Test detecting VLM backend."""
        backend = Mock()
        backend.extract_from_document = Mock()
        
        assert is_vlm_backend(backend) is True

    def test_non_vlm_backend(self):
        """Test non-VLM backend."""
        backend = Mock(spec=[])  # No extract_from_document
        
        assert is_vlm_backend(backend) is False

    def test_llm_backend_not_vlm(self):
        """Test LLM backend is not VLM."""
        backend = Mock()
        backend.extract_from_markdown = Mock()
        backend.extract_from_document = None
        
        assert is_vlm_backend(backend) is False


class TestIsLLMBackend:
    """Tests for is_llm_backend type guard."""

    def test_llm_backend_detected(self):
        """Test detecting LLM backend."""
        backend = Mock()
        backend.extract_from_markdown = Mock()
        
        assert is_llm_backend(backend) is True

    def test_non_llm_backend(self):
        """Test non-LLM backend."""
        backend = Mock(spec=[])  # No extract_from_markdown
        
        assert is_llm_backend(backend) is False

    def test_vlm_backend_not_llm(self):
        """Test VLM backend is not LLM."""
        backend = Mock()
        backend.extract_from_document = Mock()
        backend.extract_from_markdown = None
        
        assert is_llm_backend(backend) is False


class TestGetBackendType:
    """Tests for get_backend_type function."""

    def test_vlm_backend_type(self):
        """Test identifying VLM backend."""
        backend = Mock()
        backend.extract_from_document = Mock()
        backend.extract_from_markdown = None
        
        assert get_backend_type(backend) == "vlm"

    def test_llm_backend_type(self):
        """Test identifying LLM backend."""
        backend = Mock()
        backend.extract_from_markdown = Mock()
        backend.extract_from_document = None
        
        assert get_backend_type(backend) == "llm"

    def test_both_methods_returns_vlm(self):
        """Test when both methods present, VLM is returned first."""
        backend = Mock()
        backend.extract_from_document = Mock()
        backend.extract_from_markdown = Mock()
        
        assert get_backend_type(backend) == "vlm"

    def test_unknown_backend_type(self):
        """Test unknown backend type."""
        backend = Mock(spec=[])  # No methods
        
        assert get_backend_type(backend) == "unknown"
