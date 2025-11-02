"""
Mock backends for testing extractors.
"""

from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock


class MockLLMBackend(MagicMock):
    """Mock LLM backend for testing."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.process = MagicMock(return_value={"entities": [], "relations": []})
        self.batch_process = MagicMock(return_value=[{"entities": [], "relations": []}])


class MockVLMBackend(MagicMock):
    """Mock VLM backend for testing."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.extract = MagicMock(
            return_value={
                "text": "Extracted text",
                "structured": {"field1": "value1"},
            }
        )
        self.batch_extract = MagicMock(return_value=[{"text": "Extracted text", "structured": {}}])
