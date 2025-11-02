"""
Mock document processors for testing.
"""

from typing import Any, Dict, List
from unittest.mock import MagicMock


class MockDocumentProcessor(MagicMock):
    """Mock document processor."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.process_document = MagicMock(
            return_value={
                "pages": [
                    {
                        "page_num": 1,
                        "text": "Sample document text",
                        "metadata": {},
                    }
                ],
                "metadata": {"total_pages": 1},
            }
        )
        self.extract_text = MagicMock(return_value="Sample extracted text")
        self.get_pages = MagicMock(return_value=[MagicMock(text="Page 1 text", page_num=1)])
