"""Source trust stage — optional document-authenticity verification.

A :class:`~docling_graph.pipeline.stages.PipelineStage` that inspects the
source document via the Stipple API (https://www.stipple.sh) before
extraction, and attaches the verdict to the pipeline context. docling-graph
already builds a strong deterministic provenance ledger (where each claim
came from); this stage adds the complementary question of whether the
source document itself is trustworthy — a tampered contract or an
AI-generated fake act yields a perfectly traceable graph of fraudulent
facts.

The verdict (risk band, per-signal evidence, AI-text probability, and a
re-verifiable ``warrant_id``) is stored in
``context.metadata["source_trust"]`` and available to downstream stages,
exporters, and graph node attributes.

Free anonymous tier: no API key required. Set ``STIPPLE_API_KEY`` for your
own metering. The stage is best-effort: API failures are recorded as
``{"error": ...}`` and never break the pipeline.

Usage (insert before the extraction stage)::

    from docling_graph.core.stages_source_trust import SourceTrustStage
    stages = [InputNormalizationStage(), SourceTrustStage(), ExtractionStage(), ...]

Enforce a policy with ``block_above="medium"`` to raise
:class:`SourceTrustBlockedError` when the risk band meets or exceeds the
configured level.
"""

import json
import logging
import os
import urllib.request
import uuid
from pathlib import Path
from typing import Optional

from docling_graph.exceptions import DoclingGraphError
from docling_graph.pipeline.stages import PipelineStage

logger = logging.getLogger(__name__)

STIPPLE_BASE_URL = os.getenv("STIPPLE_BASE_URL", "https://www.stipple.sh")
_REQUEST_TIMEOUT = 300  # seconds


class SourceTrustBlockedError(DoclingGraphError):
    """Raised when enforce policy rejects a source document."""


def _headers() -> dict:
    headers = {
        "User-Agent": "docling-graph-source-trust/1.0",
        "Accept": "application/json",
    }
    api_key = os.getenv("STIPPLE_API_KEY", "").strip()
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    return headers


def _post_file(endpoint: str, file_path: Path) -> Optional[dict]:
    """POST a document as multipart to a Stipple endpoint. Best-effort."""
    try:
        boundary = "----docling-graph-trust" + uuid.uuid4().hex
        with file_path.open("rb") as f:
            content = f.read()
        body = b"".join(
            [
                (
                    f"--{boundary}\r\n"
                    f'Content-Disposition: form-data; name="file"; '
                    f'filename="{file_path.name}"\r\n'
                    "Content-Type: application/octet-stream\r\n\r\n"
                ).encode(),
                content,
                b"\r\n",
                f"--{boundary}--\r\n".encode(),
            ]
        )
        req = urllib.request.Request(
            STIPPLE_BASE_URL + endpoint,
            data=body,
            method="POST",
            headers={
                **_headers(),
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            },
        )
        with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT) as resp:
            return json.loads(resp.read().decode())
    except Exception:  # noqa: BLE001 - verification is best-effort by design
        return None


class SourceTrustStage(PipelineStage):
    """Inspect the source document's authenticity before extraction.

    Args:
        file_path: Path to the original source document. When ``None``,
            the stage attempts to use the pipeline context's recorded
            source (best-effort) and skips when unavailable.
        block_above: Optional risk band threshold ("medium" or "high").
            When the verdict's band meets or exceeds it,
            :class:`SourceTrustBlockedError` is raised (enforcing mode).
            Default: advisory-only (never raises).
        verify_ai_text: Also run AI-written-prose detection (default True).
    """

    def __init__(
        self,
        file_path: str | Path | None = None,
        block_above: Optional[str] = None,
        verify_ai_text: bool = True,
    ) -> None:
        if block_above is not None and block_above not in ("medium", "high"):
            raise ValueError("block_above must be 'medium' or 'high' (or None)")
        self.file_path = Path(file_path) if file_path else None
        self.block_above = block_above
        self.verify_ai_text = verify_ai_text

    def name(self) -> str:
        return "Source Trust"

    def _verify(self, path: Path) -> dict:
        block: dict = {}
        warrant = _post_file("/v1/warrants", path)
        if warrant:
            block["authenticity"] = {
                "warrant_id": warrant.get("warrant_id"),
                "risk_band": warrant.get("risk_band"),
                "risk_score": warrant.get("risk_score"),
                "inspection_quality": warrant.get("inspection_quality"),
                "recommended_action": warrant.get("recommended_action"),
                "summary": warrant.get("summary"),
            }
        else:
            block["error"] = "verification unavailable"
        if self.verify_ai_text:
            ai = _post_file("/v1/detect-ai-text", path)
            if ai:
                block["ai_text"] = (
                    {"applicable": False}
                    if ai.get("applicable") is False
                    else {
                        "applicable": True,
                        "probability": ai.get("probability"),
                        "lean": ai.get("lean"),
                        "tells": ai.get("tells"),
                    }
                )
        return block

    def execute(self, context) -> object:
        path = self.file_path
        if path is None or not path.is_file():
            logger.info("Source trust: no local source file available, skipping")
            return context
        logger.info("Verifying source document: %s", path.name)
        block = self._verify(path)
        if context.input_metadata is None:
            context.input_metadata = {}
        context.input_metadata["source_trust"] = block
        auth = block.get("authenticity") or {}
        band = auth.get("risk_band")
        if band:
            logger.info("Source trust verdict: risk_band=%s warrant=%s", band, auth.get("warrant_id"))
        if (
            self.block_above
            and band
            and {"medium": 1, "high": 2}.get(band, 0) >= {"medium": 1, "high": 2}[self.block_above]
        ):
            raise SourceTrustBlockedError(
                f"Source document rejected by trust policy: risk_band={band}, "
                f"warrant={auth.get('warrant_id')}"
            )
        return context
