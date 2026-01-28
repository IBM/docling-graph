"""
LiteLLM-backed client implementation.

This client standardizes chat/completion calls through LiteLLM's OpenAI-style
API surface while preserving the BaseLlmClient contract.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict

from ..exceptions import ClientError, ConfigurationError
from .base import BaseLlmClient

logger = logging.getLogger(__name__)

try:
    import litellm
except ImportError as e:  # pragma: no cover - handled by configuration checks
    litellm = None  # type: ignore[assignment]
    _litellm_import_error = e
else:
    _litellm_import_error = None


class LiteLLMClient(BaseLlmClient):
    """LiteLLM client implementation using OpenAI-style calls."""

    def _provider_id(self) -> str:
        return "litellm"

    def _setup_client(self, **kwargs: Any) -> None:
        if litellm is None:
            raise ConfigurationError(
                "LiteLLM client requires 'litellm' package",
                details={"install_command": "pip install litellm"},
                cause=_litellm_import_error,
            )

    def _call_api(
        self, messages: list[Dict[str, str]], **params: Any
    ) -> tuple[str, Dict[str, Any]]:
        try:
            request = self._build_request(messages)
            # #region agent log
            try:
                with open(
                    "/home/ayoub/github/docling-graph/.cursor/debug.log",
                    "a",
                    encoding="utf-8",
                ) as log_file:
                    log_file.write(
                        json.dumps(
                            {
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "H2",
                                "location": "llm_clients/litellm.py:_call_api",
                                "message": "Calling LiteLLM completion",
                                "data": {
                                    "model": request.get("model"),
                                    "api_base": request.get("api_base"),
                                    "provider_id": self.model_config.provider_id,
                                    "drop_params": request.get("drop_params"),
                                    "has_api_key": bool(request.get("api_key")),
                                },
                                "timestamp": int(time.time() * 1000),
                            }
                        )
                        + "\n"
                    )
            except Exception:
                pass
            # #endregion
            response = litellm.completion(**request)

            choices = response.get("choices", [])
            if not choices:
                raise ClientError("LiteLLM returned no choices", details={"model": self.model})

            message = choices[0].get("message", {})
            content = message.get("content")
            if not content:
                raise ClientError("LiteLLM returned empty content", details={"model": self.model})

            metadata = {
                "finish_reason": choices[0].get("finish_reason"),
                "model": response.get("model", self.model),
                "usage": response.get("usage"),
            }
            return str(content), metadata
        except Exception as e:
            if isinstance(e, ClientError):
                raise
            # #region agent log
            try:
                with open(
                    "/home/ayoub/github/docling-graph/.cursor/debug.log",
                    "a",
                    encoding="utf-8",
                ) as log_file:
                    log_file.write(
                        json.dumps(
                            {
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "H4",
                                "location": "llm_clients/litellm.py:_call_api",
                                "message": "LiteLLM completion failed",
                                "data": {
                                    "error_type": type(e).__name__,
                                    "error": str(e)[:200],
                                },
                                "timestamp": int(time.time() * 1000),
                            }
                        )
                        + "\n"
                    )
            except Exception:
                pass
            # #endregion
            raise ClientError(
                f"LiteLLM API call failed: {type(e).__name__}",
                details={"model": self.model, "error": str(e)},
                cause=e,
            ) from e

    def _build_request(self, messages: list[Dict[str, str]]) -> dict[str, Any]:
        gen = self.generation
        max_tokens = gen.max_tokens or self._max_output_tokens

        model_name = self.model_config.litellm_model or self.model
        provider_id = self.model_config.provider_id
        if provider_id == "vllm" and self.connection.base_url:
            # OpenAI-compatible vLLM server mode per LiteLLM docs
            if model_name.startswith("vllm/"):
                model_name = model_name.removeprefix("vllm/")
            if model_name.startswith("hosted_vllm/"):
                model_name = model_name.removeprefix("hosted_vllm/")
            model_name = f"hosted_vllm/{model_name}"
        elif provider_id not in {"openai"} and not model_name.startswith(f"{provider_id}/"):
            model_name = f"{provider_id}/{model_name}"

        request: dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "temperature": gen.temperature,
            "max_tokens": max_tokens,
            "timeout": self.timeout,
            "drop_params": True,
        }
        if provider_id != "vllm":
            request["response_format"] = {"type": "json_object"}

        if gen.top_p is not None:
            request["top_p"] = gen.top_p
        if gen.top_k is not None:
            request["top_k"] = gen.top_k
        if gen.frequency_penalty is not None:
            request["frequency_penalty"] = gen.frequency_penalty
        if gen.presence_penalty is not None:
            request["presence_penalty"] = gen.presence_penalty
        if gen.seed is not None:
            request["seed"] = gen.seed
        if gen.stop is not None:
            request["stop"] = gen.stop

        connection = self.connection
        api_key = connection.api_key.get_secret_value() if connection.api_key else None
        if api_key:
            request["api_key"] = api_key
        if connection.base_url:
            request["api_base"] = connection.base_url
        if connection.organization:
            request["organization"] = connection.organization
        if connection.headers:
            request["headers"] = dict(connection.headers)

        supported_fn = getattr(litellm, "get_supported_openai_params", None)
        if callable(supported_fn):
            try:
                supported = supported_fn(model=model_name)
                if supported:
                    required = {
                        "model",
                        "messages",
                        "api_base",
                        "api_key",
                        "headers",
                        "organization",
                        "timeout",
                        "drop_params",
                        "response_format",
                    }
                    filtered = {key: value for key, value in request.items() if key in required}
                    filtered.update(
                        {key: value for key, value in request.items() if key in supported}
                    )
                    request = filtered
            except Exception:
                logger.debug("LiteLLM supported params lookup failed for %s", model_name)

        # #region agent log
        try:
            with open(
                "/home/ayoub/github/docling-graph/.cursor/debug.log",
                "a",
                encoding="utf-8",
            ) as log_file:
                log_file.write(
                    json.dumps(
                        {
                            "sessionId": "debug-session",
                            "runId": "run3",
                            "hypothesisId": "H3",
                            "location": "llm_clients/litellm.py:_build_request",
                            "message": "Built LiteLLM request",
                            "data": {
                                "model": request.get("model"),
                                "api_base": request.get("api_base"),
                                "provider_id": self.model_config.provider_id,
                                "has_api_key": bool(request.get("api_key")),
                                "drop_params": request.get("drop_params"),
                                "has_response_format": "response_format" in request,
                            },
                            "timestamp": int(time.time() * 1000),
                        }
                    )
                    + "\n"
                )
        except Exception:
            pass
        # #endregion
        return request
