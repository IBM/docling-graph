"""
IBM WatsonX API client implementation.

Based on https://ibm.github.io/watsonx-ai-python-sdk/fm_chat.html
"""

import json
import os
from typing import Any, Dict, cast

from dotenv import load_dotenv
from rich import print as rich_print

from .base import BaseLlmClient
from .config import get_context_limit, get_model_config

# Load environment variables
load_dotenv()

# Requires `pip install ibm-watsonx-ai`
# Make the lazy import optional to satisfy type checkers when assigning None
_WatsonxLLM: Any | None = None
_Credentials: Any | None = None
try:
    from ibm_watsonx_ai import Credentials as WatsonxCredentials  # type: ignore[import-untyped]
    from ibm_watsonx_ai.foundation_models import ModelInference  # type: ignore[import-untyped]

    _WatsonxLLM = ModelInference
    _Credentials = WatsonxCredentials
except ImportError:
    rich_print(
        "[red]Error:[/red] `ibm-watsonx-ai` package not found. "
        "Please run `pip install ibm-watsonx-ai` to use WatsonX client."
    )
    _WatsonxLLM = None
    _Credentials = None

# Expose as Any to allow None fallback without mypy issues
WatsonxLLM: Any = _WatsonxLLM
Credentials: Any = _Credentials


class WatsonxClient(BaseLlmClient):
    """IBM WatsonX API implementation with proper message structure."""

    def __init__(self, model: str) -> None:
        # Check if the required packages are available
        if _WatsonxLLM is None or _Credentials is None:
            raise ImportError(
                "\nWatsonX client requires 'ibm-watsonx-ai' package.\n"
                "Install with: pip install 'docling-graph[watsonx]'\n"
                "Or: pip install ibm-watsonx-ai"
            )

        self.model = model
        self.api_key = os.getenv("WATSONX_API_KEY")
        self.project_id = os.getenv("WATSONX_PROJECT_ID")
        self.url = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")

        if not self.api_key:
            raise ValueError(
                "[WatsonxClient] [red]Error:[/red] WATSONX_API_KEY not set. "
                "Please set it in your environment or .env file."
            )

        if not self.project_id:
            raise ValueError(
                "[WatsonxClient] [red]Error:[/red] WATSONX_PROJECT_ID not set. "
                "Please set it in your environment or .env file."
            )

        # Initialize WatsonX credentials
        credentials = Credentials(url=self.url, api_key=self.api_key)

        # Initialize WatsonX model
        self.client = WatsonxLLM(
            model_id=self.model,
            credentials=credentials,
            project_id=self.project_id,
        )

        # Get model configuration from centralized config
        model_config = get_model_config("watsonx", model)
        if model_config:
            self._context_limit = model_config.context_limit
            self._max_new_tokens = model_config.max_new_tokens
            rich_print(f"[WatsonxClient] Initialized for [blue]{self.model}[/blue]")
            rich_print(f"[WatsonxClient] Context: [cyan]{self._context_limit:,}[/cyan] tokens")
            rich_print(f"[WatsonxClient] Max output: [cyan]{self._max_new_tokens:,}[/cyan] tokens")
        else:
            # Fallback for unknown models
            self._context_limit = 8192
            self._max_new_tokens = 2048
            rich_print(
                f"[WatsonxClient] [yellow]Warning:[/yellow] Model '{model}' not in config, using defaults"
            )
            rich_print(f"[WatsonxClient] Context: [cyan]{self._context_limit:,}[/cyan] tokens")
            rich_print(f"[WatsonxClient] Max output: [cyan]{self._max_new_tokens:,}[/cyan] tokens")

        rich_print(f"[WatsonxClient] Using endpoint: [cyan]{self.url}[/cyan]")

    def get_json_response(self, prompt: str | dict, schema_json: str) -> Dict[str, Any]:
        """
        Execute WatsonX chat completion with JSON mode.

        Official docs: https://ibm.github.io/watsonx-ai-python-sdk/fm_chat.html

        Args:
            prompt: Either a string (legacy) or dict with 'system' and 'user' keys.
            schema_json: JSON schema (for reference, not directly used by WatsonX).

        Returns:
            Parsed JSON response from WatsonX.
        """
        # Build the prompt text
        if isinstance(prompt, dict):
            # Combine system and user messages
            system_content = prompt.get("system", "")
            user_content = prompt.get("user", "")

            # Format as a conversation
            prompt_text = f"{system_content}\n\n{user_content}"
        else:
            # Legacy string prompt
            prompt_text = prompt

        # Add JSON instruction to ensure JSON output
        prompt_text += "\n\nRespond with valid JSON only."

        try:
            # Configure generation parameters
            params = {
                "decoding_method": "greedy",
                "temperature": 0.1,  # Low temperature for consistent extraction
                "max_new_tokens": self._max_new_tokens,
                "min_new_tokens": 1,
                "repetition_penalty": 1.0,
            }

            # Generate response
            response = self.client.generate_text(prompt=prompt_text, params=params)

            # Extract the generated text
            if response is None:
                rich_print("[red]Error:[/red] WatsonX returned None response")
                return {}

            if not response or (isinstance(response, str) and not response.strip()):
                rich_print("[red]Error:[/red] WatsonX returned empty response")
                return {}

            # Parse JSON from response
            try:
                # Clean the response (remove markdown code blocks if present)
                content = str(response).strip()

                # Handle markdown code blocks with optional language specifier
                if "```json" in content:
                    # Extract content between ```json and ```
                    start_idx = content.find("```json") + 7
                    end_idx = content.find("```", start_idx)
                    if end_idx != -1:
                        content = content[start_idx:end_idx]
                elif "```" in content:
                    # Extract content between ``` and ```
                    start_idx = content.find("```") + 3
                    end_idx = content.find("```", start_idx)
                    if end_idx != -1:
                        content = content[start_idx:end_idx]
                else:
                    # No markdown blocks - look for JSON object or array
                    # Find the first { or [ which indicates start of JSON
                    json_start = -1
                    for char in ["{", "["]:
                        idx = content.find(char)
                        if idx != -1 and (json_start == -1 or idx < json_start):
                            json_start = idx

                    if json_start != -1:
                        content = content[json_start:]

                content = content.strip()

                parsed_json = json.loads(content)

                # Validate it's not empty
                if not parsed_json or (
                    isinstance(parsed_json, dict) and not any(parsed_json.values())
                ):
                    rich_print("[yellow]Warning:[/yellow] WatsonX returned empty or all-null JSON")

                if isinstance(parsed_json, dict):
                    return cast(Dict[str, Any], parsed_json)
                else:
                    rich_print(
                        "[yellow]Warning:[/yellow] Expected a JSON object; got non-dict. Returning empty dict."
                    )
                    return {}

            except json.JSONDecodeError as e:
                rich_print(f"[red]Error:[/red] Failed to parse WatsonX response as JSON: {e}")
                rich_print(f"[yellow]Raw response:[/yellow] {response}")
                return {}

        except Exception as e:
            rich_print(f"[red]Error:[/red] WatsonX API call failed: {e}")
            import traceback

            traceback.print_exc()
            return {}

    @property
    def context_limit(self) -> int:
        return self._context_limit
