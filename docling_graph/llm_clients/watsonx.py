"""
IBM WatsonX API client implementation.

Based on https://ibm.github.io/watsonx-ai-python-sdk/fm_chat.html
"""

import json
import os
from typing import Any, Dict, cast

from dotenv import load_dotenv
from rich import print as rich_print

from .llm_base import BaseLlmClient

# Load environment variables
load_dotenv()

# Requires `pip install ibm-watsonx-ai`
# Make the lazy import optional to satisfy type checkers when assigning None
_WatsonxLLM: Any | None = None
_Credentials: Any | None = None
try:
    from ibm_watsonx_ai import Credentials
    from ibm_watsonx_ai.foundation_models import ModelInference

    _WatsonxLLM = ModelInference
    _Credentials = Credentials
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

        # Context limits for different models
        model_context_limits = {
            "ibm-granite/granite-4.0-h-small": 128000,
            "meta-llama/llama-3-70b-instruct": 32768,
            "meta-llama/llama-3-8b-instruct": 32768,
            "mistralai/mixtral-8x7b-instruct-v01": 32768,
        }

        self._context_limit = model_context_limits.get(model, 8192)
        rich_print(f"[WatsonxClient] Initialized for [blue]{self.model}[/blue]")
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
                "max_new_tokens": 4096,
                "min_new_tokens": 1,
                "repetition_penalty": 1.0,
            }

            # Generate response
            response = self.client.generate_text(prompt=prompt_text, params=params)

            # Extract the generated text
            if not response:
                rich_print("[red]Error:[/red] WatsonX returned empty response")
                return {}

            # Parse JSON from response
            try:
                # Clean the response (remove markdown code blocks if present)
                content = response.strip()
                if content.startswith("```json"):
                    content = content[7:]
                if content.startswith("```"):
                    content = content[3:]
                if content.endswith("```"):
                    content = content[:-3]
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

# Made with Bob, 
