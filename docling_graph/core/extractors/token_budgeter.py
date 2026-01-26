"""
Token budget management for slot-based extraction.

This module ensures that batched slot extraction requests fit within
model context windows by checking token counts and adjusting batch sizes.
"""

import json
import logging
from typing import Optional

from ...llm_clients.config import ModelConfig, get_model_config

logger = logging.getLogger(__name__)


class TokenBudgeter:
    """
    Manages token budgets for slot-based extraction.

    Ensures that extraction requests fit within model context windows
    by estimating token counts and adjusting batch sizes when needed.
    """

    def __init__(
        self,
        provider: str,
        model: str,
        system_prompt_tokens: int = 500,
        response_buffer_tokens: int = 1000,
        safety_margin: float = 0.9,
    ) -> None:
        """
        Initialize the token budgeter.

        Args:
            provider: LLM provider ID (e.g., "mistral", "openai")
            model: Model name
            system_prompt_tokens: Estimated tokens for system prompt (default: 500)
            response_buffer_tokens: Tokens reserved for response (default: 1000)
            safety_margin: Safety factor for context limit (default: 0.9 = 90%)
        """
        self.provider = provider
        self.model = model
        self.system_prompt_tokens = system_prompt_tokens
        self.response_buffer_tokens = response_buffer_tokens
        self.safety_margin = safety_margin

        # Load model config
        self.model_config: ModelConfig | None = get_model_config(provider, model)

        if not self.model_config:
            logger.warning(
                f"Model config not found for {provider}/{model}, using defaults"
            )
            self.context_limit = 8192
            self.max_new_tokens = 4096
        else:
            self.context_limit = self.model_config.context_limit
            self.max_new_tokens = self.model_config.max_new_tokens

        # Calculate available tokens for content
        self.available_tokens = int(
            (self.context_limit - system_prompt_tokens - response_buffer_tokens)
            * safety_margin
        )

        logger.info(
            f"TokenBudgeter initialized: context={self.context_limit}, "
            f"available={self.available_tokens}, max_new={self.max_new_tokens}"
        )

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.

        Uses a simple heuristic: ~3.5 characters per token for most models.
        This is conservative and works reasonably well across tokenizers.

        Args:
            text: Text to estimate

        Returns:
            Estimated token count
        """
        return int(len(text) / 3.5)

    def estimate_schema_tokens(self, schema: dict | str) -> int:
        """
        Estimate tokens for a JSON schema.

        Args:
            schema: Schema as dict or JSON string

        Returns:
            Estimated token count
        """
        if isinstance(schema, dict):
            schema_str = json.dumps(schema)
        else:
            schema_str = schema

        return self.estimate_tokens(schema_str)

    def estimate_slot_batch_tokens(
        self,
        slots: list[dict[str, str]],
        schema: dict | str,
    ) -> int:
        """
        Estimate total tokens for a slot batch extraction request.

        Includes:
        - System prompt
        - Slot contents
        - Schema
        - Response buffer

        Args:
            slots: List of slot dictionaries with 'content'
            schema: Entity schema

        Returns:
            Estimated total token count
        """
        # System prompt
        total = self.system_prompt_tokens

        # Slot contents
        for slot in slots:
            content = slot.get("content", "")
            total += self.estimate_tokens(content)
            total += 20  # Overhead for slot formatting

        # Schema
        total += self.estimate_schema_tokens(schema)

        # Response buffer
        total += self.response_buffer_tokens

        return total

    def check_fits(
        self,
        slots: list[dict[str, str]],
        schema: dict | str,
    ) -> tuple[bool, int, int]:
        """
        Check if a slot batch fits within the context window.

        Args:
            slots: List of slot dictionaries
            schema: Entity schema

        Returns:
            Tuple of (fits, estimated_tokens, available_tokens)
        """
        estimated = self.estimate_slot_batch_tokens(slots, schema)
        fits = estimated <= self.available_tokens

        if not fits:
            logger.warning(
                f"Batch too large: {estimated} tokens > {self.available_tokens} available"
            )

        return fits, estimated, self.available_tokens

    def ensure_fits(
        self,
        slots: list[dict[str, str]],
        schema: dict | str,
        max_iterations: int = 10,
    ) -> list[dict[str, str]]:
        """
        Ensure a slot batch fits by shrinking if necessary.

        Strategy:
        1. Check if batch fits
        2. If not, try reducing slot content length
        3. If still too large, reduce number of slots

        Args:
            slots: List of slot dictionaries
            schema: Entity schema
            max_iterations: Maximum shrinking iterations (default: 10)

        Returns:
            Adjusted list of slots that fits within budget

        Raises:
            ValueError: If batch cannot be made to fit
        """
        fits, estimated, available = self.check_fits(slots, schema)

        if fits:
            return slots

        logger.info(
            f"Batch doesn't fit ({estimated} > {available}), attempting to shrink..."
        )

        # Strategy 1: Truncate slot content
        adjusted_slots = slots.copy()
        for iteration in range(max_iterations):
            # Calculate how much we need to reduce
            excess = estimated - available
            tokens_per_slot = excess // len(adjusted_slots) + 100
            chars_to_remove = int(tokens_per_slot * 3.5)

            # Truncate each slot's content
            truncated = False
            for slot in adjusted_slots:
                content = slot.get("content", "")
                if len(content) > chars_to_remove:
                    slot["content"] = content[:-chars_to_remove] + "..."
                    truncated = True

            if not truncated:
                break

            # Check if it fits now
            fits, estimated, available = self.check_fits(adjusted_slots, schema)
            if fits:
                logger.info(
                    f"Batch fits after content truncation (iteration {iteration + 1})"
                )
                return adjusted_slots

        # Strategy 2: Reduce number of slots
        logger.warning(
            "Content truncation insufficient, reducing batch size..."
        )

        # Binary search for optimal batch size
        left, right = 1, len(slots)
        best_size = 1

        while left <= right:
            mid = (left + right) // 2
            test_slots = slots[:mid]

            fits, estimated, available = self.check_fits(test_slots, schema)

            if fits:
                best_size = mid
                left = mid + 1
            else:
                right = mid - 1

        if best_size < len(slots):
            logger.warning(
                f"Reduced batch size from {len(slots)} to {best_size} slots"
            )
            return slots[:best_size]

        # Last resort: single slot with truncated content
        if len(slots) > 0:
            single_slot = [slots[0].copy()]
            max_content_tokens = available - self.system_prompt_tokens - 500
            max_content_chars = int(max_content_tokens * 3.5)

            single_slot[0]["content"] = single_slot[0]["content"][:max_content_chars]

            fits, estimated, available = self.check_fits(single_slot, schema)
            if fits:
                logger.warning(
                    "Using single slot with truncated content as last resort"
                )
                return single_slot

        raise ValueError(
            f"Cannot fit batch within context window. "
            f"Context limit: {self.context_limit}, "
            f"Available: {self.available_tokens}, "
            f"Estimated: {estimated}"
        )

    def get_optimal_batch_size(
        self,
        slots: list[dict[str, str]],
        schema: dict | str,
    ) -> int:
        """
        Calculate the optimal batch size for a list of slots.

        Uses binary search to find the maximum number of slots that fit.

        Args:
            slots: List of all slots to batch
            schema: Entity schema

        Returns:
            Optimal batch size (number of slots)
        """
        if not slots:
            return 0

        # Binary search for optimal size
        left, right = 1, len(slots)
        best_size = 1

        while left <= right:
            mid = (left + right) // 2
            test_slots = slots[:mid]

            fits, _, _ = self.check_fits(test_slots, schema)

            if fits:
                best_size = mid
                left = mid + 1
            else:
                right = mid - 1

        return best_size

    def create_batches(
        self,
        slots: list[dict[str, str]],
        schema: dict | str,
    ) -> list[list[dict[str, str]]]:
        """
        Split slots into optimally-sized batches.

        Args:
            slots: List of all slots
            schema: Entity schema

        Returns:
            List of slot batches, each fitting within context window
        """
        if not slots:
            return []

        batches = []
        remaining = slots.copy()

        while remaining:
            batch_size = self.get_optimal_batch_size(remaining, schema)

            if batch_size == 0:
                logger.error(
                    "Cannot fit even a single slot. Slot content too large."
                )
                break

            batch = remaining[:batch_size]
            batches.append(batch)
            remaining = remaining[batch_size:]

            logger.debug(
                f"Created batch of {batch_size} slots, {len(remaining)} remaining"
            )

        logger.info(
            f"Created {len(batches)} batches from {len(slots)} slots"
        )

        return batches
