"""
Rate limiting and retry logic for LLM API calls.

This module provides rate limiting to respect provider limits and retry
logic with exponential backoff for handling transient failures.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, TypeVar, cast

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_retries: int = 3
    """Maximum number of retry attempts."""

    initial_delay: float = 1.0
    """Initial delay in seconds before first retry."""

    max_delay: float = 60.0
    """Maximum delay in seconds between retries."""

    exponential_base: float = 2.0
    """Base for exponential backoff calculation."""

    jitter: bool = True
    """Whether to add random jitter to delays."""


class RateLimiter:
    """
    Token bucket rate limiter for API calls.

    Implements a token bucket algorithm to enforce rate limits specified
    in requests per minute (RPM). Thread-safe for concurrent access.
    """

    def __init__(self, rpm: int | None = None) -> None:
        """
        Initialize rate limiter.

        Args:
            rpm: Requests per minute limit (None = no limit)
        """
        self.rpm = rpm
        self.enabled = rpm is not None and rpm > 0

        if self.enabled:
            assert rpm is not None  # Type narrowing for type checker
            # Convert RPM to tokens per second
            self.tokens_per_second = rpm / 60.0
            self.max_tokens = rpm  # Bucket capacity
            self.tokens = float(rpm)  # Start with full bucket
            self.last_update = time.time()
            self._lock = asyncio.Lock()

            logger.info(f"Rate limiter initialized: {rpm} RPM ({self.tokens_per_second:.2f} req/s)")
        else:
            logger.info("Rate limiter disabled (no limit)")

    async def acquire(self, tokens: int = 1) -> None:
        """
        Acquire tokens from the bucket, waiting if necessary.

        Args:
            tokens: Number of tokens to acquire (default: 1)
        """
        if not self.enabled:
            return

        async with self._lock:
            while True:
                now = time.time()
                elapsed = now - self.last_update

                # Refill tokens based on elapsed time
                self.tokens = min(
                    self.max_tokens,
                    self.tokens + elapsed * self.tokens_per_second
                )
                self.last_update = now

                if self.tokens >= tokens:
                    # Enough tokens available
                    self.tokens -= tokens
                    return

                # Not enough tokens, calculate wait time
                tokens_needed = tokens - self.tokens
                wait_time = tokens_needed / self.tokens_per_second

                logger.debug(f"Rate limit: waiting {wait_time:.2f}s for {tokens} tokens")
                await asyncio.sleep(wait_time)

    def get_stats(self) -> dict[str, Any]:
        """
        Get current rate limiter statistics.

        Returns:
            Dictionary with current state
        """
        if not self.enabled:
            return {"enabled": False}

        return {
            "enabled": True,
            "rpm": self.rpm,
            "tokens_available": self.tokens,
            "max_tokens": self.max_tokens,
            "tokens_per_second": self.tokens_per_second,
        }


class RetryHandler:
    """
    Handles retry logic with exponential backoff.

    Provides configurable retry behavior for handling transient failures
    in API calls, with exponential backoff and optional jitter.
    """

    def __init__(self, config: RetryConfig | None = None) -> None:
        """
        Initialize retry handler.

        Args:
            config: Retry configuration (uses defaults if None)
        """
        self.config = config or RetryConfig()
        self.retry_count = 0
        self.total_retries = 0

        logger.info(
            f"Retry handler initialized: max_retries={self.config.max_retries}, "
            f"initial_delay={self.config.initial_delay}s"
        )

    async def execute_with_retry(
        self,
        func: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """
        Execute a function with retry logic.

        Args:
            func: Function to execute (can be sync or async)
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Result from successful function execution

        Raises:
            Exception: If all retries are exhausted
        """
        last_exception: Exception | None = None

        for attempt in range(self.config.max_retries + 1):
            try:
                # Execute function (handle both sync and async)
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)

                # Success
                if attempt > 0:
                    logger.info(f"Retry succeeded on attempt {attempt + 1}")
                    self.total_retries += attempt

                return cast(T, result)

            except Exception as e:
                last_exception = e

                # Check if we should retry
                if attempt >= self.config.max_retries:
                    logger.error(
                        f"All {self.config.max_retries} retries exhausted: {e}"
                    )
                    self.total_retries += attempt
                    raise

                # Calculate delay with exponential backoff
                delay = min(
                    self.config.initial_delay * (self.config.exponential_base ** attempt),
                    self.config.max_delay
                )

                # Add jitter if enabled
                if self.config.jitter:
                    import random
                    delay *= (0.5 + random.random())  # 50-150% of calculated delay

                logger.warning(
                    f"Attempt {attempt + 1} failed: {e}. "
                    f"Retrying in {delay:.2f}s..."
                )

                await asyncio.sleep(delay)

        # Should never reach here, but just in case
        raise last_exception or Exception("Retry logic failed unexpectedly")

    def get_stats(self) -> dict[str, Any]:
        """
        Get retry statistics.

        Returns:
            Dictionary with retry counts
        """
        return {
            "total_retries": self.total_retries,
            "max_retries": self.config.max_retries,
        }


class ParallelExecutor:
    """
    Manages parallel execution of tasks with concurrency control.

    Provides controlled parallelism for batch processing with rate limiting
    and retry support.
    """

    def __init__(
        self,
        max_concurrency: int = 5,
        rate_limiter: RateLimiter | None = None,
        retry_handler: RetryHandler | None = None,
    ) -> None:
        """
        Initialize parallel executor.

        Args:
            max_concurrency: Maximum number of concurrent tasks
            rate_limiter: Optional rate limiter for API calls
            retry_handler: Optional retry handler for failures
        """
        self.max_concurrency = max_concurrency
        self.rate_limiter = rate_limiter
        self.retry_handler = retry_handler or RetryHandler()
        self.semaphore = asyncio.Semaphore(max_concurrency)

        logger.info(f"Parallel executor initialized: max_concurrency={max_concurrency}")

    async def execute_task(
        self,
        func: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """
        Execute a single task with rate limiting and retry.

        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Task result
        """
        async with self.semaphore:
            # Apply rate limiting
            if self.rate_limiter:
                await self.rate_limiter.acquire()

            # Execute with retry
            return await self.retry_handler.execute_with_retry(func, *args, **kwargs)

    async def execute_batch(
        self,
        func: Callable[..., T],
        items: list[Any],
        *args: Any,
        **kwargs: Any,
    ) -> list[T]:
        """
        Execute a batch of tasks in parallel.

        Args:
            func: Function to execute for each item
            items: List of items to process
            *args: Additional positional arguments for func
            **kwargs: Additional keyword arguments for func

        Returns:
            List of results in same order as items
        """
        tasks = [
            self.execute_task(func, item, *args, **kwargs)
            for item in items
        ]

        return await asyncio.gather(*tasks)

    def get_stats(self) -> dict[str, Any]:
        """
        Get executor statistics.

        Returns:
            Dictionary with executor state
        """
        stats = {
            "max_concurrency": self.max_concurrency,
            "retry_stats": self.retry_handler.get_stats(),
        }

        if self.rate_limiter:
            stats["rate_limiter"] = self.rate_limiter.get_stats()

        return stats

# Made with Bob
