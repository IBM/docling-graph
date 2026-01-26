"""
Metrics collection and reporting for extraction pipeline.

This module provides comprehensive metrics tracking for the bottom-up
extraction pipeline, including API calls, tokens, cache performance,
and stage timings.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class StageMetrics:
    """Metrics for a single extraction stage."""

    stage_name: str
    """Name of the extraction stage (e.g., 'Material', 'Component')."""

    start_time: float = field(default_factory=time.time)
    """Stage start timestamp."""

    end_time: float | None = None
    """Stage end timestamp."""

    slots_total: int = 0
    """Total number of slots processed."""

    slots_cached: int = 0
    """Number of slots served from cache."""

    slots_extracted: int = 0
    """Number of slots requiring LLM extraction."""

    batches_processed: int = 0
    """Number of batches sent to LLM."""

    api_calls: int = 0
    """Number of API calls made."""

    api_failures: int = 0
    """Number of failed API calls."""

    retries: int = 0
    """Total number of retries across all calls."""

    entities_extracted: int = 0
    """Number of entities extracted."""

    entities_assigned: int = 0
    """Number of entities successfully assigned to root model."""

    entities_unassigned: int = 0
    """Number of entities that couldn't be assigned."""

    tokens_input: int = 0
    """Total input tokens consumed."""

    tokens_output: int = 0
    """Total output tokens generated."""

    @property
    def duration(self) -> float:
        """Get stage duration in seconds."""
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time

    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate (0.0-1.0)."""
        if self.slots_total == 0:
            return 0.0
        return self.slots_cached / self.slots_total

    @property
    def success_rate(self) -> float:
        """Calculate API success rate (0.0-1.0)."""
        total_attempts = self.api_calls + self.api_failures
        if total_attempts == 0:
            return 1.0
        return self.api_calls / total_attempts

    @property
    def assignment_rate(self) -> float:
        """Calculate entity assignment rate (0.0-1.0)."""
        if self.entities_extracted == 0:
            return 1.0
        return self.entities_assigned / self.entities_extracted

    def mark_complete(self) -> None:
        """Mark stage as complete."""
        self.end_time = time.time()

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "stage_name": self.stage_name,
            "duration_seconds": round(self.duration, 2),
            "slots": {
                "total": self.slots_total,
                "cached": self.slots_cached,
                "extracted": self.slots_extracted,
                "cache_hit_rate": round(self.cache_hit_rate, 3),
            },
            "batches": {
                "processed": self.batches_processed,
                "avg_slots_per_batch": (
                    round(self.slots_extracted / self.batches_processed, 1)
                    if self.batches_processed > 0 else 0
                ),
            },
            "api": {
                "calls": self.api_calls,
                "failures": self.api_failures,
                "retries": self.retries,
                "success_rate": round(self.success_rate, 3),
            },
            "entities": {
                "extracted": self.entities_extracted,
                "assigned": self.entities_assigned,
                "unassigned": self.entities_unassigned,
                "assignment_rate": round(self.assignment_rate, 3),
            },
            "tokens": {
                "input": self.tokens_input,
                "output": self.tokens_output,
                "total": self.tokens_input + self.tokens_output,
            },
        }


@dataclass
class ExtractionMetrics:
    """
    Comprehensive metrics for entire extraction pipeline.

    Tracks metrics across all stages, providing aggregated statistics
    and detailed breakdowns per stage.
    """

    doc_fingerprint: str
    """Document fingerprint for tracking."""

    start_time: float = field(default_factory=time.time)
    """Pipeline start timestamp."""

    end_time: float | None = None
    """Pipeline end timestamp."""

    stages: dict[str, StageMetrics] = field(default_factory=dict)
    """Metrics per extraction stage."""

    conversion_time: float = 0.0
    """Time spent converting document to DoclingDocument."""

    enumeration_time: float = 0.0
    """Time spent enumerating slots."""

    hierarchy_analysis_time: float = 0.0
    """Time spent analyzing schema hierarchy."""

    assembly_time: float = 0.0
    """Time spent assembling final model."""

    total_slots: int = 0
    """Total number of slots enumerated."""

    def start_stage(self, stage_name: str) -> StageMetrics:
        """
        Start tracking a new stage.

        Args:
            stage_name: Name of the stage

        Returns:
            StageMetrics instance for this stage
        """
        stage_metrics = StageMetrics(stage_name=stage_name)
        self.stages[stage_name] = stage_metrics
        return stage_metrics

    def get_stage(self, stage_name: str) -> StageMetrics | None:
        """
        Get metrics for a specific stage.

        Args:
            stage_name: Name of the stage

        Returns:
            StageMetrics instance or None if not found
        """
        return self.stages.get(stage_name)

    def mark_complete(self) -> None:
        """Mark pipeline as complete."""
        self.end_time = time.time()

    @property
    def duration(self) -> float:
        """Get total pipeline duration in seconds."""
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time

    @property
    def total_api_calls(self) -> int:
        """Get total API calls across all stages."""
        return sum(stage.api_calls for stage in self.stages.values())

    @property
    def total_api_failures(self) -> int:
        """Get total API failures across all stages."""
        return sum(stage.api_failures for stage in self.stages.values())

    @property
    def total_retries(self) -> int:
        """Get total retries across all stages."""
        return sum(stage.retries for stage in self.stages.values())

    @property
    def total_entities_extracted(self) -> int:
        """Get total entities extracted across all stages."""
        return sum(stage.entities_extracted for stage in self.stages.values())

    @property
    def total_entities_assigned(self) -> int:
        """Get total entities assigned across all stages."""
        return sum(stage.entities_assigned for stage in self.stages.values())

    @property
    def total_entities_unassigned(self) -> int:
        """Get total unassigned entities across all stages."""
        return sum(stage.entities_unassigned for stage in self.stages.values())

    @property
    def total_tokens_input(self) -> int:
        """Get total input tokens across all stages."""
        return sum(stage.tokens_input for stage in self.stages.values())

    @property
    def total_tokens_output(self) -> int:
        """Get total output tokens across all stages."""
        return sum(stage.tokens_output for stage in self.stages.values())

    @property
    def total_tokens(self) -> int:
        """Get total tokens (input + output) across all stages."""
        return self.total_tokens_input + self.total_tokens_output

    @property
    def overall_cache_hit_rate(self) -> float:
        """Calculate overall cache hit rate."""
        total_slots = sum(stage.slots_total for stage in self.stages.values())
        if total_slots == 0:
            return 0.0
        cached_slots = sum(stage.slots_cached for stage in self.stages.values())
        return cached_slots / total_slots

    @property
    def overall_success_rate(self) -> float:
        """Calculate overall API success rate."""
        total_attempts = self.total_api_calls + self.total_api_failures
        if total_attempts == 0:
            return 1.0
        return self.total_api_calls / total_attempts

    @property
    def overall_assignment_rate(self) -> float:
        """Calculate overall entity assignment rate."""
        if self.total_entities_extracted == 0:
            return 1.0
        return self.total_entities_assigned / self.total_entities_extracted

    def to_dict(self) -> dict[str, Any]:
        """
        Convert metrics to dictionary for reporting.

        Returns:
            Comprehensive metrics dictionary
        """
        return {
            "doc_fingerprint": self.doc_fingerprint,
            "duration_seconds": round(self.duration, 2),
            "timing_breakdown": {
                "conversion": round(self.conversion_time, 2),
                "enumeration": round(self.enumeration_time, 2),
                "hierarchy_analysis": round(self.hierarchy_analysis_time, 2),
                "extraction": round(
                    sum(stage.duration for stage in self.stages.values()), 2
                ),
                "assembly": round(self.assembly_time, 2),
            },
            "slots": {
                "total": self.total_slots,
                "processed": sum(stage.slots_total for stage in self.stages.values()),
            },
            "api": {
                "calls": self.total_api_calls,
                "failures": self.total_api_failures,
                "retries": self.total_retries,
                "success_rate": round(self.overall_success_rate, 3),
            },
            "entities": {
                "extracted": self.total_entities_extracted,
                "assigned": self.total_entities_assigned,
                "unassigned": self.total_entities_unassigned,
                "assignment_rate": round(self.overall_assignment_rate, 3),
            },
            "tokens": {
                "input": self.total_tokens_input,
                "output": self.total_tokens_output,
                "total": self.total_tokens,
            },
            "cache": {
                "hit_rate": round(self.overall_cache_hit_rate, 3),
            },
            "stages": {
                name: stage.to_dict()
                for name, stage in self.stages.items()
            },
        }

    def print_summary(self) -> None:
        """Print a human-readable summary of metrics."""
        print("\n" + "=" * 80)
        print("EXTRACTION METRICS SUMMARY")
        print("=" * 80)
        print(f"Document: {self.doc_fingerprint}")
        print(f"Total Duration: {self.duration:.2f}s")
        print()

        print("Timing Breakdown:")
        print(f"  Conversion:         {self.conversion_time:>8.2f}s")
        print(f"  Enumeration:        {self.enumeration_time:>8.2f}s")
        print(f"  Hierarchy Analysis: {self.hierarchy_analysis_time:>8.2f}s")
        extraction_time = sum(stage.duration for stage in self.stages.values())
        print(f"  Extraction:         {extraction_time:>8.2f}s")
        print(f"  Assembly:           {self.assembly_time:>8.2f}s")
        print()

        print(f"Slots: {self.total_slots} total")
        print()

        print(f"API Calls: {self.total_api_calls}")
        print(f"  Failures: {self.total_api_failures}")
        print(f"  Retries:  {self.total_retries}")
        print(f"  Success Rate: {self.overall_success_rate:.1%}")
        print()

        print(f"Entities: {self.total_entities_extracted} extracted")
        print(f"  Assigned:   {self.total_entities_assigned}")
        print(f"  Unassigned: {self.total_entities_unassigned}")
        print(f"  Assignment Rate: {self.overall_assignment_rate:.1%}")
        print()

        print(f"Tokens: {self.total_tokens:,}")
        print(f"  Input:  {self.total_tokens_input:,}")
        print(f"  Output: {self.total_tokens_output:,}")
        print()

        print(f"Cache Hit Rate: {self.overall_cache_hit_rate:.1%}")
        print()

        if self.stages:
            print("Per-Stage Breakdown:")
            for name, stage in self.stages.items():
                print(f"\n  {name}:")
                print(f"    Duration:    {stage.duration:.2f}s")
                print(f"    Slots:       {stage.slots_extracted} extracted, "
                      f"{stage.slots_cached} cached")
                print(f"    Batches:     {stage.batches_processed}")
                print(f"    API Calls:   {stage.api_calls}")
                print(f"    Entities:    {stage.entities_extracted} extracted, "
                      f"{stage.entities_unassigned} unassigned")
                print(f"    Tokens:      {stage.tokens_input + stage.tokens_output:,}")

        print("\n" + "=" * 80)

# Made with Bob
