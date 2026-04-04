"""Retry queue for failed RL rollout groups.

When a rollout group fails (e.g., due to an error or all trajectories
failing), the associated :class:`~tinker_cookbook.rl.types.EnvGroupBuilder`
is saved in a :class:`RolloutRetryQueue`.  On the next training iteration
the queue is consulted, and eligible entries are returned so the builder
can produce fresh rollouts without re-doing problem selection.

Env objects are single-use and cannot be queued, but EnvGroupBuilder is
pickleable and can recreate environments via ``make_envs()``.

Note: Only error-related failures are retried (ROLLOUT_ERROR, ALL_FAILED).
Constant-reward groups are not retried because the same problem, model,
and temperature will likely produce constant rewards again.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from tinker_cookbook.rl.types import EnvGroupBuilder

logger = logging.getLogger(__name__)


class RetryReason(Enum):
    """Why a rollout group was queued for retry."""

    ROLLOUT_ERROR = "rollout_error"
    """One or more trajectories failed during rollout execution."""

    ALL_FAILED = "all_failed"
    """All trajectories in the group failed (retry budget exhausted)."""


@dataclass
class RetryEntry:
    """A single queued rollout group awaiting retry.

    Attributes:
        env_group_builder: The builder that produced the original group.
            Can recreate fresh environments via ``make_envs()``.
        reason: Why this group was queued for retry.
        cached_at_step: The training iteration when this entry was queued.
        original_step: The training step when this entry was first queued
            (preserved across re-queuing to track true age).
        cached_at_time: Wall-clock time (``time.monotonic()``) when queued.
        attempt_count: How many times this entry has been retried. Starts at 0.
    """

    env_group_builder: EnvGroupBuilder
    reason: RetryReason
    cached_at_step: int
    original_step: int | None = None
    cached_at_time: float = field(default_factory=time.monotonic)
    attempt_count: int = 0

    def __post_init__(self) -> None:
        if self.original_step is None:
            self.original_step = self.cached_at_step


@dataclass
class StepStats:
    """Per-step statistics, reset at each training iteration.

    Attributes:
        hits: Number of entries returned (retried) this step.
        misses: Number of groups that were not in the queue (new groups).
        evicted: Number of entries evicted due to age this step.
        evicted_max_attempts: Number of entries evicted due to max attempts.
        cached_this_step: Number of new entries queued this step.
        cache_operation_time: Wall-clock seconds spent on queue operations
            this step.
    """

    hits: int = 0
    misses: int = 0
    evicted: int = 0
    evicted_max_attempts: int = 0
    cached_this_step: int = 0
    cache_operation_time: float = 0.0

    @property
    def hit_rate(self) -> float:
        """Queue hit rate for this step (0.0 if no lookups)."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


@dataclass
class QueueStats:
    """Cumulative statistics for the retry queue.

    All counters are monotonically increasing across the lifetime of the
    queue.  The ``current_size`` field reflects the instantaneous queue
    occupancy.

    Attributes:
        items_cached: Total number of entries ever added to the queue.
        items_resumed: Total number of entries returned via ``get_resumable``.
        items_expired: Total number of entries evicted by ``clear_expired``.
        items_max_attempts: Total entries evicted for exceeding max attempts.
        current_size: Current number of entries in the queue.
        entries_by_reason: Breakdown of queued entries by reason.
        step: Per-step statistics (reset each iteration via
            :meth:`RolloutRetryQueue.reset_step_stats`).
    """

    items_cached: int = 0
    items_resumed: int = 0
    items_expired: int = 0
    items_max_attempts: int = 0
    current_size: int = 0
    entries_by_reason: dict[str, int] = field(default_factory=dict)
    step: StepStats = field(default_factory=StepStats)

    def as_metrics(self, prefix: str = "retry_queue") -> dict[str, float]:
        """Return queue statistics as a flat metrics dictionary.

        Suitable for passing directly to ``ml_log.log_metrics``.

        Args:
            prefix: Key prefix for all emitted metric names.

        Returns:
            Flat mapping of metric names to values.
        """
        out: dict[str, float] = {
            # Cumulative counters
            f"{prefix}/items_cached": float(self.items_cached),
            f"{prefix}/items_resumed": float(self.items_resumed),
            f"{prefix}/items_expired": float(self.items_expired),
            f"{prefix}/items_max_attempts": float(self.items_max_attempts),
            # Instantaneous
            f"{prefix}/size": float(self.current_size),
            # Per-step
            f"{prefix}/hits": float(self.step.hits),
            f"{prefix}/misses": float(self.step.misses),
            f"{prefix}/hit_rate": self.step.hit_rate,
            f"{prefix}/evicted": float(self.step.evicted),
            f"{prefix}/evicted_max_attempts": float(self.step.evicted_max_attempts),
            f"{prefix}/cached_this_step": float(self.step.cached_this_step),
            "time/retry_queue_operations": self.step.cache_operation_time,
        }
        for reason, count in self.entries_by_reason.items():
            out[f"{prefix}/queued_reason/{reason}"] = float(count)
        return out


class RolloutRetryQueue:
    """Queue for failed rollout groups that should be retried in later iterations.

    Thread-safety: This class is designed for single-threaded use within one
    asyncio event loop, matching the rollout execution model.

    Example::

        queue = RolloutRetryQueue(max_size=1000, max_attempts=3)
        # After a group fails during rollout:
        queue.enqueue(
            env_group_builder=builder,
            reason=RetryReason.ROLLOUT_ERROR,
            current_step=42,
        )
        # At the start of the next iteration:
        resumable = queue.get_resumable(current_step=43, max_age_steps=3)
        queue.clear_expired(current_step=43, max_age_steps=3)
    """

    def __init__(
        self,
        max_size: int = 1000,
        max_attempts: int = 3,
    ) -> None:
        self._entries: list[RetryEntry] = []
        self._stats: QueueStats = QueueStats()
        self._max_size = max_size
        self._max_attempts = max_attempts

    @property
    def max_size(self) -> int:
        return self._max_size

    @property
    def max_attempts(self) -> int:
        return self._max_attempts

    def reset_step_stats(self) -> None:
        """Reset per-step counters at the beginning of each training iteration.

        Call this at the start of each batch to get clean per-step metrics.
        """
        self._stats.step = StepStats()

    def enqueue(
        self,
        env_group_builder: EnvGroupBuilder,
        reason: RetryReason,
        current_step: int,
        prior_attempt_count: int = 0,
        original_step: int | None = None,
    ) -> None:
        """Queue a failed rollout group for retry.

        Args:
            env_group_builder: The builder that produced the original group.
            reason: Why this group is being queued.
            current_step: The current training iteration index.
            prior_attempt_count: Number of prior attempts (carried forward
                when re-queuing a previously retried entry).
            original_step: The step when this entry was first queued. If None,
                defaults to current_step.
        """
        t0 = time.monotonic()
        entry = RetryEntry(
            env_group_builder=env_group_builder,
            reason=reason,
            cached_at_step=current_step,
            original_step=original_step if original_step is not None else current_step,
            attempt_count=prior_attempt_count,
        )
        self._entries.append(entry)

        # Enforce max_size: drop oldest entries
        if len(self._entries) > self._max_size:
            n_to_drop = len(self._entries) - self._max_size
            self._entries = self._entries[n_to_drop:]
            self._stats.items_expired += n_to_drop
            logger.debug(
                "Retry queue overflow: dropped %d oldest entries (max_size=%d)",
                n_to_drop,
                self._max_size,
            )

        self._stats.items_cached += 1
        self._stats.current_size = len(self._entries)
        self._stats.step.cached_this_step += 1
        reason_key = reason.value
        self._stats.entries_by_reason[reason_key] = (
            self._stats.entries_by_reason.get(reason_key, 0) + 1
        )
        self._stats.step.cache_operation_time += time.monotonic() - t0
        logger.debug(
            "Queued rollout group for retry: reason=%s, step=%d, attempt=%d",
            reason.value,
            current_step,
            prior_attempt_count,
        )

    def get_resumable(
        self,
        current_step: int,
        max_age_steps: int,
        max_entries_per_step: int | None = None,
    ) -> list[RetryEntry]:
        """Return queued entries that are eligible for retry.

        An entry is eligible if it was queued within ``max_age_steps``
        training iterations of ``current_step`` and has not exceeded
        ``max_attempts``.  Returned entries are removed from the queue.

        Args:
            current_step: The current training iteration index.
            max_age_steps: Maximum age (in training steps) for an entry
                to be considered eligible for retry.
            max_entries_per_step: Maximum number of entries to return per
                step. If None, all eligible entries are returned.

        Returns:
            List of eligible retry entries, removed from the queue.
        """
        t0 = time.monotonic()
        resumable: list[RetryEntry] = []
        remaining: list[RetryEntry] = []
        evicted_max_attempts = 0
        for entry in self._entries:
            age = current_step - entry.cached_at_step
            if age <= max_age_steps:
                if entry.attempt_count >= self._max_attempts:
                    # Exceeded max attempts, evict
                    evicted_max_attempts += 1
                elif max_entries_per_step is not None and len(resumable) >= max_entries_per_step:
                    # Already hit the per-step cap, keep in queue
                    remaining.append(entry)
                else:
                    entry.attempt_count += 1
                    resumable.append(entry)
            else:
                remaining.append(entry)
        self._entries = remaining
        self._stats.items_resumed += len(resumable)
        self._stats.items_max_attempts += evicted_max_attempts
        self._stats.current_size = len(self._entries)
        # Per-step tracking
        self._stats.step.hits += len(resumable)
        self._stats.step.evicted_max_attempts += evicted_max_attempts
        self._stats.step.cache_operation_time += time.monotonic() - t0
        if resumable:
            logger.info(
                "Returning %d retry entries (step=%d, max_age=%d)",
                len(resumable),
                current_step,
                max_age_steps,
            )
        if evicted_max_attempts:
            logger.info(
                "Evicted %d entries exceeding max_attempts=%d",
                evicted_max_attempts,
                self._max_attempts,
            )
        return resumable

    def record_misses(self, count: int) -> None:
        """Record misses for groups that were freshly generated.

        Called by the training loop to record how many groups in the batch
        were *not* served from the retry queue (i.e., new groups from the dataset).

        Args:
            count: Number of groups that were not from the queue.
        """
        self._stats.step.misses += count

    def clear_expired(
        self,
        current_step: int,
        max_age_steps: int,
    ) -> int:
        """Evict queue entries older than ``max_age_steps``.

        Args:
            current_step: The current training iteration index.
            max_age_steps: Maximum age (in training steps). Entries older
                than this are removed.

        Returns:
            Number of entries evicted.
        """
        t0 = time.monotonic()
        before = len(self._entries)
        self._entries = [
            e for e in self._entries
            if (current_step - e.cached_at_step) <= max_age_steps
        ]
        evicted = before - len(self._entries)
        self._stats.items_expired += evicted
        self._stats.current_size = len(self._entries)
        self._stats.step.evicted += evicted
        self._stats.step.cache_operation_time += time.monotonic() - t0
        if evicted > 0:
            logger.debug(
                "Evicted %d expired retry queue entries (step=%d, max_age=%d)",
                evicted,
                current_step,
                max_age_steps,
            )
        return evicted

    def stats(self) -> QueueStats:
        """Return a snapshot of cumulative queue statistics.

        Returns:
            A :class:`QueueStats` with current counters including per-step
            :class:`StepStats`.
        """
        return QueueStats(
            items_cached=self._stats.items_cached,
            items_resumed=self._stats.items_resumed,
            items_expired=self._stats.items_expired,
            items_max_attempts=self._stats.items_max_attempts,
            current_size=self._stats.current_size,
            entries_by_reason=dict(self._stats.entries_by_reason),
            step=StepStats(
                hits=self._stats.step.hits,
                misses=self._stats.step.misses,
                evicted=self._stats.step.evicted,
                evicted_max_attempts=self._stats.step.evicted_max_attempts,
                cached_this_step=self._stats.step.cached_this_step,
                cache_operation_time=self._stats.step.cache_operation_time,
            ),
        )

    def __len__(self) -> int:
        return len(self._entries)

    def __bool__(self) -> bool:
        return bool(self._entries)
