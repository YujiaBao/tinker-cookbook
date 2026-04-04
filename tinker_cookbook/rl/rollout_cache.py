"""Partial rollout caching for RL training.

When a rollout group is discarded mid-generation (e.g., because all rewards in
the group are constant, or due to an error), the associated
:class:`~tinker_cookbook.rl.types.EnvGroupBuilder` and any completed
trajectories are saved in a :class:`RolloutCache`.  On the next training
iteration the cache is consulted, and resumable entries are returned so
the builder can produce fresh rollouts without re-doing problem selection.

Env objects are single-use and cannot be cached, but EnvGroupBuilder is
pickleable and can recreate environments via ``make_envs()``.

Design notes (cf. SLIME's ``RolloutDataSourceWithBuffer``):
    SLIME reuses partial response tokens from aborted rollouts and applies
    loss masking (``mask_offpolicy_in_partial_rollout``) to avoid training on
    stale completions.  Our approach instead discards partial completions and
    re-creates fresh environments from the cached ``EnvGroupBuilder``.  This
    avoids off-policy staleness entirely at the cost of re-generating tokens,
    but the ``compute_saved_estimate`` metric tracks how many prompt tokens
    we avoid re-selecting/re-encoding thanks to builder reuse.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

from tinker_cookbook.rl.types import EnvGroupBuilder, Trajectory

logger = logging.getLogger(__name__)


class CacheReason(Enum):
    """Why a rollout group was cached instead of used for training."""

    CONSTANT_REWARD = "constant_reward"
    """All trajectories in the group received identical total rewards."""

    ROLLOUT_ERROR = "rollout_error"
    """One or more trajectories failed during rollout execution."""

    ALL_FAILED = "all_failed"
    """All trajectories in the group failed (retry budget exhausted)."""


@dataclass
class CachedRolloutEntry:
    """A single cached rollout group awaiting resumption.

    Attributes:
        env_group_builder: The builder that produced the original group.
            Can recreate fresh environments via ``make_envs()``.
        partial_trajectories: Any trajectories that completed successfully
            before the group was discarded. Empty for groups that errored
            before any trajectory finished.
        reason: Why this group was cached.
        cached_at_step: The training iteration when this entry was cached.
        cached_at_time: Wall-clock time (``time.monotonic()``) when cached.
        attempt_count: How many times this entry has been retried. Starts at 0.
    """

    env_group_builder: EnvGroupBuilder
    partial_trajectories: list[Trajectory]
    reason: CacheReason
    cached_at_step: int
    cached_at_time: float = field(default_factory=time.monotonic)
    attempt_count: int = 0


@dataclass
class StepStats:
    """Per-step statistics, reset at each training iteration.

    Attributes:
        hits: Number of cache entries returned (resumed) this step.
        misses: Number of groups that were not in the cache (new groups).
        evicted: Number of entries evicted due to age this step.
        cached_this_step: Number of new entries cached this step.
        compute_saved_estimate: Estimated tokens saved by reusing cached
            builders (sum of partial trajectory token counts from resumed
            entries, representing prompt tokens that did not need to be
            re-selected/re-encoded).
        cache_operation_time: Wall-clock seconds spent on cache operations
            this step.
    """

    hits: int = 0
    misses: int = 0
    evicted: int = 0
    cached_this_step: int = 0
    compute_saved_estimate: int = 0
    cache_operation_time: float = 0.0

    @property
    def hit_rate(self) -> float:
        """Cache hit rate for this step (0.0 if no lookups)."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


@dataclass
class CacheStats:
    """Cumulative statistics for the rollout cache.

    All counters are monotonically increasing across the lifetime of the
    cache.  The ``current_size`` field reflects the instantaneous cache
    occupancy.

    Attributes:
        items_cached: Total number of entries ever added to the cache.
        items_resumed: Total number of entries returned via ``get_resumable``.
        items_expired: Total number of entries evicted by ``clear_expired``.
        current_size: Current number of entries in the cache.
        cache_hits_by_reason: Breakdown of cached entries by reason.
        step: Per-step statistics (reset each iteration via
            :meth:`RolloutCache.reset_step_stats`).
    """

    items_cached: int = 0
    items_resumed: int = 0
    items_expired: int = 0
    current_size: int = 0
    cache_hits_by_reason: dict[str, int] = field(default_factory=dict)
    step: StepStats = field(default_factory=StepStats)

    def as_metrics(self, prefix: str = "rollout_cache") -> dict[str, float]:
        """Return cache statistics as a flat metrics dictionary.

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
            # Instantaneous
            f"{prefix}/size": float(self.current_size),
            # Per-step
            f"{prefix}/hits": float(self.step.hits),
            f"{prefix}/misses": float(self.step.misses),
            f"{prefix}/hit_rate": self.step.hit_rate,
            f"{prefix}/evicted": float(self.step.evicted),
            f"{prefix}/cached_this_step": float(self.step.cached_this_step),
            f"{prefix}/compute_saved_estimate": float(self.step.compute_saved_estimate),
            "time/rollout_cache_operations": self.step.cache_operation_time,
        }
        for reason, count in self.cache_hits_by_reason.items():
            out[f"{prefix}/cached_reason/{reason}"] = float(count)
        return out


class RolloutCache:
    """Cache for partial rollout results that can be resumed in later iterations.

    Thread-safety: This class is designed for single-threaded use within one
    asyncio event loop, matching the rollout execution model.

    Example::

        cache = RolloutCache()
        # After a group is filtered due to constant rewards:
        cache.cache_partial(
            env_group_builder=builder,
            partial_trajectories=trajectories,
            reason=CacheReason.CONSTANT_REWARD,
            current_step=42,
        )
        # At the start of the next iteration:
        resumable = cache.get_resumable(current_step=43, max_age_steps=3)
        cache.clear_expired(current_step=43, max_age_steps=3)
    """

    def __init__(self) -> None:
        self._entries: list[CachedRolloutEntry] = []
        self._stats: CacheStats = CacheStats()

    def reset_step_stats(self) -> None:
        """Reset per-step counters at the beginning of each training iteration.

        Call this at the start of each batch to get clean per-step metrics.
        """
        self._stats.step = StepStats()

    def cache_partial(
        self,
        env_group_builder: EnvGroupBuilder,
        partial_trajectories: list[Trajectory],
        reason: CacheReason,
        current_step: int,
    ) -> None:
        """Save a partial rollout group for potential resumption.

        Args:
            env_group_builder: The builder that produced the original group.
            partial_trajectories: Successfully completed trajectories (may be
                empty if the group errored before any trajectory finished).
            reason: Why this group is being cached.
            current_step: The current training iteration index.
        """
        t0 = time.monotonic()
        entry = CachedRolloutEntry(
            env_group_builder=env_group_builder,
            partial_trajectories=partial_trajectories,
            reason=reason,
            cached_at_step=current_step,
        )
        self._entries.append(entry)
        self._stats.items_cached += 1
        self._stats.current_size = len(self._entries)
        self._stats.step.cached_this_step += 1
        reason_key = reason.value
        self._stats.cache_hits_by_reason[reason_key] = (
            self._stats.cache_hits_by_reason.get(reason_key, 0) + 1
        )
        self._stats.step.cache_operation_time += time.monotonic() - t0
        logger.debug(
            "Cached rollout group: reason=%s, step=%d, partial_trajectories=%d",
            reason.value,
            current_step,
            len(partial_trajectories),
        )

    def get_resumable(
        self,
        current_step: int,
        max_age_steps: int,
    ) -> list[CachedRolloutEntry]:
        """Return cached entries that are eligible for resumption.

        An entry is resumable if it was cached within ``max_age_steps``
        training iterations of ``current_step``.  Returned entries are
        removed from the cache.

        Args:
            current_step: The current training iteration index.
            max_age_steps: Maximum age (in training steps) for an entry
                to be considered resumable.

        Returns:
            List of resumable cache entries, removed from the cache.
        """
        t0 = time.monotonic()
        resumable: list[CachedRolloutEntry] = []
        remaining: list[CachedRolloutEntry] = []
        for entry in self._entries:
            age = current_step - entry.cached_at_step
            if age <= max_age_steps:
                entry.attempt_count += 1
                resumable.append(entry)
            else:
                remaining.append(entry)
        self._entries = remaining
        self._stats.items_resumed += len(resumable)
        self._stats.current_size = len(self._entries)
        # Per-step tracking
        self._stats.step.hits += len(resumable)
        # Estimate compute saved: count action tokens from partial trajectories
        # that represent generation work already done. The builder reuse also
        # saves re-selecting the problem and re-encoding the prompt.
        for entry in resumable:
            for traj in entry.partial_trajectories:
                for transition in traj.transitions:
                    self._stats.step.compute_saved_estimate += len(
                        transition.ac.tokens
                    )
        self._stats.step.cache_operation_time += time.monotonic() - t0
        if resumable:
            logger.info(
                "Returning %d resumable rollout groups (step=%d, max_age=%d)",
                len(resumable),
                current_step,
                max_age_steps,
            )
        return resumable

    def record_misses(self, count: int) -> None:
        """Record cache misses for groups that were freshly generated.

        Called by the training loop to record how many groups in the batch
        were *not* served from cache (i.e., new groups from the dataset).

        Args:
            count: Number of groups that were not from the cache.
        """
        self._stats.step.misses += count

    def clear_expired(
        self,
        current_step: int,
        max_age_steps: int,
    ) -> int:
        """Evict cache entries older than ``max_age_steps``.

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
                "Evicted %d expired rollout cache entries (step=%d, max_age=%d)",
                evicted,
                current_step,
                max_age_steps,
            )
        return evicted

    def stats(self) -> CacheStats:
        """Return a snapshot of cumulative cache statistics.

        Returns:
            A :class:`CacheStats` with current counters including per-step
            :class:`StepStats`.
        """
        # Return a copy so callers cannot mutate internal state
        return CacheStats(
            items_cached=self._stats.items_cached,
            items_resumed=self._stats.items_resumed,
            items_expired=self._stats.items_expired,
            current_size=self._stats.current_size,
            cache_hits_by_reason=dict(self._stats.cache_hits_by_reason),
            step=StepStats(
                hits=self._stats.step.hits,
                misses=self._stats.step.misses,
                evicted=self._stats.step.evicted,
                cached_this_step=self._stats.step.cached_this_step,
                compute_saved_estimate=self._stats.step.compute_saved_estimate,
                cache_operation_time=self._stats.step.cache_operation_time,
            ),
        )

    def __len__(self) -> int:
        return len(self._entries)

    def __bool__(self) -> bool:
        return bool(self._entries)
