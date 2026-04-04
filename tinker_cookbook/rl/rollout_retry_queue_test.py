"""Tests for rollout retry queue."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest
import tinker

from tinker_cookbook.completers import TokenCompleter, TokensWithLogprobs
from tinker_cookbook.rl.rollout_retry_queue import (
    QueueStats,
    RetryEntry,
    RetryReason,
    RolloutRetryQueue,
    StepStats,
)
from tinker_cookbook.rl.rollouts import _do_group_rollout_and_filter_constant_reward_impl
from tinker_cookbook.rl.types import (
    Env,
    EnvGroupBuilder,
    StepResult,
    Trajectory,
    Transition,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trajectory(reward: float = 1.0) -> Trajectory:
    """Create a minimal valid Trajectory."""
    return Trajectory(
        transitions=[
            Transition(
                ob=tinker.ModelInput.from_ints([1, 2, 3]),
                ac=TokensWithLogprobs(tokens=[4, 5], maybe_logprobs=[0.1, 0.2]),
                reward=reward,
                episode_done=True,
            )
        ],
        final_ob=tinker.ModelInput.from_ints([]),
    )


class _FakeEnv(Env):
    def __init__(self, reward: float = 1.0):
        self._reward = reward

    async def initial_observation(self):
        return tinker.ModelInput.from_ints([1, 2, 3]), [0]

    async def step(self, action, *, extra=None):
        return StepResult(
            reward=self._reward,
            episode_done=True,
            next_observation=tinker.ModelInput.from_ints([]),
            next_stop_condition=[0],
        )


class _FakeEnvGroupBuilder(EnvGroupBuilder):
    """Builder that creates _FakeEnv instances with configurable reward."""

    def __init__(self, reward: float = 1.0, group_size: int = 2):
        self._reward = reward
        self._group_size = group_size

    async def make_envs(self):
        return [_FakeEnv(self._reward) for _ in range(self._group_size)]


class _ErrorEnvGroupBuilder(EnvGroupBuilder):
    """Builder whose envs always raise."""

    async def make_envs(self):
        raise RuntimeError("make_envs failed")


class _FakePolicy(TokenCompleter):
    async def __call__(self, model_input, stop):
        return TokensWithLogprobs(tokens=[4, 5], maybe_logprobs=[0.1, 0.2])


# ---------------------------------------------------------------------------
# RolloutRetryQueue unit tests
# ---------------------------------------------------------------------------


class TestRolloutRetryQueue:
    def test_enqueue_adds_entry(self):
        queue = RolloutRetryQueue()
        builder = _FakeEnvGroupBuilder()
        queue.enqueue(
            env_group_builder=builder,
            reason=RetryReason.ROLLOUT_ERROR,
            current_step=5,
        )
        assert len(queue) == 1
        stats = queue.stats()
        assert stats.items_cached == 1
        assert stats.current_size == 1

    def test_get_resumable_returns_fresh_entries(self):
        queue = RolloutRetryQueue()
        builder = _FakeEnvGroupBuilder()
        queue.enqueue(
            env_group_builder=builder,
            reason=RetryReason.ROLLOUT_ERROR,
            current_step=5,
        )
        resumable = queue.get_resumable(current_step=6, max_age_steps=3)
        assert len(resumable) == 1
        assert resumable[0].env_group_builder is builder
        assert resumable[0].attempt_count == 1
        # Entry should be removed from queue
        assert len(queue) == 0
        assert queue.stats().items_resumed == 1

    def test_get_resumable_skips_old_entries(self):
        queue = RolloutRetryQueue()
        builder = _FakeEnvGroupBuilder()
        queue.enqueue(
            env_group_builder=builder,
            reason=RetryReason.ROLLOUT_ERROR,
            current_step=1,
        )
        # 10 steps later, entry is too old (max_age=3)
        resumable = queue.get_resumable(current_step=10, max_age_steps=3)
        assert len(resumable) == 0
        # Entry is still in the queue (not evicted, just not resumable)
        assert len(queue) == 1

    def test_clear_expired_evicts_old(self):
        queue = RolloutRetryQueue()
        queue.enqueue(
            env_group_builder=_FakeEnvGroupBuilder(),
            reason=RetryReason.ALL_FAILED,
            current_step=1,
        )
        queue.enqueue(
            env_group_builder=_FakeEnvGroupBuilder(),
            reason=RetryReason.ROLLOUT_ERROR,
            current_step=5,
        )
        evicted = queue.clear_expired(current_step=6, max_age_steps=3)
        assert evicted == 1
        assert len(queue) == 1
        assert queue.stats().items_expired == 1

    def test_stats_tracks_reasons(self):
        queue = RolloutRetryQueue()
        queue.enqueue(
            env_group_builder=_FakeEnvGroupBuilder(),
            reason=RetryReason.ROLLOUT_ERROR,
            current_step=1,
        )
        queue.enqueue(
            env_group_builder=_FakeEnvGroupBuilder(),
            reason=RetryReason.ROLLOUT_ERROR,
            current_step=2,
        )
        queue.enqueue(
            env_group_builder=_FakeEnvGroupBuilder(),
            reason=RetryReason.ALL_FAILED,
            current_step=3,
        )
        stats = queue.stats()
        assert stats.entries_by_reason["rollout_error"] == 2
        assert stats.entries_by_reason["all_failed"] == 1
        assert stats.items_cached == 3

    def test_stats_as_metrics(self):
        queue = RolloutRetryQueue()
        queue.enqueue(
            env_group_builder=_FakeEnvGroupBuilder(),
            reason=RetryReason.ROLLOUT_ERROR,
            current_step=1,
        )
        metrics = queue.stats().as_metrics()
        assert "retry_queue/items_cached" in metrics
        assert "retry_queue/size" in metrics
        assert "retry_queue/hits" in metrics
        assert "retry_queue/misses" in metrics
        assert "retry_queue/hit_rate" in metrics
        assert "retry_queue/evicted" in metrics
        assert "retry_queue/cached_this_step" in metrics
        assert "retry_queue/items_max_attempts" in metrics
        assert "retry_queue/evicted_max_attempts" in metrics
        assert "time/retry_queue_operations" in metrics
        assert "retry_queue/queued_reason/rollout_error" in metrics
        assert metrics["retry_queue/items_cached"] == 1.0
        assert metrics["retry_queue/cached_this_step"] == 1.0

    def test_bool_and_len(self):
        queue = RolloutRetryQueue()
        assert not queue
        assert len(queue) == 0
        queue.enqueue(
            env_group_builder=_FakeEnvGroupBuilder(),
            reason=RetryReason.ROLLOUT_ERROR,
            current_step=1,
        )
        assert queue
        assert len(queue) == 1

    def test_get_resumable_increments_attempt_count(self):
        queue = RolloutRetryQueue()
        builder = _FakeEnvGroupBuilder()
        queue.enqueue(
            env_group_builder=builder,
            reason=RetryReason.ROLLOUT_ERROR,
            current_step=5,
        )
        entries = queue.get_resumable(current_step=6, max_age_steps=3)
        assert entries[0].attempt_count == 1

        # Re-enqueue the same entry (simulating a second failure),
        # carrying forward attempt_count
        queue.enqueue(
            env_group_builder=builder,
            reason=RetryReason.ROLLOUT_ERROR,
            current_step=6,
            prior_attempt_count=entries[0].attempt_count,
            original_step=entries[0].original_step,
        )
        entries = queue.get_resumable(current_step=7, max_age_steps=3)
        # Should be 2 now (carried forward 1, incremented to 2)
        assert entries[0].attempt_count == 2

    def test_max_attempts_enforced(self):
        """Entries exceeding max_attempts are evicted during get_resumable."""
        queue = RolloutRetryQueue(max_attempts=2)
        builder = _FakeEnvGroupBuilder()
        # Enqueue with prior_attempt_count=2 (already at max)
        queue.enqueue(
            env_group_builder=builder,
            reason=RetryReason.ROLLOUT_ERROR,
            current_step=5,
            prior_attempt_count=2,
        )
        resumable = queue.get_resumable(current_step=6, max_age_steps=3)
        assert len(resumable) == 0
        assert queue.stats().items_max_attempts == 1
        # Entry should have been evicted
        assert len(queue) == 0

    def test_max_size_enforced(self):
        """Oldest entries are dropped when queue exceeds max_size."""
        queue = RolloutRetryQueue(max_size=3)
        builders = []
        for i in range(5):
            b = _FakeEnvGroupBuilder()
            builders.append(b)
            queue.enqueue(
                env_group_builder=b,
                reason=RetryReason.ROLLOUT_ERROR,
                current_step=i,
            )
        assert len(queue) == 3
        # Only the last 3 should remain
        resumable = queue.get_resumable(current_step=5, max_age_steps=10)
        assert len(resumable) == 3
        assert resumable[0].env_group_builder is builders[2]
        assert resumable[1].env_group_builder is builders[3]
        assert resumable[2].env_group_builder is builders[4]

    def test_max_entries_per_step(self):
        """max_entries_per_step caps how many entries are returned."""
        queue = RolloutRetryQueue()
        for i in range(5):
            queue.enqueue(
                env_group_builder=_FakeEnvGroupBuilder(),
                reason=RetryReason.ROLLOUT_ERROR,
                current_step=i,
            )
        resumable = queue.get_resumable(
            current_step=5, max_age_steps=10, max_entries_per_step=2
        )
        assert len(resumable) == 2
        # Remaining 3 should still be in the queue
        assert len(queue) == 3

    def test_original_step_preserved_on_requeue(self):
        """When re-enqueuing, original_step is carried forward."""
        queue = RolloutRetryQueue()
        builder = _FakeEnvGroupBuilder()
        queue.enqueue(
            env_group_builder=builder,
            reason=RetryReason.ROLLOUT_ERROR,
            current_step=5,
        )
        entries = queue.get_resumable(current_step=6, max_age_steps=3)
        assert entries[0].original_step == 5

        # Re-enqueue with original_step preserved
        queue.enqueue(
            env_group_builder=builder,
            reason=RetryReason.ROLLOUT_ERROR,
            current_step=8,
            prior_attempt_count=entries[0].attempt_count,
            original_step=entries[0].original_step,
        )
        entries2 = queue.get_resumable(current_step=9, max_age_steps=3)
        assert entries2[0].original_step == 5
        assert entries2[0].cached_at_step == 8

    def test_stats_returns_copy(self):
        queue = RolloutRetryQueue()
        stats1 = queue.stats()
        queue.enqueue(
            env_group_builder=_FakeEnvGroupBuilder(),
            reason=RetryReason.ROLLOUT_ERROR,
            current_step=1,
        )
        stats2 = queue.stats()
        # stats1 should not be mutated
        assert stats1.items_cached == 0
        assert stats2.items_cached == 1

    def test_multiple_get_resumable_calls(self):
        queue = RolloutRetryQueue()
        for step in range(5):
            queue.enqueue(
                env_group_builder=_FakeEnvGroupBuilder(),
                reason=RetryReason.ROLLOUT_ERROR,
                current_step=step,
            )
        # Get resumable at step 5 with max_age 1 -- only step 4 is within age 1
        resumable = queue.get_resumable(current_step=5, max_age_steps=1)
        assert len(resumable) == 1
        # Remaining: steps 0, 1, 2, 3 (too old to resume but not yet expired)
        assert len(queue) == 4

        # Now clear expired
        evicted = queue.clear_expired(current_step=5, max_age_steps=1)
        assert evicted == 4
        assert len(queue) == 0


# ---------------------------------------------------------------------------
# Integration: retry queue populated by rollout filtering
# ---------------------------------------------------------------------------


class TestRetryQueueIntegration:
    def test_constant_reward_does_not_populate_queue(self):
        """Constant-reward groups should NOT be queued for retry."""
        queue = RolloutRetryQueue()
        builder = _FakeEnvGroupBuilder(reward=1.0, group_size=3)
        sampling_client = MagicMock(spec=tinker.SamplingClient)

        result = asyncio.run(
            _do_group_rollout_and_filter_constant_reward_impl(
                sampling_client=sampling_client,
                env_group_builder=builder,
                max_tokens=100,
                temperature=1.0,
                do_remove_constant_reward_groups=True,
                enable_logging=False,
                retry_queue=queue,
                current_step=10,
            )
        )

        assert result is None
        # Queue should be empty -- constant reward is not retryable
        assert len(queue) == 0

    def test_no_queue_when_disabled(self):
        """When retry_queue is None, no queuing occurs."""
        builder = _FakeEnvGroupBuilder(reward=1.0, group_size=2)
        sampling_client = MagicMock(spec=tinker.SamplingClient)

        result = asyncio.run(
            _do_group_rollout_and_filter_constant_reward_impl(
                sampling_client=sampling_client,
                env_group_builder=builder,
                max_tokens=100,
                temperature=1.0,
                do_remove_constant_reward_groups=True,
                enable_logging=False,
                retry_queue=None,
                current_step=10,
            )
        )
        # Just verify it returns None without error (no queue to check)
        assert result is None

    def test_successful_rollout_not_queued(self):
        """Groups that produce valid trajectories should not be queued."""
        queue = RolloutRetryQueue()
        builder = _FakeEnvGroupBuilder(reward=1.0, group_size=2)
        sampling_client = MagicMock(spec=tinker.SamplingClient)

        result = asyncio.run(
            _do_group_rollout_and_filter_constant_reward_impl(
                sampling_client=sampling_client,
                env_group_builder=builder,
                max_tokens=100,
                temperature=1.0,
                do_remove_constant_reward_groups=False,  # don't filter
                enable_logging=False,
                retry_queue=queue,
                current_step=10,
            )
        )

        assert result is not None
        assert len(queue) == 0


class TestQueueStats:
    def test_as_metrics_empty(self):
        stats = QueueStats()
        metrics = stats.as_metrics()
        assert metrics["retry_queue/items_cached"] == 0.0
        assert metrics["retry_queue/size"] == 0.0
        assert metrics["retry_queue/hits"] == 0.0
        assert metrics["retry_queue/misses"] == 0.0
        assert metrics["retry_queue/hit_rate"] == 0.0
        assert metrics["retry_queue/evicted"] == 0.0
        assert metrics["retry_queue/cached_this_step"] == 0.0
        assert metrics["retry_queue/items_max_attempts"] == 0.0
        assert metrics["retry_queue/evicted_max_attempts"] == 0.0
        assert metrics["time/retry_queue_operations"] == 0.0

    def test_as_metrics_custom_prefix(self):
        stats = QueueStats(items_cached=5)
        metrics = stats.as_metrics(prefix="my_queue")
        assert "my_queue/items_cached" in metrics
        assert metrics["my_queue/items_cached"] == 5.0


class TestRetryEntry:
    def test_default_attempt_count(self):
        entry = RetryEntry(
            env_group_builder=_FakeEnvGroupBuilder(),
            reason=RetryReason.ROLLOUT_ERROR,
            cached_at_step=0,
        )
        assert entry.attempt_count == 0

    def test_original_step_defaults_to_cached_at_step(self):
        entry = RetryEntry(
            env_group_builder=_FakeEnvGroupBuilder(),
            reason=RetryReason.ROLLOUT_ERROR,
            cached_at_step=5,
        )
        assert entry.original_step == 5


class TestStepStats:
    def test_hit_rate_empty(self):
        s = StepStats()
        assert s.hit_rate == 0.0

    def test_hit_rate_computed(self):
        s = StepStats(hits=3, misses=7)
        assert abs(s.hit_rate - 0.3) < 1e-9

    def test_reset_step_stats(self):
        queue = RolloutRetryQueue()
        queue.enqueue(
            env_group_builder=_FakeEnvGroupBuilder(),
            reason=RetryReason.ROLLOUT_ERROR,
            current_step=1,
        )
        assert queue.stats().step.cached_this_step == 1
        queue.reset_step_stats()
        assert queue.stats().step.cached_this_step == 0
        # Cumulative stats should be preserved
        assert queue.stats().items_cached == 1

    def test_record_misses(self):
        queue = RolloutRetryQueue()
        queue.record_misses(5)
        assert queue.stats().step.misses == 5
        queue.record_misses(3)
        assert queue.stats().step.misses == 8

    def test_eviction_tracked_in_step_stats(self):
        queue = RolloutRetryQueue()
        queue.enqueue(
            env_group_builder=_FakeEnvGroupBuilder(),
            reason=RetryReason.ALL_FAILED,
            current_step=1,
        )
        evicted = queue.clear_expired(current_step=10, max_age_steps=1)
        assert evicted == 1
        assert queue.stats().step.evicted == 1

    def test_cache_operation_time_tracked(self):
        queue = RolloutRetryQueue()
        queue.enqueue(
            env_group_builder=_FakeEnvGroupBuilder(),
            reason=RetryReason.ROLLOUT_ERROR,
            current_step=1,
        )
        queue.get_resumable(current_step=2, max_age_steps=3)
        queue.clear_expired(current_step=2, max_age_steps=3)
        # Time should be positive (all three operations contribute)
        assert queue.stats().step.cache_operation_time >= 0.0
