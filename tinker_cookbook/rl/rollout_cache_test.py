"""Tests for partial rollout caching."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest
import tinker

from tinker_cookbook.completers import TokenCompleter, TokensWithLogprobs
from tinker_cookbook.rl.rollout_cache import (
    CachedRolloutEntry,
    CacheReason,
    CacheStats,
    RolloutCache,
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
# RolloutCache unit tests
# ---------------------------------------------------------------------------


class TestRolloutCache:
    def test_cache_partial_adds_entry(self):
        cache = RolloutCache()
        builder = _FakeEnvGroupBuilder()
        cache.cache_partial(
            env_group_builder=builder,
            partial_trajectories=[_make_trajectory()],
            reason=CacheReason.CONSTANT_REWARD,
            current_step=5,
        )
        assert len(cache) == 1
        stats = cache.stats()
        assert stats.items_cached == 1
        assert stats.current_size == 1

    def test_get_resumable_returns_fresh_entries(self):
        cache = RolloutCache()
        builder = _FakeEnvGroupBuilder()
        cache.cache_partial(
            env_group_builder=builder,
            partial_trajectories=[],
            reason=CacheReason.ROLLOUT_ERROR,
            current_step=5,
        )
        resumable = cache.get_resumable(current_step=6, max_age_steps=3)
        assert len(resumable) == 1
        assert resumable[0].env_group_builder is builder
        assert resumable[0].attempt_count == 1
        # Entry should be removed from cache
        assert len(cache) == 0
        assert cache.stats().items_resumed == 1

    def test_get_resumable_skips_old_entries(self):
        cache = RolloutCache()
        builder = _FakeEnvGroupBuilder()
        cache.cache_partial(
            env_group_builder=builder,
            partial_trajectories=[],
            reason=CacheReason.ROLLOUT_ERROR,
            current_step=1,
        )
        # 10 steps later, entry is too old (max_age=3)
        resumable = cache.get_resumable(current_step=10, max_age_steps=3)
        assert len(resumable) == 0
        # Entry is still in the cache (not evicted, just not resumable)
        assert len(cache) == 1

    def test_clear_expired_evicts_old(self):
        cache = RolloutCache()
        cache.cache_partial(
            env_group_builder=_FakeEnvGroupBuilder(),
            partial_trajectories=[],
            reason=CacheReason.ALL_FAILED,
            current_step=1,
        )
        cache.cache_partial(
            env_group_builder=_FakeEnvGroupBuilder(),
            partial_trajectories=[],
            reason=CacheReason.CONSTANT_REWARD,
            current_step=5,
        )
        evicted = cache.clear_expired(current_step=6, max_age_steps=3)
        assert evicted == 1
        assert len(cache) == 1
        assert cache.stats().items_expired == 1

    def test_stats_tracks_reasons(self):
        cache = RolloutCache()
        cache.cache_partial(
            env_group_builder=_FakeEnvGroupBuilder(),
            partial_trajectories=[],
            reason=CacheReason.CONSTANT_REWARD,
            current_step=1,
        )
        cache.cache_partial(
            env_group_builder=_FakeEnvGroupBuilder(),
            partial_trajectories=[],
            reason=CacheReason.CONSTANT_REWARD,
            current_step=2,
        )
        cache.cache_partial(
            env_group_builder=_FakeEnvGroupBuilder(),
            partial_trajectories=[],
            reason=CacheReason.ROLLOUT_ERROR,
            current_step=3,
        )
        stats = cache.stats()
        assert stats.cache_hits_by_reason["constant_reward"] == 2
        assert stats.cache_hits_by_reason["rollout_error"] == 1
        assert stats.items_cached == 3

    def test_stats_as_metrics(self):
        cache = RolloutCache()
        cache.cache_partial(
            env_group_builder=_FakeEnvGroupBuilder(),
            partial_trajectories=[],
            reason=CacheReason.CONSTANT_REWARD,
            current_step=1,
        )
        metrics = cache.stats().as_metrics()
        assert "rollout_cache/items_cached" in metrics
        assert "rollout_cache/current_size" in metrics
        assert "rollout_cache/cached_reason/constant_reward" in metrics
        assert metrics["rollout_cache/items_cached"] == 1.0

    def test_bool_and_len(self):
        cache = RolloutCache()
        assert not cache
        assert len(cache) == 0
        cache.cache_partial(
            env_group_builder=_FakeEnvGroupBuilder(),
            partial_trajectories=[],
            reason=CacheReason.CONSTANT_REWARD,
            current_step=1,
        )
        assert cache
        assert len(cache) == 1

    def test_get_resumable_increments_attempt_count(self):
        cache = RolloutCache()
        builder = _FakeEnvGroupBuilder()
        cache.cache_partial(
            env_group_builder=builder,
            partial_trajectories=[],
            reason=CacheReason.ROLLOUT_ERROR,
            current_step=5,
        )
        entries = cache.get_resumable(current_step=6, max_age_steps=3)
        assert entries[0].attempt_count == 1

        # Re-cache the same entry (simulating a second failure)
        cache.cache_partial(
            env_group_builder=builder,
            partial_trajectories=[],
            reason=CacheReason.ROLLOUT_ERROR,
            current_step=6,
        )
        entries = cache.get_resumable(current_step=7, max_age_steps=3)
        # This is a new entry, so attempt_count starts at 0 -> incremented to 1
        assert entries[0].attempt_count == 1

    def test_stats_returns_copy(self):
        cache = RolloutCache()
        stats1 = cache.stats()
        cache.cache_partial(
            env_group_builder=_FakeEnvGroupBuilder(),
            partial_trajectories=[],
            reason=CacheReason.CONSTANT_REWARD,
            current_step=1,
        )
        stats2 = cache.stats()
        # stats1 should not be mutated
        assert stats1.items_cached == 0
        assert stats2.items_cached == 1

    def test_multiple_get_resumable_calls(self):
        cache = RolloutCache()
        for step in range(5):
            cache.cache_partial(
                env_group_builder=_FakeEnvGroupBuilder(),
                partial_trajectories=[],
                reason=CacheReason.CONSTANT_REWARD,
                current_step=step,
            )
        # Get resumable at step 5 with max_age 1 -- only steps 4 and 5 are
        # within age 1, but step 5 doesn't exist so just step 4
        resumable = cache.get_resumable(current_step=5, max_age_steps=1)
        assert len(resumable) == 1
        # Remaining: steps 0, 1, 2, 3 (too old to resume but not yet expired)
        assert len(cache) == 4

        # Now clear expired
        evicted = cache.clear_expired(current_step=5, max_age_steps=1)
        assert evicted == 4
        assert len(cache) == 0


# ---------------------------------------------------------------------------
# Integration: cache populated by rollout filtering
# ---------------------------------------------------------------------------


class TestRolloutCacheIntegration:
    def test_constant_reward_populates_cache(self):
        """When all trajectories have the same reward, the cache is populated."""
        cache = RolloutCache()
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
                rollout_cache=cache,
                current_step=10,
            )
        )

        assert result is None
        assert len(cache) == 1
        stats = cache.stats()
        assert stats.cache_hits_by_reason["constant_reward"] == 1
        entry = cache.get_resumable(current_step=10, max_age_steps=3)[0]
        assert entry.env_group_builder is builder
        assert len(entry.partial_trajectories) == 3  # all trajectories were completed

    def test_no_cache_when_disabled(self):
        """When rollout_cache is None, no caching occurs."""
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
                rollout_cache=None,
                current_step=10,
            )
        )
        # Just verify it returns None without error (no cache to check)
        assert result is None

    def test_successful_rollout_not_cached(self):
        """Groups that produce valid trajectories should not be cached."""
        cache = RolloutCache()
        # Use different rewards so group is not filtered
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
                rollout_cache=cache,
                current_step=10,
            )
        )

        assert result is not None
        assert len(cache) == 0


class TestCacheStats:
    def test_as_metrics_empty(self):
        stats = CacheStats()
        metrics = stats.as_metrics()
        assert metrics["rollout_cache/items_cached"] == 0.0
        assert metrics["rollout_cache/current_size"] == 0.0

    def test_as_metrics_custom_prefix(self):
        stats = CacheStats(items_cached=5)
        metrics = stats.as_metrics(prefix="my_cache")
        assert "my_cache/items_cached" in metrics
        assert metrics["my_cache/items_cached"] == 5.0


class TestCachedRolloutEntry:
    def test_default_attempt_count(self):
        entry = CachedRolloutEntry(
            env_group_builder=_FakeEnvGroupBuilder(),
            partial_trajectories=[],
            reason=CacheReason.CONSTANT_REWARD,
            cached_at_step=0,
        )
        assert entry.attempt_count == 0
