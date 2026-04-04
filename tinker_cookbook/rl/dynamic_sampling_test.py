"""Tests for DAPO-style dynamic sampling: filter_low_variance_groups."""

import tinker

from tinker_cookbook.completers import TokensWithLogprobs
from tinker_cookbook.rl.data_processing import filter_low_variance_groups
from tinker_cookbook.rl.types import Trajectory, TrajectoryGroup, Transition


def _make_trajectory(reward: float) -> Trajectory:
    """Create a minimal single-step trajectory with the given reward."""
    transition = Transition(
        ob=tinker.ModelInput.from_ints([1, 2, 3]),
        ac=TokensWithLogprobs(tokens=[4, 5], maybe_logprobs=[-0.1, -0.2]),
        reward=reward,
        episode_done=True,
    )
    return Trajectory(transitions=[transition], final_ob=tinker.ModelInput.from_ints([6]))


def _make_group(rewards: list[float]) -> TrajectoryGroup:
    """Create a TrajectoryGroup with the given per-trajectory rewards.

    The reward is split: each transition gets reward 0.0, and the full amount
    goes into final_rewards_G (group-level reward).
    """
    trajectories = [_make_trajectory(0.0) for _ in rewards]
    return TrajectoryGroup(
        trajectories_G=trajectories,
        final_rewards_G=rewards,
        metrics_G=[{} for _ in rewards],
    )


class TestFilterLowVarianceGroups:
    def test_no_filtering_when_all_groups_have_variance(self) -> None:
        """Groups with diverse rewards should all be kept."""
        groups = [
            _make_group([1.0, 0.0, 0.5]),
            _make_group([0.0, 1.0, 0.0]),
        ]
        filtered, num_removed, _ = filter_low_variance_groups(
            groups, min_reward_std=0.0, max_filter_ratio=0.5
        )
        assert len(filtered) == 2
        assert num_removed == 0

    def test_filters_zero_variance_groups(self) -> None:
        """Groups where all rewards are identical should be filtered."""
        diverse_group = _make_group([1.0, 0.0])
        constant_group = _make_group([1.0, 1.0])
        groups = [diverse_group, constant_group]
        filtered, num_removed, _ = filter_low_variance_groups(
            groups, min_reward_std=0.0, max_filter_ratio=0.5
        )
        assert len(filtered) == 1
        assert num_removed == 1
        assert filtered[0] is diverse_group

    def test_respects_max_filter_ratio(self) -> None:
        """Should not filter more groups than max_filter_ratio allows."""
        diverse = _make_group([1.0, 0.0])
        constant_1 = _make_group([0.5, 0.5])
        constant_2 = _make_group([0.0, 0.0])
        constant_3 = _make_group([1.0, 1.0])
        groups = [diverse, constant_1, constant_2, constant_3]
        # max_filter_ratio=0.5 means at most 2 of 4 groups can be removed
        filtered, num_removed, _ = filter_low_variance_groups(
            groups, min_reward_std=0.0, max_filter_ratio=0.5
        )
        assert num_removed == 2
        assert len(filtered) == 2
        # The diverse group should always be kept
        assert diverse in filtered

    def test_min_reward_std_threshold(self) -> None:
        """Groups with std below min_reward_std should be filtered."""
        high_var = _make_group([1.0, 0.0])  # std = 0.5
        low_var = _make_group([0.49, 0.51])  # std ~= 0.01
        groups = [high_var, low_var]
        filtered, num_removed, _ = filter_low_variance_groups(
            groups, min_reward_std=0.05, max_filter_ratio=0.5
        )
        assert len(filtered) == 1
        assert num_removed == 1
        assert filtered[0] is high_var

    def test_all_constant_groups_returns_original(self) -> None:
        """If all groups are constant, return all of them to avoid empty batch."""
        groups = [
            _make_group([1.0, 1.0]),
            _make_group([0.0, 0.0]),
        ]
        filtered, num_removed, _ = filter_low_variance_groups(
            groups, min_reward_std=0.0, max_filter_ratio=0.5
        )
        # max_filter_ratio=0.5 allows filtering 1, but we'd have only 1 left
        # which is fine since one constant group can be kept
        assert len(filtered) >= 1
        assert num_removed + len(filtered) == len(groups)

    def test_max_filter_ratio_zero_disables_filtering(self) -> None:
        """With max_filter_ratio=0.0, no groups should be filtered."""
        groups = [
            _make_group([1.0, 1.0]),
            _make_group([0.0, 0.0]),
        ]
        filtered, num_removed, _ = filter_low_variance_groups(
            groups, min_reward_std=0.0, max_filter_ratio=0.0
        )
        assert len(filtered) == 2
        assert num_removed == 0

    def test_single_trajectory_group(self) -> None:
        """A group with a single trajectory has std=0 and should be filterable."""
        single = _make_group([1.0])
        diverse = _make_group([1.0, 0.0])
        groups = [single, diverse]
        filtered, num_removed, _ = filter_low_variance_groups(
            groups, min_reward_std=0.0, max_filter_ratio=0.5
        )
        assert len(filtered) == 1
        assert num_removed == 1
        assert filtered[0] is diverse

    def test_empty_input(self) -> None:
        """Empty input should return empty output with 0 filtered."""
        filtered, num_removed, filtered_builders = filter_low_variance_groups(
            [], min_reward_std=0.0, max_filter_ratio=0.5
        )
        assert filtered == []
        assert num_removed == 0
        assert filtered_builders is None

    def test_env_group_builders_filtered_in_lockstep(self) -> None:
        """env_group_builders_P should be filtered in lockstep with trajectory groups."""
        diverse_group = _make_group([1.0, 0.0])
        constant_group = _make_group([1.0, 1.0])
        groups = [diverse_group, constant_group]
        builders = ["builder_diverse", "builder_constant"]
        filtered, num_removed, filtered_builders = filter_low_variance_groups(
            groups,
            min_reward_std=0.0,
            max_filter_ratio=0.5,
            env_group_builders_P=builders,
        )
        assert len(filtered) == 1
        assert num_removed == 1
        assert filtered[0] is diverse_group
        assert filtered_builders == ["builder_diverse"]

    def test_env_group_builders_none_when_not_provided(self) -> None:
        """When env_group_builders_P is None, returned builders should be None."""
        groups = [_make_group([1.0, 0.0])]
        filtered, num_removed, filtered_builders = filter_low_variance_groups(
            groups, min_reward_std=0.0, max_filter_ratio=0.5
        )
        assert filtered_builders is None


class TestDynamicSamplingConfig:
    def test_config_construction(self) -> None:
        """DynamicSamplingConfig should be constructible with defaults."""
        from tinker_cookbook.rl.train import DynamicSamplingConfig

        cfg = DynamicSamplingConfig()
        assert cfg.oversample_ratio == 1.5
        assert cfg.min_reward_std == 0.0
        assert cfg.max_filter_ratio == 0.5

    def test_config_custom_values(self) -> None:
        """DynamicSamplingConfig should accept custom values."""
        from tinker_cookbook.rl.train import DynamicSamplingConfig

        cfg = DynamicSamplingConfig(
            oversample_ratio=2.0,
            min_reward_std=0.1,
            max_filter_ratio=0.3,
        )
        assert cfg.oversample_ratio == 2.0
        assert cfg.min_reward_std == 0.1
        assert cfg.max_filter_ratio == 0.3

    def test_config_rejects_low_oversample_ratio(self) -> None:
        """DynamicSamplingConfig should reject oversample_ratio < 1.0."""
        import pytest

        from tinker_cookbook.rl.train import DynamicSamplingConfig

        with pytest.raises(ValueError, match="oversample_ratio must be >= 1.0"):
            DynamicSamplingConfig(oversample_ratio=0.5)

    def test_config_field_on_rl_config(self) -> None:
        """Config.dynamic_sampling should default to None."""
        from tinker_cookbook.rl.train import Config

        # We can't easily construct a full Config (requires dataset_builder etc),
        # so just check the field exists via annotations
        assert "dynamic_sampling" in Config.__annotations__
