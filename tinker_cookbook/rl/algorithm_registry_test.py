"""Unit tests for the algorithm registry system."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from tinker_cookbook.rl.algorithm_registry import (
    AdvantageEstimator,
    PolicyLossConfigurator,
    Registry,
    advantage_registry,
    get_advantage_fn,
    get_policy_loss_config,
    grpo_advantage,
    normalized_grpo_advantage,
    policy_loss_registry,
    register_advantage,
    register_policy_loss,
)
from tinker_cookbook.rl.types import (
    Trajectory,
    TrajectoryGroup,
    Transition,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trajectory(reward: float) -> Trajectory:
    """Create a minimal single-step trajectory with the given reward."""
    import tinker

    ob = tinker.ModelInput.from_ints([1, 2, 3])
    from tinker_cookbook.completers import TokensWithLogprobs

    ac = TokensWithLogprobs(tokens=[4, 5], maybe_logprobs=[-0.1, -0.2], stop_reason="stop")
    transition = Transition(
        ob=ob,
        ac=ac,
        reward=reward,
        episode_done=True,
    )
    return Trajectory(transitions=[transition], final_ob=ob)


def _make_group(rewards: list[float]) -> TrajectoryGroup:
    """Create a TrajectoryGroup with trajectories having the given rewards."""
    trajectories = [_make_trajectory(r) for r in rewards]
    return TrajectoryGroup(
        trajectories_G=trajectories,
        final_rewards_G=[0.0] * len(rewards),
        metrics_G=[{} for _ in rewards],
    )


# ---------------------------------------------------------------------------
# Registry class tests
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_register_and_get(self) -> None:
        reg: Registry[AdvantageEstimator] = Registry("test")

        @reg.register("my_fn")
        def my_fn(trajectory_groups_P: list[TrajectoryGroup], **kwargs: Any) -> list[torch.Tensor]:
            return []

        assert "my_fn" in reg
        assert reg.get("my_fn") is my_fn

    def test_duplicate_registration_raises(self) -> None:
        reg: Registry[AdvantageEstimator] = Registry("test")

        @reg.register("dup")
        def fn1(trajectory_groups_P: list[TrajectoryGroup], **kwargs: Any) -> list[torch.Tensor]:
            return []

        with pytest.raises(ValueError, match="already registered"):

            @reg.register("dup")
            def fn2(
                trajectory_groups_P: list[TrajectoryGroup], **kwargs: Any
            ) -> list[torch.Tensor]:
                return []

    def test_get_unknown_raises(self) -> None:
        reg: Registry[AdvantageEstimator] = Registry("test")
        with pytest.raises(KeyError, match="Unknown test 'nonexistent'"):
            reg.get("nonexistent")

    def test_list_names(self) -> None:
        reg: Registry[AdvantageEstimator] = Registry("test")

        @reg.register("a")
        def fn_a(trajectory_groups_P: list[TrajectoryGroup], **kwargs: Any) -> list[torch.Tensor]:
            return []

        @reg.register("b")
        def fn_b(trajectory_groups_P: list[TrajectoryGroup], **kwargs: Any) -> list[torch.Tensor]:
            return []

        assert reg.list_names() == ["a", "b"]

    def test_contains(self) -> None:
        reg: Registry[AdvantageEstimator] = Registry("test")

        @reg.register("present")
        def fn(trajectory_groups_P: list[TrajectoryGroup], **kwargs: Any) -> list[torch.Tensor]:
            return []

        assert "present" in reg
        assert "absent" not in reg

    def test_repr(self) -> None:
        reg: Registry[AdvantageEstimator] = Registry("widget")
        assert "widget" in repr(reg)


# ---------------------------------------------------------------------------
# Built-in advantage estimator tests
# ---------------------------------------------------------------------------


class TestGrpoAdvantage:
    def test_single_group_centering(self) -> None:
        group = _make_group([1.0, 3.0, 5.0])
        result = grpo_advantage([group])
        assert len(result) == 1
        advantages = result[0]
        assert len(advantages) == 3
        # Mean is 3.0, so advantages should be [-2, 0, 2]
        assert torch.allclose(advantages, torch.tensor([-2.0, 0.0, 2.0]))

    def test_multiple_groups(self) -> None:
        g1 = _make_group([0.0, 4.0])
        g2 = _make_group([10.0, 20.0, 30.0])
        result = grpo_advantage([g1, g2])
        assert len(result) == 2
        # Group 1: mean=2, advantages=[-2, 2]
        assert torch.allclose(result[0], torch.tensor([-2.0, 2.0]))
        # Group 2: mean=20, advantages=[-10, 0, 10]
        assert torch.allclose(result[1], torch.tensor([-10.0, 0.0, 10.0]))

    def test_uniform_rewards_zero_advantages(self) -> None:
        group = _make_group([5.0, 5.0, 5.0])
        result = grpo_advantage([group])
        assert torch.allclose(result[0], torch.tensor([0.0, 0.0, 0.0]))


class TestNormalizedGrpoAdvantage:
    def test_unit_variance(self) -> None:
        group = _make_group([1.0, 3.0, 5.0])
        result = normalized_grpo_advantage([group])
        advantages = result[0]
        # Should have mean ~0 and std ~1
        assert abs(float(advantages.mean())) < 1e-6
        assert abs(float(advantages.std()) - 1.0) < 1e-5

    def test_uniform_rewards_zero_advantages(self) -> None:
        group = _make_group([5.0, 5.0, 5.0])
        result = normalized_grpo_advantage([group])
        assert torch.allclose(result[0], torch.tensor([0.0, 0.0, 0.0]))


# ---------------------------------------------------------------------------
# Built-in policy loss tests
# ---------------------------------------------------------------------------


class TestPolicyLossRegistry:
    def test_importance_sampling_registered(self) -> None:
        assert "importance_sampling" in policy_loss_registry

    def test_ppo_registered(self) -> None:
        assert "ppo" in policy_loss_registry

    def test_importance_sampling_returns_correct_tuple(self) -> None:
        loss_fn, config = get_policy_loss_config("importance_sampling")
        assert loss_fn == "importance_sampling"
        assert config is None

    def test_ppo_returns_correct_tuple(self) -> None:
        loss_fn, config = get_policy_loss_config("ppo")
        assert loss_fn == "ppo"
        assert config is not None
        assert config["clip_param"] == 0.2

    def test_ppo_custom_clip_param(self) -> None:
        loss_fn, config = get_policy_loss_config("ppo", clip_param=0.3)
        assert loss_fn == "ppo"
        assert config is not None
        assert config["clip_param"] == 0.3

    def test_unknown_loss_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown policy loss"):
            get_policy_loss_config("nonexistent_loss")


# ---------------------------------------------------------------------------
# Global registry state tests
# ---------------------------------------------------------------------------


class TestGlobalRegistries:
    def test_advantage_registry_has_builtins(self) -> None:
        names = advantage_registry.list_names()
        assert "grpo" in names
        assert "normalized_grpo" in names

    def test_policy_loss_registry_has_builtins(self) -> None:
        names = policy_loss_registry.list_names()
        assert "importance_sampling" in names
        assert "ppo" in names

    def test_get_advantage_fn_returns_callable(self) -> None:
        fn = get_advantage_fn("grpo")
        assert callable(fn)


# ---------------------------------------------------------------------------
# Decorator API tests
# ---------------------------------------------------------------------------


class TestDecoratorApi:
    def test_register_advantage_decorator(self) -> None:
        # The decorator should return the original function
        @register_advantage("_test_custom_advantage")
        def custom(
            trajectory_groups_P: list[TrajectoryGroup], **kwargs: Any
        ) -> list[torch.Tensor]:
            return [torch.tensor([42.0]) for _ in trajectory_groups_P]

        fn = get_advantage_fn("_test_custom_advantage")
        assert fn is custom

        group = _make_group([1.0])
        result = fn([group])
        assert torch.allclose(result[0], torch.tensor([42.0]))

    def test_register_policy_loss_decorator(self) -> None:

        @register_policy_loss("_test_custom_loss")
        def custom(**kwargs: Any) -> tuple[str, dict[str, Any] | None]:
            return "importance_sampling", {"custom_key": True}

        loss_fn, config = get_policy_loss_config("_test_custom_loss")
        assert loss_fn == "importance_sampling"
        assert config is not None
        assert config["custom_key"] is True
