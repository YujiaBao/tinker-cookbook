"""Registry system for RL advantage estimators and policy loss functions.

Provides a plugin architecture so that new RL algorithms can be added by
registering new functions, without modifying the training loop directly.

Usage::

    from tinker_cookbook.rl.algorithm_registry import (
        register_advantage,
        register_policy_loss,
        get_advantage_fn,
        resolve_policy_loss_config,
    )

    # Register a custom advantage estimator
    @register_advantage("my_advantage")
    def my_advantage(trajectory_groups_P, **kwargs):
        ...

    # Register a custom policy loss
    @register_policy_loss("my_loss")
    def my_loss(**kwargs):
        return "importance_sampling", {"clip_ratio": 0.5}

    # Look up registered functions
    advantage_fn = get_advantage_fn("grpo")
    loss_fn, loss_fn_config = resolve_policy_loss_config("ppo")
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Generic, TypeVar

import torch
from tinker.types import LossFnType

from tinker_cookbook.exceptions import ConfigurationError
from tinker_cookbook.rl.types import TrajectoryGroup

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Type aliases for registered functions
# ---------------------------------------------------------------------------

AdvantageEstimator = Callable[..., list[torch.Tensor]]
"""Signature: (trajectory_groups_P: list[TrajectoryGroup], **kwargs) -> list[torch.Tensor]

Each returned tensor has shape ``(G,)`` where ``G`` is the group size, giving
the per-trajectory advantage within that group.
"""

PolicyLossConfigurator = Callable[..., tuple[LossFnType, dict[str, Any] | None]]
"""Signature: (**kwargs) -> tuple[LossFnType, dict[str, Any] | None]

Returns the Tinker loss function name and optional config dict.
"""


# ---------------------------------------------------------------------------
# Registry class
# ---------------------------------------------------------------------------


T = TypeVar("T")


class Registry(Generic[T]):
    """A simple dict-based registry mapping string names to callables.

    Type parameter ``T`` is the callable type stored in the registry
    (e.g. :data:`AdvantageEstimator` or :data:`PolicyLossConfigurator`).
    """

    def __init__(self, kind: str) -> None:
        """Initialize a new registry.

        Args:
            kind: Human-readable category name used in error messages
                (e.g. ``"advantage estimator"``).
        """
        self._kind = kind
        self._entries: dict[str, T] = {}

    def register(self, name: str) -> Callable[[T], T]:
        """Decorator that registers ``fn`` under ``name``.

        Args:
            name: Unique key for this entry.

        Returns:
            A decorator that stores the wrapped callable in the registry.

        Raises:
            ConfigurationError: If ``name`` is already registered.
        """

        def decorator(fn: T) -> T:
            if name in self._entries:
                raise ConfigurationError(
                    f"{self._kind} '{name}' is already registered. "
                    f"Registered names: {list(self._entries)}"
                )
            self._entries[name] = fn
            return fn

        return decorator

    def get(self, name: str) -> T:
        """Look up a registered entry by name.

        Args:
            name: The registered key to retrieve.

        Returns:
            The callable previously registered under ``name``.

        Raises:
            KeyError: If ``name`` has not been registered.
        """
        if name not in self._entries:
            raise KeyError(
                f"Unknown {self._kind} '{name}'. "
                f"Available: {list(self._entries)}"
            )
        return self._entries[name]

    def _remove(self, name: str) -> None:
        """Remove a registered entry by name (for testing only).

        Args:
            name: The registered key to remove.

        Raises:
            KeyError: If ``name`` is not registered.
        """
        if name not in self._entries:
            raise KeyError(
                f"Cannot remove unknown {self._kind} '{name}'. "
                f"Available: {list(self._entries)}"
            )
        del self._entries[name]

    def list_names(self) -> list[str]:
        """Return all registered names in insertion order.

        Returns:
            List of registered name strings.
        """
        return list(self._entries)

    def __contains__(self, name: str) -> bool:
        return name in self._entries

    def __repr__(self) -> str:
        return f"Registry({self._kind!r}, entries={list(self._entries)})"


# ---------------------------------------------------------------------------
# Global registry instances
# ---------------------------------------------------------------------------

advantage_registry: Registry[AdvantageEstimator] = Registry("advantage estimator")
policy_loss_registry: Registry[PolicyLossConfigurator] = Registry("policy loss")


# ---------------------------------------------------------------------------
# Convenience decorators
# ---------------------------------------------------------------------------


def register_advantage(name: str) -> Callable[[AdvantageEstimator], AdvantageEstimator]:
    """Register an advantage estimator function.

    The decorated function must have the signature::

        def fn(trajectory_groups_P: list[TrajectoryGroup], **kwargs) -> list[torch.Tensor]

    Example::

        @register_advantage("grpo")
        def grpo_advantage(trajectory_groups_P, **kwargs):
            ...
    """
    return advantage_registry.register(name)


def register_policy_loss(name: str) -> Callable[[PolicyLossConfigurator], PolicyLossConfigurator]:
    """Register a policy loss configurator function.

    The decorated function must have the signature::

        def fn(**kwargs) -> tuple[LossFnType, dict[str, Any] | None]

    Example::

        @register_policy_loss("ppo")
        def ppo_loss(**kwargs):
            return "ppo", {"clip_param": 0.2}
    """
    return policy_loss_registry.register(name)


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------


def get_advantage_fn(name: str) -> AdvantageEstimator:
    """Retrieve a registered advantage estimator by name.

    Args:
        name: The registered name (e.g. ``"grpo"``).

    Returns:
        The advantage estimator callable.

    Raises:
        KeyError: If ``name`` is not registered.
    """
    return advantage_registry.get(name)


def resolve_policy_loss_config(name: str, **kwargs: Any) -> tuple[LossFnType, dict[str, Any] | None]:
    """Retrieve and call a registered policy loss configurator.

    Args:
        name: The registered name (e.g. ``"ppo"``).
        **kwargs: Forwarded to the configurator function.

    Returns:
        A ``(loss_fn, loss_fn_config)`` tuple ready for
        :func:`tinker_cookbook.rl.train.train_step`.

    Raises:
        KeyError: If ``name`` is not registered.
    """
    configurator = policy_loss_registry.get(name)
    return configurator(**kwargs)


# ---------------------------------------------------------------------------
# Built-in advantage estimators
# ---------------------------------------------------------------------------


@register_advantage("grpo")
def grpo_advantage(
    trajectory_groups_P: list[TrajectoryGroup],
    **kwargs: Any,
) -> list[torch.Tensor]:
    """GRPO advantage: mean-centered rewards within each group (no variance normalization).

    For each trajectory group, the advantage of trajectory *i* is
    ``reward_i - mean(rewards)``.  This subtracts the per-group mean but does
    **not** divide by the standard deviation, which is equivalent to what
    VERL calls "Dr.GRPO" (``norm_adv_by_std_in_grpo=False``).

    This is the default estimator in tinker-cookbook.

    .. note::

       Normalization is **per-group**, not batch-global.  SLIME/VERL whiten
       advantages across the entire DP batch, which requires distributed
       coordination not available in Tinker's API architecture.

    Args:
        trajectory_groups_P: Groups of trajectories whose rewards
            are centered independently.

    Returns:
        Per-group advantage tensors of shape ``(G,)``.
    """
    advantages_P: list[torch.Tensor] = []
    for traj_group in trajectory_groups_P:
        rewards_G = torch.tensor(traj_group.get_total_rewards())
        advantages_G = rewards_G - rewards_G.mean()
        advantages_P.append(advantages_G)
    return advantages_P


@register_advantage("normalized_grpo")
def normalized_grpo_advantage(
    trajectory_groups_P: list[TrajectoryGroup],
    eps: float = 1e-8,
    **kwargs: Any,
) -> list[torch.Tensor]:
    """Normalized GRPO advantage: center and scale rewards by per-group std.

    Computes ``(reward_i - mean(rewards)) / std(rewards)`` within each group,
    producing unit-variance advantages.  This is the standard GRPO formulation
    from the original paper and corresponds to VERL's default behavior
    (``norm_adv_by_std_in_grpo=True``).

    Falls back to zero advantages when the standard deviation is below ``eps``.

    .. note::

       Normalization is **per-group**, not batch-global.  SLIME/VERL whiten
       advantages across the entire DP batch, which requires distributed
       coordination not available in Tinker's API architecture.

    Args:
        trajectory_groups_P: Groups of trajectories.
        eps: Small constant to avoid division by zero.

    Returns:
        Per-group advantage tensors of shape ``(G,)``.
    """
    advantages_P: list[torch.Tensor] = []
    for traj_group in trajectory_groups_P:
        rewards_G = torch.tensor(traj_group.get_total_rewards())
        mean = rewards_G.mean()
        std = rewards_G.std()
        if std < eps:
            advantages_G = torch.zeros_like(rewards_G)
        else:
            advantages_G = (rewards_G - mean) / std
        advantages_P.append(advantages_G)
    return advantages_P


# ---------------------------------------------------------------------------
# Built-in policy loss configurators
# ---------------------------------------------------------------------------


@register_policy_loss("importance_sampling")
def importance_sampling_loss(**kwargs: Any) -> tuple[LossFnType, dict[str, Any] | None]:
    """Standard importance-sampling policy gradient loss (REINFORCE-style).

    This is the default loss function used by the RL trainer.
    """
    return "importance_sampling", None


@register_policy_loss("ppo")
def ppo_loss(
    clip_param: float = 0.2,
    **kwargs: Any,
) -> tuple[LossFnType, dict[str, Any] | None]:
    """PPO clipped surrogate loss.

    Args:
        clip_param: Clipping parameter for the surrogate objective.
            Defaults to 0.2.
    """
    return "ppo", {"clip_param": clip_param}
