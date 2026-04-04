"""Reinforcement learning: environment protocol, types, and training loops."""

from tinker_cookbook.rl.algorithm_registry import (
    AdvantageEstimator,
    PolicyLossConfigurator,
    advantage_registry,
    get_advantage_fn,
    resolve_policy_loss_config,
    policy_loss_registry,
    register_advantage,
    register_policy_loss,
)
from tinker_cookbook.rl.interleaved import InterleavedRLDatasetBuilder
from tinker_cookbook.rl.rollout_strategy import FailFast, RetryOnFailure, RolloutStrategy
from tinker_cookbook.rl.types import (
    Action,
    ActionExtra,
    Env,
    EnvGroupBuilder,
    Logs,
    Metrics,
    Observation,
    RLDataset,
    RLDatasetBuilder,
    RolloutError,
    StepResult,
    Trajectory,
    TrajectoryGroup,
    Transition,
)

__all__ = [
    # Algorithm registry (algorithm_registry.py)
    "AdvantageEstimator",
    "PolicyLossConfigurator",
    "advantage_registry",
    "get_advantage_fn",
    "resolve_policy_loss_config",
    "policy_loss_registry",
    "register_advantage",
    "register_policy_loss",
    # Core protocol and types (types.py)
    "Action",
    "ActionExtra",
    "Env",
    "EnvGroupBuilder",
    "Logs",
    "Metrics",
    "Observation",
    "RLDataset",
    "RLDatasetBuilder",
    "RolloutError",
    "StepResult",
    "Trajectory",
    "TrajectoryGroup",
    "Transition",
    # Interleaved datasets (interleaved.py)
    "InterleavedRLDatasetBuilder",
    # Rollout strategies (rollout_strategy.py)
    "FailFast",
    "RetryOnFailure",
    "RolloutStrategy",
]
