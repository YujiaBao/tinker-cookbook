"""Reinforcement learning: environment protocol, types, and training loops."""

from tinker_cookbook.rl.agentic_telemetry import (
    compute_agentic_metrics,
    log_agentic_batch_summary,
    log_agentic_episode,
    log_agentic_loss_masking,
)
from tinker_cookbook.rl.interleaved import InterleavedRLDatasetBuilder
from tinker_cookbook.rl.loss_masking import TrajectoryMaskSummary, TurnMaskInfo, compute_trajectory_mask_summary
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
    # Loss masking utilities (loss_masking.py)
    "TrajectoryMaskSummary",
    "TurnMaskInfo",
    "compute_trajectory_mask_summary",
    # Agentic telemetry (agentic_telemetry.py)
    "compute_agentic_metrics",
    "log_agentic_batch_summary",
    "log_agentic_episode",
    "log_agentic_loss_masking",
    # Rollout strategies (rollout_strategy.py)
    "FailFast",
    "RetryOnFailure",
    "RolloutStrategy",
]
