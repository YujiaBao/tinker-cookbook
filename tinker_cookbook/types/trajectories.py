"""Typed schemas for stored trajectories.

Two trajectory types serve different use cases:
- ``StoredTrainingTrajectory``: Token-level data from RL training rollouts
  (observation/action lengths, per-step rewards, numeric metrics).
- ``StoredEvalTrajectory``: Conversation-level data from benchmark evaluation
  (decoded text turns, timing, error info).
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class StoredStep:
    """One step in a training trajectory (token-level data)."""

    step_idx: int
    ob_len: int
    ac_len: int
    reward: float
    episode_done: bool
    metrics: dict[str, float | int] = field(default_factory=dict)
    logs: dict[str, str | int | float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_idx": self.step_idx,
            "ob_len": self.ob_len,
            "ac_len": self.ac_len,
            "reward": self.reward,
            "episode_done": self.episode_done,
            "metrics": dict(self.metrics),
            "logs": dict(self.logs),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StoredStep":
        return cls(
            step_idx=d["step_idx"],
            ob_len=d["ob_len"],
            ac_len=d["ac_len"],
            reward=d["reward"],
            episode_done=d["episode_done"],
            metrics=d.get("metrics", {}),
            logs=d.get("logs", {}),
        )


@dataclass
class StoredTrainingTrajectory:
    """A serialized training trajectory for storage and visualization.

    Written by ``rl/rollout_logging.py``, read by Tinker Chef.
    """

    schema_version: int = 1
    split: str = "train"
    iteration: int = 0
    group_idx: int = 0
    traj_idx: int = 0
    tags: list[str] = field(default_factory=list)
    sampling_client_step: int | None = None
    total_reward: float = 0.0
    final_reward: float = 0.0
    trajectory_metrics: dict[str, Any] = field(default_factory=dict)
    steps: list[StoredStep] = field(default_factory=list)
    final_ob_len: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "split": self.split,
            "iteration": self.iteration,
            "group_idx": self.group_idx,
            "traj_idx": self.traj_idx,
            "tags": list(self.tags),
            "sampling_client_step": self.sampling_client_step,
            "total_reward": self.total_reward,
            "final_reward": self.final_reward,
            "trajectory_metrics": dict(self.trajectory_metrics),
            "steps": [s.to_dict() for s in self.steps],
            "final_ob_len": self.final_ob_len,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StoredTrainingTrajectory":
        return cls(
            schema_version=d.get("schema_version", 1),
            split=d.get("split", "train"),
            iteration=d.get("iteration", 0),
            group_idx=d.get("group_idx", 0),
            traj_idx=d.get("traj_idx", 0),
            tags=d.get("tags", []),
            sampling_client_step=d.get("sampling_client_step"),
            total_reward=d.get("total_reward", 0.0),
            final_reward=d.get("final_reward", 0.0),
            trajectory_metrics=d.get("trajectory_metrics", {}),
            steps=[StoredStep.from_dict(s) for s in d.get("steps", [])],
            final_ob_len=d.get("final_ob_len", 0),
        )


@dataclass
class StoredTurn:
    """One turn in an eval trajectory (conversation-level data)."""

    role: str
    content: str
    token_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "token_count": self.token_count,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StoredTurn":
        return cls(
            role=d["role"],
            content=d["content"],
            token_count=d.get("token_count", 0),
            metadata=d.get("metadata", {}),
        )


@dataclass
class StoredEvalTrajectory:
    """A serialized eval trajectory for storage and visualization.

    Written by ``eval/benchmarks/_runner.py``, read by Tinker Chef.
    """

    idx: int = 0
    benchmark: str = ""
    example_id: str = ""
    turns: list[StoredTurn] = field(default_factory=list)
    reward: float = 0.0
    metrics: dict[str, Any] = field(default_factory=dict)
    logs: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    time_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "idx": self.idx,
            "benchmark": self.benchmark,
            "example_id": self.example_id,
            "turns": [t.to_dict() for t in self.turns],
            "reward": self.reward,
            "metrics": dict(self.metrics),
            "logs": dict(self.logs),
            "time_seconds": self.time_seconds,
        }
        if self.error is not None:
            d["error"] = self.error
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StoredEvalTrajectory":
        return cls(
            idx=d.get("idx", 0),
            benchmark=d.get("benchmark", ""),
            example_id=d.get("example_id", ""),
            turns=[StoredTurn.from_dict(t) for t in d.get("turns", [])],
            reward=d.get("reward", 0.0),
            metrics=d.get("metrics", {}),
            logs=d.get("logs", {}),
            error=d.get("error"),
            time_seconds=d.get("time_seconds", 0.0),
        )
