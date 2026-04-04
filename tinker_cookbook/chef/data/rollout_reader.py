"""Reader for per-iteration rollout summary JSONL files."""

from functools import lru_cache
from pathlib import Path
from typing import Any

from tinker_cookbook.chef.data.io import read_jsonl


class RolloutReader:
    """Reads rollout summary JSONL files from iteration directories.

    Results are cached with an LRU cache since rollout files for past
    iterations are immutable once the iteration completes.
    """

    def __init__(self, run_path: Path) -> None:
        self._run_path = run_path

    def read_rollouts(
        self,
        iteration: int,
        split: str = "train",
        label: str | None = None,
    ) -> list[dict[str, Any]]:
        """Read rollout summaries for a specific iteration and split."""
        iter_dir = self._run_path / f"iteration_{iteration:06d}"

        if split == "train":
            filename = "train_rollout_summaries.jsonl"
        elif label:
            filename = f"eval_{label}_rollout_summaries.jsonl"
        else:
            filename = f"{split}_rollout_summaries.jsonl"

        return self._read_cached(iter_dir / filename)

    def read_single_rollout(
        self,
        iteration: int,
        group_idx: int,
        traj_idx: int,
        split: str = "train",
        label: str | None = None,
    ) -> dict[str, Any] | None:
        """Read a single rollout by group and trajectory index."""
        rollouts = self.read_rollouts(iteration, split, label)
        for record in rollouts:
            if record.get("group_idx") == group_idx and record.get("traj_idx") == traj_idx:
                return record
        return None

    @staticmethod
    @lru_cache(maxsize=64)
    def _read_cached(path: Path) -> list[dict[str, Any]]:
        return read_jsonl(path)
