"""Reader for per-iteration rollout summary JSONL files."""

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


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
        """Read rollout summaries for a specific iteration and split.

        Args:
            iteration: The iteration number.
            split: Dataset split ("train" or "eval").
            label: For eval splits, the evaluation label (e.g. "test").

        Returns:
            List of rollout summary records.
        """
        iter_dir = self._run_path / f"iteration_{iteration:06d}"
        if not iter_dir.is_dir():
            return []

        if split == "train":
            filename = "train_rollout_summaries.jsonl"
        elif label:
            filename = f"eval_{label}_rollout_summaries.jsonl"
        else:
            filename = f"{split}_rollout_summaries.jsonl"

        path = iter_dir / filename
        return self._read_jsonl_cached(path)

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
    def _read_jsonl_cached(path: Path) -> list[dict[str, Any]]:
        """Read and cache a JSONL file. Cached by path (immutable past iterations)."""
        if not path.exists():
            return []

        records: list[dict[str, Any]] = []
        try:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to read rollouts from %s: %s", path, e)

        return records
