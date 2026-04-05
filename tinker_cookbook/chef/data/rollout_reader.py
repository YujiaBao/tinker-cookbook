"""Reader for per-iteration rollout summary JSONL files."""

from typing import Any

from tinker_cookbook.storage import Storage, storage_join, storage_read_jsonl


class RolloutReader:
    """Reads rollout summary JSONL files from iteration directories.

    Past iterations are immutable, so results are cached by
    (iteration, split, label) to avoid re-reading on each API call.
    """

    def __init__(self, storage: Storage, run_prefix: str) -> None:
        self._storage = storage
        self._prefix = run_prefix
        self._cache: dict[str, list[dict[str, Any]]] = {}

    def read_rollouts(
        self,
        iteration: int,
        split: str = "train",
        label: str | None = None,
    ) -> list[dict[str, Any]]:
        if split == "train":
            filename = "train_rollout_summaries.jsonl"
        elif label:
            filename = f"eval_{label}_rollout_summaries.jsonl"
        else:
            filename = f"{split}_rollout_summaries.jsonl"

        cache_key = f"{iteration}/{filename}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        path = storage_join(self._prefix, f"iteration_{iteration:06d}", filename)
        records = storage_read_jsonl(self._storage, path)
        self._cache[cache_key] = records
        return records

    def read_single_rollout(
        self,
        iteration: int,
        group_idx: int,
        traj_idx: int,
        split: str = "train",
        label: str | None = None,
    ) -> dict[str, Any] | None:
        for record in self.read_rollouts(iteration, split, label):
            if record.get("group_idx") == group_idx and record.get("traj_idx") == traj_idx:
                return record
        return None
