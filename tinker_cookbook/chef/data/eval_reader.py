"""Reader for eval benchmark data from the eval-benchmark-framework.

Reads the EvalStore directory structure:
    eval_store/
        runs.jsonl                    # Run index
        runs/{run_id}/
            metadata.json             # RunMetadata
            {benchmark}/
                result.json           # BenchmarkResult
                trajectories.jsonl    # StoredTrajectory per line
"""

from functools import lru_cache
from pathlib import Path
from typing import Any

from tinker_cookbook.chef.data.io import read_json, read_jsonl


class EvalReader:
    """Reads eval benchmark data from an EvalStore directory."""

    def __init__(self, eval_root: Path) -> None:
        self._root = eval_root

    @property
    def root(self) -> Path:
        return self._root

    def list_eval_runs(self) -> list[dict[str, Any]]:
        """Read the runs.jsonl index file, returning all eval run metadata."""
        return read_jsonl(self._root / "runs.jsonl")

    def get_eval_run_metadata(self, run_id: str) -> dict[str, Any] | None:
        """Read metadata.json for a specific eval run."""
        return read_json(self._root / "runs" / run_id / "metadata.json")

    def list_benchmarks(self, run_id: str) -> list[str]:
        """List benchmark names that have results for an eval run."""
        run_dir = self._root / "runs" / run_id
        if not run_dir.is_dir():
            return []
        return sorted(
            child.name
            for child in run_dir.iterdir()
            if child.is_dir() and (child / "result.json").exists()
        )

    def get_benchmark_result(self, run_id: str, benchmark: str) -> dict[str, Any] | None:
        """Read result.json for a specific benchmark in an eval run."""
        return self._read_json_cached(self._root / "runs" / run_id / benchmark / "result.json")

    def get_benchmark_trajectories(
        self,
        run_id: str,
        benchmark: str,
    ) -> list[dict[str, Any]]:
        """Read all stored trajectories for a benchmark in an eval run."""
        return self._read_jsonl_cached(
            self._root / "runs" / run_id / benchmark / "trajectories.jsonl"
        )

    def get_single_trajectory(
        self,
        run_id: str,
        benchmark: str,
        idx: int,
    ) -> dict[str, Any] | None:
        """Read a single trajectory by index."""
        for traj in self.get_benchmark_trajectories(run_id, benchmark):
            if traj.get("idx") == idx:
                return traj
        return None

    def get_summary(self, run_id: str) -> dict[str, Any] | None:
        """Read the cross-benchmark summary.json for an eval run."""
        return read_json(self._root / "runs" / run_id / "summary.json")

    def get_scores_table(self) -> list[dict[str, Any]]:
        """Build a scores table: [{run_id, model_name, checkpoint, scores: {...}}, ...]."""
        table: list[dict[str, Any]] = []
        for run_entry in self.list_eval_runs():
            run_id = run_entry.get("run_id", "")
            if not run_id:
                continue
            metadata = self.get_eval_run_metadata(run_id)
            row: dict[str, Any] = {
                "run_id": run_id,
                "model_name": metadata.get("model_name", "") if metadata else "",
                "checkpoint_name": metadata.get("checkpoint_name") if metadata else None,
                "timestamp": metadata.get("timestamp") if metadata else None,
            }
            if metadata and "scores" in metadata:
                row["scores"] = metadata["scores"]
            else:
                scores: dict[str, float] = {}
                for benchmark in self.list_benchmarks(run_id):
                    result = self.get_benchmark_result(run_id, benchmark)
                    if result and "score" in result:
                        scores[benchmark] = result["score"]
                row["scores"] = scores
            table.append(row)
        return table

    @staticmethod
    @lru_cache(maxsize=32)
    def _read_json_cached(path: Path) -> dict[str, Any] | None:
        return read_json(path)

    @staticmethod
    @lru_cache(maxsize=32)
    def _read_jsonl_cached(path: Path) -> list[dict[str, Any]]:
        return read_jsonl(path)
