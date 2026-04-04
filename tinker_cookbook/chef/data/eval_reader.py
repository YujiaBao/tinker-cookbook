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

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class EvalReader:
    """Reads eval benchmark data from an EvalStore directory.

    Supports both the EvalStore layout (runs.jsonl + runs/{run_id}/...)
    and standalone eval directories with result.json and trajectories.jsonl.
    """

    def __init__(self, eval_root: Path) -> None:
        self._root = eval_root

    @property
    def root(self) -> Path:
        return self._root

    def list_eval_runs(self) -> list[dict[str, Any]]:
        """Read the runs.jsonl index file, returning all eval run metadata."""
        index_path = self._root / "runs.jsonl"
        if not index_path.exists():
            return []

        records: list[dict[str, Any]] = []
        try:
            with open(index_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to read eval runs index: %s", e)
        return records

    def get_eval_run_metadata(self, run_id: str) -> dict[str, Any] | None:
        """Read metadata.json for a specific eval run."""
        path = self._root / "runs" / run_id / "metadata.json"
        if not path.exists():
            return None
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to read eval metadata %s: %s", path, e)
            return None

    def list_benchmarks(self, run_id: str) -> list[str]:
        """List benchmark names that have results for an eval run."""
        run_dir = self._root / "runs" / run_id
        if not run_dir.is_dir():
            return []
        benchmarks = []
        for child in sorted(run_dir.iterdir()):
            if child.is_dir() and (child / "result.json").exists():
                benchmarks.append(child.name)
        return benchmarks

    def get_benchmark_result(self, run_id: str, benchmark: str) -> dict[str, Any] | None:
        """Read result.json for a specific benchmark in an eval run."""
        path = self._root / "runs" / run_id / benchmark / "result.json"
        return self._read_json_cached(path)

    def get_benchmark_trajectories(
        self,
        run_id: str,
        benchmark: str,
    ) -> list[dict[str, Any]]:
        """Read all stored trajectories for a benchmark in an eval run."""
        path = self._root / "runs" / run_id / benchmark / "trajectories.jsonl"
        return self._read_jsonl_cached(path)

    def get_single_trajectory(
        self,
        run_id: str,
        benchmark: str,
        idx: int,
    ) -> dict[str, Any] | None:
        """Read a single trajectory by index."""
        trajectories = self.get_benchmark_trajectories(run_id, benchmark)
        for traj in trajectories:
            if traj.get("idx") == idx:
                return traj
        return None

    def get_summary(self, run_id: str) -> dict[str, Any] | None:
        """Read the cross-benchmark summary.json for an eval run."""
        path = self._root / "runs" / run_id / "summary.json"
        return self._read_json_cached(path)

    def get_scores_table(self) -> list[dict[str, Any]]:
        """Build a scores table: [{run_id, model_name, checkpoint, benchmark: score, ...}].

        Reads metadata from each run and collects scores from all benchmarks.
        """
        runs = self.list_eval_runs()
        table: list[dict[str, Any]] = []

        for run_entry in runs:
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

            # Add per-benchmark scores
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
        if not path.exists():
            return None
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to read %s: %s", path, e)
            return None

    @staticmethod
    @lru_cache(maxsize=32)
    def _read_jsonl_cached(path: Path) -> list[dict[str, Any]]:
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
            logger.warning("Failed to read %s: %s", path, e)
        return records
