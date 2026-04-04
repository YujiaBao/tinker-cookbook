"""Reader for eval benchmark data from the eval-benchmark-framework."""

from typing import Any

from tinker_cookbook.storage import Storage, storage_join, storage_read_json, storage_read_jsonl


class EvalReader:
    """Reads eval benchmark data from an EvalStore directory."""

    def __init__(self, storage: Storage, prefix: str) -> None:
        self._storage = storage
        self._prefix = prefix

    def list_eval_runs(self) -> list[dict[str, Any]]:
        return storage_read_jsonl(self._storage, storage_join(self._prefix, "runs.jsonl"))

    def get_eval_run_metadata(self, run_id: str) -> dict[str, Any] | None:
        return storage_read_json(self._storage, storage_join(self._prefix, "runs", run_id, "metadata.json"))

    def list_benchmarks(self, run_id: str) -> list[str]:
        items = self._storage.list_dir(storage_join(self._prefix, "runs", run_id))
        return sorted(
            name for name in items
            if self._storage.exists(storage_join(self._prefix, "runs", run_id, name, "result.json"))
        )

    def get_benchmark_result(self, run_id: str, benchmark: str) -> dict[str, Any] | None:
        return storage_read_json(
            self._storage, storage_join(self._prefix, "runs", run_id, benchmark, "result.json")
        )

    def get_benchmark_trajectories(self, run_id: str, benchmark: str) -> list[dict[str, Any]]:
        return storage_read_jsonl(
            self._storage, storage_join(self._prefix, "runs", run_id, benchmark, "trajectories.jsonl")
        )

    def get_single_trajectory(self, run_id: str, benchmark: str, idx: int) -> dict[str, Any] | None:
        for traj in self.get_benchmark_trajectories(run_id, benchmark):
            if traj.get("idx") == idx:
                return traj
        return None

    def get_summary(self, run_id: str) -> dict[str, Any] | None:
        return storage_read_json(self._storage, storage_join(self._prefix, "runs", run_id, "summary.json"))

    def get_scores_table(self) -> list[dict[str, Any]]:
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
