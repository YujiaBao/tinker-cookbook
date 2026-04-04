"""RunStore — unified data access layer for Tinker Chef.

Orchestrates individual readers and provides a single interface
for the API routes to query run data. All file access goes through
a ``Storage`` instance.
"""

import logging
from typing import Any

from tinker_cookbook.chef.data.config_reader import ConfigReader
from tinker_cookbook.chef.data.eval_reader import EvalReader
from tinker_cookbook.chef.data.logtree_reader import LogtreeReader
from tinker_cookbook.chef.data.metrics_reader import MetricsReader
from tinker_cookbook.chef.data.rollout_reader import RolloutReader
from tinker_cookbook.chef.data.run_discovery import (
    IterationInfo,
    RunInfo,
    discover_runs,
    list_iterations,
)
from tinker_cookbook.chef.data.timing_reader import TimingReader
from tinker_cookbook.storage import Storage, storage_join, storage_read_jsonl

logger = logging.getLogger(__name__)


class RunStore:
    """Manages data access for all discovered training runs.

    Supports multiple storage backends (e.g., multiple local directories).
    Each run is tracked with its source storage so readers access the right one.
    """

    def __init__(self, storages: Storage | list[Storage]) -> None:
        if isinstance(storages, list):
            self._storages = storages
        else:
            self._storages = [storages]
        self._runs: dict[str, RunInfo] | None = None
        self._run_storage: dict[str, Storage] = {}  # run_id -> its storage
        self._metrics_readers: dict[str, MetricsReader] = {}
        self._config_readers: dict[str, ConfigReader] = {}
        self._rollout_readers: dict[str, RolloutReader] = {}
        self._logtree_readers: dict[str, LogtreeReader] = {}
        self._timing_readers: dict[str, TimingReader] = {}
        self._eval_readers: dict[str, EvalReader | object] = {}

    @property
    def storage(self) -> Storage:
        """Primary storage (first in list). Used for global operations."""
        return self._storages[0]

    def storage_for(self, run_id: str) -> Storage:
        """Get the storage that contains a specific run."""
        return self._run_storage.get(run_id) or self._storages[0]

    def _storage_for(self, run_id: str) -> Storage | None:
        return self._run_storage.get(run_id)

    def refresh_runs(self) -> list[RunInfo]:
        all_runs: dict[str, RunInfo] = {}
        self._run_storage.clear()
        for storage in self._storages:
            source = ""
            if hasattr(storage, "root"):
                source = getattr(storage, "root").name
            for run in discover_runs(storage):
                # Deduplicate by run_id; first storage wins
                uid = f"{source}--{run.run_id}" if run.run_id in all_runs else run.run_id
                all_runs[uid] = run
                self._run_storage[uid] = storage
        self._runs = all_runs
        return list(all_runs.values())

    def get_runs(self) -> list[RunInfo]:
        if self._runs is None:
            self.refresh_runs()
        assert self._runs is not None
        return list(self._runs.values())

    def get_run(self, run_id: str) -> RunInfo | None:
        if self._runs is None:
            self.refresh_runs()
        assert self._runs is not None
        return self._runs.get(run_id)

    def _run_prefix(self, run_id: str) -> str | None:
        run = self.get_run(run_id)
        return run.prefix if run else None

    # --- Metrics ---

    def get_metrics_reader(self, run_id: str) -> MetricsReader | None:
        if run_id not in self._metrics_readers:
            prefix = self._run_prefix(run_id)
            if prefix is None:
                return None
            path = storage_join(prefix, "metrics.jsonl")
            storage = self._storage_for(run_id) or self.storage
            self._metrics_readers[run_id] = MetricsReader(storage, path)
        return self._metrics_readers[run_id]

    def get_metrics(self, run_id: str) -> list[dict[str, Any]]:
        reader = self.get_metrics_reader(run_id)
        if reader is None:
            return []
        reader.read()
        return reader.records

    def get_new_metrics(self, run_id: str) -> list[dict[str, Any]]:
        reader = self.get_metrics_reader(run_id)
        if reader is None:
            return []
        return reader.read()

    # --- Config ---

    def get_config(self, run_id: str) -> dict[str, Any] | None:
        if run_id not in self._config_readers:
            prefix = self._run_prefix(run_id)
            if prefix is None:
                return None
            path = storage_join(prefix, "config.json")
            storage = self._storage_for(run_id) or self.storage
            self._config_readers[run_id] = ConfigReader(storage, path)
        return self._config_readers[run_id].read()

    # --- Iterations ---

    def get_iterations(self, run_id: str) -> list[IterationInfo]:
        prefix = self._run_prefix(run_id)
        if prefix is None:
            return []
        storage = self._storage_for(run_id) or self.storage
        return list_iterations(storage, prefix)

    # --- Rollouts ---

    def _get_rollout_reader(self, run_id: str) -> RolloutReader | None:
        if run_id not in self._rollout_readers:
            prefix = self._run_prefix(run_id)
            if prefix is None:
                return None
            storage = self._storage_for(run_id) or self.storage
            self._rollout_readers[run_id] = RolloutReader(storage, prefix)
        return self._rollout_readers[run_id]

    def get_rollouts(self, run_id: str, iteration: int, split: str = "train",
                     label: str | None = None) -> list[dict[str, Any]]:
        reader = self._get_rollout_reader(run_id)
        return reader.read_rollouts(iteration, split, label) if reader else []

    def get_single_rollout(self, run_id: str, iteration: int, group_idx: int,
                           traj_idx: int, split: str = "train",
                           label: str | None = None) -> dict[str, Any] | None:
        reader = self._get_rollout_reader(run_id)
        return reader.read_single_rollout(iteration, group_idx, traj_idx, split, label) if reader else None

    # --- Logtree ---

    def _get_logtree_reader(self, run_id: str) -> LogtreeReader | None:
        if run_id not in self._logtree_readers:
            prefix = self._run_prefix(run_id)
            if prefix is None:
                return None
            storage = self._storage_for(run_id) or self.storage
            self._logtree_readers[run_id] = LogtreeReader(storage, prefix)
        return self._logtree_readers[run_id]

    def get_logtree(self, run_id: str, iteration: int, base_name: str = "train") -> dict[str, Any] | None:
        reader = self._get_logtree_reader(run_id)
        return reader.read_logtree(iteration, base_name) if reader else None

    # --- Timing ---

    def get_timing_reader(self, run_id: str) -> TimingReader | None:
        if run_id not in self._timing_readers:
            prefix = self._run_prefix(run_id)
            if prefix is None:
                return None
            path = storage_join(prefix, "timing_spans.jsonl")
            storage = self._storage_for(run_id) or self.storage
            self._timing_readers[run_id] = TimingReader(storage, path)
        return self._timing_readers[run_id]

    def get_timing(self, run_id: str) -> list[dict[str, Any]]:
        reader = self.get_timing_reader(run_id)
        if reader is None:
            return []
        reader.read()
        return reader.records

    # --- Eval Benchmarks ---

    _EVAL_MISS = object()

    def get_eval_reader(self, run_id: str) -> EvalReader | None:
        cached = self._eval_readers.get(run_id)
        if cached is self._EVAL_MISS:
            return None
        if isinstance(cached, EvalReader):
            return cached

        prefix = self._run_prefix(run_id)
        if prefix is None:
            self._eval_readers[run_id] = self._EVAL_MISS
            return None

        for suffix in ["eval", "eval_store"]:
            candidate = storage_join(prefix, suffix)
            runs_jsonl = storage_join(candidate, "runs.jsonl")
            runs_dir = storage_join(candidate, "runs")
            storage = self._storage_for(run_id) or self.storage
            if storage.exists(runs_jsonl) or (
                storage.exists(runs_dir) and len(storage.list_dir(runs_dir)) > 0
            ):
                reader = EvalReader(storage, candidate)
                self._eval_readers[run_id] = reader
                return reader

        self._eval_readers[run_id] = self._EVAL_MISS
        return None

    def get_global_eval_reader(self) -> EvalReader | None:
        cached = self._eval_readers.get("__global__")
        if cached is self._EVAL_MISS:
            return None
        if isinstance(cached, EvalReader):
            return cached

        for storage in self._storages:
            for prefix in ["eval", "eval_store", ""]:
                runs_path = storage_join(prefix, "runs.jsonl")
                if storage.exists(runs_path):
                    reader = EvalReader(storage, prefix)
                    self._eval_readers["__global__"] = reader
                    return reader

        self._eval_readers["__global__"] = self._EVAL_MISS
        return None
