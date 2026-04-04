"""RunStore — unified data access layer for Tinker Chef.

Orchestrates individual readers and provides a single interface
for the API routes to query run data.
"""

import logging
from pathlib import Path
from typing import Any

from tinker_cookbook.chef.data.config_reader import ConfigReader
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

logger = logging.getLogger(__name__)


class RunStore:
    """Manages data access for all discovered training runs.

    Lazily creates readers for each run on first access. Readers are
    cached for the lifetime of the store.
    """

    def __init__(self, root: Path) -> None:
        self._root = root.resolve()
        self._runs: dict[str, RunInfo] | None = None
        self._metrics_readers: dict[str, MetricsReader] = {}
        self._config_readers: dict[str, ConfigReader] = {}
        self._rollout_readers: dict[str, RolloutReader] = {}
        self._logtree_readers: dict[str, LogtreeReader] = {}
        self._timing_readers: dict[str, TimingReader] = {}

    @property
    def root(self) -> Path:
        return self._root

    def refresh_runs(self) -> list[RunInfo]:
        """Re-scan the root directory for runs."""
        runs = discover_runs(self._root)
        self._runs = {r.run_id: r for r in runs}
        return runs

    def get_runs(self) -> list[RunInfo]:
        """Return discovered runs, scanning on first call."""
        if self._runs is None:
            self.refresh_runs()
        assert self._runs is not None
        return list(self._runs.values())

    def get_run(self, run_id: str) -> RunInfo | None:
        """Look up a single run by ID."""
        if self._runs is None:
            self.refresh_runs()
        assert self._runs is not None
        return self._runs.get(run_id)

    def _run_path(self, run_id: str) -> Path | None:
        run = self.get_run(run_id)
        return run.path if run else None

    # --- Metrics ---

    def get_metrics_reader(self, run_id: str) -> MetricsReader | None:
        if run_id not in self._metrics_readers:
            path = self._run_path(run_id)
            if path is None:
                return None
            self._metrics_readers[run_id] = MetricsReader(path / "metrics.jsonl")
        return self._metrics_readers[run_id]

    def get_metrics(self, run_id: str) -> list[dict[str, Any]]:
        """Read all metrics for a run (incremental)."""
        reader = self.get_metrics_reader(run_id)
        if reader is None:
            return []
        reader.read()
        return reader.records

    def get_new_metrics(self, run_id: str) -> list[dict[str, Any]]:
        """Read only new metrics since last call."""
        reader = self.get_metrics_reader(run_id)
        if reader is None:
            return []
        return reader.read()

    # --- Config ---

    def get_config(self, run_id: str) -> dict[str, Any] | None:
        if run_id not in self._config_readers:
            path = self._run_path(run_id)
            if path is None:
                return None
            self._config_readers[run_id] = ConfigReader(path / "config.json")
        return self._config_readers[run_id].read()

    # --- Iterations ---

    def get_iterations(self, run_id: str) -> list[IterationInfo]:
        path = self._run_path(run_id)
        if path is None:
            return []
        return list_iterations(path)

    # --- Rollouts ---

    def _get_rollout_reader(self, run_id: str) -> RolloutReader | None:
        if run_id not in self._rollout_readers:
            path = self._run_path(run_id)
            if path is None:
                return None
            self._rollout_readers[run_id] = RolloutReader(path)
        return self._rollout_readers[run_id]

    def get_rollouts(
        self,
        run_id: str,
        iteration: int,
        split: str = "train",
        label: str | None = None,
    ) -> list[dict[str, Any]]:
        reader = self._get_rollout_reader(run_id)
        if reader is None:
            return []
        return reader.read_rollouts(iteration, split, label)

    def get_single_rollout(
        self,
        run_id: str,
        iteration: int,
        group_idx: int,
        traj_idx: int,
        split: str = "train",
        label: str | None = None,
    ) -> dict[str, Any] | None:
        reader = self._get_rollout_reader(run_id)
        if reader is None:
            return None
        return reader.read_single_rollout(iteration, group_idx, traj_idx, split, label)

    # --- Logtree ---

    def _get_logtree_reader(self, run_id: str) -> LogtreeReader | None:
        if run_id not in self._logtree_readers:
            path = self._run_path(run_id)
            if path is None:
                return None
            self._logtree_readers[run_id] = LogtreeReader(path)
        return self._logtree_readers[run_id]

    def get_logtree(
        self,
        run_id: str,
        iteration: int,
        base_name: str = "train",
    ) -> dict[str, Any] | None:
        reader = self._get_logtree_reader(run_id)
        if reader is None:
            return None
        return reader.read_logtree(iteration, base_name)

    # --- Timing ---

    def get_timing_reader(self, run_id: str) -> TimingReader | None:
        if run_id not in self._timing_readers:
            path = self._run_path(run_id)
            if path is None:
                return None
            self._timing_readers[run_id] = TimingReader(path / "timing_spans.jsonl")
        return self._timing_readers[run_id]

    def get_timing(self, run_id: str) -> list[dict[str, Any]]:
        reader = self.get_timing_reader(run_id)
        if reader is None:
            return []
        reader.read()
        return reader.records
