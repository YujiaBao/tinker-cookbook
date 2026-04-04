"""Discover training runs by scanning directories for metrics.jsonl files."""

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Literal

from tinker_cookbook.storage import Storage, storage_join, storage_read_json, storage_read_jsonl

logger = logging.getLogger(__name__)

_ITERATION_DIR_RE = re.compile(r"^iteration_(\d+)$")
_ACTIVE_THRESHOLD_SECONDS = 120

Status = Literal["running", "completed", "idle"]
TrainingType = Literal["rl", "sl", "dpo"]


@dataclass(frozen=True)
class RunInfo:
    """Metadata about a discovered training run."""

    run_id: str
    prefix: str  # storage-relative path to the run directory
    has_config: bool
    has_metrics: bool
    has_checkpoints: bool
    has_timing: bool
    iteration_count: int
    status: Status
    last_updated: float | None
    training_type: TrainingType | None


@dataclass
class IterationInfo:
    """Metadata about a single training iteration directory."""

    iteration: int
    has_train_rollouts: bool = False
    has_train_logtree: bool = False
    eval_labels: list[str] = field(default_factory=list)


def discover_runs(storage: Storage, root_prefix: str = "") -> list[RunInfo]:
    """Scan storage for directories containing metrics.jsonl or config.json."""
    runs: list[RunInfo] = []

    # Check if root itself is a run
    if _is_run_dir(storage, root_prefix):
        name = root_prefix.rstrip("/").rsplit("/", 1)[-1] if root_prefix else "root"
        runs.append(_build_run_info(storage, name, root_prefix))
        return runs

    # Scan immediate subdirectories
    for child in sorted(storage.list_dir(root_prefix)):
        child_prefix = storage_join(root_prefix, child) if root_prefix else child
        if _is_run_dir(storage, child_prefix):
            runs.append(_build_run_info(storage, child, child_prefix))

    return runs


def list_iterations(storage: Storage, run_prefix: str) -> list[IterationInfo]:
    """List iteration directories within a run, sorted by iteration number."""
    iterations: list[IterationInfo] = []

    for child in storage.list_dir(run_prefix):
        match = _ITERATION_DIR_RE.match(child)
        if not match:
            continue

        iteration_num = int(match.group(1))
        info = IterationInfo(iteration=iteration_num)
        iter_prefix = storage_join(run_prefix, child)

        for f in storage.list_dir(iter_prefix):
            if f == "train_rollout_summaries.jsonl":
                info.has_train_rollouts = True
            elif f == "train_logtree.json":
                info.has_train_logtree = True
            elif f.startswith("eval_") and f.endswith("_rollout_summaries.jsonl"):
                label = f[len("eval_") : -len("_rollout_summaries.jsonl")]
                info.eval_labels.append(label)

        iterations.append(info)

    iterations.sort(key=lambda x: x.iteration)
    return iterations


def _is_run_dir(storage: Storage, prefix: str) -> bool:
    metrics_path = storage_join(prefix, "metrics.jsonl")
    config_path = storage_join(prefix, "config.json")
    return storage.exists(metrics_path) or storage.exists(config_path)


def _detect_status(storage: Storage, prefix: str) -> tuple[Status, float | None]:
    metrics_path = storage_join(prefix, "metrics.jsonl")
    stat = storage.stat(metrics_path)
    if stat is None:
        return "idle", None

    age = time.time() - stat.mtime
    if age < _ACTIVE_THRESHOLD_SECONDS:
        return "running", stat.mtime

    ckpt_path = storage_join(prefix, "checkpoints.jsonl")
    for ckpt in reversed(storage_read_jsonl(storage, ckpt_path)):
        if ckpt.get("final"):
            return "completed", stat.mtime

    return "idle", stat.mtime


def _infer_training_type(storage: Storage, prefix: str) -> TrainingType | None:
    config_path = storage_join(prefix, "config.json")
    config = storage_read_json(storage, config_path)
    if config is None:
        return None

    if "dpo_beta" in config:
        return "dpo"
    if "loss_fn" in config:
        return "rl"
    if "num_epochs" in config:
        return "sl"

    dataset_builder = config.get("dataset_builder")
    if isinstance(dataset_builder, dict):
        db_type = dataset_builder.get("__type__", "")
        if "RL" in db_type:
            return "rl"
        if "Supervised" in db_type or "SL" in db_type:
            return "sl"

    return None


def _build_run_info(storage: Storage, run_id: str, prefix: str) -> RunInfo:
    iteration_count = sum(
        1 for child in storage.list_dir(prefix) if _ITERATION_DIR_RE.match(child)
    )
    status, last_updated = _detect_status(storage, prefix)
    training_type = _infer_training_type(storage, prefix)

    return RunInfo(
        run_id=run_id,
        prefix=prefix,
        has_config=storage.exists(storage_join(prefix, "config.json")),
        has_metrics=storage.exists(storage_join(prefix, "metrics.jsonl")),
        has_checkpoints=storage.exists(storage_join(prefix, "checkpoints.jsonl")),
        has_timing=storage.exists(storage_join(prefix, "timing_spans.jsonl")),
        iteration_count=iteration_count,
        status=status,
        last_updated=last_updated,
        training_type=training_type,
    )
