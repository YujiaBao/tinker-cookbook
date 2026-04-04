"""Discover training runs by scanning directories for metrics.jsonl files."""

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from tinker_cookbook.chef.data.io import read_json, read_jsonl

logger = logging.getLogger(__name__)

_ITERATION_DIR_RE = re.compile(r"^iteration_(\d+)$")

# A run is considered "running" if its metrics file was updated within this window
_ACTIVE_THRESHOLD_SECONDS = 120


@dataclass(frozen=True)
class RunInfo:
    """Metadata about a discovered training run."""

    run_id: str
    path: Path
    has_config: bool
    has_metrics: bool
    has_checkpoints: bool
    has_timing: bool
    iteration_count: int
    status: str  # "running" | "completed" | "idle"
    last_updated: float | None  # mtime of metrics.jsonl (epoch seconds)
    training_type: str | None  # "rl" | "sl" | "dpo" | None


@dataclass
class IterationInfo:
    """Metadata about a single training iteration directory."""

    iteration: int
    path: Path
    has_train_rollouts: bool = False
    has_train_logtree: bool = False
    eval_labels: list[str] = field(default_factory=list)


def discover_runs(root: Path) -> list[RunInfo]:
    """Scan *root* for directories containing metrics.jsonl or config.json."""
    if not root.is_dir():
        return []

    runs: list[RunInfo] = []

    if _is_run_dir(root):
        runs.append(_build_run_info(root.name, root))
        return runs

    for child in sorted(root.iterdir()):
        if child.is_dir() and _is_run_dir(child):
            runs.append(_build_run_info(child.name, child))

    return runs


def list_iterations(run_path: Path) -> list[IterationInfo]:
    """List iteration directories within a run, sorted by iteration number."""
    iterations: list[IterationInfo] = []

    if not run_path.is_dir():
        return iterations

    for child in run_path.iterdir():
        if not child.is_dir():
            continue
        match = _ITERATION_DIR_RE.match(child.name)
        if not match:
            continue

        iteration_num = int(match.group(1))
        info = IterationInfo(iteration=iteration_num, path=child)

        for f in child.iterdir():
            name = f.name
            if name == "train_rollout_summaries.jsonl":
                info.has_train_rollouts = True
            elif name == "train_logtree.json":
                info.has_train_logtree = True
            elif name.startswith("eval_") and name.endswith("_rollout_summaries.jsonl"):
                label = name[len("eval_") : -len("_rollout_summaries.jsonl")]
                info.eval_labels.append(label)

        iterations.append(info)

    iterations.sort(key=lambda x: x.iteration)
    return iterations


def _is_run_dir(path: Path) -> bool:
    return (path / "metrics.jsonl").exists() or (path / "config.json").exists()


def _detect_status(path: Path) -> tuple[str, float | None]:
    """Detect run status from file system state."""
    metrics_path = path / "metrics.jsonl"
    last_updated: float | None = None

    try:
        last_updated = metrics_path.stat().st_mtime
    except FileNotFoundError:
        return "idle", None

    # Check if metrics file was recently updated
    age = time.time() - last_updated
    if age < _ACTIVE_THRESHOLD_SECONDS:
        return "running", last_updated

    # Check for final checkpoint
    checkpoints = read_jsonl(path / "checkpoints.jsonl")
    for ckpt in reversed(checkpoints):
        if ckpt.get("final"):
            return "completed", last_updated

    # Old metrics but no final checkpoint — might have been interrupted
    return "idle", last_updated


def _infer_training_type(path: Path) -> str | None:
    """Infer training type (rl/sl/dpo) from config.json fields."""
    config = read_json(path / "config.json")
    if config is None:
        return None

    # DPO has dpo_beta
    if "dpo_beta" in config:
        return "dpo"

    # RL has loss_fn (importance_sampling, etc.) and no num_epochs
    if "loss_fn" in config:
        return "rl"

    # SL has num_epochs
    if "num_epochs" in config:
        return "sl"

    # Check for nested config patterns
    if "dataset_builder" in config:
        db = config["dataset_builder"]
        if isinstance(db, dict):
            db_type = db.get("__type__", "")
            if "RL" in db_type:
                return "rl"
            if "Supervised" in db_type or "SL" in db_type:
                return "sl"

    return None


def _build_run_info(run_id: str, path: Path) -> RunInfo:
    """Build RunInfo by checking which files exist."""
    iteration_count = sum(
        1 for child in path.iterdir() if child.is_dir() and _ITERATION_DIR_RE.match(child.name)
    )

    status, last_updated = _detect_status(path)
    training_type = _infer_training_type(path)

    return RunInfo(
        run_id=run_id,
        path=path,
        has_config=(path / "config.json").exists(),
        has_metrics=(path / "metrics.jsonl").exists(),
        has_checkpoints=(path / "checkpoints.jsonl").exists(),
        has_timing=(path / "timing_spans.jsonl").exists(),
        iteration_count=iteration_count,
        status=status,
        last_updated=last_updated,
        training_type=training_type,
    )
