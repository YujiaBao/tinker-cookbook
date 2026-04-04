"""Discover training runs by scanning directories for metrics.jsonl files."""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Matches iteration directory names like "iteration_000050"
_ITERATION_DIR_RE = re.compile(r"^iteration_(\d+)$")


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


@dataclass
class IterationInfo:
    """Metadata about a single training iteration directory."""

    iteration: int
    path: Path
    has_train_rollouts: bool = False
    has_train_logtree: bool = False
    eval_labels: list[str] = field(default_factory=list)


def discover_runs(root: Path) -> list[RunInfo]:
    """Scan *root* for directories containing metrics.jsonl or config.json.

    If *root* itself is a run directory (contains metrics.jsonl), it is
    returned as a single-element list with run_id derived from the
    directory name.

    Otherwise, immediate subdirectories of *root* are scanned.
    """
    if not root.is_dir():
        return []

    runs: list[RunInfo] = []

    # Check if root itself is a run
    if _is_run_dir(root):
        runs.append(_build_run_info(root.name, root))
        return runs

    # Scan immediate subdirectories
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

        # Check for rollout summaries
        for f in child.iterdir():
            name = f.name
            if name == "train_rollout_summaries.jsonl":
                info.has_train_rollouts = True
            elif name == "train_logtree.json":
                info.has_train_logtree = True
            elif name.startswith("eval_") and name.endswith("_rollout_summaries.jsonl"):
                # Extract label from "eval_LABEL_rollout_summaries.jsonl"
                label = name[len("eval_") : -len("_rollout_summaries.jsonl")]
                info.eval_labels.append(label)

        iterations.append(info)

    iterations.sort(key=lambda x: x.iteration)
    return iterations


def _is_run_dir(path: Path) -> bool:
    """True if the directory looks like a training run output."""
    return (path / "metrics.jsonl").exists() or (path / "config.json").exists()


def _build_run_info(run_id: str, path: Path) -> RunInfo:
    """Build RunInfo by checking which files exist."""
    iteration_count = sum(
        1 for child in path.iterdir() if child.is_dir() and _ITERATION_DIR_RE.match(child.name)
    )

    return RunInfo(
        run_id=run_id,
        path=path,
        has_config=(path / "config.json").exists(),
        has_metrics=(path / "metrics.jsonl").exists(),
        has_checkpoints=(path / "checkpoints.jsonl").exists(),
        has_timing=(path / "timing_spans.jsonl").exists(),
        iteration_count=iteration_count,
    )
