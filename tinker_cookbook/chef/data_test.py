"""Tests for Tinker Chef data readers."""

import json
import os
import tempfile
from pathlib import Path

import pytest

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
from tinker_cookbook.chef.data.store import RunStore
from tinker_cookbook.storage import LocalStorage
from tinker_cookbook.chef.data.timing_reader import TimingReader


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def tmp_run(tmp_path: Path) -> Path:
    """Create a minimal training run directory with fixture data."""
    run_dir = tmp_path / "my_run"
    run_dir.mkdir()

    # config.json
    config = {
        "model_name": "Llama-3.1-8B",
        "learning_rate": 1e-4,
        "batch_size": 32,
        "n_batches": 100,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    # metrics.jsonl
    metrics_lines = []
    for step in range(5):
        metrics_lines.append(
            json.dumps({
                "step": step,
                "train_mean_nll": 2.5 - step * 0.1,
                "env/all/reward/total": step * 0.2,
                "time/forward_backward": 1.0 + step * 0.1,
            })
        )
    (run_dir / "metrics.jsonl").write_text("\n".join(metrics_lines) + "\n")

    # checkpoints.jsonl
    ckpt = {
        "state_path": "tinker:///ckpt/000002",
        "name": "000002",
        "kind": "both",
        "timestamp": 1700000000.0,
        "loop_state": {"epoch": 0, "batch": 2},
    }
    (run_dir / "checkpoints.jsonl").write_text(json.dumps(ckpt) + "\n")

    # timing_spans.jsonl (nested format from trace.py write_spans_jsonl)
    timing_lines = [
        {
            "step": 0,
            "spans": [
                {"name": "forward_backward", "duration": 1.5, "wall_start": 0.0, "wall_end": 1.5},
                {"name": "optim_step", "duration": 0.5, "wall_start": 1.5, "wall_end": 2.0},
                # policy_sample overlaps with env_step (concurrent)
                {"name": "policy_sample", "duration": 0.8, "wall_start": 0.0, "wall_end": 0.8},
                {"name": "env_step", "duration": 0.6, "wall_start": 0.2, "wall_end": 0.8},
            ],
        },
        {
            "step": 1,
            "spans": [
                {"name": "forward_backward", "duration": 1.3, "wall_start": 2.0, "wall_end": 3.3},
            ],
        },
    ]
    (run_dir / "timing_spans.jsonl").write_text(
        "\n".join(json.dumps(s) for s in timing_lines) + "\n"
    )

    # iteration directories with rollout summaries
    for iteration in [0, 2, 4]:
        iter_dir = run_dir / f"iteration_{iteration:06d}"
        iter_dir.mkdir()

        rollouts = []
        for group_idx in range(2):
            for traj_idx in range(3):
                rollouts.append({
                    "schema_version": 1,
                    "split": "train",
                    "iteration": iteration,
                    "group_idx": group_idx,
                    "traj_idx": traj_idx,
                    "tags": ["math", "gsm8k"] if group_idx == 0 else ["code", "humaneval"],
                    "sampling_client_step": iteration,
                    "total_reward": (group_idx + traj_idx) * 0.3,
                    "final_reward": (group_idx + traj_idx) * 0.1,
                    "trajectory_metrics": {"custom": 1.5},
                    "steps": [
                        {
                            "step_idx": 0,
                            "ob_len": 128,
                            "ac_len": 45,
                            "reward": 0.0,
                            "episode_done": False,
                            "metrics": {"step_time": 0.01},
                            "logs": {},
                        },
                        {
                            "step_idx": 1,
                            "ob_len": 200,
                            "ac_len": 60,
                            "reward": 1.0,
                            "episode_done": True,
                            "metrics": {"step_time": 0.02},
                            "logs": {},
                        },
                    ],
                    "final_ob_len": 512,
                })
        (iter_dir / "train_rollout_summaries.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rollouts) + "\n"
        )

        # logtree JSON
        logtree = {
            "title": f"RL Iteration {iteration}",
            "started_at": "2024-04-04T12:34:56.789123",
            "root": {
                "tag": "div",
                "attrs": {"class": "lt-root"},
                "children": [
                    {"tag": "p", "children": ["Episode completed"]},
                ],
                "data": {"type": "conversation", "messages": [
                    {"role": "user", "content": "What is 2+2?"},
                    {"role": "assistant", "content": "4"},
                ]},
            },
        }
        (iter_dir / "train_logtree.json").write_text(json.dumps(logtree))

    # Add an eval rollout to iteration 4
    iter4 = run_dir / "iteration_000004"
    eval_rollouts = [{
        "schema_version": 1,
        "split": "eval",
        "iteration": 4,
        "group_idx": 0,
        "traj_idx": 0,
        "tags": ["test_set"],
        "total_reward": 0.9,
        "final_reward": 0.9,
        "steps": [{"step_idx": 0, "ob_len": 100, "ac_len": 30, "reward": 0.9,
                    "episode_done": True, "metrics": {}, "logs": {}}],
        "final_ob_len": 200,
    }]
    (iter4 / "eval_test_rollout_summaries.jsonl").write_text(
        json.dumps(eval_rollouts[0]) + "\n"
    )

    return run_dir


@pytest.fixture
def tmp_multi_run(tmp_path: Path, tmp_run: Path) -> Path:
    """Create a parent directory containing multiple runs."""
    parent = tmp_path / "experiments"
    parent.mkdir()
    # Move the run into the parent
    target = parent / "run_001"
    os.rename(tmp_run, target)

    # Create a second minimal run
    run2 = parent / "run_002"
    run2.mkdir()
    (run2 / "metrics.jsonl").write_text(
        json.dumps({"step": 0, "loss": 3.0}) + "\n"
    )
    return parent


# ── MetricsReader tests ───────────────────────────────────────────────


class TestMetricsReader:
    def test_read_all(self, tmp_run: Path) -> None:
        reader = MetricsReader(LocalStorage(tmp_run), "metrics.jsonl")
        new = reader.read()
        assert len(new) == 5
        assert reader.records == new
        assert new[0]["step"] == 0
        assert new[4]["step"] == 4

    def test_incremental_read(self, tmp_run: Path) -> None:
        metrics_path = tmp_run / "metrics.jsonl"
        reader = MetricsReader(LocalStorage(metrics_path.parent), metrics_path.name)

        # First read
        reader.read()
        assert len(reader.records) == 5

        # Append new data
        with open(metrics_path, "a") as f:
            f.write(json.dumps({"step": 5, "loss": 1.0}) + "\n")

        # Incremental read should return only the new record
        new = reader.read()
        assert len(new) == 1
        assert new[0]["step"] == 5
        assert len(reader.records) == 6

    def test_no_new_data(self, tmp_run: Path) -> None:
        reader = MetricsReader(LocalStorage(tmp_run), "metrics.jsonl")
        reader.read()
        new = reader.read()
        assert new == []

    def test_nonexistent_file(self, tmp_path: Path) -> None:
        reader = MetricsReader(LocalStorage(tmp_path), "nonexistent.jsonl")
        assert reader.read() == []
        assert not reader.has_data()

    def test_partial_write_ignored(self, tmp_run: Path) -> None:
        metrics_path = tmp_run / "metrics.jsonl"
        reader = MetricsReader(LocalStorage(metrics_path.parent), metrics_path.name)
        reader.read()

        # Append a partial line (no trailing newline)
        with open(metrics_path, "a") as f:
            f.write('{"step": 99, "partial": tru')

        new = reader.read()
        assert new == []

    def test_metric_keys(self, tmp_run: Path) -> None:
        reader = MetricsReader(LocalStorage(tmp_run), "metrics.jsonl")
        reader.read()
        keys = reader.metric_keys()
        assert "train_mean_nll" in keys
        assert "env/all/reward/total" in keys
        assert "time/forward_backward" in keys
        assert "step" not in keys

    def test_malformed_line_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "metrics.jsonl"
        path.write_text('{"step": 0, "ok": 1}\nnot json\n{"step": 1, "ok": 2}\n')
        reader = MetricsReader(LocalStorage(path.parent), path.name)
        records = reader.read()
        assert len(records) == 2
        assert records[0]["step"] == 0
        assert records[1]["step"] == 1


# ── ConfigReader tests ────────────────────────────────────────────────


class TestConfigReader:
    def test_read_config(self, tmp_run: Path) -> None:
        reader = ConfigReader(LocalStorage(tmp_run), "config.json")
        config = reader.read()
        assert config is not None
        assert config["model_name"] == "Llama-3.1-8B"
        assert config["learning_rate"] == 1e-4

    def test_cached(self, tmp_run: Path) -> None:
        reader = ConfigReader(LocalStorage(tmp_run), "config.json")
        c1 = reader.read()
        c2 = reader.read()
        assert c1 is c2  # Same object, not re-read

    def test_nonexistent(self, tmp_path: Path) -> None:
        reader = ConfigReader(LocalStorage(tmp_path), "nope.json")
        assert reader.read() is None


# ── RunDiscovery tests ────────────────────────────────────────────────


class TestRunDiscovery:
    def test_discover_single_run(self, tmp_run: Path) -> None:
        runs = discover_runs(LocalStorage(tmp_run))
        assert len(runs) == 1
        # When root itself is a run, run_id is derived from directory name or "root"
        assert runs[0].run_id in (tmp_run.name, "root")
        assert runs[0].has_config is True
        assert runs[0].has_metrics is True
        assert runs[0].has_checkpoints is True
        assert runs[0].has_timing is True
        assert runs[0].iteration_count == 3

    def test_discover_multi_run(self, tmp_multi_run: Path) -> None:
        runs = discover_runs(LocalStorage(tmp_multi_run))
        assert len(runs) == 2
        ids = {r.run_id for r in runs}
        assert "run_001" in ids
        assert "run_002" in ids

    def test_discover_empty_dir(self, tmp_path: Path) -> None:
        runs = discover_runs(LocalStorage(tmp_path))
        assert runs == []

    def test_discover_nonexistent(self, tmp_path: Path) -> None:
        nope = tmp_path / "nope"
        nope.mkdir()  # LocalStorage needs an existing dir
        runs = discover_runs(LocalStorage(nope))
        assert runs == []

    def test_list_iterations(self, tmp_run: Path) -> None:
        iterations = list_iterations(LocalStorage(tmp_run), "")
        assert len(iterations) == 3
        assert iterations[0].iteration == 0
        assert iterations[1].iteration == 2
        assert iterations[2].iteration == 4

        # Check iteration 0 has train rollouts
        assert iterations[0].has_train_rollouts is True
        assert iterations[0].has_train_logtree is True

        # Check iteration 4 has eval rollouts
        assert "test" in iterations[2].eval_labels


# ── RolloutReader tests ───────────────────────────────────────────────


class TestRolloutReader:
    def test_read_train_rollouts(self, tmp_run: Path) -> None:
        reader = RolloutReader(LocalStorage(tmp_run), "")
        rollouts = reader.read_rollouts(0, "train")
        assert len(rollouts) == 6  # 2 groups * 3 trajectories
        assert rollouts[0]["schema_version"] == 1
        assert rollouts[0]["group_idx"] == 0
        assert rollouts[0]["traj_idx"] == 0
        assert len(rollouts[0]["steps"]) == 2

    def test_read_eval_rollouts(self, tmp_run: Path) -> None:
        reader = RolloutReader(LocalStorage(tmp_run), "")
        rollouts = reader.read_rollouts(4, "eval", label="test")
        assert len(rollouts) == 1
        assert rollouts[0]["total_reward"] == 0.9

    def test_read_single_rollout(self, tmp_run: Path) -> None:
        reader = RolloutReader(LocalStorage(tmp_run), "")
        rollout = reader.read_single_rollout(0, group_idx=1, traj_idx=2)
        assert rollout is not None
        assert rollout["group_idx"] == 1
        assert rollout["traj_idx"] == 2
        assert rollout["tags"] == ["code", "humaneval"]

    def test_nonexistent_iteration(self, tmp_run: Path) -> None:
        reader = RolloutReader(LocalStorage(tmp_run), "")
        assert reader.read_rollouts(999) == []

    def test_nonexistent_rollout(self, tmp_run: Path) -> None:
        reader = RolloutReader(LocalStorage(tmp_run), "")
        assert reader.read_single_rollout(0, group_idx=99, traj_idx=99) is None


# ── LogtreeReader tests ───────────────────────────────────────────────


class TestLogtreeReader:
    def test_read_logtree(self, tmp_run: Path) -> None:
        reader = LogtreeReader(LocalStorage(tmp_run), "")
        tree = reader.read_logtree(0)
        assert tree is not None
        assert tree["title"] == "RL Iteration 0"
        assert "root" in tree

    def test_list_logtrees(self, tmp_run: Path) -> None:
        reader = LogtreeReader(LocalStorage(tmp_run), "")
        names = reader.list_logtrees(0)
        assert "train" in names

    def test_nonexistent(self, tmp_run: Path) -> None:
        reader = LogtreeReader(LocalStorage(tmp_run), "")
        assert reader.read_logtree(999) is None


# ── TimingReader tests ────────────────────────────────────────────────


class TestTimingReader:
    def test_read_all(self, tmp_run: Path) -> None:
        reader = TimingReader(LocalStorage(tmp_run), "timing_spans.jsonl")
        new = reader.read()
        assert len(new) == 2  # Two step records
        assert new[0]["step"] == 0
        assert len(new[0]["spans"]) == 4  # 4 spans in step 0

    def test_get_spans_for_step(self, tmp_run: Path) -> None:
        reader = TimingReader(LocalStorage(tmp_run), "timing_spans.jsonl")
        reader.read()
        step0_spans = reader.get_spans_for_step(0)
        assert len(step0_spans) == 4
        names = {s["name"] for s in step0_spans}
        assert names == {"forward_backward", "optim_step", "policy_sample", "env_step"}

    def test_get_all_spans_flat(self, tmp_run: Path) -> None:
        reader = TimingReader(LocalStorage(tmp_run), "timing_spans.jsonl")
        reader.read()
        flat = reader.get_all_spans_flat()
        assert len(flat) == 5  # 4 spans in step 0 + 1 in step 1
        # All should have step annotation
        steps = {s["step"] for s in flat}
        assert steps == {0, 1}

    def test_concurrency_analysis(self, tmp_run: Path) -> None:
        reader = TimingReader(LocalStorage(tmp_run), "timing_spans.jsonl")
        reader.read()
        analysis = reader.get_concurrency_analysis(0)
        # policy_sample (0.0-0.8) overlaps with env_step (0.2-0.8) and forward_backward (0.0-1.5)
        assert analysis["max_concurrency"] >= 3
        assert len(analysis["spans"]) == 4
        assert len(analysis["timeline"]) > 0

    def test_incremental(self, tmp_run: Path) -> None:
        path = tmp_run / "timing_spans.jsonl"
        reader = TimingReader(LocalStorage(path.parent), path.name)
        reader.read()

        with open(path, "a") as f:
            f.write(json.dumps({"step": 2, "spans": [
                {"name": "new_span", "duration": 1.0, "wall_start": 5.0, "wall_end": 6.0}
            ]}) + "\n")

        new = reader.read()
        assert len(new) == 1
        assert new[0]["step"] == 2
        assert len(reader.records) == 3


# ── RunStore integration tests ────────────────────────────────────────


class TestRunStore:
    def test_discover_and_get_runs(self, tmp_run: Path) -> None:
        store = RunStore(LocalStorage(tmp_run))
        runs = store.get_runs()
        assert len(runs) == 1

    def test_get_metrics(self, tmp_run: Path) -> None:
        store = RunStore(LocalStorage(tmp_run))
        run_id = store.get_runs()[0].run_id
        metrics = store.get_metrics(run_id)
        assert len(metrics) == 5

    def test_get_new_metrics(self, tmp_run: Path) -> None:
        store = RunStore(LocalStorage(tmp_run))
        run_id = store.get_runs()[0].run_id

        # First call reads all
        store.get_metrics(run_id)

        # No new data
        new = store.get_new_metrics(run_id)
        assert new == []

        # Append
        with open(tmp_run / "metrics.jsonl", "a") as f:
            f.write(json.dumps({"step": 10, "x": 1}) + "\n")

        new = store.get_new_metrics(run_id)
        assert len(new) == 1

    def test_get_config(self, tmp_run: Path) -> None:
        store = RunStore(LocalStorage(tmp_run))
        run_id = store.get_runs()[0].run_id
        config = store.get_config(run_id)
        assert config is not None
        assert config["model_name"] == "Llama-3.1-8B"

    def test_get_iterations(self, tmp_run: Path) -> None:
        store = RunStore(LocalStorage(tmp_run))
        run_id = store.get_runs()[0].run_id
        iterations = store.get_iterations(run_id)
        assert len(iterations) == 3

    def test_get_rollouts(self, tmp_run: Path) -> None:
        store = RunStore(LocalStorage(tmp_run))
        run_id = store.get_runs()[0].run_id
        rollouts = store.get_rollouts(run_id, 0)
        assert len(rollouts) == 6

    def test_get_single_rollout(self, tmp_run: Path) -> None:
        store = RunStore(LocalStorage(tmp_run))
        run_id = store.get_runs()[0].run_id
        rollout = store.get_single_rollout(run_id, 0, 0, 0)
        assert rollout is not None
        assert rollout["tags"] == ["math", "gsm8k"]

    def test_get_logtree(self, tmp_run: Path) -> None:
        store = RunStore(LocalStorage(tmp_run))
        run_id = store.get_runs()[0].run_id
        tree = store.get_logtree(run_id, 0)
        assert tree is not None
        assert "root" in tree

    def test_get_timing(self, tmp_run: Path) -> None:
        store = RunStore(LocalStorage(tmp_run))
        run_id = store.get_runs()[0].run_id
        timing = store.get_timing(run_id)
        assert len(timing) == 2  # Two step records

    def test_nonexistent_run(self, tmp_run: Path) -> None:
        store = RunStore(LocalStorage(tmp_run))
        assert store.get_run("nonexistent") is None
        assert store.get_metrics("nonexistent") == []
        assert store.get_config("nonexistent") is None

    def test_multi_run(self, tmp_multi_run: Path) -> None:
        store = RunStore(LocalStorage(tmp_multi_run))
        runs = store.get_runs()
        assert len(runs) == 2

        # Each run is accessible by ID
        for run in runs:
            assert store.get_run(run.run_id) is not None

    def test_refresh_runs(self, tmp_multi_run: Path) -> None:
        store = RunStore(LocalStorage(tmp_multi_run))
        runs = store.get_runs()
        assert len(runs) == 2

        # Add a new run
        run3 = tmp_multi_run / "run_003"
        run3.mkdir()
        (run3 / "metrics.jsonl").write_text(json.dumps({"step": 0}) + "\n")

        # Refresh should find the new run
        runs = store.refresh_runs()
        assert len(runs) == 3


# ── EvalReader tests ──────────────────────────────────────────────────


@pytest.fixture
def tmp_eval_store(tmp_path: Path) -> Path:
    """Create a minimal eval store directory."""
    eval_dir = tmp_path / "eval_store"
    eval_dir.mkdir()

    # runs.jsonl index
    runs = [
        {"run_id": "eval_001", "model_name": "Llama-3.1-8B", "timestamp": "2024-04-01T00:00:00"},
        {"run_id": "eval_002", "model_name": "Llama-3.1-8B-LoRA", "timestamp": "2024-04-02T00:00:00"},
    ]
    (eval_dir / "runs.jsonl").write_text(
        "\n".join(json.dumps(r) for r in runs) + "\n"
    )

    # eval_001 run
    run1 = eval_dir / "runs" / "eval_001"
    run1.mkdir(parents=True)
    (run1 / "metadata.json").write_text(json.dumps({
        "run_id": "eval_001",
        "model_name": "Llama-3.1-8B",
        "checkpoint_path": None,
        "checkpoint_name": "base",
        "benchmarks": ["gsm8k", "mmlu_pro"],
        "timestamp": "2024-04-01T00:00:00",
        "scores": {"gsm8k": 0.75, "mmlu_pro": 0.62},
    }))

    # gsm8k benchmark for eval_001
    gsm8k_dir = run1 / "gsm8k"
    gsm8k_dir.mkdir()
    (gsm8k_dir / "result.json").write_text(json.dumps({
        "name": "gsm8k",
        "score": 0.75,
        "num_examples": 100,
        "num_correct": 75,
        "num_errors": 2,
        "time_seconds": 120.5,
    }))

    trajectories = []
    for i in range(3):
        trajectories.append({
            "idx": i,
            "benchmark": "gsm8k",
            "example_id": f"gsm8k_{i:04d}",
            "turns": [
                {"role": "user", "content": f"What is {i}+{i}?", "token_count": 10, "metadata": {}},
                {"role": "assistant", "content": f"{2*i}", "token_count": 5, "metadata": {}},
            ],
            "reward": 1.0 if i < 2 else 0.0,
            "metrics": {"correct": float(i < 2)},
            "logs": {"expected": str(2 * i)},
            "error": None if i < 2 else "Wrong answer",
            "time_seconds": 1.5 + i * 0.1,
        })
    (gsm8k_dir / "trajectories.jsonl").write_text(
        "\n".join(json.dumps(t) for t in trajectories) + "\n"
    )

    # eval_002 run (minimal)
    run2 = eval_dir / "runs" / "eval_002"
    run2.mkdir(parents=True)
    (run2 / "metadata.json").write_text(json.dumps({
        "run_id": "eval_002",
        "model_name": "Llama-3.1-8B-LoRA",
        "checkpoint_path": "tinker:///ckpt/lora_001",
        "checkpoint_name": "step_500",
        "benchmarks": ["gsm8k"],
        "timestamp": "2024-04-02T00:00:00",
        "scores": {"gsm8k": 0.85},
    }))
    gsm8k2 = run2 / "gsm8k"
    gsm8k2.mkdir()
    (gsm8k2 / "result.json").write_text(json.dumps({
        "name": "gsm8k",
        "score": 0.85,
        "num_examples": 100,
        "num_correct": 85,
        "num_errors": 1,
        "time_seconds": 110.0,
    }))

    return eval_dir


class TestEvalReader:
    def test_list_eval_runs(self, tmp_eval_store: Path) -> None:
        from tinker_cookbook.chef.data.eval_reader import EvalReader
        reader = EvalReader(LocalStorage(tmp_eval_store), "")
        runs = reader.list_eval_runs()
        assert len(runs) == 2
        assert runs[0]["run_id"] == "eval_001"

    def test_get_metadata(self, tmp_eval_store: Path) -> None:
        from tinker_cookbook.chef.data.eval_reader import EvalReader
        reader = EvalReader(LocalStorage(tmp_eval_store), "")
        metadata = reader.get_eval_run_metadata("eval_001")
        assert metadata is not None
        assert metadata["model_name"] == "Llama-3.1-8B"
        assert metadata["scores"]["gsm8k"] == 0.75

    def test_list_benchmarks(self, tmp_eval_store: Path) -> None:
        from tinker_cookbook.chef.data.eval_reader import EvalReader
        reader = EvalReader(LocalStorage(tmp_eval_store), "")
        benchmarks = reader.list_benchmarks("eval_001")
        assert "gsm8k" in benchmarks

    def test_get_benchmark_result(self, tmp_eval_store: Path) -> None:
        from tinker_cookbook.chef.data.eval_reader import EvalReader
        reader = EvalReader(LocalStorage(tmp_eval_store), "")
        result = reader.get_benchmark_result("eval_001", "gsm8k")
        assert result is not None
        assert result["score"] == 0.75
        assert result["num_correct"] == 75

    def test_get_trajectories(self, tmp_eval_store: Path) -> None:
        from tinker_cookbook.chef.data.eval_reader import EvalReader
        reader = EvalReader(LocalStorage(tmp_eval_store), "")
        trajectories = reader.get_benchmark_trajectories("eval_001", "gsm8k")
        assert len(trajectories) == 3
        assert trajectories[0]["example_id"] == "gsm8k_0000"
        assert len(trajectories[0]["turns"]) == 2

    def test_get_single_trajectory(self, tmp_eval_store: Path) -> None:
        from tinker_cookbook.chef.data.eval_reader import EvalReader
        reader = EvalReader(LocalStorage(tmp_eval_store), "")
        traj = reader.get_single_trajectory("eval_001", "gsm8k", 1)
        assert traj is not None
        assert traj["reward"] == 1.0

    def test_get_scores_table(self, tmp_eval_store: Path) -> None:
        from tinker_cookbook.chef.data.eval_reader import EvalReader
        reader = EvalReader(LocalStorage(tmp_eval_store), "")
        table = reader.get_scores_table()
        assert len(table) == 2
        assert table[0]["model_name"] == "Llama-3.1-8B"
        assert table[0]["scores"]["gsm8k"] == 0.75
        assert table[1]["scores"]["gsm8k"] == 0.85

    def test_nonexistent_run(self, tmp_eval_store: Path) -> None:
        from tinker_cookbook.chef.data.eval_reader import EvalReader
        reader = EvalReader(LocalStorage(tmp_eval_store), "")
        assert reader.get_eval_run_metadata("nonexistent") is None
        assert reader.list_benchmarks("nonexistent") == []
