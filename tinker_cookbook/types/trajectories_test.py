"""Tests for trajectory typed schemas — round-trip serialization."""

from tinker_cookbook.eval.benchmarks._types import (
    StoredTrajectory as StoredEvalTrajectory,
    StoredTurn,
)
from tinker_cookbook.types.trajectories import (
    StoredStep,
    StoredTrainingTrajectory,
)


class TestStoredStep:
    def test_round_trip(self) -> None:
        step = StoredStep(step_idx=0, ob_len=128, ac_len=45, reward=0.5, episode_done=False,
                          metrics={"step_time": 0.01}, logs={"debug": "ok"})
        d = step.to_dict()
        restored = StoredStep.from_dict(d)
        assert restored.step_idx == 0
        assert restored.ob_len == 128
        assert restored.reward == 0.5
        assert restored.metrics == {"step_time": 0.01}
        assert restored.to_dict() == d

    def test_defaults(self) -> None:
        step = StoredStep(step_idx=0, ob_len=10, ac_len=5, reward=0.0, episode_done=True)
        d = step.to_dict()
        assert d["metrics"] == {}
        assert d["logs"] == {}

    def test_from_dict_missing_optional(self) -> None:
        d = {"step_idx": 1, "ob_len": 50, "ac_len": 20, "reward": 1.0, "episode_done": True}
        step = StoredStep.from_dict(d)
        assert step.metrics == {}
        assert step.logs == {}


class TestStoredTrainingTrajectory:
    def test_round_trip(self) -> None:
        traj = StoredTrainingTrajectory(
            iteration=42, group_idx=1, traj_idx=3,
            tags=["math", "gsm8k"],
            sampling_client_step=42,
            total_reward=0.85, final_reward=0.4,
            steps=[
                StoredStep(step_idx=0, ob_len=128, ac_len=45, reward=0.0, episode_done=False),
                StoredStep(step_idx=1, ob_len=200, ac_len=60, reward=0.85, episode_done=True),
            ],
            final_ob_len=350,
        )
        d = traj.to_dict()
        restored = StoredTrainingTrajectory.from_dict(d)
        assert restored.iteration == 42
        assert restored.tags == ["math", "gsm8k"]
        assert len(restored.steps) == 2
        assert restored.steps[0].ob_len == 128
        assert restored.steps[1].reward == 0.85
        assert restored.to_dict() == d

    def test_from_dict_minimal(self) -> None:
        d = {}
        traj = StoredTrainingTrajectory.from_dict(d)
        assert traj.schema_version == 1
        assert traj.split == "train"
        assert traj.steps == []

    def test_schema_version(self) -> None:
        traj = StoredTrainingTrajectory()
        assert traj.to_dict()["schema_version"] == 1


class TestStoredTurn:
    def test_fields(self) -> None:
        turn = StoredTurn(role="assistant", content="The answer is 42.",
                          token_count=15, metadata={"latency_ms": 120})
        assert turn.role == "assistant"
        assert turn.content == "The answer is 42."
        assert turn.token_count == 15
        assert turn.metadata == {"latency_ms": 120}

    def test_defaults(self) -> None:
        turn = StoredTurn(role="user", content="hello")
        assert turn.token_count == 0
        assert turn.metadata == {}


class TestStoredEvalTrajectory:
    """Tests for the canonical StoredTrajectory from eval/benchmarks/_types."""

    def test_round_trip(self) -> None:
        traj = StoredEvalTrajectory(
            idx=5, benchmark="gsm8k", example_id="abc123",
            turns=[
                StoredTurn(role="user", content="What is 2+2?", token_count=10),
                StoredTurn(role="assistant", content="4", token_count=5),
            ],
            reward=1.0,
            metrics={"correct": 1.0},
            logs={"expected": "4", "extracted": "4"},
            time_seconds=1.5,
        )
        d = traj.to_dict()
        restored = StoredEvalTrajectory.from_dict(d)
        assert restored.idx == 5
        assert restored.benchmark == "gsm8k"
        assert len(restored.turns) == 2
        assert restored.turns[0].role == "user"
        assert restored.reward == 1.0
        assert restored.error is None

    def test_with_error(self) -> None:
        traj = StoredEvalTrajectory(idx=0, benchmark="test", error="Timeout after 300s")
        d = traj.to_dict()
        assert d["error"] == "Timeout after 300s"
        restored = StoredEvalTrajectory.from_dict(d)
        assert restored.error == "Timeout after 300s"

    def test_from_dict_with_required_fields(self) -> None:
        d = {"idx": 0, "benchmark": "test"}
        traj = StoredEvalTrajectory.from_dict(d)
        assert traj.turns == []
        assert traj.error is None


class TestSchemaRoundTripWithRealData:
    """Verify that the typed schema matches what rollout_logging.py actually writes."""

    def test_training_trajectory_matches_writer_output(self) -> None:
        """Build a dict the same way rollout_logging.py does, verify from_dict works."""
        # This is the exact structure written by write_rollout_summaries_jsonl
        raw_record = {
            "schema_version": 1,
            "split": "train",
            "iteration": 42,
            "group_idx": 2,
            "traj_idx": 1,
            "tags": ["math", "gsm8k"],
            "sampling_client_step": 42,
            "total_reward": 0.85,
            "final_reward": 0.4,
            "trajectory_metrics": {"custom": 1.5},
            "steps": [
                {
                    "step_idx": 0,
                    "ob_len": 128,
                    "ac_len": 45,
                    "reward": 0.0,
                    "episode_done": False,
                    "metrics": {"step_time": 0.5},
                    "logs": {},
                },
                {
                    "step_idx": 1,
                    "ob_len": 200,
                    "ac_len": 60,
                    "reward": 0.85,
                    "episode_done": True,
                    "metrics": {},
                    "logs": {"debug": "ok"},
                },
            ],
            "final_ob_len": 350,
        }
        traj = StoredTrainingTrajectory.from_dict(raw_record)
        assert traj.schema_version == 1
        assert traj.group_idx == 2
        assert len(traj.steps) == 2
        assert traj.steps[1].reward == 0.85
        # Round-trip should be identical
        assert traj.to_dict() == raw_record

    def test_eval_trajectory_matches_eval_runner_output(self) -> None:
        """Build a dict the same way the eval runner writes, verify from_dict works."""
        raw_record = {
            "idx": 5,
            "benchmark": "gsm8k",
            "example_id": "abc123",
            "turns": [
                {"role": "user", "content": "What is 2+2?", "token_count": 10, "metadata": {}},
                {"role": "assistant", "content": "4", "token_count": 5, "metadata": {}},
            ],
            "reward": 1.0,
            "metrics": {"correct": 1.0},
            "logs": {"expected": "4", "extracted": "4"},
            "error": None,
            "time_seconds": 1.5,
        }
        traj = StoredEvalTrajectory.from_dict(raw_record)
        assert traj.idx == 5
        assert len(traj.turns) == 2
        assert traj.turns[0].role == "user"
        assert traj.error is None
        assert traj.to_dict()["reward"] == 1.0
