"""Tests for trajectory typed schemas — round-trip serialization."""

from tinker_cookbook.types.trajectories import (
    StoredEvalTrajectory,
    StoredStep,
    StoredTrainingTrajectory,
    StoredTurn,
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
    def test_round_trip(self) -> None:
        turn = StoredTurn(role="assistant", content="The answer is 42.",
                          token_count=15, metadata={"latency_ms": 120})
        d = turn.to_dict()
        restored = StoredTurn.from_dict(d)
        assert restored.role == "assistant"
        assert restored.content == "The answer is 42."
        assert restored.token_count == 15
        assert restored.metadata == {"latency_ms": 120}

    def test_defaults(self) -> None:
        turn = StoredTurn(role="user", content="hello")
        assert turn.token_count == 0
        assert turn.metadata == {}


class TestStoredEvalTrajectory:
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
        assert "error" not in d  # None errors omitted from dict

    def test_with_error(self) -> None:
        traj = StoredEvalTrajectory(idx=0, error="Timeout after 300s")
        d = traj.to_dict()
        assert d["error"] == "Timeout after 300s"
        restored = StoredEvalTrajectory.from_dict(d)
        assert restored.error == "Timeout after 300s"

    def test_from_dict_minimal(self) -> None:
        d = {}
        traj = StoredEvalTrajectory.from_dict(d)
        assert traj.turns == []
        assert traj.error is None
