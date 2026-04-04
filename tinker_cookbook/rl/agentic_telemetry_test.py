"""Tests for agentic_telemetry module."""

import tinker

from tinker_cookbook.completers import TokensWithLogprobs
from tinker_cookbook.rl.agentic_telemetry import (
    _count_tool_calls_in_trajectory,
    _count_tool_errors_in_trajectory,
    _is_error_tool_result,
    compute_agentic_metrics,
    log_agentic_batch_summary,
    log_agentic_episode,
    log_agentic_loss_masking,
)
from tinker_cookbook.rl.loss_masking import TrajectoryMaskSummary, TurnMaskInfo
from tinker_cookbook.rl.types import Trajectory, TrajectoryGroup, Transition
from tinker_cookbook.utils import logtree


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_transition(
    *,
    ob_len: int = 10,
    ac_tokens: list[int] | None = None,
    reward: float = 0.0,
    episode_done: bool = False,
    metrics: dict | None = None,
    logs: dict | None = None,
) -> Transition:
    """Create a Transition with minimal boilerplate."""
    if ac_tokens is None:
        ac_tokens = [1, 2, 3]
    return Transition(
        ob=tinker.ModelInput.from_ints(list(range(ob_len))),
        ac=TokensWithLogprobs(
            tokens=ac_tokens,
            maybe_logprobs=[-0.1] * len(ac_tokens),
            stop_reason="stop",
        ),
        reward=reward,
        episode_done=episode_done,
        metrics=metrics or {},
        logs=logs or {},
    )


def _make_trajectory(transitions: list[Transition]) -> Trajectory:
    return Trajectory(
        transitions=transitions,
        final_ob=tinker.ModelInput.from_ints([0]),
    )


def _make_trajectory_group(
    trajectories: list[Trajectory],
    final_rewards: list[float] | None = None,
) -> TrajectoryGroup:
    if final_rewards is None:
        final_rewards = [0.0] * len(trajectories)
    return TrajectoryGroup(
        trajectories_G=trajectories,
        final_rewards_G=final_rewards,
        metrics_G=[{} for _ in trajectories],
    )


# ---------------------------------------------------------------------------
# Tests: tool counting
# ---------------------------------------------------------------------------


class TestToolCounting:
    def test_no_tool_calls(self):
        traj = _make_trajectory([_make_transition()])
        assert _count_tool_calls_in_trajectory(traj) == 0

    def test_single_tool_call(self):
        traj = _make_trajectory([
            _make_transition(logs={
                "tool_call_0": "search({\"q\": \"test\"})",
                "tool_result_0": "result",
            }),
        ])
        assert _count_tool_calls_in_trajectory(traj) == 1

    def test_multiple_tool_calls_across_turns(self):
        traj = _make_trajectory([
            _make_transition(logs={
                "tool_call_0": "search({})",
                "tool_result_0": "r1",
            }),
            _make_transition(logs={
                "tool_call_0": "calc({})",
                "tool_call_1": "fetch({})",
                "tool_result_0": "r2",
                "tool_result_1": "r3",
            }),
        ])
        assert _count_tool_calls_in_trajectory(traj) == 3

    def test_tool_error_counting(self):
        traj = _make_trajectory([
            _make_transition(logs={
                "tool_call_0": "search({})",
                "tool_result_0": '{"error": "not found"}',
            }),
            _make_transition(logs={
                "tool_call_0": "calc({})",
                "tool_result_0": "42",
            }),
        ])
        assert _count_tool_errors_in_trajectory(traj) == 1
        assert _count_tool_calls_in_trajectory(traj) == 2

    def test_no_tool_errors(self):
        traj = _make_trajectory([
            _make_transition(logs={
                "tool_call_0": "search({})",
                "tool_result_0": "success result",
            }),
        ])
        assert _count_tool_errors_in_trajectory(traj) == 0

    def test_no_false_positive_on_error_word_in_output(self):
        """Output containing 'error' as normal text should NOT be counted."""
        traj = _make_trajectory([
            _make_transition(logs={
                "tool_call_0": "python_exec({})",
                "tool_result_0": '{"output": "The error rate is 0.01"}',
            }),
        ])
        assert _count_tool_errors_in_trajectory(traj) == 0


# ---------------------------------------------------------------------------
# Tests: _is_error_tool_result
# ---------------------------------------------------------------------------


class TestIsErrorToolResult:
    def test_structured_error_json(self):
        """error_tool_result format: JSON with 'error' key."""
        import json

        val = json.dumps({"error": "Execution error: NameError"})
        assert _is_error_tool_result(val) is True

    def test_normal_json_output(self):
        import json

        val = json.dumps({"output": "42"})
        assert _is_error_tool_result(val) is False

    def test_json_with_error_word_in_value(self):
        """Should NOT match when 'error' is a value, not a key."""
        import json

        val = json.dumps({"output": "The error rate is 0.01"})
        assert _is_error_tool_result(val) is False

    def test_plain_string(self):
        assert _is_error_tool_result("just a plain string") is False

    def test_non_string_input(self):
        assert _is_error_tool_result(12345) is False

    def test_error_key_with_extra_fields(self):
        import json

        val = json.dumps({"error": "timeout", "details": "5s limit"})
        assert _is_error_tool_result(val) is True


# ---------------------------------------------------------------------------
# Tests: compute_agentic_metrics
# ---------------------------------------------------------------------------


class TestComputeAgenticMetrics:
    def test_empty_input(self):
        metrics = compute_agentic_metrics([])
        assert metrics == {}

    def test_single_turn_episodes(self):
        """Single-turn episodes should have multi_turn_frac=0."""
        traj = _make_trajectory([
            _make_transition(reward=1.0, episode_done=True),
        ])
        tg = _make_trajectory_group([traj], final_rewards=[0.5])
        metrics = compute_agentic_metrics([tg])

        assert metrics["agentic/turns_per_episode_mean"] == 1.0
        assert metrics["agentic/turns_per_episode_std"] == 0.0
        assert metrics["agentic/multi_turn_frac"] == 0.0
        assert metrics["agentic/episode_reward_mean"] == 1.5  # 1.0 step + 0.5 final

    def test_multi_turn_episodes(self):
        """Multi-turn episodes with tool calls."""
        traj = _make_trajectory([
            _make_transition(
                reward=0.0,
                logs={"tool_call_0": "search({})", "tool_result_0": "ok"},
            ),
            _make_transition(
                reward=0.0,
                logs={"tool_call_0": "calc({})", "tool_result_0": "42"},
            ),
            _make_transition(reward=1.0, episode_done=True),
        ])
        tg = _make_trajectory_group([traj], final_rewards=[0.0])
        metrics = compute_agentic_metrics([tg])

        assert metrics["agentic/turns_per_episode_mean"] == 3.0
        assert metrics["agentic/tool_calls_per_episode"] == 2.0
        assert metrics["agentic/tool_success_rate"] == 1.0
        assert metrics["agentic/multi_turn_frac"] == 1.0

    def test_tool_errors_affect_success_rate(self):
        """Tool errors reduce the success rate."""
        traj = _make_trajectory([
            _make_transition(logs={
                "tool_call_0": "search({})",
                "tool_result_0": '{"error": "timeout"}',
            }),
            _make_transition(logs={
                "tool_call_0": "search({})",
                "tool_result_0": "success",
            }),
            _make_transition(reward=0.5, episode_done=True),
        ])
        tg = _make_trajectory_group([traj])
        metrics = compute_agentic_metrics([tg])

        assert metrics["agentic/tool_success_rate"] == 0.5

    def test_multiple_groups(self):
        """Metrics aggregate across multiple trajectory groups."""
        traj1 = _make_trajectory([
            _make_transition(reward=1.0, episode_done=True),
        ])
        traj2 = _make_trajectory([
            _make_transition(reward=0.0),
            _make_transition(reward=0.5, episode_done=True),
        ])
        tg1 = _make_trajectory_group([traj1], final_rewards=[0.0])
        tg2 = _make_trajectory_group([traj2], final_rewards=[0.0])
        metrics = compute_agentic_metrics([tg1, tg2])

        assert metrics["agentic/turns_per_episode_mean"] == 1.5
        assert metrics["agentic/multi_turn_frac"] == 0.5

    def test_custom_prefix(self):
        """Custom prefix is applied to all metric keys."""
        traj = _make_trajectory([_make_transition(reward=1.0, episode_done=True)])
        tg = _make_trajectory_group([traj])
        metrics = compute_agentic_metrics([tg], prefix="my_agent")

        assert all(k.startswith("my_agent/") for k in metrics)

    def test_no_tool_calls_success_rate_is_one(self):
        """When no tool calls exist, success rate should be 1.0."""
        traj = _make_trajectory([
            _make_transition(reward=1.0, episode_done=True),
        ])
        tg = _make_trajectory_group([traj])
        metrics = compute_agentic_metrics([tg])

        assert metrics["agentic/tool_success_rate"] == 1.0


# ---------------------------------------------------------------------------
# Tests: logtree logging (smoke tests -- verify no exceptions)
# ---------------------------------------------------------------------------


class TestLogtreeLogging:
    """Smoke tests that logtree logging functions run without error."""

    def test_log_agentic_episode(self):
        traj = _make_trajectory([
            _make_transition(
                reward=0.0,
                logs={
                    "assistant_content": "Let me search.",
                    "tool_call_0": "search({})",
                    "tool_result_0": "found it",
                },
                metrics={"parse_error": 0.0},
            ),
            _make_transition(
                reward=1.0,
                episode_done=True,
                logs={"assistant_content": "The answer is 42."},
            ),
        ])
        tg = _make_trajectory_group([traj], final_rewards=[0.5])

        with logtree.init_trace("test", path=None):
            log_agentic_episode(tg, episode_idx=0)

    def test_log_agentic_loss_masking(self):
        summary = TrajectoryMaskSummary(
            turns=[
                TurnMaskInfo(turn_index=0, observation_tokens=50, action_tokens=20),
                TurnMaskInfo(turn_index=1, observation_tokens=30, action_tokens=15),
            ],
            total_tokens=115,
            masked_tokens=80,
            unmasked_tokens=35,
            n_datums=1,
        )
        with logtree.init_trace("test", path=None):
            log_agentic_loss_masking(summary, trajectory_idx=0)

    def test_log_agentic_batch_summary(self):
        traj = _make_trajectory([
            _make_transition(reward=1.0, episode_done=True),
        ])
        tg = _make_trajectory_group([traj])
        metrics = compute_agentic_metrics([tg])

        with logtree.init_trace("test", path=None):
            log_agentic_batch_summary([tg], metrics, step=0)

    def test_log_batch_summary_no_agentic_metrics(self):
        """Batch summary is a no-op when no agentic metrics are present."""
        traj = _make_trajectory([_make_transition(reward=1.0, episode_done=True)])
        tg = _make_trajectory_group([traj])

        with logtree.init_trace("test", path=None):
            log_agentic_batch_summary([tg], {"some/other_metric": 0.5}, step=0)

    def test_log_episode_no_tool_calls(self):
        """Episode logging works when there are no tool calls."""
        traj = _make_trajectory([
            _make_transition(
                reward=1.0,
                episode_done=True,
                logs={"assistant_content": "Final answer."},
            ),
        ])
        tg = _make_trajectory_group([traj])

        with logtree.init_trace("test", path=None):
            log_agentic_episode(tg, episode_idx=0)

    def test_log_loss_masking_empty(self):
        """Loss masking logging works with zero tokens."""
        summary = TrajectoryMaskSummary(
            turns=[],
            total_tokens=0,
            masked_tokens=0,
            unmasked_tokens=0,
            n_datums=0,
        )
        with logtree.init_trace("test", path=None):
            log_agentic_loss_masking(summary)
