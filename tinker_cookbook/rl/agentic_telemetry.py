"""Tracing, metrics, and logtree telemetry for multi-turn agentic RL.

This module provides utilities to instrument agentic RL episodes with:

- **Trace spans** via ``scope_span`` for Perfetto-compatible timing traces
  (initial observation, each agent turn, policy sampling, env steps, tool
  execution).
- **Aggregated metrics** for logging to ml_log (turns per episode, tool calls
  per episode, tool success rate, episode rewards, tool execution time).
- **Logtree logging** for human-readable HTML reports showing per-turn model
  responses, tool calls with inputs/outputs, rewards, and loss mask summaries.

These are designed to compose with the existing ``do_single_rollout`` and
``do_group_rollout`` functions in ``rollouts.py``. The core rollout loop
already emits trace spans for ``env_initial_observation``, ``policy_sample``,
and ``env_step``. This module adds agentic-specific instrumentation on top.

Usage in a training loop::

    from tinker_cookbook.rl.agentic_telemetry import (
        compute_agentic_metrics,
        log_agentic_episode,
        log_agentic_loss_masking,
    )

    # After gathering rollouts:
    metrics.update(compute_agentic_metrics(trajectory_groups_P))

    # Inside logtree scope for an episode:
    log_agentic_episode(trajectory_group, episode_idx=0)

    # After computing mask summaries:
    log_agentic_loss_masking(mask_summary)
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence

import numpy as np

from tinker_cookbook.rl.loss_masking import TrajectoryMaskSummary, compute_trajectory_mask_summary
from tinker_cookbook.rl.types import Trajectory, TrajectoryGroup
from tinker_cookbook.utils import logtree

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------


def _count_tool_calls_in_trajectory(trajectory: Trajectory) -> int:
    """Count tool calls across all transitions in a trajectory.

    Tool calls are detected by the presence of ``tool_call_*`` keys in
    transition logs, which are populated by ``AgentToolMessageEnv.step()``.

    Args:
        trajectory: A completed trajectory from a rollout.

    Returns:
        Total number of tool calls across all transitions.
    """
    count = 0
    for transition in trajectory.transitions:
        for key in transition.logs:
            if isinstance(key, str) and key.startswith("tool_call_"):
                count += 1
    return count


def _is_error_tool_result(value: object) -> bool:
    """Check whether a tool result represents an error.

    ``error_tool_result`` in ``tinker_cookbook.tool_use`` formats errors as
    JSON with an ``"error"`` key (e.g., ``{"error": "some message"}``).
    We detect this structured pattern rather than naively substring-matching
    on the word "error", which would false-positive on benign output that
    happens to contain "error" (e.g., "error rate: 0.01").
    """
    s = str(value)
    try:
        import json

        data = json.loads(s)
        return isinstance(data, dict) and "error" in data
    except (json.JSONDecodeError, TypeError, ValueError):
        return False


def _count_tool_errors_in_trajectory(trajectory: Trajectory) -> int:
    """Count tool execution errors in a trajectory.

    An error is detected by checking whether the tool result is a JSON
    object with an ``"error"`` key, which matches the format produced by
    ``error_tool_result``.

    Args:
        trajectory: A completed trajectory from a rollout.

    Returns:
        Total number of tool errors across all transitions.
    """
    errors = 0
    for transition in trajectory.transitions:
        for key, value in transition.logs.items():
            if isinstance(key, str) and key.startswith("tool_result_"):
                if _is_error_tool_result(value):
                    errors += 1
    return errors


def compute_agentic_metrics(
    trajectory_groups_P: Sequence[TrajectoryGroup],
    *,
    prefix: str = "agentic",
) -> dict[str, float]:
    """Compute agentic RL metrics from a batch of trajectory groups.

    Aggregates multi-turn episode statistics across all trajectories in the
    batch. Intended to be called once per training iteration and logged via
    ``ml_log.log_metrics``.

    Emitted metrics (all prefixed with *prefix*):

    - ``{prefix}/turns_per_episode_mean`` -- average turns per episode
    - ``{prefix}/turns_per_episode_std`` -- standard deviation of turns
    - ``{prefix}/tool_calls_per_episode`` -- average tool calls per episode
    - ``{prefix}/tool_success_rate`` -- fraction of tool calls that succeeded
    - ``{prefix}/episode_reward_mean`` -- mean total reward across episodes
    - ``{prefix}/episode_reward_std`` -- standard deviation of total reward
    - ``{prefix}/multi_turn_frac`` -- fraction of episodes with >1 turn

    Args:
        trajectory_groups_P: One trajectory group per problem in the batch.
        prefix: Metric key prefix (default ``"agentic"``).

    Returns:
        Flat dictionary of metric names to values.

    Example::

        metrics = compute_agentic_metrics(trajectory_groups_P)
        ml_logger.log_metrics(metrics, step=i_batch)
    """
    flat_trajs = [
        traj for tg in trajectory_groups_P for traj in tg.trajectories_G
    ]
    if not flat_trajs:
        return {}

    # Turns per episode
    turns = [len(traj.transitions) for traj in flat_trajs]
    turns_arr = np.array(turns, dtype=np.float64)

    # Tool calls per episode
    tool_calls = [_count_tool_calls_in_trajectory(traj) for traj in flat_trajs]
    tool_calls_arr = np.array(tool_calls, dtype=np.float64)

    # Tool errors per episode
    tool_errors = [_count_tool_errors_in_trajectory(traj) for traj in flat_trajs]
    total_tool_calls = int(tool_calls_arr.sum())
    total_tool_errors = sum(tool_errors)
    tool_success_rate = (
        (total_tool_calls - total_tool_errors) / total_tool_calls
        if total_tool_calls > 0
        else 1.0
    )

    # Episode rewards
    rewards = [
        reward for tg in trajectory_groups_P for reward in tg.get_total_rewards()
    ]
    rewards_arr = np.array(rewards, dtype=np.float64)

    # Multi-turn fraction
    multi_turn_count = sum(1 for t in turns if t > 1)

    metrics: dict[str, float] = {
        f"{prefix}/turns_per_episode_mean": float(turns_arr.mean()),
        f"{prefix}/turns_per_episode_std": float(turns_arr.std()),
        f"{prefix}/tool_calls_per_episode": float(tool_calls_arr.mean()),
        f"{prefix}/tool_success_rate": tool_success_rate,
        f"{prefix}/episode_reward_mean": float(rewards_arr.mean()),
        f"{prefix}/episode_reward_std": float(rewards_arr.std()),
        f"{prefix}/multi_turn_frac": multi_turn_count / len(flat_trajs),
    }

    return metrics


# ---------------------------------------------------------------------------
# Logtree logging
# ---------------------------------------------------------------------------


def log_agentic_episode(
    trajectory_group: TrajectoryGroup,
    episode_idx: int,
) -> None:
    """Log a single agentic episode to logtree with per-turn detail.

    Creates a nested logtree structure showing each turn's model response,
    tool calls (with inputs/outputs), step metrics, and rewards. This gives
    a human-readable HTML view of agentic episodes for debugging.

    Should be called inside a logtree trace context (e.g., within
    ``init_trace`` or inside a ``scope_header`` block).

    Structure::

        Episode {episode_idx}
          Turn 1
            Step Stats (ob_len, ac_len, step_reward)
            Tool Call (if present)
              Tool: {name}
              Input: {arguments}
              Output: {result}
            Step Metrics (if present)
          Turn 2
            ...
          Episode Summary
            num_turns, total_reward, ...

    Args:
        trajectory_group: A trajectory group containing one or more
            trajectories.
        episode_idx: Index of this episode within the batch, used in the
            header.
    """
    with logtree.scope_header(f"Episode {episode_idx}"):
        for traj_idx, trajectory in enumerate(trajectory_group.trajectories_G):
            total_reward = trajectory_group.get_total_rewards()[traj_idx]

            with logtree.scope_header(f"Trajectory {traj_idx}"):
                for turn_idx, transition in enumerate(trajectory.transitions):
                    with logtree.scope_header(f"Turn {turn_idx}"):
                        # Step stats
                        logtree.table_from_dict(
                            {
                                "ob_len": transition.ob.length,
                                "ac_len": len(transition.ac.tokens),
                                "step_reward": f"{transition.reward:.3f}",
                                "episode_done": str(transition.episode_done),
                            },
                            caption="Step stats",
                        )

                        # Log tool calls from transition logs
                        tool_call_keys = sorted(
                            k for k in transition.logs
                            if isinstance(k, str) and k.startswith("tool_call_")
                        )
                        for tc_key in tool_call_keys:
                            idx_str = tc_key.replace("tool_call_", "")
                            result_key = f"tool_result_{idx_str}"
                            tool_call_str = str(transition.logs[tc_key])
                            tool_result_str = str(
                                transition.logs.get(result_key, "<no result>")
                            )

                            with logtree.scope_details(f"Tool Call: {tool_call_str}"):
                                logtree.log_text(f"Input: {tool_call_str}")
                                logtree.log_text(f"Output: {tool_result_str}")

                        # Log assistant content if present
                        if "assistant_content" in transition.logs:
                            logtree.details(
                                str(transition.logs["assistant_content"]),
                                summary="Model Response",
                                pre=True,
                            )

                        # Transition metrics
                        if transition.metrics:
                            logtree.table_from_dict(
                                {
                                    k: f"{v:.3f}" if isinstance(v, float) else str(v)
                                    for k, v in transition.metrics.items()
                                },
                                caption="Step metrics",
                            )

                # Episode summary
                n_turns = len(trajectory.transitions)
                step_rewards_sum = sum(
                    t.reward for t in trajectory.transitions
                )
                n_tool_calls = _count_tool_calls_in_trajectory(trajectory)
                n_tool_errors = _count_tool_errors_in_trajectory(trajectory)

                logtree.table_from_dict(
                    {
                        "num_turns": n_turns,
                        "tool_calls": n_tool_calls,
                        "tool_errors": n_tool_errors,
                        "sum_step_rewards": f"{step_rewards_sum:.3f}",
                        "final_group_reward": f"{trajectory_group.final_rewards_G[traj_idx]:.3f}",
                        "total_reward": f"{total_reward:.3f}",
                    },
                    caption="Episode summary",
                )


def log_agentic_loss_masking(
    mask_summary: TrajectoryMaskSummary,
    *,
    trajectory_idx: int | None = None,
) -> None:
    """Log loss masking information for a trajectory to logtree.

    Shows per-turn token counts and mask ratios, making it easy to verify
    that environment tokens (tool results, user messages) are correctly
    masked out and only model-generated tokens receive gradient.

    Args:
        mask_summary: A ``TrajectoryMaskSummary`` from
            ``compute_trajectory_mask_summary()``.
        trajectory_idx: Optional trajectory index for the header.

    Example::

        summary = compute_trajectory_mask_summary(trajectory)
        log_agentic_loss_masking(summary)
    """
    header = "Loss Masking"
    if trajectory_idx is not None:
        header = f"Loss Masking (Trajectory {trajectory_idx})"

    with logtree.scope_header(header):
        # Overall summary
        total = mask_summary.total_tokens
        mask_ratio = (
            mask_summary.masked_tokens / total if total > 0 else 0.0
        )
        logtree.table_from_dict(
            {
                "total_tokens": mask_summary.total_tokens,
                "model_tokens (unmasked)": mask_summary.unmasked_tokens,
                "env_tokens (masked)": mask_summary.masked_tokens,
                "mask_ratio": f"{mask_ratio:.2%}",
                "n_datums": mask_summary.n_datums,
            },
            caption="Trajectory mask summary",
        )

        # Per-turn breakdown
        if mask_summary.turns:
            turn_data: dict[str, list[str]] = {
                "turn": [],
                "obs_tokens": [],
                "action_tokens": [],
                "obs_mask": [],
                "action_mask": [],
            }
            for turn in mask_summary.turns:
                turn_data["turn"].append(str(turn.turn_index))
                turn_data["obs_tokens"].append(str(turn.observation_tokens))
                turn_data["action_tokens"].append(str(turn.action_tokens))
                turn_data["obs_mask"].append(str(turn.observation_mask))
                turn_data["action_mask"].append(str(turn.action_mask))

            logtree.table_from_dict_of_lists(
                turn_data,
                caption="Per-turn token counts and masks",
            )


def log_agentic_batch_summary(
    trajectory_groups_P: Sequence[TrajectoryGroup],
    metrics: dict[str, float],
    step: int,
) -> None:
    """Log a batch-level summary of agentic RL to logtree.

    Creates a top-level logtree section with aggregate metrics and
    optionally per-episode details. Call this once per training iteration
    inside a logtree trace context.

    Args:
        trajectory_groups_P: Trajectory groups from the current batch.
        metrics: The metrics dict (from ``compute_agentic_metrics`` or the
            full training metrics). Only ``agentic/*`` keys are displayed.
        step: Training step number.
    """
    agentic_metrics = {
        k: v for k, v in metrics.items() if k.startswith("agentic/")
    }
    if not agentic_metrics:
        return

    with logtree.scope_header(f"Agentic Summary (Step {step})"):
        logtree.table_from_dict(
            {
                k: f"{v:.4f}" if isinstance(v, float) else str(v)
                for k, v in sorted(agentic_metrics.items())
            },
            caption="Agentic metrics",
        )

        # Log loss masking for the first few trajectories
        n_to_log = min(3, len(trajectory_groups_P))
        if n_to_log > 0:
            with logtree.scope_header("Loss Mask Diagnostics"):
                traj_count = 0
                for tg in trajectory_groups_P:
                    for traj in tg.trajectories_G:
                        if traj_count >= n_to_log:
                            break
                        summary = compute_trajectory_mask_summary(traj)
                        log_agentic_loss_masking(
                            summary, trajectory_idx=traj_count
                        )
                        traj_count += 1
                    if traj_count >= n_to_log:
                        break
