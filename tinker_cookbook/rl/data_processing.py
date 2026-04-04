"""
Data processing functions for RL training.

Contains functions for computing advantages, converting trajectories to training data,
and assembling training batches.
"""

import logging

import tinker
import torch
from tinker import TensorData

from tinker_cookbook.exceptions import ConfigurationError
from tinker_cookbook.rl.types import EnvGroupBuilder, Trajectory, TrajectoryGroup
from tinker_cookbook.supervised.common import (
    create_rightshifted_model_input_and_leftshifted_targets,
)
from tinker_cookbook.utils.misc_utils import all_same, safezip

logger = logging.getLogger(__name__)


def compute_advantages(trajectory_groups_P: list[TrajectoryGroup]) -> list[torch.Tensor]:
    """Compute advantages for each trajectory, centered within groups.

    Args:
        trajectory_groups_P (list[TrajectoryGroup]): Groups of trajectories,
            where each group's rewards are centered independently.

    Returns:
        list[torch.Tensor]: Per-group advantage tensors of shape ``(G,)``,
            where ``G`` is the number of trajectories in each group.
    """
    advantages_P: list[torch.Tensor] = []

    for traj_group in trajectory_groups_P:
        rewards_G = torch.tensor(traj_group.get_total_rewards())
        # Center advantages within the group
        advantages_G = rewards_G - rewards_G.mean()
        advantages_P.append(advantages_G)

    return advantages_P


FlatObElem = int | tinker.ModelInputChunk
FlatOb = list[FlatObElem]


def _is_prefix(seq1: FlatOb, seq2: FlatOb) -> bool:
    """
    Check if seq1 is a prefix of seq2.
    """
    return len(seq1) <= len(seq2) and seq2[: len(seq1)] == seq1


def _flat_ob_token_len(flat_ob: FlatOb) -> int:
    out = 0
    for elem in flat_ob:
        if isinstance(elem, int):
            out += 1
        else:
            out += elem.length
    return out


def _flat_ob_to_model_input(flat_ob: FlatOb) -> tinker.ModelInput:
    out: list[tinker.ModelInputChunk] = []
    current_text_chunk: list[int] = []

    def flush_text_chunk():
        if current_text_chunk:
            out.append(tinker.EncodedTextChunk(tokens=current_text_chunk))
            current_text_chunk.clear()

    for elem in flat_ob:
        if isinstance(elem, int):
            current_text_chunk.append(elem)
        else:
            flush_text_chunk()
            out.append(elem)
    flush_text_chunk()
    return tinker.ModelInput(chunks=out)


def _flatten_chunks(chunks: list[tinker.ModelInputChunk]) -> FlatOb:
    out: FlatOb = []
    for chunk in chunks:
        if isinstance(chunk, tinker.EncodedTextChunk):
            out.extend(chunk.tokens)
        else:
            out.append(chunk)
    return out


def trajectory_to_data(traj: Trajectory, traj_advantage: float) -> list[tinker.Datum]:
    """Return one or more Datum objects corresponding to the trajectory.

    If the sequence grows by appending, i.e., each successive observation contains
    the previous observation+action as a prefix, then we can return a single Datum.
    However, if we get a sequence that's not an extension of the previous sequence,
    then that results in a new Datum.

    For example, let O1 denote a chunk of observation tokens, and let A1 denote an action.

    Then let's say ob_ac_pairs is as follows.

    (O1, A1)
    (O1+A1+O2, A2)
    (O3, A3)

    Then we will merge the first two observation-action pairs into a single Datum,
    and the last observation-action pair into a separate Datum.

    Args:
        traj (Trajectory): A single trajectory containing transitions
            (observation-action pairs).
        traj_advantage (float): The scalar advantage to assign to all action
            tokens in this trajectory.

    Returns:
        list[tinker.Datum]: One or more training datums, each containing
            model input, targets, sampled log-probs, advantages, and masks.
    """

    class SequenceAccumulator:
        full_sequence: list[FlatObElem] = []
        sampled_logprobs: list[float] = []
        advantages: list[float] = []
        mask: list[float] = []

        @classmethod
        def clear(cls):
            cls.full_sequence = []
            cls.sampled_logprobs = []
            cls.advantages = []
            cls.mask = []

    def make_datum_from_state():
        all_tokens_T = _flat_ob_to_model_input(SequenceAccumulator.full_sequence)
        input_tokens_T, target_tokens_T = create_rightshifted_model_input_and_leftshifted_targets(
            list(all_tokens_T.chunks)
        )
        sampled_logprobs_T = SequenceAccumulator.sampled_logprobs[1:]
        advantages_T = SequenceAccumulator.advantages[1:]
        mask_T = SequenceAccumulator.mask[1:]
        assert (
            input_tokens_T.length
            == len(target_tokens_T)
            == len(sampled_logprobs_T)
            == len(advantages_T)
            == len(mask_T)
        )
        return tinker.Datum(
            model_input=input_tokens_T,
            loss_fn_inputs={
                "target_tokens": TensorData.from_torch(torch.tensor(target_tokens_T)),
                "logprobs": TensorData.from_torch(torch.tensor(sampled_logprobs_T)),
                "advantages": TensorData.from_torch(torch.tensor(advantages_T)),
                "mask": TensorData.from_torch(torch.tensor(mask_T)),
            },
        )

    data: list[tinker.Datum] = []
    for transition in traj.transitions:
        ob = transition.ob
        ob_flat = _flatten_chunks(ob.chunks)
        ac_with_logprobs = transition.ac
        if len(SequenceAccumulator.full_sequence) == 0:
            delta_ob_flat = ob_flat
        elif _is_prefix(SequenceAccumulator.full_sequence, ob_flat):
            delta_ob_flat = ob_flat[len(SequenceAccumulator.full_sequence) :]
        else:
            data.append(make_datum_from_state())
            SequenceAccumulator.clear()
            delta_ob_flat = ob_flat
        delta_ob_len = _flat_ob_token_len(delta_ob_flat)
        SequenceAccumulator.full_sequence.extend(delta_ob_flat)
        SequenceAccumulator.full_sequence.extend(ac_with_logprobs.tokens)
        SequenceAccumulator.sampled_logprobs.extend(
            [0.0] * delta_ob_len + ac_with_logprobs.logprobs
        )
        SequenceAccumulator.advantages.extend(
            [0] * delta_ob_len + [traj_advantage] * len(ac_with_logprobs.tokens)
        )
        SequenceAccumulator.mask.extend([0.0] * delta_ob_len + [1.0] * len(ac_with_logprobs.tokens))

    if SequenceAccumulator.full_sequence:
        data.append(make_datum_from_state())

    return data


def assemble_training_data(
    trajectory_groups_P: list[TrajectoryGroup],
    advantages_P: list[torch.Tensor],
) -> tuple[list[tinker.Datum], list[dict[str, int]]]:
    """Convert trajectories to training data format.

    Args:
        trajectory_groups_P (list[TrajectoryGroup]): Groups of trajectories
            to convert into training datums.
        advantages_P (list[torch.Tensor]): Per-group advantage tensors,
            one per trajectory group, as returned by :func:`compute_advantages`.

    Returns:
        tuple[list[tinker.Datum], list[dict[str, int]]]: A flat list of
            training datums and a parallel list of metadata dicts mapping
            each datum back to its ``group_idx`` and ``traj_idx``.
    """
    data_D: list[tinker.Datum] = []
    metadata_D: list[dict[str, int]] = []

    for i_group, (traj_group, advantages_G) in enumerate(
        safezip(trajectory_groups_P, advantages_P)
    ):
        for i_traj, (traj, traj_advantage) in enumerate(
            safezip(traj_group.trajectories_G, advantages_G)
        ):
            # Build the full sequence from the trajectory
            new_data = trajectory_to_data(traj, float(traj_advantage))
            data_D.extend(new_data)
            metadata_D.extend([{"group_idx": i_group, "traj_idx": i_traj} for _ in new_data])

    return data_D, metadata_D


def remove_constant_reward_groups(
    trajectory_groups_P: list[TrajectoryGroup],
) -> list[TrajectoryGroup]:
    """Filter out trajectory groups where all trajectories received the same reward.

    Groups with uniform rewards produce zero advantage for every trajectory,
    contributing no gradient signal.  Removing them avoids wasted compute.
    If *all* groups are uniform, a single group is returned so downstream code
    that expects a non-empty list does not break.

    Args:
        trajectory_groups_P (list[TrajectoryGroup]): Groups of trajectories
            to filter.

    Returns:
        list[TrajectoryGroup]: The subset of groups that contain at least two
            distinct reward values, or a singleton list if every group was
            uniform.
    """
    new_groups: list[TrajectoryGroup] = []
    for group in trajectory_groups_P:
        if not all_same(group.get_total_rewards()):
            new_groups.append(group)
    if not new_groups:
        logger.warning("All rewards are uniform. There will be no gradient")
        return trajectory_groups_P[0:1]  # return singleton list in case empty
        # list will cause problems
    return new_groups


def filter_low_variance_groups(
    trajectory_groups_P: list[TrajectoryGroup],
    min_reward_std: float,
    max_filter_ratio: float,
    env_group_builders_P: list[EnvGroupBuilder] | None = None,
) -> tuple[list[TrajectoryGroup], int, list[EnvGroupBuilder] | None]:
    """Filter trajectory groups with reward standard deviation below a threshold.

    This implements the DAPO-style dynamic sampling filter: groups where the
    reward standard deviation is at or below ``min_reward_std`` provide little
    learning signal and are candidates for removal. At most
    ``max_filter_ratio`` of groups will be filtered to ensure the batch does
    not become too small.

    When ``env_group_builders_P`` is provided, it is filtered in lockstep so
    that the returned builders remain aligned with the returned trajectory
    groups.

    Args:
        trajectory_groups_P (list[TrajectoryGroup]): Groups of trajectories
            to filter.
        min_reward_std (float): Minimum reward standard deviation to keep a
            group. Groups with std <= this value are candidates for filtering.
        max_filter_ratio (float): Maximum fraction of groups that may be
            filtered. Must be in [0, 1). E.g. 0.5 means at most half the
            groups can be removed.
        env_group_builders_P (list | None): Optional parallel list of
            EnvGroupBuilder instances to filter in lockstep. If provided, must
            have the same length as ``trajectory_groups_P``.

    Returns:
        tuple[list[TrajectoryGroup], int, list | None]: A tuple of
            (filtered trajectory groups, number of groups removed, filtered
            builders or None). If all groups would be filtered, returns the
            original lists with 0 filtered.
    """
    if not (0.0 <= max_filter_ratio < 1.0):
        raise ConfigurationError(
            f"max_filter_ratio must be in [0, 1), got {max_filter_ratio}"
        )
    if env_group_builders_P is not None:
        if len(env_group_builders_P) != len(trajectory_groups_P):
            raise ConfigurationError(
                f"env_group_builders_P length ({len(env_group_builders_P)}) must match "
                f"trajectory_groups_P length ({len(trajectory_groups_P)})"
            )

    # Compute per-group reward std and classify as keep/filter by index
    keep_indices: list[int] = []
    filter_candidate_indices: list[int] = []
    for i, group in enumerate(trajectory_groups_P):
        rewards = group.get_total_rewards()
        std = float(torch.tensor(rewards).std().item()) if len(rewards) > 1 else 0.0
        if std <= min_reward_std:
            filter_candidate_indices.append(i)
        else:
            keep_indices.append(i)

    # Cap the number of groups we actually filter
    max_to_filter = int(len(trajectory_groups_P) * max_filter_ratio)
    num_to_filter = min(len(filter_candidate_indices), max_to_filter)

    if num_to_filter == 0:
        return trajectory_groups_P, 0, env_group_builders_P

    # Keep some of the candidate groups if we'd exceed the filter cap
    num_to_keep_from_candidates = len(filter_candidate_indices) - num_to_filter
    keep_indices.extend(filter_candidate_indices[:num_to_keep_from_candidates])

    if not keep_indices:
        logger.warning(
            "Dynamic sampling: all groups have low reward variance. "
            "Returning original groups to avoid empty batch."
        )
        return trajectory_groups_P, 0, env_group_builders_P

    # Sort to preserve original ordering
    keep_indices.sort()
    kept_groups = [trajectory_groups_P[i] for i in keep_indices]
    kept_builders = (
        [env_group_builders_P[i] for i in keep_indices]
        if env_group_builders_P is not None
        else None
    )

    return kept_groups, num_to_filter, kept_builders
