"""Full-GRPO and planner COT objective helpers.

This module keeps the experimental group-ranking objective separate from PPO
training orchestration. It intentionally receives PPO-local helpers as callbacks
so it does not import ``ppo_training`` and create circular dependencies.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

_PLANNER_MODE_STOP = 0
_PLANNER_MODE_COMPACT = 1
_PLANNER_MODE_EXPLORE = 3


def _compute_full_grpo_episode_scores(
    trajectories: Sequence[Any],
    episode_contexts: Sequence[Any],
    config: Any,
    *,
    final_membership_mask_fn: Callable[..., np.ndarray],
    evidence_state_fn: Callable[..., dict[str, float]],
    key_node_score_fn: Callable[..., float],
) -> np.ndarray:
    """Compute no-GT terminal scores used for strict full-GRPO ranking."""
    scores = np.zeros((len(trajectories),), dtype=np.float64)
    for i, traj in enumerate(trajectories):
        ctx = episode_contexts[int(traj.episode_slot)]
        final_mask = final_membership_mask_fn(ctx=ctx, trajectory=traj)
        mean_step_reward = float(traj.total_reward / max(1, len(traj.steps)))
        evidence = evidence_state_fn(ctx=ctx, membership_mask=final_mask, config=config)
        stopped = bool(traj.steps and int(traj.steps[-1].action) == 0)
        stop_score = _full_grpo_stop_score(
            stopped=stopped,
            frontier_quality=float(evidence["frontier_quality"]),
            overgrowth_risk=float(evidence["overgrowth_risk"]),
        )

        scores[i] = (
            float(config.full_grpo_reward_weight) * mean_step_reward
            + float(config.full_grpo_evidence_growth_weight) * float(evidence["frontier_quality"])
            + float(config.full_grpo_stop_weight) * stop_score
            - float(config.full_grpo_overgrowth_weight) * float(evidence["overgrowth_risk"])
            + float(config.full_grpo_compact_weight) * float(evidence["compactness"])
        )
        if bool(config.planner_enabled) and traj.planner_steps:
            node_scores = [
                key_node_score_fn(
                    ctx=ctx,
                    planner_step=planner_step,
                    config=config,
                )
                for planner_step in traj.planner_steps
            ]
            scores[i] += float(config.planner_cot_weight) * float(np.mean(node_scores))
    return scores


def _compute_full_grpo_episode_advantages(
    trajectories: Sequence[Any],
    episode_contexts: Sequence[Any],
    *,
    group_size: int,
    norm_epsilon: float,
    config: Any,
    score_fn: Callable[[Sequence[Any], Sequence[Any], Any], np.ndarray] | None = None,
    final_membership_mask_fn: Callable[..., np.ndarray] | None = None,
    evidence_state_fn: Callable[..., dict[str, float]] | None = None,
    key_node_score_fn: Callable[..., float] | None = None,
) -> np.ndarray:
    """Standardize full-GRPO terminal scores within each same-cell group."""
    if score_fn is None:
        if final_membership_mask_fn is None or evidence_state_fn is None or key_node_score_fn is None:
            raise ValueError("score_fn or all full-GRPO helper callbacks must be provided")
        scores = _compute_full_grpo_episode_scores(
            trajectories,
            episode_contexts,
            config,
            final_membership_mask_fn=final_membership_mask_fn,
            evidence_state_fn=evidence_state_fn,
            key_node_score_fn=key_node_score_fn,
        )
    else:
        scores = score_fn(trajectories, episode_contexts, config)

    if len(scores) % int(group_size) != 0:
        raise ValueError("trajectory count must be divisible by group_size")
    advantages = np.zeros_like(scores, dtype=np.float64)
    for start in range(0, len(scores), int(group_size)):
        stop = start + int(group_size)
        group = scores[start:stop]
        advantages[start:stop] = (group - float(np.mean(group))) / (float(np.std(group, ddof=0)) + norm_epsilon)
    return advantages


def _full_grpo_frontier_quality(*, topk_mean: float, has_frontier: bool, config: Any) -> float:
    if not bool(has_frontier):
        return 0.0
    x = (float(topk_mean) - float(config.full_grpo_tau_frontier)) / max(
        float(config.full_grpo_frontier_temp),
        1.0e-8,
    )
    x = float(np.clip(x, -60.0, 60.0))
    return float(1.0 / (1.0 + np.exp(-x)))


def _full_grpo_evidence_state(
    *,
    ctx: Any,
    membership_mask: np.ndarray,
    config: Any,
    state_feature_bundle_fn: Callable[..., Any],
) -> dict[str, float]:
    bundle = state_feature_bundle_fn(ctx=ctx, membership_mask=membership_mask, step_index=0)
    has_frontier = bool(np.any(bundle.frontier_mask))
    frontier_quality = _full_grpo_frontier_quality(
        topk_mean=float(bundle.frontier_add_reward_topk_mean),
        has_frontier=has_frontier,
        config=config,
    )
    overgrowth_risk = float(bundle.grow_ratio_scaled) * (1.0 - frontier_quality)
    support_topk = _masked_topk_mean(bundle.neighbor_support, bundle.frontier_mask, k=int(ctx.stop_top_k))
    return {
        "frontier_quality": float(frontier_quality),
        "overgrowth_risk": float(overgrowth_risk),
        "compactness": float(bundle.compactness_proxy),
        "centroid_drift": float(bundle.centroid_drift_scaled),
        "frontier_add_reward_max": float(bundle.frontier_add_reward_max),
        "frontier_add_reward_mean": float(bundle.frontier_add_reward_mean),
        "frontier_add_reward_topk_mean": float(bundle.frontier_add_reward_topk_mean),
        "frontier_neighbor_support_topk_mean": float(support_topk),
    }


def _full_grpo_stop_score(*, stopped: bool, frontier_quality: float, overgrowth_risk: float) -> float:
    q = float(np.clip(frontier_quality, 0.0, 1.0))
    risk = float(max(overgrowth_risk, 0.0))
    if bool(stopped):
        return float(1.0 - 2.0 * q + risk)
    return float(q - risk)


def _masked_topk_mean(values: np.ndarray, mask: np.ndarray, *, k: int) -> float:
    active = np.asarray(mask, dtype=bool)
    if not np.any(active):
        return 0.0
    arr = np.asarray(values, dtype=np.float64)[active]
    if arr.size == 0:
        return 0.0
    kk = min(max(int(k), 1), int(arr.size))
    top = np.partition(arr, arr.size - kk)[arr.size - kk :]
    return float(np.mean(top))


def _full_grpo_key_node_score(
    *,
    ctx: Any,
    planner_step: Any,
    config: Any,
    unpack_mask_fn: Callable[..., np.ndarray],
    evidence_state_fn: Callable[..., dict[str, float]],
    compact_streak_scale_fn: Callable[[int], float],
) -> float:
    """Score a high-level planner checkpoint without using GT labels."""
    mask = unpack_mask_fn(planner_step.packed_membership_mask, n_bits=ctx.n_bins)
    evidence = evidence_state_fn(ctx=ctx, membership_mask=mask, config=config)
    frontier_quality = float(evidence["frontier_quality"])
    overgrowth_risk = float(evidence["overgrowth_risk"])
    compactness = float(evidence["compactness"])
    support_topk = float(evidence["frontier_neighbor_support_topk_mean"])
    compact_streak_scaled = compact_streak_scale_fn(int(planner_step.compact_streak))
    mode = int(planner_step.mode)

    if mode == _PLANNER_MODE_STOP:
        mode_score = _full_grpo_stop_score(
            stopped=True,
            frontier_quality=frontier_quality,
            overgrowth_risk=overgrowth_risk,
        )
    elif mode == _PLANNER_MODE_COMPACT:
        mode_score = (
            float(config.full_grpo_compact_weight) * compactness
            + support_topk
            - float(config.full_grpo_overgrowth_weight) * overgrowth_risk
            - float(config.full_grpo_compact_streak_weight) * compact_streak_scaled
        )
    elif mode == _PLANNER_MODE_EXPLORE:
        mode_score = (
            float(config.full_grpo_explore_weight) * frontier_quality
            + float(evidence["frontier_add_reward_max"])
            - float(evidence["centroid_drift"])
            - 0.5 * float(config.full_grpo_overgrowth_weight) * overgrowth_risk
        )
    else:
        mode_score = frontier_quality + 0.5 * compactness - 0.5 * overgrowth_risk

    evidence_score = frontier_quality - overgrowth_risk
    return float(float(config.full_grpo_evidence_growth_weight) * evidence_score + mode_score)
