"""Rollout transition dataclasses and PPO buffer construction."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .ppo_config import PLANNER_MODE_BALANCED, PPOTrainingConfig
from .ppo_state import (
    ACTION_FEATURE_DIM,
    GLOBAL_FEATURE_DIM,
    EpisodeContext,
    EpisodeStaticPolicyFeatures,
    _build_episode_static_policy_features,
    _compute_state_summary_from_mask,
    _scale_compact_streak_feature,
    _zscore_1d,
)

EpisodeAdvantageBonusFn = Callable[..., np.ndarray | None]


@dataclass(frozen=True)
class EpisodeStep:
    """One transition sampled during rollout."""

    packed_membership_mask: np.ndarray
    step_index: int
    action: int
    reward: float
    done: bool
    old_log_prob: float
    old_value: float
    planner_mode: int = PLANNER_MODE_BALANCED
    compact_streak: int = 0
    low_level_train: bool = True


@dataclass(frozen=True)
class PlannerStep:
    """One high-level planner decision sampled at a checkpoint state."""

    packed_membership_mask: np.ndarray
    step_index: int
    mode: int
    old_log_prob: float
    compact_streak: int = 0


@dataclass(frozen=True)
class EpisodeTrajectory:
    """One variable-length trajectory (one cell episode)."""

    episode_slot: int
    steps: tuple[EpisodeStep, ...]
    total_reward: float
    planner_steps: tuple[PlannerStep, ...] = tuple()


@dataclass(frozen=True)
class RolloutTransition:
    """Flattened transition used by PPO updates."""

    episode_slot: int
    packed_membership_mask: np.ndarray
    step_index: int
    action: int
    reward: float
    done: bool
    old_log_prob: float
    old_value: float
    return_t: float
    advantage: float
    planner_mode: int = PLANNER_MODE_BALANCED
    compact_streak: int = 0


@dataclass(frozen=True)
class PlannerRolloutTransition:
    """Flattened high-level planner transition used by PPO updates."""

    episode_slot: int
    packed_membership_mask: np.ndarray
    step_index: int
    mode: int
    old_log_prob: float
    advantage: float
    compact_streak: int = 0


@dataclass(frozen=True)
class RolloutFeatureCache:
    """Cached transition features reused across PPO epochs/minibatches."""

    episode_static: tuple[EpisodeStaticPolicyFeatures, ...]
    episode_slot: np.ndarray  # (N,), int32
    packed_membership_masks: tuple[np.ndarray, ...]  # each packed uint8 bitset
    assigned_frac: np.ndarray  # (N,), float32
    step_frac: np.ndarray  # (N,), float32
    remaining_frac: np.ndarray  # (N,), float32
    grow_ratio_scaled: np.ndarray  # (N,), float32
    positive_frontier_fraction: np.ndarray  # (N,), float32
    centroid_drift_scaled: np.ndarray  # (N,), float32
    compactness_proxy: np.ndarray  # (N,), float32
    compact_streak_scaled: np.ndarray  # (N,), float32
    frontier_add_reward_topk_mean: np.ndarray  # (N,), float32
    frontier_add_reward_mean: np.ndarray  # (N,), float32
    frontier_add_reward_std: np.ndarray  # (N,), float32
    frontier_add_reward_max: np.ndarray  # (N,), float32
    assigned_ll_mean: np.ndarray  # (N,), float32
    assigned_ll_max: np.ndarray  # (N,), float32
    actions: np.ndarray  # (N,), int64
    planner_modes: np.ndarray  # (N,), int64
    old_log_probs: np.ndarray  # (N,), float32
    returns: np.ndarray  # (N,), float32
    advantages: np.ndarray  # (N,), float32

    @property
    def n_transitions(self) -> int:
        return int(self.actions.shape[0])


@dataclass(frozen=True)
class PlannerRolloutFeatureCache:
    """Cached high-level planner features reused across PPO epochs/minibatches."""

    episode_static: tuple[EpisodeStaticPolicyFeatures, ...]
    episode_slot: np.ndarray  # (N,), int32
    packed_membership_masks: tuple[np.ndarray, ...]
    assigned_frac: np.ndarray  # (N,), float32
    step_frac: np.ndarray  # (N,), float32
    remaining_frac: np.ndarray  # (N,), float32
    grow_ratio_scaled: np.ndarray  # (N,), float32
    positive_frontier_fraction: np.ndarray  # (N,), float32
    centroid_drift_scaled: np.ndarray  # (N,), float32
    compactness_proxy: np.ndarray  # (N,), float32
    compact_streak_scaled: np.ndarray  # (N,), float32
    frontier_add_reward_topk_mean: np.ndarray  # (N,), float32
    frontier_add_reward_mean: np.ndarray  # (N,), float32
    frontier_add_reward_std: np.ndarray  # (N,), float32
    frontier_add_reward_max: np.ndarray  # (N,), float32
    assigned_ll_mean: np.ndarray  # (N,), float32
    assigned_ll_max: np.ndarray  # (N,), float32
    actions: np.ndarray  # (N,), int64
    old_log_probs: np.ndarray  # (N,), float32
    advantages: np.ndarray  # (N,), float32

    @property
    def n_transitions(self) -> int:
        return int(self.actions.shape[0])


def _pack_mask(mask: np.ndarray) -> np.ndarray:
    bits = np.asarray(mask, dtype=np.uint8)
    return np.packbits(bits, bitorder="little")


def _unpack_mask(packed: np.ndarray, n_bits: int) -> np.ndarray:
    out = np.unpackbits(np.asarray(packed, dtype=np.uint8), count=int(n_bits), bitorder="little")
    return out.astype(np.uint8, copy=False)


def compute_discounted_returns(rewards: np.ndarray, gamma: float) -> np.ndarray:
    """Compute Monte-Carlo discounted returns for one episode."""
    r = np.asarray(rewards, dtype=np.float64)
    if r.ndim != 1:
        raise ValueError("rewards must be a 1D array")
    if not (0.0 < gamma <= 1.0):
        raise ValueError("gamma must be in (0, 1]")

    out = np.zeros_like(r, dtype=np.float64)
    running = 0.0
    for i in range(len(r) - 1, -1, -1):
        running = float(r[i]) + gamma * running
        out[i] = running
    return out


def compute_gae_returns_and_advantages(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute GAE(lambda) advantages and returns for one episode."""
    r = np.asarray(rewards, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    d = np.asarray(dones, dtype=bool)
    if r.ndim != 1 or v.ndim != 1 or d.ndim != 1:
        raise ValueError("rewards, values, and dones must be 1D arrays")
    if r.shape != v.shape or r.shape != d.shape:
        raise ValueError("rewards, values, and dones must have the same shape")
    if not (0.0 < gamma <= 1.0):
        raise ValueError("gamma must be in (0, 1]")
    if not (0.0 <= gae_lambda <= 1.0):
        raise ValueError("gae_lambda must be in [0, 1]")

    advantages = np.zeros_like(r, dtype=np.float64)
    gae = 0.0
    next_value = 0.0
    for i in range(len(r) - 1, -1, -1):
        nonterminal = 0.0 if bool(d[i]) else 1.0
        delta = float(r[i]) + float(gamma) * next_value * nonterminal - float(v[i])
        gae = delta + float(gamma) * float(gae_lambda) * nonterminal * gae
        advantages[i] = gae
        next_value = float(v[i])
    returns = advantages + v
    return returns, advantages


def compute_advantages(returns: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Compute advantages as A_t = R_t - V(s_t)."""
    ret = np.asarray(returns, dtype=np.float64)
    val = np.asarray(values, dtype=np.float64)
    if ret.shape != val.shape:
        raise ValueError("returns and values must have same shape")
    return ret - val


def _compute_group_relative_episode_advantages(
    trajectories: list["EpisodeTrajectory"],
    *,
    group_size: int,
    norm_epsilon: float,
    score: str,
) -> np.ndarray:
    """Compute one standardized same-cell scalar per trajectory."""
    if group_size <= 1:
        raise ValueError("group_size must be > 1")
    if norm_epsilon <= 0:
        raise ValueError("norm_epsilon must be > 0")
    if score != "episode_total_reward":
        raise ValueError(f"unsupported group-relative score: {score!r}")
    if len(trajectories) % int(group_size) != 0:
        raise ValueError("trajectory count must be divisible by group_size")

    bonuses = np.zeros((len(trajectories),), dtype=np.float64)
    for start in range(0, len(trajectories), int(group_size)):
        stop = start + int(group_size)
        scores = np.asarray([trajectories[i].total_reward for i in range(start, stop)], dtype=np.float64)
        mu = float(np.mean(scores))
        sigma = float(np.std(scores, ddof=0))
        bonuses[start:stop] = (scores - mu) / (sigma + float(norm_epsilon))
    return bonuses


def _batch_observations_for_rollout(
    observations: list[dict[str, Any]],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad variable-size action tensors into one batched forward input."""
    n = len(observations)
    if n == 0:
        raise ValueError("cannot batch empty observations list")

    max_actions = max(int(np.asarray(obs["action_features"]).shape[0]) for obs in observations)
    global_batch = np.zeros((n, GLOBAL_FEATURE_DIM), dtype=np.float32)
    action_batch = np.zeros((n, max_actions, ACTION_FEATURE_DIM), dtype=np.float32)
    mask_batch = np.zeros((n, max_actions), dtype=bool)

    for i, obs in enumerate(observations):
        global_features = np.asarray(obs["global_features"], dtype=np.float32)
        action_features = np.asarray(obs["action_features"], dtype=np.float32)
        action_mask = np.asarray(obs["action_mask"], dtype=bool)
        n_actions = int(action_features.shape[0])

        global_batch[i] = global_features
        action_batch[i, :n_actions, :] = action_features
        mask_batch[i, :n_actions] = action_mask

    return (
        torch.as_tensor(global_batch, device=device, dtype=torch.float32),
        torch.as_tensor(action_batch, device=device, dtype=torch.float32),
        torch.as_tensor(mask_batch, device=device, dtype=torch.bool),
    )


def _batch_global_features_for_rollout(
    observations: list[dict[str, Any]],
    *,
    device: torch.device,
) -> torch.Tensor:
    """Batch only compact state-level features for vectorized rollout."""
    n = len(observations)
    if n == 0:
        raise ValueError("cannot batch empty observations list")

    global_batch = np.zeros((n, GLOBAL_FEATURE_DIM), dtype=np.float32)
    for i, obs in enumerate(observations):
        global_batch[i] = np.asarray(obs["global_features"], dtype=np.float32)
    return torch.as_tensor(global_batch, device=device, dtype=torch.float32)


def _build_rollout_buffer(
    *,
    trajectories: list[EpisodeTrajectory],
    gamma: float,
    gae_lambda: float,
    normalize_returns_per_episode: bool,
    normalize_advantages: bool,
    group_relative_enabled: bool,
    group_relative_group_size: int,
    group_relative_mix_alpha: float,
    group_relative_norm_epsilon: float,
    group_relative_score: str,
    training_mode: str = "ppo",
    episode_contexts: list[EpisodeContext] | None = None,
    config: PPOTrainingConfig | None = None,
    episode_advantage_bonus: np.ndarray | None = None,
    episode_advantage_bonus_fn: EpisodeAdvantageBonusFn | None = None,
) -> list[RolloutTransition]:
    out: list[RolloutTransition] = []
    mode = str(training_mode).strip().lower()
    if episode_advantage_bonus is not None:
        group_relative_bonus = episode_advantage_bonus
    elif episode_advantage_bonus_fn is not None:
        group_relative_bonus = episode_advantage_bonus_fn(
            trajectories=trajectories,
            group_relative_enabled=group_relative_enabled,
            group_relative_group_size=group_relative_group_size,
            group_relative_norm_epsilon=group_relative_norm_epsilon,
            group_relative_score=group_relative_score,
            training_mode=training_mode,
            episode_contexts=episode_contexts,
            config=config,
        )
    elif mode == "full_grpo":
        raise ValueError("full_grpo rollout buffer requires episode advantage bonuses")
    elif group_relative_enabled and trajectories:
        group_relative_bonus = _compute_group_relative_episode_advantages(
            trajectories=trajectories,
            group_size=group_relative_group_size,
            norm_epsilon=group_relative_norm_epsilon,
            score=group_relative_score,
        )
    else:
        group_relative_bonus = None
    for traj_idx, traj in enumerate(trajectories):
        train_steps = tuple(s for s in traj.steps if bool(s.low_level_train))
        rewards = np.asarray([s.reward for s in train_steps], dtype=np.float64)
        values = np.asarray([s.old_value for s in train_steps], dtype=np.float64)
        dones = np.asarray([s.done for s in train_steps], dtype=bool)
        if mode == "full_grpo":
            returns = np.zeros_like(rewards, dtype=np.float64)
            advantages = np.full_like(rewards, fill_value=float(group_relative_bonus[traj_idx]), dtype=np.float64)
        else:
            returns, advantages = compute_gae_returns_and_advantages(
                rewards,
                values,
                dones,
                gamma=gamma,
                gae_lambda=gae_lambda,
            )
            if normalize_returns_per_episode and returns.size > 1:
                returns = _zscore_1d(returns)
            if group_relative_bonus is not None:
                advantages = (
                    (1.0 - float(group_relative_mix_alpha)) * advantages
                    + float(group_relative_mix_alpha) * float(group_relative_bonus[traj_idx])
                )
        for i, step in enumerate(train_steps):
            out.append(
                RolloutTransition(
                    episode_slot=int(traj.episode_slot),
                    packed_membership_mask=step.packed_membership_mask,
                    step_index=int(step.step_index),
                    action=int(step.action),
                    reward=float(step.reward),
                    done=bool(step.done),
                    old_log_prob=float(step.old_log_prob),
                    old_value=float(step.old_value),
                    return_t=float(returns[i]),
                    advantage=float(advantages[i]),
                    planner_mode=int(step.planner_mode),
                    compact_streak=int(step.compact_streak),
                )
            )

    if normalize_advantages and out:
        adv = np.asarray([t.advantage for t in out], dtype=np.float64)
        adv = _zscore_1d(adv)
        out = [
            RolloutTransition(
                episode_slot=t.episode_slot,
                packed_membership_mask=t.packed_membership_mask,
                step_index=t.step_index,
                action=t.action,
                reward=t.reward,
                done=t.done,
                old_log_prob=t.old_log_prob,
                old_value=t.old_value,
                return_t=t.return_t,
                advantage=float(adv[i]),
                planner_mode=t.planner_mode,
                compact_streak=t.compact_streak,
            )
            for i, t in enumerate(out)
        ]
    return out


def _build_planner_rollout_buffer(
    *,
    trajectories: list[EpisodeTrajectory],
    episode_advantage_bonus: np.ndarray | None,
) -> list[PlannerRolloutTransition]:
    if episode_advantage_bonus is None:
        return []
    out: list[PlannerRolloutTransition] = []
    for traj_idx, traj in enumerate(trajectories):
        for planner_step in traj.planner_steps:
            out.append(
                PlannerRolloutTransition(
                    episode_slot=int(traj.episode_slot),
                    packed_membership_mask=np.asarray(planner_step.packed_membership_mask, dtype=np.uint8).copy(),
                    step_index=int(planner_step.step_index),
                    mode=int(planner_step.mode),
                    old_log_prob=float(planner_step.old_log_prob),
                    advantage=float(episode_advantage_bonus[traj_idx]),
                    compact_streak=int(planner_step.compact_streak),
                )
            )
    return out


def _build_rollout_feature_cache(
    *,
    transitions: list[RolloutTransition],
    episode_contexts: list[EpisodeContext],
) -> RolloutFeatureCache:
    """Precompute transition-dependent features once per outer PPO update."""
    episode_static = tuple(_build_episode_static_policy_features(ctx) for ctx in episode_contexts)
    n = len(transitions)
    if n == 0:
        return RolloutFeatureCache(
            episode_static=episode_static,
            episode_slot=np.zeros((0,), dtype=np.int32),
            packed_membership_masks=tuple(),
            assigned_frac=np.zeros((0,), dtype=np.float32),
            step_frac=np.zeros((0,), dtype=np.float32),
            remaining_frac=np.zeros((0,), dtype=np.float32),
            grow_ratio_scaled=np.zeros((0,), dtype=np.float32),
            positive_frontier_fraction=np.zeros((0,), dtype=np.float32),
            centroid_drift_scaled=np.zeros((0,), dtype=np.float32),
            compactness_proxy=np.zeros((0,), dtype=np.float32),
            compact_streak_scaled=np.zeros((0,), dtype=np.float32),
            frontier_add_reward_topk_mean=np.zeros((0,), dtype=np.float32),
            frontier_add_reward_mean=np.zeros((0,), dtype=np.float32),
            frontier_add_reward_std=np.zeros((0,), dtype=np.float32),
            frontier_add_reward_max=np.zeros((0,), dtype=np.float32),
            assigned_ll_mean=np.zeros((0,), dtype=np.float32),
            assigned_ll_max=np.zeros((0,), dtype=np.float32),
            actions=np.zeros((0,), dtype=np.int64),
            planner_modes=np.zeros((0,), dtype=np.int64),
            old_log_probs=np.zeros((0,), dtype=np.float32),
            returns=np.zeros((0,), dtype=np.float32),
            advantages=np.zeros((0,), dtype=np.float32),
        )

    episode_slot = np.zeros((n,), dtype=np.int32)
    packed_membership_masks: list[np.ndarray] = []
    assigned_frac = np.zeros((n,), dtype=np.float32)
    step_frac = np.zeros((n,), dtype=np.float32)
    remaining_frac = np.zeros((n,), dtype=np.float32)
    grow_ratio_scaled = np.zeros((n,), dtype=np.float32)
    positive_frontier_fraction = np.zeros((n,), dtype=np.float32)
    centroid_drift_scaled = np.zeros((n,), dtype=np.float32)
    compactness_proxy = np.zeros((n,), dtype=np.float32)
    compact_streak_scaled = np.zeros((n,), dtype=np.float32)
    frontier_add_reward_topk_mean = np.zeros((n,), dtype=np.float32)
    frontier_add_reward_mean = np.zeros((n,), dtype=np.float32)
    frontier_add_reward_std = np.zeros((n,), dtype=np.float32)
    frontier_add_reward_max = np.zeros((n,), dtype=np.float32)
    assigned_ll_mean = np.zeros((n,), dtype=np.float32)
    assigned_ll_max = np.zeros((n,), dtype=np.float32)
    actions = np.zeros((n,), dtype=np.int64)
    planner_modes = np.zeros((n,), dtype=np.int64)
    old_log_probs = np.zeros((n,), dtype=np.float32)
    returns = np.zeros((n,), dtype=np.float32)
    advantages = np.zeros((n,), dtype=np.float32)

    for i, t in enumerate(transitions):
        ep = int(t.episode_slot)
        ctx = episode_contexts[ep]
        packed_membership = np.asarray(t.packed_membership_mask, dtype=np.uint8).copy()
        membership = _unpack_mask(packed_membership, n_bits=ctx.n_bins)
        summary = _compute_state_summary_from_mask(ctx=ctx, membership_mask=membership, step_index=int(t.step_index))

        episode_slot[i] = ep
        packed_membership_masks.append(packed_membership)
        assigned_frac[i] = np.float32(summary["assigned_frac"])
        step_frac[i] = np.float32(summary["step_frac"])
        remaining_frac[i] = np.float32(summary["remaining_frac"])
        grow_ratio_scaled[i] = np.float32(summary["grow_ratio_scaled"])
        positive_frontier_fraction[i] = np.float32(summary["positive_frontier_fraction"])
        centroid_drift_scaled[i] = np.float32(summary["centroid_drift_scaled"])
        compactness_proxy[i] = np.float32(summary["compactness_proxy"])
        compact_streak_scaled[i] = np.float32(_scale_compact_streak_feature(int(t.compact_streak)))
        frontier_add_reward_topk_mean[i] = np.float32(summary["frontier_add_reward_topk_mean"])
        frontier_add_reward_mean[i] = np.float32(summary["frontier_add_reward_mean"])
        frontier_add_reward_std[i] = np.float32(summary["frontier_add_reward_std"])
        frontier_add_reward_max[i] = np.float32(summary["frontier_add_reward_max"])
        assigned_ll_mean[i] = np.float32(summary["assigned_ll_mean"])
        assigned_ll_max[i] = np.float32(summary["assigned_ll_max"])
        actions[i] = int(t.action)
        planner_modes[i] = int(t.planner_mode)
        old_log_probs[i] = np.float32(t.old_log_prob)
        returns[i] = np.float32(t.return_t)
        advantages[i] = np.float32(t.advantage)

    return RolloutFeatureCache(
        episode_static=episode_static,
        episode_slot=episode_slot,
        packed_membership_masks=tuple(packed_membership_masks),
        assigned_frac=assigned_frac,
        step_frac=step_frac,
        remaining_frac=remaining_frac,
        grow_ratio_scaled=grow_ratio_scaled,
        positive_frontier_fraction=positive_frontier_fraction,
        centroid_drift_scaled=centroid_drift_scaled,
        compactness_proxy=compactness_proxy,
        compact_streak_scaled=compact_streak_scaled,
        frontier_add_reward_topk_mean=frontier_add_reward_topk_mean,
        frontier_add_reward_mean=frontier_add_reward_mean,
        frontier_add_reward_std=frontier_add_reward_std,
        frontier_add_reward_max=frontier_add_reward_max,
        assigned_ll_mean=assigned_ll_mean,
        assigned_ll_max=assigned_ll_max,
        actions=actions,
        planner_modes=planner_modes,
        old_log_probs=old_log_probs,
        returns=returns,
        advantages=advantages,
    )


def _build_planner_rollout_feature_cache(
    *,
    transitions: list[PlannerRolloutTransition],
    episode_contexts: list[EpisodeContext],
) -> PlannerRolloutFeatureCache:
    episode_static = tuple(_build_episode_static_policy_features(ctx) for ctx in episode_contexts)
    n = len(transitions)
    if n == 0:
        return PlannerRolloutFeatureCache(
            episode_static=episode_static,
            episode_slot=np.zeros((0,), dtype=np.int32),
            packed_membership_masks=tuple(),
            assigned_frac=np.zeros((0,), dtype=np.float32),
            step_frac=np.zeros((0,), dtype=np.float32),
            remaining_frac=np.zeros((0,), dtype=np.float32),
            grow_ratio_scaled=np.zeros((0,), dtype=np.float32),
            positive_frontier_fraction=np.zeros((0,), dtype=np.float32),
            centroid_drift_scaled=np.zeros((0,), dtype=np.float32),
            compactness_proxy=np.zeros((0,), dtype=np.float32),
            compact_streak_scaled=np.zeros((0,), dtype=np.float32),
            frontier_add_reward_topk_mean=np.zeros((0,), dtype=np.float32),
            frontier_add_reward_mean=np.zeros((0,), dtype=np.float32),
            frontier_add_reward_std=np.zeros((0,), dtype=np.float32),
            frontier_add_reward_max=np.zeros((0,), dtype=np.float32),
            assigned_ll_mean=np.zeros((0,), dtype=np.float32),
            assigned_ll_max=np.zeros((0,), dtype=np.float32),
            actions=np.zeros((0,), dtype=np.int64),
            old_log_probs=np.zeros((0,), dtype=np.float32),
            advantages=np.zeros((0,), dtype=np.float32),
        )

    episode_slot = np.zeros((n,), dtype=np.int32)
    packed_membership_masks: list[np.ndarray] = []
    assigned_frac = np.zeros((n,), dtype=np.float32)
    step_frac = np.zeros((n,), dtype=np.float32)
    remaining_frac = np.zeros((n,), dtype=np.float32)
    grow_ratio_scaled = np.zeros((n,), dtype=np.float32)
    positive_frontier_fraction = np.zeros((n,), dtype=np.float32)
    centroid_drift_scaled = np.zeros((n,), dtype=np.float32)
    compactness_proxy = np.zeros((n,), dtype=np.float32)
    compact_streak_scaled = np.zeros((n,), dtype=np.float32)
    frontier_add_reward_topk_mean = np.zeros((n,), dtype=np.float32)
    frontier_add_reward_mean = np.zeros((n,), dtype=np.float32)
    frontier_add_reward_std = np.zeros((n,), dtype=np.float32)
    frontier_add_reward_max = np.zeros((n,), dtype=np.float32)
    assigned_ll_mean = np.zeros((n,), dtype=np.float32)
    assigned_ll_max = np.zeros((n,), dtype=np.float32)
    actions = np.zeros((n,), dtype=np.int64)
    old_log_probs = np.zeros((n,), dtype=np.float32)
    advantages = np.zeros((n,), dtype=np.float32)

    for i, t in enumerate(transitions):
        ep = int(t.episode_slot)
        ctx = episode_contexts[ep]
        packed_membership = np.asarray(t.packed_membership_mask, dtype=np.uint8).copy()
        membership = _unpack_mask(packed_membership, n_bits=ctx.n_bins)
        summary = _compute_state_summary_from_mask(ctx=ctx, membership_mask=membership, step_index=int(t.step_index))

        episode_slot[i] = ep
        packed_membership_masks.append(packed_membership)
        assigned_frac[i] = np.float32(summary["assigned_frac"])
        step_frac[i] = np.float32(summary["step_frac"])
        remaining_frac[i] = np.float32(summary["remaining_frac"])
        grow_ratio_scaled[i] = np.float32(summary["grow_ratio_scaled"])
        positive_frontier_fraction[i] = np.float32(summary["positive_frontier_fraction"])
        centroid_drift_scaled[i] = np.float32(summary["centroid_drift_scaled"])
        compactness_proxy[i] = np.float32(summary["compactness_proxy"])
        compact_streak_scaled[i] = np.float32(_scale_compact_streak_feature(int(t.compact_streak)))
        frontier_add_reward_topk_mean[i] = np.float32(summary["frontier_add_reward_topk_mean"])
        frontier_add_reward_mean[i] = np.float32(summary["frontier_add_reward_mean"])
        frontier_add_reward_std[i] = np.float32(summary["frontier_add_reward_std"])
        frontier_add_reward_max[i] = np.float32(summary["frontier_add_reward_max"])
        assigned_ll_mean[i] = np.float32(summary["assigned_ll_mean"])
        assigned_ll_max[i] = np.float32(summary["assigned_ll_max"])
        actions[i] = int(t.mode)
        old_log_probs[i] = np.float32(t.old_log_prob)
        advantages[i] = np.float32(t.advantage)

    return PlannerRolloutFeatureCache(
        episode_static=episode_static,
        episode_slot=episode_slot,
        packed_membership_masks=tuple(packed_membership_masks),
        assigned_frac=assigned_frac,
        step_frac=step_frac,
        remaining_frac=remaining_frac,
        grow_ratio_scaled=grow_ratio_scaled,
        positive_frontier_fraction=positive_frontier_fraction,
        centroid_drift_scaled=centroid_drift_scaled,
        compactness_proxy=compactness_proxy,
        compact_streak_scaled=compact_streak_scaled,
        frontier_add_reward_topk_mean=frontier_add_reward_topk_mean,
        frontier_add_reward_mean=frontier_add_reward_mean,
        frontier_add_reward_std=frontier_add_reward_std,
        frontier_add_reward_max=frontier_add_reward_max,
        assigned_ll_mean=assigned_ll_mean,
        assigned_ll_max=assigned_ll_max,
        actions=actions,
        old_log_probs=old_log_probs,
        advantages=advantages,
    )
