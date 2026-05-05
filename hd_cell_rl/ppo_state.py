"""PPO state feature and observation construction helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .ppo_config import PPOTrainingConfig, _PLANNER_MODE_TO_INDEX
from .ppo_feature_schema import (
    ACTION_FEATURE_DIM,
    GLOBAL_FEATURE_DIM,
    A_CANDIDATE_CENTROID_DISTANCE,
    A_CANDIDATE_COMPACTNESS_GAIN,
    A_CANDIDATE_NEIGHBOR_SUPPORT,
    A_FEATURE_1,
    A_FEATURE_2,
    A_FEATURE_3,
    A_FEATURE_4,
    A_FEATURE_5,
    A_FEATURE_6,
    A_FEATURE_7,
    A_FEATURE_8,
    A_FEATURE_9,
    A_FEATURE_10,
    A_IS_STOP_ACTION,
    G_ASSIGNED_FRAC,
    G_ASSIGNED_LL_MAX,
    G_ASSIGNED_LL_MEAN,
    G_CENTROID_DRIFT_SCALED,
    G_COMPACTNESS_PROXY,
    G_COMPACT_STREAK_SCALED,
    G_FRONTIER_ADD_REWARD_MAX,
    G_FRONTIER_ADD_REWARD_MEAN,
    G_FRONTIER_ADD_REWARD_TOPK_MEAN,
    G_GROW_RATIO_SCALED,
    G_N_BINS_SCALED,
    G_POSITIVE_FRONTIER_FRACTION,
    G_REMAINING_FRAC,
    G_SEED_SIZE_SCALED,
    G_STEP_FRAC,
)
from .reward import (
    compute_frontier_eligible_mask,
    compute_neighbor_support_fraction,
    compute_stop_delta,
)


@dataclass(frozen=True)
class EpisodeContext:
    """Static per-cell tensors needed for rollout + PPO reconstruction."""

    cell_id: str
    candidate_bin_ids: tuple[str, ...]
    initial_membership_mask: np.ndarray  # (B,), uint8; nuclear seed bins already assigned at reset
    candidate_bin_xy_um: np.ndarray  # (B, 2), float32
    nucleus_center_xy_um: np.ndarray  # (2,), float32
    ll: np.ndarray  # (B, K), float32
    p_dis: np.ndarray  # (B,), float32
    p_overlap: np.ndarray  # (B,), float32
    ll_mean_z: np.ndarray  # (B,), float32
    ll_max_z: np.ndarray  # (B,), float32
    base_penalty: np.ndarray  # (B,), float32 = w2*p_dis + w3*p_overlap
    expression_confidence: np.ndarray  # (B,), float32
    bin_count_totals: np.ndarray  # (B,), float32
    neighbor_index: np.ndarray  # (B, 8), int32
    max_steps: int
    log_prior: float
    r_max_um: float
    w1: float
    w2: float
    w3: float
    w4: float
    w5: float
    stop_lambda: float
    stop_stat: str
    stop_top_k: int
    expression_confidence_pseudocount: float
    normalize_expression_zscore: bool
    zscore_delta: float

    @property
    def n_bins(self) -> int:
        return int(self.ll.shape[0])

    @property
    def n_cell_types(self) -> int:
        return int(self.ll.shape[1])


@dataclass(frozen=True)
class EpisodeStaticPolicyFeatures:
    """Per-episode static policy feature template reused across many transitions."""

    context: EpisodeContext
    action_template: np.ndarray  # (A, F), float32
    n_bins: int
    n_bins_scaled: float
    seed_size_scaled: float
    max_steps: int


@dataclass(frozen=True)
class StateFeatureBundle:
    """Dynamic state features computed once per membership mask."""

    assigned_frac: float
    step_frac: float
    remaining_frac: float
    grow_ratio_scaled: float
    positive_frontier_fraction: float
    centroid_drift_scaled: float
    compactness_proxy: float
    assigned_ll_mean: float
    assigned_ll_max: float
    frontier_add_reward_topk_mean: float
    frontier_add_reward_mean: float
    frontier_add_reward_std: float
    frontier_add_reward_max: float
    frontier_mask: np.ndarray
    neighbor_support: np.ndarray
    expr_raw: np.ndarray
    expr_term: np.ndarray
    add_rewards: np.ndarray
    candidate_centroid_distance: np.ndarray
    candidate_compactness_gain: np.ndarray
    dx_from_current_centroid_scaled: np.ndarray
    dy_from_current_centroid_scaled: np.ndarray
    radial_alignment_with_centroid_drift: np.ndarray

def _build_policy_observation_from_state(
    *,
    ctx: EpisodeContext,
    membership_mask: np.ndarray,
    step_index: int,
) -> dict[str, np.ndarray]:
    n_bins_scaled = float(np.log1p(ctx.n_bins) / 8.0)
    seed_size_scaled = _scale_seed_size_feature(int(np.sum(ctx.initial_membership_mask)))
    static_template = _build_static_action_template(ctx, n_bins_scaled, seed_size_scaled)
    mask = np.asarray(membership_mask, dtype=np.uint8)
    bundle = _compute_state_feature_bundle(ctx=ctx, membership_mask=mask, step_index=step_index)

    action_features = static_template.copy()
    action_mask = np.zeros(ctx.n_bins + 1, dtype=bool)
    action_mask[0] = True
    if ctx.n_bins > 0:
        action_mask[1:] = bundle.frontier_mask
    _fill_dynamic_action_features(
        action_features=action_features,
        ctx=ctx,
        membership_mask=mask,
        bundle=bundle,
    )
    global_features = _global_features_from_bundle(
        bundle=bundle,
        n_bins_scaled=n_bins_scaled,
        seed_size_scaled=seed_size_scaled,
    )
    return {"global_features": global_features, "action_features": action_features, "action_mask": action_mask}


def _build_episode_static_policy_features(ctx: EpisodeContext) -> EpisodeStaticPolicyFeatures:
    """Build per-episode static action template reused for all transitions."""
    n_bins = int(ctx.n_bins)
    n_bins_scaled = float(np.log1p(n_bins) / 8.0)
    seed_size_scaled = _scale_seed_size_feature(int(np.sum(ctx.initial_membership_mask)))
    template = _build_static_action_template(ctx, n_bins_scaled, seed_size_scaled)
    return EpisodeStaticPolicyFeatures(
        context=ctx,
        action_template=template,
        n_bins=n_bins,
        n_bins_scaled=n_bins_scaled,
        seed_size_scaled=seed_size_scaled,
        max_steps=int(ctx.max_steps),
    )


def _build_static_action_template(
    ctx: EpisodeContext,
    n_bins_scaled: float,
    seed_size_scaled: float,
) -> np.ndarray:
    """Build action feature matrix with static columns only.

    Dynamic fields are filled per state. Static columns keep cheap per-bin
    geometry/expression summaries so ADD scoring can use more than scalar
    distance penalties without storing new episode artifacts.
    """
    n_bins = int(ctx.n_bins)
    template = np.zeros((n_bins + 1, ACTION_FEATURE_DIM), dtype=np.float32)
    template[0, A_IS_STOP_ACTION] = np.float32(1.0)
    template[0, A_FEATURE_3] = np.float32(n_bins_scaled)
    template[0, A_FEATURE_6] = np.float32(seed_size_scaled)
    if n_bins > 0:
        template[1:, A_IS_STOP_ACTION] = np.float32(0.0)
        template[1:, A_FEATURE_1] = ctx.ll_mean_z.astype(np.float32, copy=False)
        template[1:, A_FEATURE_2] = ctx.ll_max_z.astype(np.float32, copy=False)
        template[1:, A_FEATURE_3] = ctx.p_dis.astype(np.float32, copy=False)
        template[1:, A_FEATURE_4] = ctx.p_overlap.astype(np.float32, copy=False)
    return template


def _compute_state_summary_from_mask(
    *,
    ctx: EpisodeContext,
    membership_mask: np.ndarray,
    step_index: int,
) -> dict[str, float]:
    """Compute dynamic scalar features for one state."""
    bundle = _compute_state_feature_bundle(ctx=ctx, membership_mask=membership_mask, step_index=step_index)
    return {
        "assigned_frac": bundle.assigned_frac,
        "step_frac": bundle.step_frac,
        "remaining_frac": bundle.remaining_frac,
        "grow_ratio_scaled": bundle.grow_ratio_scaled,
        "positive_frontier_fraction": bundle.positive_frontier_fraction,
        "centroid_drift_scaled": bundle.centroid_drift_scaled,
        "compactness_proxy": bundle.compactness_proxy,
        "assigned_ll_mean": bundle.assigned_ll_mean,
        "assigned_ll_max": bundle.assigned_ll_max,
        "frontier_add_reward_topk_mean": bundle.frontier_add_reward_topk_mean,
        "frontier_add_reward_mean": bundle.frontier_add_reward_mean,
        "frontier_add_reward_std": bundle.frontier_add_reward_std,
        "frontier_add_reward_max": bundle.frontier_add_reward_max,
    }


def _compute_state_feature_bundle(
    *,
    ctx: EpisodeContext,
    membership_mask: np.ndarray,
    step_index: int,
    frontier_mask: np.ndarray | None = None,
) -> StateFeatureBundle:
    """Compute all dynamic state/action features from one current mask."""
    n_bins = int(ctx.n_bins)
    mask = np.asarray(membership_mask, dtype=np.uint8)
    if mask.shape != (n_bins,):
        raise ValueError(f"membership mask shape mismatch: expected {(n_bins,)}, got {mask.shape}")

    assigned_count = int(mask.sum())
    if n_bins > 0:
        assigned_frac = float(assigned_count / n_bins)
        remaining_frac = float((n_bins - assigned_count) / n_bins)
    else:
        assigned_frac = 0.0
        remaining_frac = 0.0
    step_frac = float(step_index / max(1, int(ctx.max_steps)))
    assigned = mask.astype(bool, copy=False)

    if assigned_count > 0:
        m = mask.astype(np.float32, copy=False)
        assigned_ll_mean = float(np.dot(m, ctx.ll_mean_z) / assigned_count)
        assigned_ll_max = float(np.dot(m, ctx.ll_max_z) / assigned_count)
    else:
        assigned_ll_mean = 0.0
        assigned_ll_max = 0.0

    initial_seed_count = int(np.sum(ctx.initial_membership_mask))
    grow_ratio_scaled = _scale_grow_ratio_feature(assigned_count, initial_seed_count)
    neighbor_support = compute_neighbor_support_fraction(mask, ctx.neighbor_index).astype(np.float32, copy=False)

    if assigned_count > 0:
        compactness_proxy = float(np.mean(neighbor_support[assigned]))
        current_centroid_xy = np.mean(ctx.candidate_bin_xy_um[assigned], axis=0, dtype=np.float64)
        drift_vec = current_centroid_xy - np.asarray(ctx.nucleus_center_xy_um, dtype=np.float64)
        drift_um = float(np.sqrt(np.sum(drift_vec * drift_vec)))
        centroid_drift_scaled = float(min(max(drift_um / max(float(ctx.r_max_um), 1.0e-8), 0.0), 1.0))
    else:
        compactness_proxy = 0.0
        current_centroid_xy = np.asarray(ctx.nucleus_center_xy_um, dtype=np.float64)
        drift_vec = np.zeros((2,), dtype=np.float64)
        drift_um = 0.0
        centroid_drift_scaled = 0.0

    if frontier_mask is None:
        frontier = compute_frontier_eligible_mask(mask, ctx.neighbor_index)
    else:
        frontier = np.asarray(frontier_mask, dtype=bool)
        if frontier.shape != (n_bins,):
            raise ValueError(f"frontier_mask shape mismatch: expected {(n_bins,)}, got {frontier.shape}")

    posterior = _posterior_from_membership_mask(ctx=ctx, membership_mask=mask)
    expr_raw, expr_term, _, expr_old_term = _expression_reward_terms_per_bin(
        ctx=ctx,
        membership_mask=mask,
        posterior=posterior,
        frontier_mask=frontier,
    )
    add_rewards = (
        ctx.w1 * expr_term
        + ctx.w5 * expr_old_term
        - ctx.base_penalty
        + ctx.w4 * neighbor_support
    ).astype(np.float32, copy=False)

    if np.any(frontier):
        frontier_rewards = add_rewards[frontier].astype(np.float64, copy=False)
        positive_frontier_fraction = float(np.mean(frontier_rewards > 0.0))
        frontier_add_reward_mean = float(np.mean(frontier_rewards))
        frontier_add_reward_std = float(np.std(frontier_rewards, ddof=0))
        frontier_add_reward_max = float(np.max(frontier_rewards))
        frontier_add_reward_topk_mean = float(
            compute_stop_delta(
                add_rewards,
                frontier,
                stop_stat="topk_mean",
                stop_top_k=int(ctx.stop_top_k),
            )
        )
    else:
        positive_frontier_fraction = 0.0
        frontier_add_reward_mean = 0.0
        frontier_add_reward_std = 0.0
        frontier_add_reward_max = 0.0
        frontier_add_reward_topk_mean = 0.0

    candidate_centroid_distance, candidate_compactness_gain = _compute_dynamic_add_action_features_from_support(
        ctx=ctx,
        membership_mask=mask,
        frontier_mask=frontier,
        neighbor_support=neighbor_support,
        current_centroid_xy=current_centroid_xy,
        assigned_count=assigned_count,
    )
    dx_current = np.zeros((n_bins,), dtype=np.float32)
    dy_current = np.zeros((n_bins,), dtype=np.float32)
    radial_alignment = np.zeros((n_bins,), dtype=np.float32)
    if n_bins > 0 and np.any(frontier):
        r_max = max(float(ctx.r_max_um), 1.0e-8)
        xy = np.asarray(ctx.candidate_bin_xy_um, dtype=np.float64)
        delta_current = (xy - current_centroid_xy) / r_max
        dx_current[frontier] = np.clip(delta_current[frontier, 0], -1.0, 1.0).astype(np.float32, copy=False)
        dy_current[frontier] = np.clip(delta_current[frontier, 1], -1.0, 1.0).astype(np.float32, copy=False)
        if drift_um > 1.0e-8:
            candidate_vec = xy - current_centroid_xy
            candidate_norm = np.sqrt(np.sum(candidate_vec * candidate_vec, axis=1))
            valid = frontier & (candidate_norm > 1.0e-8)
            if np.any(valid):
                drift_unit = drift_vec / drift_um
                radial_alignment[valid] = np.clip(
                    np.sum((candidate_vec[valid] / candidate_norm[valid, None]) * drift_unit, axis=1),
                    -1.0,
                    1.0,
                ).astype(np.float32, copy=False)

    return StateFeatureBundle(
        assigned_frac=assigned_frac,
        step_frac=step_frac,
        remaining_frac=remaining_frac,
        grow_ratio_scaled=grow_ratio_scaled,
        positive_frontier_fraction=positive_frontier_fraction,
        centroid_drift_scaled=centroid_drift_scaled,
        compactness_proxy=compactness_proxy,
        assigned_ll_mean=assigned_ll_mean,
        assigned_ll_max=assigned_ll_max,
        frontier_add_reward_topk_mean=frontier_add_reward_topk_mean,
        frontier_add_reward_mean=frontier_add_reward_mean,
        frontier_add_reward_std=frontier_add_reward_std,
        frontier_add_reward_max=frontier_add_reward_max,
        frontier_mask=frontier.astype(bool, copy=False),
        neighbor_support=neighbor_support.astype(np.float32, copy=False),
        expr_raw=expr_raw.astype(np.float32, copy=False),
        expr_term=expr_term.astype(np.float32, copy=False),
        add_rewards=add_rewards.astype(np.float32, copy=False),
        candidate_centroid_distance=candidate_centroid_distance.astype(np.float32, copy=False),
        candidate_compactness_gain=candidate_compactness_gain.astype(np.float32, copy=False),
        dx_from_current_centroid_scaled=dx_current,
        dy_from_current_centroid_scaled=dy_current,
        radial_alignment_with_centroid_drift=radial_alignment,
    )


def _compute_shape_frontier_features(
    *,
    ctx: EpisodeContext,
    membership_mask: np.ndarray,
) -> tuple[float, float, float]:
    """Compute dynamic shape/frontier summaries used by STOP/value features."""
    bundle = _compute_state_feature_bundle(ctx=ctx, membership_mask=membership_mask, step_index=0)
    return bundle.positive_frontier_fraction, bundle.centroid_drift_scaled, bundle.compactness_proxy


def _compute_dynamic_add_action_features(
    *,
    ctx: EpisodeContext,
    membership_mask: np.ndarray,
    frontier_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute dynamic per-candidate ADD features from the current mask.

    Returns
    -------
    candidate_to_current_centroid_distance:
        Distance from each frontier bin to the current assigned centroid, scaled by
        ``r_max_um`` and clipped to ``[0, 1]``.
    candidate_compactness_gain:
        Change in compactness proxy if that frontier bin were added next.
    """
    bundle = _compute_state_feature_bundle(
        ctx=ctx,
        membership_mask=membership_mask,
        step_index=0,
        frontier_mask=frontier_mask,
    )
    return bundle.candidate_centroid_distance, bundle.candidate_compactness_gain


def _compute_dynamic_add_action_features_from_support(
    *,
    ctx: EpisodeContext,
    membership_mask: np.ndarray,
    frontier_mask: np.ndarray,
    neighbor_support: np.ndarray,
    current_centroid_xy: np.ndarray,
    assigned_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_bins = int(ctx.n_bins)
    candidate_to_current_centroid_distance = np.zeros((n_bins,), dtype=np.float32)
    candidate_compactness_gain = np.zeros((n_bins,), dtype=np.float32)
    if n_bins == 0:
        return candidate_to_current_centroid_distance, candidate_compactness_gain

    frontier = np.asarray(frontier_mask, dtype=bool)
    if frontier.shape != (n_bins,):
        raise ValueError(f"frontier_mask shape mismatch: expected {(n_bins,)}, got {frontier.shape}")
    if not np.any(frontier):
        return candidate_to_current_centroid_distance, candidate_compactness_gain

    xy = np.asarray(ctx.candidate_bin_xy_um, dtype=np.float64)
    centroid_dist_um = np.sqrt(np.sum((xy - np.asarray(current_centroid_xy, dtype=np.float64)) ** 2, axis=1))
    scaled_dist = centroid_dist_um / max(float(ctx.r_max_um), 1.0e-8)
    candidate_to_current_centroid_distance[frontier] = np.clip(scaled_dist[frontier], 0.0, 1.0).astype(
        np.float32,
        copy=False,
    )

    if int(assigned_count) > 0:
        mask = np.asarray(membership_mask, dtype=np.uint8)
        assigned = mask.astype(bool, copy=False)
        current_compactness_sum = float(np.sum(neighbor_support[assigned], dtype=np.float64))
        current_compactness = current_compactness_sum / float(assigned_count)
        new_compactness = (current_compactness_sum + 2.0 * neighbor_support) / float(assigned_count + 1)
        gains = new_compactness - current_compactness
        candidate_compactness_gain[frontier] = gains[frontier].astype(np.float32, copy=False)

    return candidate_to_current_centroid_distance, candidate_compactness_gain


def _fill_dynamic_action_features(
    *,
    action_features: np.ndarray,
    ctx: EpisodeContext,
    membership_mask: np.ndarray,
    bundle: StateFeatureBundle,
) -> None:
    """Fill dynamic STOP and ADD columns in-place."""
    action_features[0, A_FEATURE_1] = np.float32(bundle.assigned_frac)
    action_features[0, A_FEATURE_2] = np.float32(bundle.step_frac)
    action_features[0, A_FEATURE_4] = np.float32(bundle.assigned_ll_mean)
    action_features[0, A_FEATURE_5] = np.float32(bundle.remaining_frac)
    action_features[0, A_FEATURE_7] = np.float32(bundle.grow_ratio_scaled)
    action_features[0, A_FEATURE_8] = np.float32(bundle.positive_frontier_fraction)
    action_features[0, A_FEATURE_9] = np.float32(bundle.centroid_drift_scaled)
    action_features[0, A_FEATURE_10] = np.float32(bundle.compactness_proxy)

    n_bins = int(ctx.n_bins)
    if n_bins == 0:
        return

    mask = np.asarray(membership_mask, dtype=np.uint8)
    action_features[1:, A_FEATURE_5] = mask.astype(np.float32, copy=False)
    action_features[1:, A_CANDIDATE_CENTROID_DISTANCE] = bundle.candidate_centroid_distance
    action_features[1:, A_CANDIDATE_COMPACTNESS_GAIN] = bundle.candidate_compactness_gain
    action_features[1:, A_CANDIDATE_NEIGHBOR_SUPPORT] = bundle.neighbor_support


def _global_features_from_bundle(
    *,
    bundle: StateFeatureBundle,
    n_bins_scaled: float,
    seed_size_scaled: float,
    compact_streak_scaled: float = 0.0,
) -> np.ndarray:
    out = np.zeros((GLOBAL_FEATURE_DIM,), dtype=np.float32)
    out[G_ASSIGNED_FRAC] = np.float32(bundle.assigned_frac)
    out[G_STEP_FRAC] = np.float32(bundle.step_frac)
    out[G_N_BINS_SCALED] = np.float32(n_bins_scaled)
    out[G_ASSIGNED_LL_MEAN] = np.float32(bundle.assigned_ll_mean)
    out[G_ASSIGNED_LL_MAX] = np.float32(bundle.assigned_ll_max)
    out[G_REMAINING_FRAC] = np.float32(bundle.remaining_frac)
    out[G_SEED_SIZE_SCALED] = np.float32(seed_size_scaled)
    out[G_GROW_RATIO_SCALED] = np.float32(bundle.grow_ratio_scaled)
    out[G_POSITIVE_FRONTIER_FRACTION] = np.float32(bundle.positive_frontier_fraction)
    out[G_CENTROID_DRIFT_SCALED] = np.float32(bundle.centroid_drift_scaled)
    out[G_COMPACTNESS_PROXY] = np.float32(bundle.compactness_proxy)
    out[G_FRONTIER_ADD_REWARD_TOPK_MEAN] = np.float32(bundle.frontier_add_reward_topk_mean)
    out[G_FRONTIER_ADD_REWARD_MAX] = np.float32(bundle.frontier_add_reward_max)
    out[G_FRONTIER_ADD_REWARD_MEAN] = np.float32(bundle.frontier_add_reward_mean)
    out[G_COMPACT_STREAK_SCALED] = np.float32(compact_streak_scaled)
    return out


def _compute_seed_shape_features(ctx: EpisodeContext) -> tuple[float, float, float]:
    n_bins = int(ctx.n_bins)
    if n_bins == 0:
        return 0.0, 0.0, 0.0
    seed = np.asarray(ctx.initial_membership_mask, dtype=np.uint8).astype(bool, copy=False)
    if not np.any(seed):
        return 0.0, 0.0, 0.0

    seed_support = compute_neighbor_support_fraction(np.asarray(ctx.initial_membership_mask, dtype=np.uint8), ctx.neighbor_index)
    seed_compactness = float(np.mean(seed_support[seed]))

    xy = np.asarray(ctx.candidate_bin_xy_um, dtype=np.float64)
    nucleus_xy = np.asarray(ctx.nucleus_center_xy_um, dtype=np.float64)
    seed_xy = xy[seed]
    radius = np.sqrt(np.sum((seed_xy - nucleus_xy) ** 2, axis=1))
    seed_radius_p90_scaled = float(np.clip(np.percentile(radius, 90) / max(float(ctx.r_max_um), 1.0e-8), 0.0, 1.0))

    if seed_xy.shape[0] < 3:
        seed_aspect_ratio_scaled = 0.0
    else:
        centered = seed_xy - np.mean(seed_xy, axis=0, dtype=np.float64)
        cov = np.cov(centered, rowvar=False, bias=True)
        eig = np.linalg.eigvalsh(np.asarray(cov, dtype=np.float64))
        eig = np.maximum(eig, 0.0)
        major = float(np.max(eig))
        minor = float(np.min(eig))
        if major <= 1.0e-12:
            seed_aspect_ratio_scaled = 0.0
        else:
            seed_aspect_ratio_scaled = float(np.clip(1.0 - np.sqrt((minor + 1.0e-12) / (major + 1.0e-12)), 0.0, 1.0))

    return seed_compactness, seed_radius_p90_scaled, seed_aspect_ratio_scaled


def _compute_ll_distribution_features(ll: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(ll, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("ll must have shape (B, K)")
    n_bins, n_types = arr.shape
    if n_bins == 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    if n_types <= 1:
        margin = arr[:, 0] if n_types == 1 else np.zeros((n_bins,), dtype=np.float64)
        return _zscore_1d(margin).astype(np.float32, copy=False), np.zeros((n_bins,), dtype=np.float32)

    top1_idx = np.argmax(arr, axis=1)
    top1 = arr[np.arange(n_bins), top1_idx]
    masked = arr.copy()
    masked[np.arange(n_bins), top1_idx] = -np.inf
    top2 = np.max(masked, axis=1)
    margin = top1 - top2
    shifted = arr - np.max(arr, axis=1, keepdims=True)
    ex = np.exp(shifted)
    probs = ex / np.sum(ex, axis=1, keepdims=True)
    entropy = -np.sum(probs * np.log(np.maximum(probs, 1.0e-12)), axis=1) / np.log(float(n_types))
    return _zscore_1d(margin).astype(np.float32, copy=False), entropy.astype(np.float32, copy=False)


def _scale_count_feature(counts: np.ndarray, *, cap: float = 256.0) -> np.ndarray:
    arr = np.asarray(counts, dtype=np.float64)
    clipped = np.clip(arr, 0.0, float(cap))
    return (np.log1p(clipped) / np.log1p(float(cap))).astype(np.float32, copy=False)


def _posterior_from_membership_mask(
    *,
    ctx: EpisodeContext,
    membership_mask: np.ndarray,
) -> np.ndarray:
    """Compute posterior p(k|S_t) from current assigned membership."""
    mask = np.asarray(membership_mask, dtype=np.uint8).astype(bool, copy=False)
    if np.any(mask):
        log_p_st_given_k = np.sum(ctx.ll[mask], axis=0, dtype=np.float64)
    else:
        log_p_st_given_k = np.zeros((ctx.n_cell_types,), dtype=np.float64)
    return _softmax_1d(log_p_st_given_k + ctx.log_prior)


def _posterior_confidence_delta_per_bin(
    *,
    ctx: EpisodeContext,
    membership_mask: np.ndarray,
) -> np.ndarray:
    """Score each candidate by how much adding it sharpens cell-type posterior confidence."""
    mask = np.asarray(membership_mask, dtype=np.uint8).astype(bool, copy=False)
    if ctx.n_bins == 0:
        return np.zeros((0,), dtype=np.float32)
    if np.any(mask):
        current_scores = np.sum(ctx.ll[mask], axis=0, dtype=np.float64) + float(ctx.log_prior)
    else:
        current_scores = np.full((ctx.n_cell_types,), float(ctx.log_prior), dtype=np.float64)

    current_posterior = _softmax_1d(current_scores)
    current_confidence = float(np.max(current_posterior)) if current_posterior.size > 0 else 0.0

    next_scores = current_scores[None, :] + np.asarray(ctx.ll, dtype=np.float64)
    next_scores = next_scores - np.max(next_scores, axis=1, keepdims=True)
    next_exp = np.exp(next_scores)
    next_den = np.sum(next_exp, axis=1, keepdims=True)
    next_posterior = next_exp / np.maximum(next_den, 1.0e-300)
    next_confidence = np.max(next_posterior, axis=1)
    delta = (next_confidence - current_confidence) * np.asarray(ctx.expression_confidence, dtype=np.float64)
    return delta.astype(np.float32, copy=False)


def _old_bin_compatibility_per_bin(
    *,
    ctx: EpisodeContext,
    posterior: np.ndarray,
) -> np.ndarray:
    """Old expression score: candidate likelihood under the current cell-type posterior."""
    scores = np.asarray(ctx.ll, dtype=np.float64) @ np.asarray(posterior, dtype=np.float64)
    scores *= np.asarray(ctx.expression_confidence, dtype=np.float64)
    return scores.astype(np.float32, copy=False)


def _zscore_over_frontier(values: np.ndarray, frontier_mask: np.ndarray, zscore_delta: float) -> np.ndarray:
    values_arr = np.asarray(values, dtype=np.float32)
    frontier = np.asarray(frontier_mask, dtype=bool)
    if values_arr.shape != frontier.shape:
        raise ValueError("values and frontier_mask must have the same shape")
    if not np.any(frontier):
        return np.zeros_like(values_arr, dtype=np.float32)
    frontier_values = values_arr[frontier]
    mu = float(np.mean(frontier_values))
    sigma = float(np.std(frontier_values, ddof=0))
    return ((values_arr - mu) / (sigma + float(zscore_delta))).astype(np.float32, copy=False)


def _expression_reward_terms_per_bin(
    *,
    ctx: EpisodeContext,
    membership_mask: np.ndarray,
    posterior: np.ndarray,
    frontier_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return raw/normalized new and old expression reward terms.

    New term: posterior-confidence gain after adding a bin.
    Old term: bin compatibility with the current posterior, kept as a small w5 helper.
    """
    expr_new_raw = _posterior_confidence_delta_per_bin(ctx=ctx, membership_mask=membership_mask)
    expr_old_raw = _old_bin_compatibility_per_bin(ctx=ctx, posterior=posterior)
    if ctx.normalize_expression_zscore:
        expr_new_term = _zscore_over_frontier(expr_new_raw, frontier_mask, ctx.zscore_delta)
        expr_old_term = _zscore_over_frontier(expr_old_raw, frontier_mask, ctx.zscore_delta)
    else:
        expr_new_term = expr_new_raw.astype(np.float32, copy=False)
        expr_old_term = expr_old_raw.astype(np.float32, copy=False)
    return (
        expr_new_raw.astype(np.float32, copy=False),
        expr_new_term.astype(np.float32, copy=False),
        expr_old_raw.astype(np.float32, copy=False),
        expr_old_term.astype(np.float32, copy=False),
    )


def _add_rewards_from_membership_mask(
    *,
    ctx: EpisodeContext,
    membership_mask: np.ndarray,
    posterior: np.ndarray,
    neighbor_support: np.ndarray,
    frontier_mask: np.ndarray,
) -> np.ndarray:
    """Recompute current ADD rewards from one state summary path."""
    _, expr_new_term, _, expr_old_term = _expression_reward_terms_per_bin(
        ctx=ctx,
        membership_mask=membership_mask,
        posterior=posterior,
        frontier_mask=frontier_mask,
    )
    return ctx.w1 * expr_new_term + ctx.w5 * expr_old_term - ctx.base_penalty + ctx.w4 * neighbor_support


def _scale_seed_size_feature(seed_count: int, *, cap: int = 32) -> float:
    """Map initial nuclear seed size into a stable [0, 1] feature range."""
    capped = float(min(max(int(seed_count), 0), int(cap)))
    return float(np.log1p(capped) / np.log1p(float(cap)))


def _scale_grow_ratio_feature(assigned_count: int, initial_seed_count: int, *, cap: float = 8.0) -> float:
    """Map current growth ratio assigned/seed into a stable [0, 1] feature range."""
    denom = max(int(initial_seed_count), 1)
    ratio = float(max(int(assigned_count), 0)) / float(denom)
    capped = min(max(ratio, 0.0), float(cap))
    return float(np.log1p(capped) / np.log1p(float(cap)))


def _scale_compact_streak_feature(compact_streak: int, *, cap: int = 8) -> float:
    """Map consecutive compact planner decisions into a stable [0, 1] feature."""
    capped = float(min(max(int(compact_streak), 0), int(cap)))
    return float(capped / float(max(int(cap), 1)))


def _observation_with_compact_streak(obs: dict[str, Any], compact_streak: int) -> dict[str, Any]:
    """Return a shallow observation copy with planner-history state injected."""
    out = dict(obs)
    global_features = np.asarray(obs["global_features"], dtype=np.float32).copy()
    if global_features.shape[0] >= GLOBAL_FEATURE_DIM:
        global_features[G_COMPACT_STREAK_SCALED] = np.float32(_scale_compact_streak_feature(compact_streak))
    out["global_features"] = global_features
    return out


def _observation_to_tensors(obs: dict[str, Any], *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if "global_features" not in obs or "action_features" not in obs or "action_mask" not in obs:
        raise KeyError(
            "observation must contain keys: global_features, action_features, action_mask. "
            "Update your environment observation adapter if needed."
        )
    global_features = torch.as_tensor(np.asarray(obs["global_features"], dtype=np.float32), device=device).unsqueeze(0)
    action_features = torch.as_tensor(np.asarray(obs["action_features"], dtype=np.float32), device=device).unsqueeze(0)
    action_mask = torch.as_tensor(np.asarray(obs["action_mask"], dtype=bool), device=device).unsqueeze(0)
    return global_features, action_features, action_mask


def _observation_without_stop_action(obs: dict[str, Any]) -> dict[str, Any]:
    """Copy an observation and disable low-level STOP for planner-controlled rollouts."""
    out = dict(obs)
    mask = np.asarray(obs["action_mask"], dtype=bool).copy()
    if mask.size > 0:
        mask[0] = False
    out["action_mask"] = mask
    return out


def _planner_logit_bias_from_action_features(
    action_features: torch.Tensor,
    planner_modes: torch.Tensor,
    config: PPOTrainingConfig,
) -> torch.Tensor | None:
    """Build soft ADD-logit bias for the current high-level growth mode."""
    if not bool(config.planner_enabled):
        return None
    if action_features.ndim != 3:
        raise ValueError("action_features must have shape (N, A, F)")

    modes = planner_modes.to(device=action_features.device, dtype=torch.long).reshape(-1)
    if modes.shape[0] != action_features.shape[0]:
        raise ValueError("planner_modes batch size must match action_features")

    bias = action_features.new_zeros(action_features.shape[:2])
    expression = action_features[:, :, A_FEATURE_2]
    centroid_distance = action_features[:, :, A_CANDIDATE_CENTROID_DISTANCE]
    compactness_gain = action_features[:, :, A_CANDIDATE_COMPACTNESS_GAIN]
    neighbor_support = action_features[:, :, A_CANDIDATE_NEIGHBOR_SUPPORT]
    add_rows = action_features[:, :, A_IS_STOP_ACTION] < 0.5

    for mode_name, mode_idx in _PLANNER_MODE_TO_INDEX.items():
        weights = config.planner_logit_bias.get(mode_name, {})
        if not weights:
            continue
        row_bias = (
            float(weights.get("expression", 0.0)) * expression
            + float(weights.get("centroid_distance", 0.0)) * centroid_distance
            + float(weights.get("neighbor_support", 0.0)) * neighbor_support
            + float(weights.get("compactness_gain", 0.0)) * compactness_gain
        )
        active = (modes[:, None] == int(mode_idx)) & add_rows
        bias = torch.where(active, row_bias, bias)

    if bias.shape[1] > 0:
        bias[:, 0] = 0.0
    return bias


def _softmax_1d(scores: np.ndarray) -> np.ndarray:
    s = np.asarray(scores, dtype=np.float64)
    m = float(np.max(s))
    ex = np.exp(s - m)
    den = float(np.sum(ex))
    if den <= 0.0 or not np.isfinite(den):
        raise ValueError("softmax denominator is non-finite or non-positive")
    return ex / den


def _zscore_1d(values: np.ndarray, delta: float = 1e-8) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("zscore input must be 1D")
    if arr.size <= 1:
        return np.zeros_like(arr, dtype=np.float64)
    mu = float(np.mean(arr))
    sigma = float(np.std(arr, ddof=0))
    return (arr - mu) / (sigma + float(delta))
