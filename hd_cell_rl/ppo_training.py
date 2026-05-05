"""PyTorch PPO training pipeline for HD cell ADD/STOP environments.

This module trains one shared actor-critic policy across many single-cell episodes.
Each outer update samples a random batch of cells, rolls out one episode per cell,
then runs PPO-Clip updates on the collected transitions.
"""

from __future__ import annotations

from dataclasses import dataclass
import datetime as dt
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
import torch

from . import grpo_objective as _grpo_objective
from .ppo_config import (
    ConfigError,
    PLANNER_MODES,
    PLANNER_MODE_STOP,
    PLANNER_MODE_COMPACT,
    PLANNER_MODE_BALANCED,
    PLANNER_MODE_EXPLORE,
    _PLANNER_MODE_TO_INDEX,
    PPOTrainingConfig,
    _parse_planner_logit_bias,
    load_ppo_training_config,
)
from .ppo_checkpoint import build_actor_critic_from_config
from .ppo_model import ActorCritic
from .ppo_run_io import (
    _append_step_log,
    _build_metadata,
    _save_checkpoint,
    _slugify,
    _write_json,
    _write_yaml,
)
from .ppo_feature_schema import (
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
from .ppo_buffers import (
    EpisodeStep,
    PlannerStep,
    EpisodeTrajectory,
    RolloutTransition,
    PlannerRolloutTransition,
    RolloutFeatureCache,
    PlannerRolloutFeatureCache,
    compute_discounted_returns,
    compute_gae_returns_and_advantages,
    compute_advantages,
    _batch_global_features_for_rollout,
    _batch_observations_for_rollout,
    _build_planner_rollout_buffer,
    _build_planner_rollout_feature_cache,
    _build_rollout_buffer,
    _build_rollout_feature_cache,
    _compute_group_relative_episode_advantages,
    _pack_mask,
    _unpack_mask,
)
from .ppo_rollout import (
    _collect_episode_contexts,
    _collect_trajectories,
    _expand_group_relative_contexts,
    _run_episode_with_planner,
    run_episode,
)
from .ppo_dataset import (
    EpisodeDataset,
    _build_initial_membership_mask,
    _load_nuclear_barcode_assignment_lookup,
    _load_reference_counts,
    _load_table,
    _normalize_cell_id,
    _validate_reference_matrix,
)
from .ppo_state import (
    ACTION_FEATURE_DIM as POLICY_ACTION_FEATURE_DIM,
    GLOBAL_FEATURE_DIM as POLICY_GLOBAL_FEATURE_DIM,
    EpisodeContext,
    EpisodeStaticPolicyFeatures,
    StateFeatureBundle,
    _add_rewards_from_membership_mask,
    _build_episode_static_policy_features,
    _build_policy_observation_from_state,
    _build_static_action_template,
    _compute_dynamic_add_action_features,
    _compute_dynamic_add_action_features_from_support,
    _compute_ll_distribution_features,
    _compute_seed_shape_features,
    _compute_shape_frontier_features,
    _compute_state_feature_bundle,
    _compute_state_summary_from_mask,
    _expression_reward_terms_per_bin,
    _fill_dynamic_action_features,
    _global_features_from_bundle,
    _observation_to_tensors,
    _observation_with_compact_streak,
    _observation_without_stop_action,
    _old_bin_compatibility_per_bin,
    _planner_logit_bias_from_action_features,
    _posterior_confidence_delta_per_bin,
    _posterior_from_membership_mask,
    _scale_compact_streak_feature,
    _scale_count_feature,
    _scale_grow_ratio_feature,
    _scale_seed_size_feature,
    _softmax_1d,
    _zscore_1d,
)
from .reward import (
    compute_frontier_eligible_mask,
    compute_neighbor_support_fraction,
    compute_stop_delta,
)


@dataclass(frozen=True)
class PPOUpdateMetrics:
    """Aggregate metrics from one PPO parameter-update call."""

    policy_loss: float
    value_loss: float
    entropy: float
    total_loss: float
    approx_kl: float


@dataclass(frozen=True)
class TrainUpdateMetrics:
    """Metrics tracked after one outer training update."""

    update_index: int
    average_batch_reward: float
    moving_average_reward: float | None
    policy_loss: float
    value_loss: float
    entropy: float
    total_loss: float
    approx_kl: float
    no_improve_count: int
    n_episodes: int
    n_transitions: int


@dataclass(frozen=True)
class PPOTrainingResult:
    """Training outputs returned to caller."""

    run_dir: Path
    best_checkpoint_path: Path | None
    best_moving_average_reward: float | None
    updates_completed: int
    stopped_early: bool
    logs: tuple[TrainUpdateMetrics, ...]


class AddStopCellEnv:
    """Fast ADD/STOP environment using precomputed episode context.

    Notes for customization:
    - This is where you can change action semantics (e.g., add REMOVE later).
    - `_build_policy_observation` defines policy input features.
    - reward formulas are implemented in `_add_reward_for_bin` and `_stop_reward`.
    """

    ACTION_FEATURE_DIM = POLICY_ACTION_FEATURE_DIM
    GLOBAL_FEATURE_DIM = POLICY_GLOBAL_FEATURE_DIM

    def __init__(self, context: EpisodeContext) -> None:
        self._ctx = context
        self._initial_membership_mask = np.asarray(self._ctx.initial_membership_mask, dtype=np.uint8).copy()
        if self._initial_membership_mask.shape != (self._ctx.n_bins,):
            raise ValueError(
                f"initial_membership_mask must have shape ({self._ctx.n_bins},), "
                f"got {self._initial_membership_mask.shape}"
            )
        self._membership_mask = np.zeros(self._ctx.n_bins, dtype=np.uint8)
        self._step_index = 0
        self._terminated = False
        self._truncated = False
        self._log_p_st_given_k = np.zeros(self._ctx.n_cell_types, dtype=np.float64)
        self._assigned_count = 0
        self._assigned_ll_mean_sum = 0.0
        self._assigned_ll_max_sum = 0.0
        self._n_bins = int(self._ctx.n_bins)
        self._n_bins_scaled = float(np.log1p(self._n_bins) / 8.0)
        self._seed_size_scaled = _scale_seed_size_feature(int(np.sum(self._initial_membership_mask)))
        self._action_template = _build_static_action_template(
            self._ctx,
            self._n_bins_scaled,
            self._seed_size_scaled,
        )
        self._action_features = self._action_template.copy()
        self._action_mask = np.ones(self._n_bins + 1, dtype=bool)
        self._action_mask[0] = True
        self._global_features = np.zeros((self.GLOBAL_FEATURE_DIM,), dtype=np.float32)
        seed_idx = self._initial_membership_mask.astype(bool, copy=False)
        if np.any(seed_idx):
            self._initial_log_p_st_given_k = np.sum(self._ctx.ll[seed_idx], axis=0, dtype=np.float64)
            self._initial_assigned_count = int(np.sum(seed_idx))
            self._initial_assigned_ll_mean_sum = float(np.sum(self._ctx.ll_mean_z[seed_idx], dtype=np.float64))
            self._initial_assigned_ll_max_sum = float(np.sum(self._ctx.ll_max_z[seed_idx], dtype=np.float64))
        else:
            self._initial_log_p_st_given_k = np.zeros(self._ctx.n_cell_types, dtype=np.float64)
            self._initial_assigned_count = 0
            self._initial_assigned_ll_mean_sum = 0.0
            self._initial_assigned_ll_max_sum = 0.0

    def reset(self) -> tuple[dict[str, Any], dict[str, Any]]:
        np.copyto(self._membership_mask, self._initial_membership_mask)
        self._step_index = 0
        self._terminated = False
        self._truncated = False
        np.copyto(self._log_p_st_given_k, self._initial_log_p_st_given_k)
        self._assigned_count = int(self._initial_assigned_count)
        self._assigned_ll_mean_sum = float(self._initial_assigned_ll_mean_sum)
        self._assigned_ll_max_sum = float(self._initial_assigned_ll_max_sum)
        np.copyto(self._action_features, self._action_template)
        self._refresh_dynamic_action_state()
        return self._build_policy_observation(), self._build_info()

    def step(self, action: int) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if self._terminated or self._truncated:
            raise RuntimeError("cannot call step() on a finished episode; call reset() first")
        if not isinstance(action, (int, np.integer)):
            raise TypeError("action must be an integer index")

        action_i = int(action)
        if action_i < 0 or action_i > self._ctx.n_bins:
            raise ValueError(f"action index out of range: {action_i}")

        if action_i == 0:
            reward = float(self._stop_reward())
            self._terminated = True
        else:
            bin_idx = action_i - 1
            if self._membership_mask[bin_idx] == 1:
                raise ValueError("invalid ADD action: bin already assigned")

            reward = float(self._add_reward_for_bin(bin_idx))
            self._membership_mask[bin_idx] = 1
            self._log_p_st_given_k += self._ctx.ll[bin_idx]
            self._assigned_count += 1
            self._assigned_ll_mean_sum += float(self._ctx.ll_mean_z[bin_idx])
            self._assigned_ll_max_sum += float(self._ctx.ll_max_z[bin_idx])
            self._refresh_dynamic_action_state()

        self._step_index += 1
        if self._step_index >= self._ctx.max_steps and not self._terminated:
            self._truncated = True

        return self._build_policy_observation(), reward, self._terminated, self._truncated, self._build_info()

    def _posterior(self) -> np.ndarray:
        scores = self._log_p_st_given_k + self._ctx.log_prior
        return _softmax_1d(scores)

    def _add_reward_for_bin(self, bin_idx: int) -> float:
        neighbor_support = float(
            compute_neighbor_support_fraction(self._membership_mask, self._ctx.neighbor_index)[bin_idx]
        )
        _, expr_new_term, _, expr_old_term = _expression_reward_terms_per_bin(
            ctx=self._ctx,
            membership_mask=self._membership_mask,
            posterior=self._posterior(),
            frontier_mask=self._action_mask[1:],
        )
        return float(
            self._ctx.w1 * float(expr_new_term[bin_idx])
            + self._ctx.w5 * float(expr_old_term[bin_idx])
            - self._ctx.base_penalty[bin_idx]
            + self._ctx.w4 * neighbor_support
        )

    def _all_add_rewards(self, posterior: np.ndarray) -> np.ndarray:
        _, expr_new_term, _, expr_old_term = _expression_reward_terms_per_bin(
            ctx=self._ctx,
            membership_mask=self._membership_mask,
            posterior=posterior,
            frontier_mask=self._action_mask[1:],
        )
        neighbor_support = compute_neighbor_support_fraction(self._membership_mask, self._ctx.neighbor_index).astype(
            np.float32,
            copy=False,
        )
        return (
            self._ctx.w1 * expr_new_term
            + self._ctx.w5 * expr_old_term
            - self._ctx.base_penalty
            + self._ctx.w4 * neighbor_support
        )

    def _stop_reward(self) -> float:
        eligible = self._action_mask[1:]
        posterior = self._posterior()
        r_add = self._all_add_rewards(posterior)
        delta_t = compute_stop_delta(
            r_add,
            eligible,
            stop_stat=self._ctx.stop_stat,
            stop_top_k=self._ctx.stop_top_k,
        )
        return float(-self._ctx.stop_lambda * delta_t)

    def _build_policy_observation(self) -> dict[str, Any]:
        """Build action/global features consumed by ActorCritic.

        Replace this feature construction if you want richer HD-specific inputs.
        """
        bundle = _compute_state_feature_bundle(
            ctx=self._ctx,
            membership_mask=self._membership_mask,
            step_index=self._step_index,
        )

        np.copyto(
            self._global_features,
            _global_features_from_bundle(
                bundle=bundle,
                n_bins_scaled=self._n_bins_scaled,
                seed_size_scaled=self._seed_size_scaled,
            ),
        )
        _fill_dynamic_action_features(
            action_features=self._action_features,
            ctx=self._ctx,
            membership_mask=self._membership_mask,
            bundle=bundle,
        )

        return {
            "global_features": self._global_features,
            "action_features": self._action_features,
            "action_mask": self._action_mask,
            "membership_mask": self._membership_mask,
            "step_index": int(self._step_index),
        }

    def _refresh_dynamic_action_state(self) -> None:
        """Refresh frontier mask plus dynamic ADD-row feature columns."""
        self._action_mask.fill(False)
        self._action_mask[0] = True
        if self._n_bins > 0:
            frontier = compute_frontier_eligible_mask(self._membership_mask, self._ctx.neighbor_index)
            self._action_mask[1:] = frontier

    def _build_info(self) -> dict[str, Any]:
        return {
            "cell_id": self._ctx.cell_id,
            "n_candidate_bins": self._ctx.n_bins,
            "n_assigned_bins": int(self._assigned_count),
            "step_index": int(self._step_index),
            "max_steps": int(self._ctx.max_steps),
            "terminated": bool(self._terminated),
            "truncated": bool(self._truncated),
        }


def _compute_episode_advantage_bonuses(
    *,
    trajectories: list[EpisodeTrajectory],
    group_relative_enabled: bool,
    group_relative_group_size: int,
    group_relative_norm_epsilon: float,
    group_relative_score: str,
    training_mode: str,
    episode_contexts: list[EpisodeContext] | None,
    config: PPOTrainingConfig | None,
) -> np.ndarray | None:
    mode = str(training_mode).strip().lower()
    if mode == "full_grpo":
        if episode_contexts is None or config is None:
            raise ValueError("full_grpo rollout buffer requires episode_contexts and config")
        return _compute_full_grpo_episode_advantages(
            trajectories,
            episode_contexts,
            group_size=int(group_relative_group_size),
            norm_epsilon=float(group_relative_norm_epsilon),
            config=config,
        )
    if group_relative_enabled and trajectories:
        return _compute_group_relative_episode_advantages(
            trajectories,
            group_size=int(group_relative_group_size),
            norm_epsilon=float(group_relative_norm_epsilon),
            score=str(group_relative_score),
        )
    return None


def _final_membership_mask_from_trajectory(
    *,
    ctx: EpisodeContext,
    trajectory: EpisodeTrajectory,
) -> np.ndarray:
    mask = np.asarray(ctx.initial_membership_mask, dtype=np.uint8).copy()
    for step in trajectory.steps:
        if int(step.action) > 0:
            mask[int(step.action) - 1] = 1
    return mask


def _compute_full_grpo_episode_scores(
    trajectories: list[EpisodeTrajectory],
    episode_contexts: list[EpisodeContext],
    config: PPOTrainingConfig,
) -> np.ndarray:
    return _grpo_objective._compute_full_grpo_episode_scores(
        trajectories,
        episode_contexts,
        config,
        final_membership_mask_fn=_final_membership_mask_from_trajectory,
        evidence_state_fn=_full_grpo_evidence_state,
        key_node_score_fn=_full_grpo_key_node_score,
    )


def _compute_full_grpo_episode_advantages(
    trajectories: list[EpisodeTrajectory],
    episode_contexts: list[EpisodeContext],
    *,
    group_size: int,
    norm_epsilon: float,
    config: PPOTrainingConfig,
) -> np.ndarray:
    return _grpo_objective._compute_full_grpo_episode_advantages(
        trajectories,
        episode_contexts,
        group_size=group_size,
        norm_epsilon=norm_epsilon,
        config=config,
        score_fn=_compute_full_grpo_episode_scores,
    )


def _full_grpo_frontier_quality(*, topk_mean: float, has_frontier: bool, config: PPOTrainingConfig) -> float:
    return _grpo_objective._full_grpo_frontier_quality(
        topk_mean=topk_mean,
        has_frontier=has_frontier,
        config=config,
    )


def _full_grpo_evidence_state(
    *,
    ctx: EpisodeContext,
    membership_mask: np.ndarray,
    config: PPOTrainingConfig,
) -> dict[str, float]:
    return _grpo_objective._full_grpo_evidence_state(
        ctx=ctx,
        membership_mask=membership_mask,
        config=config,
        state_feature_bundle_fn=_compute_state_feature_bundle,
    )


def _full_grpo_stop_score(*, stopped: bool, frontier_quality: float, overgrowth_risk: float) -> float:
    return _grpo_objective._full_grpo_stop_score(
        stopped=stopped,
        frontier_quality=frontier_quality,
        overgrowth_risk=overgrowth_risk,
    )


def _masked_topk_mean(values: np.ndarray, mask: np.ndarray, *, k: int) -> float:
    return _grpo_objective._masked_topk_mean(values, mask, k=k)


def _full_grpo_key_node_score(
    *,
    ctx: EpisodeContext,
    planner_step: PlannerStep,
    config: PPOTrainingConfig,
) -> float:
    return _grpo_objective._full_grpo_key_node_score(
        ctx=ctx,
        planner_step=planner_step,
        config=config,
        unpack_mask_fn=_unpack_mask,
        evidence_state_fn=_full_grpo_evidence_state,
        compact_streak_scale_fn=_scale_compact_streak_feature,
    )


def ppo_update(
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    rollout_cache: RolloutFeatureCache,
    *,
    planner_cache: PlannerRolloutFeatureCache | None,
    config: PPOTrainingConfig,
    eps_clip: float,
    ppo_epochs: int,
    minibatch_size: int,
    vf_coef: float,
    ent_coef: float,
    max_grad_norm: float,
    target_kl: float | None,
    include_value_loss: bool,
    device: torch.device,
    rng: np.random.Generator,
) -> PPOUpdateMetrics:
    """Run PPO-Clip optimization over one flattened rollout buffer."""
    planner_n = 0 if planner_cache is None else int(planner_cache.n_transitions)
    if rollout_cache.n_transitions == 0 and planner_n == 0:
        raise ValueError("transitions must not be empty")

    n = int(rollout_cache.n_transitions)
    all_indices = np.arange(n, dtype=np.int64)
    planner_indices = np.arange(planner_n, dtype=np.int64)

    policy_losses: list[float] = []
    value_losses: list[float] = []
    entropies: list[float] = []
    total_losses: list[float] = []
    approx_kls: list[float] = []

    stop_early = False
    for _epoch in range(int(ppo_epochs)):
        if stop_early:
            break
        if n > 0:
            perm = rng.permutation(all_indices)
            for start in range(0, n, int(minibatch_size)):
                mb_idx = perm[start : start + int(minibatch_size)]
                if mb_idx.size == 0:
                    continue

                batch = _evaluate_minibatch_from_cache_grouped(
                    model=model,
                    rollout_cache=rollout_cache,
                    indices=mb_idx,
                    device=device,
                    config=config,
                )
                new_log_prob = batch["new_log_probs"]
                values = batch["values"]
                entropy = batch["entropy"].mean()

                ratio = torch.exp(new_log_prob - batch["old_log_probs"])
                surr1 = ratio * batch["advantages"]
                surr2 = torch.clamp(ratio, 1.0 - eps_clip, 1.0 + eps_clip) * batch["advantages"]
                policy_loss = -torch.min(surr1, surr2).mean()
                if include_value_loss:
                    value_loss = torch.mean((values - batch["returns"]) ** 2)
                    total_loss = policy_loss + vf_coef * value_loss - ent_coef * entropy
                else:
                    value_loss = values.new_tensor(0.0)
                    total_loss = policy_loss - ent_coef * entropy

                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                optimizer.step()

                with torch.no_grad():
                    approx_kl = torch.mean(batch["old_log_probs"] - new_log_prob)

                policy_losses.append(float(policy_loss.item()))
                value_losses.append(float(value_loss.item()))
                entropies.append(float(entropy.item()))
                total_losses.append(float(total_loss.item()))
                approx_kls.append(float(approx_kl.item()))

                if target_kl is not None and float(approx_kl.item()) > 1.5 * float(target_kl):
                    stop_early = True
                    break

        if stop_early:
            break

        if planner_cache is not None and planner_n > 0:
            planner_perm = rng.permutation(planner_indices)
            for start in range(0, planner_n, int(minibatch_size)):
                mb_idx = planner_perm[start : start + int(minibatch_size)]
                if mb_idx.size == 0:
                    continue
                batch = _evaluate_planner_minibatch_from_cache(
                    model=model,
                    planner_cache=planner_cache,
                    indices=mb_idx,
                    device=device,
                )
                new_log_prob = batch["new_log_probs"]
                entropy = batch["entropy"].mean()
                ratio = torch.exp(new_log_prob - batch["old_log_probs"])
                surr1 = ratio * batch["advantages"]
                surr2 = torch.clamp(ratio, 1.0 - eps_clip, 1.0 + eps_clip) * batch["advantages"]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = new_log_prob.new_tensor(0.0)
                total_loss = policy_loss - float(config.planner_entropy_coef) * entropy

                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                optimizer.step()

                with torch.no_grad():
                    approx_kl = torch.mean(batch["old_log_probs"] - new_log_prob)

                policy_losses.append(float(policy_loss.item()))
                value_losses.append(float(value_loss.item()))
                entropies.append(float(entropy.item()))
                total_losses.append(float(total_loss.item()))
                approx_kls.append(float(approx_kl.item()))

                if target_kl is not None and float(approx_kl.item()) > 1.5 * float(target_kl):
                    stop_early = True
                    break

    return PPOUpdateMetrics(
        policy_loss=float(np.mean(policy_losses)),
        value_loss=float(np.mean(value_losses)),
        entropy=float(np.mean(entropies)),
        total_loss=float(np.mean(total_losses)),
        approx_kl=float(np.mean(approx_kls)),
    )


def train_one_update(
    *,
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    dataset: EpisodeDataset,
    config: PPOTrainingConfig,
    device: torch.device,
    rng: np.random.Generator,
) -> tuple[PPOUpdateMetrics, float, int, int, dict[str, float]]:
    """Collect one 100-cell rollout batch, then run PPO updates."""
    t_start = time.perf_counter()

    t0 = time.perf_counter()
    sampled_cells = (
        int(config.batch_cells) // int(config.group_relative_group_size)
        if bool(config.group_relative_enabled) or str(config.training_mode) == "full_grpo"
        else int(config.batch_cells)
    )
    contexts = _collect_episode_contexts(
        dataset=dataset,
        batch_cells=sampled_cells,
        max_steps_per_episode=config.max_steps_per_episode,
    )
    if bool(config.group_relative_enabled) or str(config.training_mode) == "full_grpo":
        contexts = _expand_group_relative_contexts(
            contexts=contexts,
            group_size=int(config.group_relative_group_size),
        )
    t_context = time.perf_counter() - t0
    if not contexts:
        raise RuntimeError("failed to collect any valid trajectories for this update")

    t0 = time.perf_counter()
    model.eval()
    trajectories = _collect_trajectories(
        contexts=contexts,
        model=model,
        device=device,
        rng=rng,
        rollout_mode=str(config.rollout_mode),
        n_rollout_workers=int(config.n_rollout_workers),
        config=config,
        env_cls=AddStopCellEnv,
    )
    model.train()
    t_rollout = time.perf_counter() - t0

    t0 = time.perf_counter()
    episode_advantage_bonus = _compute_episode_advantage_bonuses(
        trajectories=trajectories,
        group_relative_enabled=bool(config.group_relative_enabled),
        group_relative_group_size=int(config.group_relative_group_size),
        group_relative_norm_epsilon=float(config.group_relative_norm_epsilon),
        group_relative_score=str(config.group_relative_score),
        training_mode=str(config.training_mode),
        episode_contexts=contexts,
        config=config,
    )
    transitions = _build_rollout_buffer(
        trajectories=trajectories,
        gamma=float(config.gamma),
        gae_lambda=float(config.gae_lambda),
        normalize_returns_per_episode=bool(config.normalize_returns_per_episode),
        normalize_advantages=bool(config.normalize_advantages),
        group_relative_enabled=bool(config.group_relative_enabled),
        group_relative_group_size=int(config.group_relative_group_size),
        group_relative_mix_alpha=float(config.group_relative_mix_alpha),
        group_relative_norm_epsilon=float(config.group_relative_norm_epsilon),
        group_relative_score=str(config.group_relative_score),
        training_mode=str(config.training_mode),
        episode_contexts=contexts,
        config=config,
        episode_advantage_bonus=episode_advantage_bonus,
        episode_advantage_bonus_fn=_compute_episode_advantage_bonuses,
    )
    planner_transitions = (
        _build_planner_rollout_buffer(
            trajectories=trajectories,
            episode_advantage_bonus=episode_advantage_bonus,
        )
        if bool(config.planner_enabled)
        else []
    )
    t_buffer = time.perf_counter() - t0

    t0 = time.perf_counter()
    rollout_cache = _build_rollout_feature_cache(
        transitions=transitions,
        episode_contexts=contexts,
    )
    planner_cache = (
        _build_planner_rollout_feature_cache(
            transitions=planner_transitions,
            episode_contexts=contexts,
        )
        if bool(config.planner_enabled)
        else None
    )
    t_cache = time.perf_counter() - t0

    t0 = time.perf_counter()
    ppo_metrics = ppo_update(
        model=model,
        optimizer=optimizer,
        rollout_cache=rollout_cache,
        planner_cache=planner_cache,
        config=config,
        eps_clip=float(config.eps_clip),
        ppo_epochs=int(config.ppo_epochs),
        minibatch_size=int(config.minibatch_size),
        vf_coef=float(config.vf_coef),
        ent_coef=float(config.ent_coef),
        max_grad_norm=float(config.max_grad_norm),
        target_kl=config.target_kl,
        include_value_loss=str(config.training_mode) != "full_grpo",
        device=device,
        rng=rng,
    )
    t_ppo = time.perf_counter() - t0

    if str(config.training_mode) == "full_grpo":
        avg_batch_reward = float(np.mean(_compute_full_grpo_episode_scores(trajectories, contexts, config)))
    else:
        avg_batch_reward = float(np.mean([t.total_reward for t in trajectories]))
    timing = {
        "time_context_sec": float(t_context),
        "time_rollout_sec": float(t_rollout),
        "time_buffer_sec": float(t_buffer),
        "time_cache_sec": float(t_cache),
        "time_ppo_sec": float(t_ppo),
        "time_total_sec": float(time.perf_counter() - t_start),
    }
    return ppo_metrics, avg_batch_reward, len(trajectories), len(transitions), timing


def run_ppo_training(config: PPOTrainingConfig) -> PPOTrainingResult:
    """Execute full PPO training loop and return run artifacts."""
    config.output_root.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = config.output_root / f"{_slugify(config.run_name)}_{timestamp}"
    run_dir.mkdir(parents=False, exist_ok=False)

    config_dir = run_dir / "config"
    logs_dir = run_dir / "logs"
    checkpoints_dir = run_dir / "checkpoints"
    config_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    _write_yaml(config_dir / "config_resolved.yaml", config.to_serializable_dict())
    _write_json(config_dir / "metadata.json", _build_metadata(run_dir=run_dir, seed=config.seed))

    device = _resolve_device(config.device)
    if device.type == "cpu" and config.rollout_mode == "legacy" and int(config.n_rollout_workers) > 1:
        # Avoid severe CPU oversubscription when many rollout workers call PyTorch.
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    _set_global_seeds(config.seed)
    rng = np.random.default_rng(config.seed)

    dataset = EpisodeDataset(config=config, rng=rng)
    model = build_actor_critic_from_config(config, device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )

    steps_log_path = logs_dir / "steps.jsonl"
    _append_step_log(
        steps_log_path,
        event="run_start",
        payload={
            "run_name": config.run_name,
            "seed": config.seed,
            "device": str(device),
            "n_cells_total": int(dataset.n_cells),
            "training_mode": str(config.training_mode),
            "rollout_mode": str(config.rollout_mode),
            "n_rollout_workers": int(config.n_rollout_workers),
            "gae_lambda": float(config.gae_lambda),
            "group_relative_enabled": bool(config.group_relative_enabled),
            "group_relative_group_size": int(config.group_relative_group_size),
            "group_relative_mix_alpha": float(config.group_relative_mix_alpha),
            "planner_enabled": bool(config.planner_enabled),
            "planner_interval": int(config.planner_interval),
            "planner_cot_weight": float(config.planner_cot_weight),
            "torch_num_threads": int(torch.get_num_threads()),
            "torch_num_interop_threads": int(torch.get_num_interop_threads()),
        },
    )

    reward_history: list[float] = []
    logs: list[TrainUpdateMetrics] = []
    best_moving_avg: float | None = None
    best_checkpoint_path: Path | None = None
    no_improve_count = 0
    stopped_early = False

    try:
        for update_idx in range(1, int(config.max_updates) + 1):
            ppo_metrics, avg_batch_reward, n_episodes, n_transitions, timing = train_one_update(
                model=model,
                optimizer=optimizer,
                dataset=dataset,
                config=config,
                device=device,
                rng=rng,
            )

            reward_history.append(avg_batch_reward)
            moving_avg_reward: float | None = None
            improved = False

            if len(reward_history) >= int(config.moving_avg_window):
                moving_avg_reward = float(np.mean(reward_history[-int(config.moving_avg_window) :]))
                if best_moving_avg is None or moving_avg_reward > best_moving_avg + float(config.min_improvement):
                    best_moving_avg = moving_avg_reward
                    no_improve_count = 0
                    improved = True
                    best_checkpoint_path = checkpoints_dir / "best_model.pt"
                    _save_checkpoint(
                        path=best_checkpoint_path,
                        model=model,
                        optimizer=optimizer,
                        update_index=update_idx,
                        best_moving_avg_reward=best_moving_avg,
                        config=config,
                    )
                else:
                    no_improve_count += 1

            row = TrainUpdateMetrics(
                update_index=int(update_idx),
                average_batch_reward=float(avg_batch_reward),
                moving_average_reward=None if moving_avg_reward is None else float(moving_avg_reward),
                policy_loss=float(ppo_metrics.policy_loss),
                value_loss=float(ppo_metrics.value_loss),
                entropy=float(ppo_metrics.entropy),
                total_loss=float(ppo_metrics.total_loss),
                approx_kl=float(ppo_metrics.approx_kl),
                no_improve_count=int(no_improve_count),
                n_episodes=int(n_episodes),
                n_transitions=int(n_transitions),
            )
            logs.append(row)

            _append_step_log(
                steps_log_path,
                event="update_complete",
                payload={
                    "update_index": row.update_index,
                    "average_batch_reward": row.average_batch_reward,
                    "moving_average_reward": row.moving_average_reward,
                    "policy_loss": row.policy_loss,
                    "value_loss": row.value_loss,
                    "entropy": row.entropy,
                    "total_loss": row.total_loss,
                    "approx_kl": row.approx_kl,
                    "no_improve_count": row.no_improve_count,
                    "n_episodes": row.n_episodes,
                    "n_transitions": row.n_transitions,
                    "best_moving_avg_reward": best_moving_avg,
                    "checkpoint_saved": bool(improved),
                    "training_mode": str(config.training_mode),
                    "gae_lambda": float(config.gae_lambda),
                    "group_relative_enabled": bool(config.group_relative_enabled),
                    "group_relative_group_size": int(config.group_relative_group_size),
                    "group_relative_mix_alpha": float(config.group_relative_mix_alpha),
                    "planner_enabled": bool(config.planner_enabled),
                    "planner_interval": int(config.planner_interval),
                    "planner_cot_weight": float(config.planner_cot_weight),
                    "time_context_sec": float(timing["time_context_sec"]),
                    "time_rollout_sec": float(timing["time_rollout_sec"]),
                    "time_buffer_sec": float(timing["time_buffer_sec"]),
                    "time_cache_sec": float(timing["time_cache_sec"]),
                    "time_ppo_sec": float(timing["time_ppo_sec"]),
                    "time_total_sec": float(timing["time_total_sec"]),
                },
            )

            if len(reward_history) >= int(config.moving_avg_window) and no_improve_count >= int(config.patience):
                stopped_early = True
                _append_step_log(
                    steps_log_path,
                    event="early_stop",
                    payload={
                        "update_index": int(update_idx),
                        "no_improve_count": int(no_improve_count),
                        "patience": int(config.patience),
                    },
                )
                break
    finally:
        dataset.close()

    final_ckpt = checkpoints_dir / "final_model.pt"
    _save_checkpoint(
        path=final_ckpt,
        model=model,
        optimizer=optimizer,
        update_index=len(logs),
        best_moving_avg_reward=best_moving_avg,
        config=config,
    )

    logs_df = pd.DataFrame(
        [
            {
                "update_index": row.update_index,
                "average_batch_reward": row.average_batch_reward,
                "moving_average_reward": row.moving_average_reward,
                "policy_loss": row.policy_loss,
                "value_loss": row.value_loss,
                "entropy": row.entropy,
                "total_loss": row.total_loss,
                "approx_kl": row.approx_kl,
                "no_improve_count": row.no_improve_count,
                "n_episodes": row.n_episodes,
                "n_transitions": row.n_transitions,
            }
            for row in logs
        ]
    )
    logs_df.to_csv(logs_dir / "training_metrics.csv", index=False)

    summary = {
        "updates_completed": int(len(logs)),
        "stopped_early": bool(stopped_early),
        "best_moving_average_reward": best_moving_avg,
        "best_checkpoint_path": None if best_checkpoint_path is None else str(best_checkpoint_path),
        "final_checkpoint_path": str(final_ckpt),
        "n_cells_total": int(dataset.n_cells),
    }
    _write_json(run_dir / "summary.json", summary)

    return PPOTrainingResult(
        run_dir=run_dir,
        best_checkpoint_path=best_checkpoint_path,
        best_moving_average_reward=best_moving_avg,
        updates_completed=len(logs),
        stopped_early=stopped_early,
        logs=tuple(logs),
    )


def run_ppo_training_from_config(config_path: str | Path) -> PPOTrainingResult:
    """Load config and run PPO training."""
    config = load_ppo_training_config(config_path)
    return run_ppo_training(config)




def _collate_minibatch_from_cache(
    indices: np.ndarray,
    rollout_cache: RolloutFeatureCache,
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Build one minibatch by slicing cached transition features."""
    idx_arr = np.asarray(indices, dtype=np.int64)
    n = int(idx_arr.size)
    if n == 0:
        raise ValueError("cannot collate empty minibatch")

    max_actions = 0
    for idx in idx_arr.tolist():
        ep = int(rollout_cache.episode_slot[idx])
        max_actions = max(max_actions, int(rollout_cache.episode_static[ep].action_template.shape[0]))

    g_batch = np.zeros((n, AddStopCellEnv.GLOBAL_FEATURE_DIM), dtype=np.float32)
    af_padded = np.zeros((n, max_actions, AddStopCellEnv.ACTION_FEATURE_DIM), dtype=np.float32)
    am_padded = np.zeros((n, max_actions), dtype=bool)

    for i, idx in enumerate(idx_arr.tolist()):
        ep = int(rollout_cache.episode_slot[idx])
        static = rollout_cache.episode_static[ep]
        template = static.action_template
        ai = int(template.shape[0])

        af_padded[i, :ai, :] = template
        am_padded[i, 0] = True
        if ai > 1:
            membership = _unpack_mask(
                rollout_cache.packed_membership_masks[idx],
                n_bits=static.n_bins,
            )
            frontier = compute_frontier_eligible_mask(membership, static.context.neighbor_index)
            am_padded[i, 1:ai] = frontier
            bundle = _compute_state_feature_bundle(
                ctx=static.context,
                membership_mask=membership,
                step_index=int(rollout_cache.step_frac[idx] * max(1, static.max_steps)),
                frontier_mask=frontier,
            )
            _fill_dynamic_action_features(
                action_features=af_padded[i, :ai, :],
                ctx=static.context,
                membership_mask=membership,
                bundle=bundle,
            )

        assigned_frac_i = float(rollout_cache.assigned_frac[idx])
        step_frac_i = float(rollout_cache.step_frac[idx])
        remaining_frac_i = float(rollout_cache.remaining_frac[idx])
        grow_ratio_scaled_i = float(rollout_cache.grow_ratio_scaled[idx])
        positive_frontier_fraction_i = float(rollout_cache.positive_frontier_fraction[idx])
        centroid_drift_scaled_i = float(rollout_cache.centroid_drift_scaled[idx])
        compactness_proxy_i = float(rollout_cache.compactness_proxy[idx])
        frontier_add_reward_topk_mean_i = float(rollout_cache.frontier_add_reward_topk_mean[idx])
        frontier_add_reward_max_i = float(rollout_cache.frontier_add_reward_max[idx])
        frontier_add_reward_mean_i = float(rollout_cache.frontier_add_reward_mean[idx])
        compact_streak_scaled_i = float(rollout_cache.compact_streak_scaled[idx])
        assigned_ll_mean_i = float(rollout_cache.assigned_ll_mean[idx])
        assigned_ll_max_i = float(rollout_cache.assigned_ll_max[idx])

        af_padded[i, 0, 1] = np.float32(assigned_frac_i)
        af_padded[i, 0, 2] = np.float32(step_frac_i)
        af_padded[i, 0, 4] = np.float32(assigned_ll_mean_i)
        af_padded[i, 0, 5] = np.float32(remaining_frac_i)
        af_padded[i, 0, 7] = np.float32(grow_ratio_scaled_i)
        af_padded[i, 0, 8] = np.float32(positive_frontier_fraction_i)
        af_padded[i, 0, 9] = np.float32(centroid_drift_scaled_i)
        af_padded[i, 0, 10] = np.float32(compactness_proxy_i)

        g_batch[i] = np.asarray(
            [
                assigned_frac_i,
                step_frac_i,
                float(static.n_bins_scaled),
                assigned_ll_mean_i,
                assigned_ll_max_i,
                remaining_frac_i,
                float(static.seed_size_scaled),
                grow_ratio_scaled_i,
                positive_frontier_fraction_i,
                centroid_drift_scaled_i,
                compactness_proxy_i,
                frontier_add_reward_topk_mean_i,
                frontier_add_reward_max_i,
                frontier_add_reward_mean_i,
                compact_streak_scaled_i,
            ],
            dtype=np.float32,
        )

    return {
        "global_features": torch.as_tensor(g_batch, device=device, dtype=torch.float32),
        "action_features": torch.as_tensor(af_padded, device=device, dtype=torch.float32),
        "action_mask": torch.as_tensor(am_padded, device=device, dtype=torch.bool),
        "actions": torch.as_tensor(rollout_cache.actions[idx_arr], device=device, dtype=torch.int64),
        "planner_modes": torch.as_tensor(rollout_cache.planner_modes[idx_arr], device=device, dtype=torch.int64),
        "old_log_probs": torch.as_tensor(rollout_cache.old_log_probs[idx_arr], device=device, dtype=torch.float32),
        "returns": torch.as_tensor(rollout_cache.returns[idx_arr], device=device, dtype=torch.float32),
        "advantages": torch.as_tensor(rollout_cache.advantages[idx_arr], device=device, dtype=torch.float32),
    }


def _build_global_feature_batch_from_cache(
    indices: np.ndarray,
    rollout_cache: RolloutFeatureCache,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct compact global feature batch for one PPO minibatch."""
    idx_arr = np.asarray(indices, dtype=np.int64)
    if idx_arr.ndim != 1:
        raise ValueError("minibatch indices must be 1D")
    episode_slot = rollout_cache.episode_slot[idx_arr]
    n = int(idx_arr.size)
    global_batch = np.empty((n, AddStopCellEnv.GLOBAL_FEATURE_DIM), dtype=np.float32)
    global_batch[:, G_ASSIGNED_FRAC] = rollout_cache.assigned_frac[idx_arr]
    global_batch[:, G_STEP_FRAC] = rollout_cache.step_frac[idx_arr]
    global_batch[:, G_N_BINS_SCALED] = np.fromiter(
        (rollout_cache.episode_static[int(ep)].n_bins_scaled for ep in episode_slot.tolist()),
        dtype=np.float32,
        count=n,
    )
    global_batch[:, G_ASSIGNED_LL_MEAN] = rollout_cache.assigned_ll_mean[idx_arr]
    global_batch[:, G_ASSIGNED_LL_MAX] = rollout_cache.assigned_ll_max[idx_arr]
    global_batch[:, G_REMAINING_FRAC] = rollout_cache.remaining_frac[idx_arr]
    global_batch[:, G_SEED_SIZE_SCALED] = np.fromiter(
        (rollout_cache.episode_static[int(ep)].seed_size_scaled for ep in episode_slot.tolist()),
        dtype=np.float32,
        count=n,
    )
    global_batch[:, G_GROW_RATIO_SCALED] = rollout_cache.grow_ratio_scaled[idx_arr]
    global_batch[:, G_POSITIVE_FRONTIER_FRACTION] = rollout_cache.positive_frontier_fraction[idx_arr]
    global_batch[:, G_CENTROID_DRIFT_SCALED] = rollout_cache.centroid_drift_scaled[idx_arr]
    global_batch[:, G_COMPACTNESS_PROXY] = rollout_cache.compactness_proxy[idx_arr]
    global_batch[:, G_FRONTIER_ADD_REWARD_TOPK_MEAN] = rollout_cache.frontier_add_reward_topk_mean[idx_arr]
    global_batch[:, G_FRONTIER_ADD_REWARD_MAX] = rollout_cache.frontier_add_reward_max[idx_arr]
    global_batch[:, G_FRONTIER_ADD_REWARD_MEAN] = rollout_cache.frontier_add_reward_mean[idx_arr]
    global_batch[:, G_COMPACT_STREAK_SCALED] = rollout_cache.compact_streak_scaled[idx_arr]
    return global_batch, episode_slot


def _build_global_feature_batch_from_planner_cache(
    indices: np.ndarray,
    planner_cache: PlannerRolloutFeatureCache,
) -> np.ndarray:
    idx_arr = np.asarray(indices, dtype=np.int64)
    episode_slot = planner_cache.episode_slot[idx_arr]
    n = int(idx_arr.size)
    global_batch = np.empty((n, AddStopCellEnv.GLOBAL_FEATURE_DIM), dtype=np.float32)
    global_batch[:, G_ASSIGNED_FRAC] = planner_cache.assigned_frac[idx_arr]
    global_batch[:, G_STEP_FRAC] = planner_cache.step_frac[idx_arr]
    global_batch[:, G_N_BINS_SCALED] = np.fromiter(
        (planner_cache.episode_static[int(ep)].n_bins_scaled for ep in episode_slot.tolist()),
        dtype=np.float32,
        count=n,
    )
    global_batch[:, G_ASSIGNED_LL_MEAN] = planner_cache.assigned_ll_mean[idx_arr]
    global_batch[:, G_ASSIGNED_LL_MAX] = planner_cache.assigned_ll_max[idx_arr]
    global_batch[:, G_REMAINING_FRAC] = planner_cache.remaining_frac[idx_arr]
    global_batch[:, G_SEED_SIZE_SCALED] = np.fromiter(
        (planner_cache.episode_static[int(ep)].seed_size_scaled for ep in episode_slot.tolist()),
        dtype=np.float32,
        count=n,
    )
    global_batch[:, G_GROW_RATIO_SCALED] = planner_cache.grow_ratio_scaled[idx_arr]
    global_batch[:, G_POSITIVE_FRONTIER_FRACTION] = planner_cache.positive_frontier_fraction[idx_arr]
    global_batch[:, G_CENTROID_DRIFT_SCALED] = planner_cache.centroid_drift_scaled[idx_arr]
    global_batch[:, G_COMPACTNESS_PROXY] = planner_cache.compactness_proxy[idx_arr]
    global_batch[:, G_FRONTIER_ADD_REWARD_TOPK_MEAN] = planner_cache.frontier_add_reward_topk_mean[idx_arr]
    global_batch[:, G_FRONTIER_ADD_REWARD_MAX] = planner_cache.frontier_add_reward_max[idx_arr]
    global_batch[:, G_FRONTIER_ADD_REWARD_MEAN] = planner_cache.frontier_add_reward_mean[idx_arr]
    global_batch[:, G_COMPACT_STREAK_SCALED] = planner_cache.compact_streak_scaled[idx_arr]
    return global_batch


def _evaluate_planner_minibatch_from_cache(
    *,
    model: ActorCritic,
    planner_cache: PlannerRolloutFeatureCache,
    indices: np.ndarray,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    idx_arr = np.asarray(indices, dtype=np.int64)
    global_batch = _build_global_feature_batch_from_planner_cache(idx_arr, planner_cache)
    global_t = torch.as_tensor(global_batch, device=device, dtype=torch.float32)
    actions_t = torch.as_tensor(planner_cache.actions[idx_arr], device=device, dtype=torch.int64)
    dist = model.planner_distribution(global_t)
    return {
        "new_log_probs": dist.log_prob(actions_t),
        "entropy": dist.entropy(),
        "old_log_probs": torch.as_tensor(planner_cache.old_log_probs[idx_arr], device=device, dtype=torch.float32),
        "advantages": torch.as_tensor(planner_cache.advantages[idx_arr], device=device, dtype=torch.float32),
    }


def _evaluate_minibatch_from_cache_grouped(
    *,
    model: ActorCritic,
    rollout_cache: RolloutFeatureCache,
    indices: np.ndarray,
    device: torch.device,
    config: PPOTrainingConfig,
) -> dict[str, torch.Tensor]:
    """Evaluate one PPO minibatch with exact reconstructed dynamic action features."""
    batch = _collate_minibatch_from_cache(indices=indices, rollout_cache=rollout_cache, device=device)
    if bool(config.planner_enabled) and batch["action_mask"].shape[1] > 0:
        batch["action_mask"][:, 0] = False
    action_bias = _planner_logit_bias_from_action_features(batch["action_features"], batch["planner_modes"], config)
    dist, values = model(
        batch["global_features"],
        batch["action_features"],
        batch["action_mask"],
        action_logit_bias=action_bias,
    )
    new_log_probs = dist.log_prob(batch["actions"])
    entropy_values = dist.entropy()
    return {
        "new_log_probs": new_log_probs,
        "values": values,
        "entropy": entropy_values,
        "old_log_probs": batch["old_log_probs"],
        "returns": batch["returns"],
        "advantages": batch["advantages"],
    }


def _set_global_seeds(seed: int | None) -> None:
    if seed is None:
        return
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("run.device is 'cuda' but CUDA is not available")
        return torch.device("cuda")
    return torch.device("cpu")
