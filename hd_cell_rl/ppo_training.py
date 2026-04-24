"""PyTorch PPO training pipeline for HD cell ADD/STOP environments.

This module trains one shared actor-critic policy across many single-cell episodes.
Each outer update samples a random batch of cells, rolls out one episode per cell,
then runs PPO-Clip updates on the collected transitions.
"""

from __future__ import annotations

from dataclasses import dataclass
import datetime as dt
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
from typing import Any
from concurrent.futures import ThreadPoolExecutor
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical
import yaml

from .reward import (
    build_eight_neighbor_index,
    compute_expression_confidence,
    compute_stop_delta,
    compute_frontier_eligible_mask,
    compute_neighbor_support_fraction,
    compute_reference_distribution,
)
from .reward_grid_search import (
    _MatrixOnDemandExpressionLoader,
    _build_nuclei_centers,
    _build_nuclei_spatial_index,
    _load_episode_build_expression_context,
    _load_one_episode_artifact,
)

_LOCAL_TIMEZONE = ZoneInfo("America/Chicago")
_LOCAL_TIMEZONE_NAME = "America/Chicago"


class ConfigError(ValueError):
    """Raised when PPO config is invalid."""


@dataclass(frozen=True)
class PPOTrainingConfig:
    """Resolved configuration for one PPO training run."""

    run_name: str
    output_root: Path
    seed: int | None
    device: str
    batch_cells: int
    rollout_mode: str
    n_rollout_workers: int
    max_updates: int
    max_steps_per_episode: int | None
    training_mode: str

    episodes_index_path: Path
    reference_path: Path
    reference_format: str
    reference_array_key: str
    reference_genes_key: str
    nuclei_path: Path
    nuclei_format: str
    nuclei_columns: dict[str, str]
    expression_cache_size: int | None

    gamma: float
    gae_lambda: float
    normalize_returns_per_episode: bool
    normalize_advantages: bool
    eps_clip: float
    ppo_epochs: int
    minibatch_size: int
    learning_rate: float
    weight_decay: float
    vf_coef: float
    ent_coef: float
    max_grad_norm: float
    hidden_dim: int
    target_kl: float | None

    group_relative_enabled: bool
    group_relative_group_size: int
    group_relative_mix_alpha: float
    group_relative_norm_epsilon: float
    group_relative_score: str

    full_grpo_reward_weight: float
    full_grpo_size_weight: float
    full_grpo_stop_weight: float
    full_grpo_compact_weight: float
    full_grpo_small_over_weight: float
    full_grpo_large_under_weight: float
    full_grpo_small_seed_max: int
    full_grpo_large_seed_min: int
    full_grpo_small_target_min: float
    full_grpo_small_target_max: float
    full_grpo_medium_target_min: float
    full_grpo_medium_target_max: float
    full_grpo_large_target_min: float
    full_grpo_large_target_max: float

    epsilon: float
    r_max_um: float
    w1: float
    w2: float
    w3: float
    w4: float
    stop_lambda: float
    stop_stat: str
    stop_top_k: int
    expression_confidence_pseudocount: float
    normalize_expression_zscore: bool
    zscore_delta: float

    moving_avg_window: int
    min_improvement: float
    patience: int

    def to_serializable_dict(self) -> dict[str, Any]:
        """Return config as plain Python types for YAML/JSON output."""
        return {
            "run": {
                "name": self.run_name,
                "output_root": str(self.output_root),
                "seed": self.seed,
                "device": self.device,
                "batch_cells": self.batch_cells,
                "rollout_mode": self.rollout_mode,
                "n_rollout_workers": self.n_rollout_workers,
                "max_updates": self.max_updates,
                "max_steps_per_episode": self.max_steps_per_episode,
                "training_mode": self.training_mode,
            },
            "inputs": {
                "episodes_index_path": str(self.episodes_index_path),
                "reference": {
                    "path": str(self.reference_path),
                    "format": self.reference_format,
                    "array_key": self.reference_array_key,
                    "genes_key": self.reference_genes_key,
                },
                "nuclei": {
                    "path": str(self.nuclei_path),
                    "format": self.nuclei_format,
                    "columns": self.nuclei_columns,
                },
                "expression_cache_size": self.expression_cache_size,
            },
            "ppo": {
                "gamma": self.gamma,
                "gae_lambda": self.gae_lambda,
                "normalize_returns_per_episode": self.normalize_returns_per_episode,
                "normalize_advantages": self.normalize_advantages,
                "eps_clip": self.eps_clip,
                "ppo_epochs": self.ppo_epochs,
                "minibatch_size": self.minibatch_size,
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "vf_coef": self.vf_coef,
                "ent_coef": self.ent_coef,
                "max_grad_norm": self.max_grad_norm,
                "hidden_dim": self.hidden_dim,
                "target_kl": self.target_kl,
            },
            "group_relative": {
                "enabled": self.group_relative_enabled,
                "group_size": self.group_relative_group_size,
                "mix_alpha": self.group_relative_mix_alpha,
                "norm_epsilon": self.group_relative_norm_epsilon,
                "score": self.group_relative_score,
            },
            "full_grpo": {
                "reward_weight": self.full_grpo_reward_weight,
                "size_weight": self.full_grpo_size_weight,
                "stop_weight": self.full_grpo_stop_weight,
                "compact_weight": self.full_grpo_compact_weight,
                "small_over_weight": self.full_grpo_small_over_weight,
                "large_under_weight": self.full_grpo_large_under_weight,
                "small_seed_max": self.full_grpo_small_seed_max,
                "large_seed_min": self.full_grpo_large_seed_min,
                "targets": {
                    "small": [self.full_grpo_small_target_min, self.full_grpo_small_target_max],
                    "medium": [self.full_grpo_medium_target_min, self.full_grpo_medium_target_max],
                    "large": [self.full_grpo_large_target_min, self.full_grpo_large_target_max],
                },
            },
            "reward": {
                "epsilon": self.epsilon,
                "r_max_um": self.r_max_um,
                "w1": self.w1,
                "w2": self.w2,
                "w3": self.w3,
                "w4": self.w4,
                "stop_lambda": self.stop_lambda,
                "stop_stat": self.stop_stat,
                "stop_top_k": self.stop_top_k,
                "expression_confidence_pseudocount": self.expression_confidence_pseudocount,
                "normalize_expression_zscore": self.normalize_expression_zscore,
                "zscore_delta": self.zscore_delta,
            },
            "stopping": {
                "moving_avg_window": self.moving_avg_window,
                "min_improvement": self.min_improvement,
                "patience": self.patience,
            },
        }


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
    neighbor_index: np.ndarray  # (B, 8), int32
    max_steps: int
    log_prior: float
    r_max_um: float
    w1: float
    w2: float
    w3: float
    w4: float
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
class EpisodeStep:
    """One transition sampled during rollout."""

    packed_membership_mask: np.ndarray
    step_index: int
    action: int
    reward: float
    done: bool
    old_log_prob: float
    old_value: float


@dataclass(frozen=True)
class EpisodeTrajectory:
    """One variable-length trajectory (one cell episode)."""

    episode_slot: int
    steps: tuple[EpisodeStep, ...]
    total_reward: float


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
    assigned_ll_mean: np.ndarray  # (N,), float32
    assigned_ll_max: np.ndarray  # (N,), float32
    actions: np.ndarray  # (N,), int64
    old_log_probs: np.ndarray  # (N,), float32
    returns: np.ndarray  # (N,), float32
    advantages: np.ndarray  # (N,), float32

    @property
    def n_transitions(self) -> int:
        return int(self.actions.shape[0])


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


class ActorCritic(nn.Module):
    """Masked actor-critic model over variable-size action sets."""

    def __init__(self, global_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.global_dim = int(global_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.global_encoder = nn.Sequential(
            nn.Linear(global_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.logit_head = nn.Linear(hidden_dim * 2, 1)
        self.value_head = nn.Linear(hidden_dim, 1)

    def encode_global(self, global_features: torch.Tensor) -> torch.Tensor:
        """Encode state-level features for critic/value prediction."""
        return self.global_encoder(global_features)

    def encode_action_features(self, action_features: torch.Tensor) -> torch.Tensor:
        """Encode per-action features for masked actor scoring."""
        return self.action_encoder(action_features)

    def value_from_global_latent(self, global_latent: torch.Tensor) -> torch.Tensor:
        """Predict scalar state value from encoded global features."""
        return self.value_head(global_latent).squeeze(-1)

    def policy_logits_from_action_latent(self, action_latent: torch.Tensor) -> torch.Tensor:
        """Score actions from action latents only.

        The original actor head is linear over [action_latent, global_latent]. The
        global half contributes the same scalar offset to every action within one
        state, so it cancels exactly inside the categorical softmax. Using only the
        action half is therefore mathematically equivalent for policy probabilities
        while avoiding repeated global-action concatenation.
        """
        actor_weight = self.logit_head.weight[:, : self.hidden_dim]
        return F.linear(action_latent, actor_weight, self.logit_head.bias).squeeze(-1)

    def stop_logits_from_global_features(self, global_features: torch.Tensor) -> torch.Tensor:
        """Build the STOP row from global features and score it with the actor."""
        if global_features.ndim != 2 or global_features.shape[1] != self.global_dim:
            raise ValueError(
                f"global_features must have shape (N, {self.global_dim}) to score STOP logits"
            )
        stop_features = global_features.new_zeros((global_features.shape[0], self.action_dim))
        stop_features[:, 0] = 1.0
        stop_features[:, 1] = global_features[:, 0]
        stop_features[:, 2] = global_features[:, 1]
        stop_features[:, 3] = global_features[:, 2]
        stop_features[:, 4] = global_features[:, 3]
        stop_features[:, 5] = global_features[:, 5]
        stop_features[:, 6] = global_features[:, 6]
        stop_features[:, 7] = global_features[:, 7]
        stop_features[:, 8] = global_features[:, 8]
        stop_features[:, 9] = global_features[:, 9]
        stop_features[:, 10] = global_features[:, 10]
        stop_latent = self.encode_action_features(stop_features)
        return self.policy_logits_from_action_latent(stop_latent)

    def forward(
        self,
        global_features: torch.Tensor,
        action_features: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple[Categorical, torch.Tensor]:
        """Return masked action distribution and state values.

        Parameters
        ----------
        global_features:
            Shape (N, G)
        action_features:
            Shape (N, A, F)
        action_mask:
            Shape (N, A), True for valid actions.
        """
        if global_features.ndim != 2:
            raise ValueError("global_features must have shape (N, G)")
        if action_features.ndim != 3:
            raise ValueError("action_features must have shape (N, A, F)")
        if action_mask.ndim != 2:
            raise ValueError("action_mask must have shape (N, A)")
        if action_features.shape[0] != global_features.shape[0]:
            raise ValueError("batch dimension mismatch between global_features and action_features")
        if action_mask.shape[0] != global_features.shape[0]:
            raise ValueError("batch dimension mismatch between global_features and action_mask")
        if action_mask.shape[1] != action_features.shape[1]:
            raise ValueError("action dimension mismatch between action_features and action_mask")

        n_batch, n_actions, _ = action_features.shape

        g = self.encode_global(global_features)  # (N, H)
        a = self.encode_action_features(action_features.reshape(n_batch * n_actions, -1)).reshape(n_batch, n_actions, -1)
        logits = self.policy_logits_from_action_latent(a)  # (N, A)

        # Use a large negative value so invalid actions have effectively zero probability.
        neg_large = torch.finfo(logits.dtype).min
        masked_logits = logits.masked_fill(~action_mask, neg_large)
        dist = Categorical(logits=masked_logits)
        values = self.value_from_global_latent(g)  # (N,)
        return dist, values


class EpisodeDataset:
    """On-demand loader for episode artifacts, optimized for many-cell datasets."""

    def __init__(self, config: PPOTrainingConfig, rng: np.random.Generator) -> None:
        self._config = config
        self._rng = rng

        if not config.episodes_index_path.exists():
            raise FileNotFoundError(f"episodes index file not found: {config.episodes_index_path}")
        index_df = pd.read_csv(config.episodes_index_path)
        required = ["cell_id", "artifact_path"]
        missing = [col for col in required if col not in index_df.columns]
        if missing:
            raise ValueError(f"episodes index missing required columns: {missing}")
        self._index_df = index_df.reset_index(drop=True)

        self._reference_counts = _load_reference_counts(
            path=config.reference_path,
            reference_format=config.reference_format,
            array_key=config.reference_array_key,
        )
        self._theta = compute_reference_distribution(self._reference_counts, epsilon=config.epsilon)
        self._log_theta = np.log(self._theta)

        nuclei_df = _load_table(config.nuclei_path, config.nuclei_format)
        centers = _build_nuclei_centers(df=nuclei_df, columns=config.nuclei_columns)
        self._nuclei_spatial_index = _build_nuclei_spatial_index(centers)

        expression_ctx = _load_episode_build_expression_context(config.episodes_index_path)
        self._expression_loader: _MatrixOnDemandExpressionLoader | None = None
        self._nuclear_barcode_to_cell: dict[str, str] = {}
        if expression_ctx is not None and config.reference_format == "npz":
            cache_size = (
                int(config.expression_cache_size)
                if config.expression_cache_size is not None
                else int(expression_ctx["cache_size"])
            )
            self._expression_loader = _MatrixOnDemandExpressionLoader(
                matrix_path=Path(expression_ctx["matrix_path"]),
                reference_npz_path=config.reference_path,
                reference_genes_key=config.reference_genes_key,
                cache_size=cache_size,
            )
            bins_path = expression_ctx.get("bins_path")
            if bins_path is not None:
                self._nuclear_barcode_to_cell = _load_nuclear_barcode_assignment_lookup(Path(bins_path))

    @property
    def n_cells(self) -> int:
        return int(len(self._index_df))

    def close(self) -> None:
        if self._expression_loader is not None:
            self._expression_loader.close()

    def sample_rows(self, n_rows: int) -> pd.DataFrame:
        """Sample rows from training set; with replacement if needed."""
        if n_rows <= 0:
            raise ValueError("n_rows must be > 0")
        if len(self._index_df) == 0:
            raise ValueError("episodes index is empty")

        replace = n_rows > len(self._index_df)
        sampled = self._rng.choice(len(self._index_df), size=n_rows, replace=replace)
        return self._index_df.iloc[np.asarray(sampled, dtype=np.int64)].reset_index(drop=True)

    def load_episode_context(
        self,
        cell_id: str,
        artifact_path: Path,
        max_steps_per_episode: int | None,
        *,
        include_candidate_bin_ids: bool = False,
    ) -> EpisodeContext | None:
        """Load one episode artifact and convert it into training-ready static context."""
        prepared = _load_one_episode_artifact(
            artifact_path=artifact_path,
            cell_id=cell_id,
            expression_loader=self._expression_loader,
            theta=self._theta,
            log_theta=self._log_theta,
            nuclei_spatial_index=self._nuclei_spatial_index,
            include_candidate_bin_ids=True,
        )
        if prepared is None:
            return None

        ll = np.asarray(prepared.precomputed_ll, dtype=np.float32)
        if ll.ndim != 2 or ll.shape[0] == 0 or ll.shape[1] == 0:
            return None

        bin_xy = np.asarray(prepared.candidate_bin_xy_um, dtype=np.float32)
        nucleus_center = np.asarray(prepared.nucleus_center_xy_um, dtype=np.float32)
        delta = bin_xy - nucleus_center[None, :]
        d_n = np.sqrt(np.sum(delta * delta, axis=1, dtype=np.float32))
        p_dis = (d_n / float(self._config.r_max_um)).astype(np.float32, copy=False)
        d_other = np.asarray(prepared.precomputed_d_other_um, dtype=np.float32)
        p_overlap = np.maximum(0.0, (d_n - d_other) / float(self._config.r_max_um)).astype(np.float32, copy=False)

        ll_mean = np.mean(ll, axis=1)
        ll_max = np.max(ll, axis=1)
        ll_mean_z = _zscore_1d(ll_mean).astype(np.float32, copy=False)
        ll_max_z = _zscore_1d(ll_max).astype(np.float32, copy=False)
        w2 = float(self._config.w2)
        w3 = float(self._config.w3)
        base_penalty = (w2 * p_dis + w3 * p_overlap).astype(np.float32, copy=False)
        expression_confidence = compute_expression_confidence(
            bin_count_totals=np.asarray(prepared.bin_count_totals, dtype=np.float64),
            pseudocount=float(self._config.expression_confidence_pseudocount),
        ).astype(np.float32, copy=False)
        neighbor_index = build_eight_neighbor_index(prepared.candidate_bin_ids, bin_xy).astype(np.int32, copy=False)
        initial_membership_mask = _build_initial_membership_mask(
            candidate_bin_ids=prepared.candidate_bin_ids,
            cell_id=str(prepared.cell_id),
            nuclear_barcode_to_cell=self._nuclear_barcode_to_cell,
        )

        max_steps = int(max_steps_per_episode) if max_steps_per_episode is not None else max(1, ll.shape[0] * 3)
        return EpisodeContext(
            cell_id=str(prepared.cell_id),
            candidate_bin_ids=tuple(prepared.candidate_bin_ids),
            initial_membership_mask=initial_membership_mask,
            candidate_bin_xy_um=bin_xy.astype(np.float32, copy=False),
            nucleus_center_xy_um=nucleus_center.astype(np.float32, copy=False),
            ll=ll,
            p_dis=p_dis,
            p_overlap=p_overlap,
            ll_mean_z=ll_mean_z,
            ll_max_z=ll_max_z,
            base_penalty=base_penalty,
            expression_confidence=expression_confidence,
            neighbor_index=neighbor_index,
            max_steps=max_steps,
            log_prior=-np.log(float(ll.shape[1])),
            r_max_um=float(self._config.r_max_um),
            w1=float(self._config.w1),
            w2=w2,
            w3=w3,
            w4=float(self._config.w4),
            stop_lambda=float(self._config.stop_lambda),
            stop_stat=str(self._config.stop_stat),
            stop_top_k=int(self._config.stop_top_k),
            expression_confidence_pseudocount=float(self._config.expression_confidence_pseudocount),
            normalize_expression_zscore=bool(self._config.normalize_expression_zscore),
            zscore_delta=float(self._config.zscore_delta),
        )


def _normalize_cell_id(value: Any) -> str | None:
    """Normalize mixed int/float/string cell IDs into one stable string form."""
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        if not np.isfinite(value):
            return None
        if float(value).is_integer():
            return str(int(value))
        return str(value)
    text = str(value).strip()
    if not text:
        return None
    if re.fullmatch(r"[+-]?\\d+\\.0+", text):
        return text.split(".", 1)[0]
    return text


def _load_nuclear_barcode_assignment_lookup(bins_path: Path) -> dict[str, str]:
    """Load confident nuclear barcode -> cell_id assignments from merged bins metadata."""
    if not bins_path.exists():
        raise FileNotFoundError(f"episode-build bins metadata not found: {bins_path}")

    df = pd.read_parquet(
        bins_path,
        columns=[
            "barcode",
            "has_nuclear_annotation",
            "dominant_cell_id",
            "ambiguous_nuclear_assignment",
        ],
    )
    if len(df) == 0:
        return {}

    has_nuclear = df["has_nuclear_annotation"].fillna(False).astype(bool)
    ambiguous = df["ambiguous_nuclear_assignment"].fillna(False).astype(bool)
    out = df.loc[has_nuclear & ~ambiguous, ["barcode", "dominant_cell_id"]].copy()
    if out.empty:
        return {}

    out["cell_id"] = out["dominant_cell_id"].map(_normalize_cell_id)
    out = out.loc[out["cell_id"].notna()].copy()
    if out.empty:
        return {}

    return {
        str(barcode): str(cell_id)
        for barcode, cell_id in zip(out["barcode"].astype(str), out["cell_id"].astype(str), strict=False)
    }


def _build_initial_membership_mask(
    *,
    candidate_bin_ids: tuple[str, ...],
    cell_id: str,
    nuclear_barcode_to_cell: dict[str, str],
) -> np.ndarray:
    """Seed episode state with all confident nuclear bins for this cell."""
    if not candidate_bin_ids:
        return np.zeros((0,), dtype=np.uint8)
    normalized_cell_id = _normalize_cell_id(cell_id)
    return np.asarray(
        [
            1 if normalized_cell_id is not None and nuclear_barcode_to_cell.get(str(bin_id)) == normalized_cell_id else 0
            for bin_id in candidate_bin_ids
        ],
        dtype=np.uint8,
    )


class AddStopCellEnv:
    """Fast ADD/STOP environment using precomputed episode context.

    Notes for customization:
    - This is where you can change action semantics (e.g., add REMOVE later).
    - `_build_policy_observation` defines policy input features.
    - reward formulas are implemented in `_add_reward_for_bin` and `_stop_reward`.
    """

    ACTION_FEATURE_DIM = 13
    GLOBAL_FEATURE_DIM = 11

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
        posterior = self._posterior()
        neighbor_support = float(
            compute_neighbor_support_fraction(self._membership_mask, self._ctx.neighbor_index)[bin_idx]
        )
        r_expr = (self._ctx.ll @ posterior) * self._ctx.expression_confidence
        if self._ctx.normalize_expression_zscore:
            eligible = self._action_mask[1:]
            if np.any(eligible):
                expr_eligible = r_expr[eligible]
                mu = float(np.mean(expr_eligible))
                sigma = float(np.std(expr_eligible, ddof=0))
                expr_term = (float(r_expr[bin_idx]) - mu) / (sigma + self._ctx.zscore_delta)
            else:
                expr_term = 0.0
            return float(self._ctx.w1 * expr_term - self._ctx.base_penalty[bin_idx] + self._ctx.w4 * neighbor_support)
        return float(self._ctx.w1 * float(r_expr[bin_idx]) - self._ctx.base_penalty[bin_idx] + self._ctx.w4 * neighbor_support)

    def _all_add_rewards(self, posterior: np.ndarray) -> np.ndarray:
        r_expr = (self._ctx.ll @ posterior) * self._ctx.expression_confidence
        if self._ctx.normalize_expression_zscore:
            eligible = self._action_mask[1:]
            if np.any(eligible):
                mu = float(np.mean(r_expr[eligible]))
                sigma = float(np.std(r_expr[eligible], ddof=0))
                expr_term = (r_expr - mu) / (sigma + self._ctx.zscore_delta)
            else:
                expr_term = np.zeros_like(r_expr)
        else:
            expr_term = r_expr
        neighbor_support = compute_neighbor_support_fraction(self._membership_mask, self._ctx.neighbor_index).astype(
            np.float32,
            copy=False,
        )
        return self._ctx.w1 * expr_term - self._ctx.base_penalty + self._ctx.w4 * neighbor_support

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
        summary = _compute_state_summary_from_mask(
            ctx=self._ctx,
            membership_mask=self._membership_mask,
            step_index=self._step_index,
        )

        global_features = self._global_features
        global_features[0] = np.float32(summary["assigned_frac"])
        global_features[1] = np.float32(summary["step_frac"])
        global_features[2] = np.float32(self._n_bins_scaled)
        global_features[3] = np.float32(summary["assigned_ll_mean"])
        global_features[4] = np.float32(summary["assigned_ll_max"])
        global_features[5] = np.float32(summary["remaining_frac"])
        global_features[6] = np.float32(self._seed_size_scaled)
        global_features[7] = np.float32(summary["grow_ratio_scaled"])
        global_features[8] = np.float32(summary["positive_frontier_fraction"])
        global_features[9] = np.float32(summary["centroid_drift_scaled"])
        global_features[10] = np.float32(summary["compactness_proxy"])

        action_features = self._action_features
        # STOP action dynamic fields.
        action_features[0, 1] = np.float32(summary["assigned_frac"])
        action_features[0, 2] = np.float32(summary["step_frac"])
        action_features[0, 4] = np.float32(summary["assigned_ll_mean"])
        action_features[0, 5] = np.float32(summary["remaining_frac"])
        action_features[0, 7] = np.float32(summary["grow_ratio_scaled"])
        action_features[0, 8] = np.float32(summary["positive_frontier_fraction"])
        action_features[0, 9] = np.float32(summary["centroid_drift_scaled"])
        action_features[0, 10] = np.float32(summary["compactness_proxy"])

        return {
            "global_features": global_features,
            "action_features": action_features,
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
            self._action_features[1:, 5] = self._membership_mask.astype(np.float32, copy=False)
            self._action_features[1:, 11:13] = np.float32(0.0)
            candidate_centroid_distance, candidate_compactness_gain = _compute_dynamic_add_action_features(
                ctx=self._ctx,
                membership_mask=self._membership_mask,
                frontier_mask=frontier,
            )
            self._action_features[1:, 11] = candidate_centroid_distance
            self._action_features[1:, 12] = candidate_compactness_gain

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


def run_episode(
    env: AddStopCellEnv,
    model: ActorCritic,
    device: torch.device,
    rng: np.random.Generator,
) -> EpisodeTrajectory:
    """Roll out exactly one episode and collect PPO-required fields."""
    obs, _ = env.reset()
    steps: list[EpisodeStep] = []
    total_reward = 0.0

    while True:
        global_t, action_t, mask_t = _observation_to_tensors(obs, device=device)
        with torch.inference_mode():
            dist, value = model(global_t, action_t, mask_t)
            probs = dist.probs.squeeze(0).detach().cpu().numpy()
            prob_sum = float(np.sum(probs))
            if prob_sum <= 0.0 or not np.isfinite(prob_sum):
                raise RuntimeError("action probabilities became non-finite or non-positive")
            probs = probs / prob_sum
            action = int(rng.choice(probs.shape[0], p=probs))
            action_tensor = torch.as_tensor([action], device=device, dtype=torch.int64)
            old_log_prob = float(dist.log_prob(action_tensor).item())

        old_value = float(value.item())

        packed_mask = _pack_mask(np.asarray(obs["membership_mask"], dtype=np.uint8))
        step_index = int(obs["step_index"])

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)
        total_reward += float(reward)
        steps.append(
            EpisodeStep(
                packed_membership_mask=packed_mask,
                step_index=step_index,
                action=action,
                reward=float(reward),
                done=done,
                old_log_prob=old_log_prob,
                old_value=old_value,
            )
        )
        obs = next_obs
        if done:
            break

    # Caller fills the actual episode slot index after collection.
    return EpisodeTrajectory(episode_slot=-1, steps=tuple(steps), total_reward=float(total_reward))


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


def _compute_full_grpo_episode_scores(
    trajectories: list["EpisodeTrajectory"],
    episode_contexts: list[EpisodeContext],
    config: PPOTrainingConfig,
) -> np.ndarray:
    """Compute no-GT terminal scores used for strict full-GRPO ranking."""
    scores = np.zeros((len(trajectories),), dtype=np.float64)
    for i, traj in enumerate(trajectories):
        ctx = episode_contexts[int(traj.episode_slot)]
        final_mask = _final_membership_mask_from_trajectory(ctx=ctx, trajectory=traj)
        assigned_count = int(np.sum(final_mask))
        seed_count = max(1, int(np.sum(ctx.initial_membership_mask)))
        grow_ratio = float(assigned_count / seed_count)
        lower, upper, bucket = _full_grpo_size_target_interval(seed_count, config)

        mean_step_reward = float(traj.total_reward / max(1, len(traj.steps)))
        size_score = _full_grpo_size_score(
            grow_ratio=grow_ratio,
            lower=lower,
            upper=upper,
            bucket=bucket,
            config=config,
        )
        stop_score = _full_grpo_stop_score(
            ctx=ctx,
            trajectory=traj,
            final_mask=final_mask,
            grow_ratio=grow_ratio,
            lower=lower,
            upper=upper,
        )
        _, _, compactness = _compute_shape_frontier_features(ctx=ctx, membership_mask=final_mask)

        scores[i] = (
            float(config.full_grpo_reward_weight) * mean_step_reward
            + float(config.full_grpo_size_weight) * size_score
            + float(config.full_grpo_stop_weight) * stop_score
            + float(config.full_grpo_compact_weight) * compactness
        )
    return scores


def _compute_full_grpo_episode_advantages(
    trajectories: list["EpisodeTrajectory"],
    episode_contexts: list[EpisodeContext],
    *,
    group_size: int,
    norm_epsilon: float,
    config: PPOTrainingConfig,
) -> np.ndarray:
    """Standardize full-GRPO terminal scores within each same-cell group."""
    scores = _compute_full_grpo_episode_scores(trajectories, episode_contexts, config)
    if len(scores) % int(group_size) != 0:
        raise ValueError("trajectory count must be divisible by group_size")
    advantages = np.zeros_like(scores, dtype=np.float64)
    for start in range(0, len(scores), int(group_size)):
        stop = start + int(group_size)
        group = scores[start:stop]
        advantages[start:stop] = (group - float(np.mean(group))) / (float(np.std(group, ddof=0)) + norm_epsilon)
    return advantages


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


def _full_grpo_size_target_interval(seed_count: int, config: PPOTrainingConfig) -> tuple[float, float, str]:
    if int(seed_count) <= int(config.full_grpo_small_seed_max):
        return float(config.full_grpo_small_target_min), float(config.full_grpo_small_target_max), "small"
    if int(seed_count) >= int(config.full_grpo_large_seed_min):
        return float(config.full_grpo_large_target_min), float(config.full_grpo_large_target_max), "large"
    return float(config.full_grpo_medium_target_min), float(config.full_grpo_medium_target_max), "medium"


def _full_grpo_size_score(
    *,
    grow_ratio: float,
    lower: float,
    upper: float,
    bucket: str,
    config: PPOTrainingConfig,
) -> float:
    under = max(0.0, float(lower) - float(grow_ratio)) / max(float(lower), 1.0e-8)
    over = max(0.0, float(grow_ratio) - float(upper)) / max(float(upper), 1.0e-8)
    under_weight = float(config.full_grpo_large_under_weight) if bucket == "large" else 1.0
    over_weight = float(config.full_grpo_small_over_weight) if bucket == "small" else 1.0
    return -float(under_weight * under * under + over_weight * over * over)


def _full_grpo_stop_score(
    *,
    ctx: EpisodeContext,
    trajectory: EpisodeTrajectory,
    final_mask: np.ndarray,
    grow_ratio: float,
    lower: float,
    upper: float,
) -> float:
    if not trajectory.steps:
        return 0.0

    final_action = int(trajectory.steps[-1].action)
    stopped = final_action == 0
    frontier = compute_frontier_eligible_mask(final_mask, ctx.neighbor_index)
    frontier_delta = 0.0
    if np.any(frontier):
        posterior = _posterior_from_membership_mask(ctx=ctx, membership_mask=final_mask)
        neighbor_support = compute_neighbor_support_fraction(final_mask, ctx.neighbor_index)
        add_rewards = _add_rewards_from_membership_mask(
            ctx=ctx,
            membership_mask=final_mask,
            posterior=posterior,
            neighbor_support=neighbor_support,
            frontier_mask=frontier,
        )
        frontier_delta = max(
            0.0,
            float(
                compute_stop_delta(
                    add_rewards,
                    frontier,
                    stop_stat=ctx.stop_stat,
                    stop_top_k=ctx.stop_top_k,
                )
            ),
        )

    score = 0.0
    if stopped and float(grow_ratio) < float(lower):
        score -= frontier_delta
    if not stopped and float(grow_ratio) > float(upper):
        score -= (float(grow_ratio) - float(upper)) / max(float(upper), 1.0e-8)
    return score


def ppo_update(
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    rollout_cache: RolloutFeatureCache,
    *,
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
    if rollout_cache.n_transitions == 0:
        raise ValueError("transitions must not be empty")

    n = int(rollout_cache.n_transitions)
    all_indices = np.arange(n, dtype=np.int64)

    policy_losses: list[float] = []
    value_losses: list[float] = []
    entropies: list[float] = []
    total_losses: list[float] = []
    approx_kls: list[float] = []

    stop_early = False
    for _epoch in range(int(ppo_epochs)):
        if stop_early:
            break
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
    )
    model.train()
    t_rollout = time.perf_counter() - t0

    t0 = time.perf_counter()
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
    )
    t_buffer = time.perf_counter() - t0

    t0 = time.perf_counter()
    rollout_cache = _build_rollout_feature_cache(
        transitions=transitions,
        episode_contexts=contexts,
    )
    t_cache = time.perf_counter() - t0

    t0 = time.perf_counter()
    ppo_metrics = ppo_update(
        model=model,
        optimizer=optimizer,
        rollout_cache=rollout_cache,
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


def _collect_episode_contexts(
    *,
    dataset: EpisodeDataset,
    batch_cells: int,
    max_steps_per_episode: int | None,
) -> list[EpisodeContext]:
    """Sample episode rows and load static contexts for one outer update."""
    contexts: list[EpisodeContext] = []
    max_attempts = max(int(batch_cells) * 10, 100)
    attempts = 0

    while len(contexts) < int(batch_cells) and attempts < max_attempts:
        needed = int(batch_cells) - len(contexts)
        sampled = dataset.sample_rows(needed)
        for row in sampled.itertuples(index=False):
            attempts += 1
            cell_id = str(row.cell_id)
            artifact_path = Path(str(row.artifact_path)).expanduser().resolve()
            ctx = dataset.load_episode_context(
                cell_id=cell_id,
                artifact_path=artifact_path,
                max_steps_per_episode=max_steps_per_episode,
            )
            if ctx is None:
                continue
            contexts.append(ctx)
            if len(contexts) >= int(batch_cells):
                break
    return contexts


def _expand_group_relative_contexts(
    *,
    contexts: list[EpisodeContext],
    group_size: int,
) -> list[EpisodeContext]:
    """Duplicate sampled contexts contiguously for grouped rollouts."""
    if group_size <= 1:
        raise ValueError("group_size must be > 1")
    expanded: list[EpisodeContext] = []
    for ctx in contexts:
        expanded.extend([ctx] * int(group_size))
    return expanded


def _collect_trajectories(
    *,
    contexts: list[EpisodeContext],
    model: ActorCritic,
    device: torch.device,
    rng: np.random.Generator,
    rollout_mode: str,
    n_rollout_workers: int,
) -> list[EpisodeTrajectory]:
    """Collect one rollout trajectory per context."""
    if not contexts:
        return []

    mode = str(rollout_mode).strip().lower()
    if mode == "vectorized":
        return _collect_trajectories_vectorized(contexts=contexts, model=model, device=device, rng=rng)
    if mode == "legacy":
        return _collect_trajectories_legacy(
            contexts=contexts,
            model=model,
            device=device,
            rng=rng,
            n_rollout_workers=n_rollout_workers,
        )
    raise ValueError(f"unsupported rollout mode: {rollout_mode!r}")


def _collect_trajectories_legacy(
    *,
    contexts: list[EpisodeContext],
    model: ActorCritic,
    device: torch.device,
    rng: np.random.Generator,
    n_rollout_workers: int,
) -> list[EpisodeTrajectory]:
    """Legacy rollout path: one env per worker with optional threading."""
    worker_count = max(1, int(n_rollout_workers))
    seeds = rng.integers(low=0, high=np.iinfo(np.uint32).max, size=len(contexts), dtype=np.uint64)
    specs = [(i, contexts[i], int(seeds[i])) for i in range(len(contexts))]

    if worker_count == 1 or len(contexts) == 1:
        trajectories: list[EpisodeTrajectory] = []
        for episode_slot, ctx, seed in specs:
            trajectories.append(_rollout_worker(episode_slot, ctx, model, device, seed))
        return trajectories

    trajectories = []
    max_workers = min(worker_count, len(contexts))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_rollout_worker, slot, ctx, model, device, seed) for slot, ctx, seed in specs]
        for future in futures:
            trajectories.append(future.result())

    trajectories.sort(key=lambda x: x.episode_slot)
    return trajectories


def _collect_trajectories_vectorized(
    *,
    contexts: list[EpisodeContext],
    model: ActorCritic,
    device: torch.device,
    rng: np.random.Generator,
) -> list[EpisodeTrajectory]:
    """Synchronous vectorized rollout: batch active envs each step."""
    n_envs = len(contexts)
    envs = [AddStopCellEnv(ctx) for ctx in contexts]
    observations: list[dict[str, Any]] = []
    for env in envs:
        obs, _ = env.reset()
        observations.append(obs)

    episode_steps: list[list[EpisodeStep]] = [[] for _ in range(n_envs)]
    total_rewards = np.zeros((n_envs,), dtype=np.float64)
    active_slots = list(range(n_envs))

    while active_slots:
        active_obs = [observations[slot] for slot in active_slots]
        global_t, action_t, mask_t = _batch_observations_for_rollout(active_obs, device=device)

        with torch.inference_mode():
            dist, values = model(global_t, action_t, mask_t)
            probs = dist.probs
            if probs.ndim != 2:
                raise RuntimeError("expected batched action probabilities with shape (N, A)")
            if not torch.isfinite(probs).all():
                raise RuntimeError("action probabilities became non-finite")
            actions_t = dist.sample()
            log_probs_t = dist.log_prob(actions_t)

        actions = actions_t.detach().cpu().numpy().astype(np.int64, copy=False)
        old_log_probs = log_probs_t.detach().cpu().numpy().astype(np.float32, copy=False)
        old_values = values.detach().cpu().numpy().astype(np.float32, copy=False)

        next_active_slots: list[int] = []
        for local_idx, slot in enumerate(active_slots):
            obs = observations[slot]
            packed_mask = _pack_mask(np.asarray(obs["membership_mask"], dtype=np.uint8))
            step_index = int(obs["step_index"])
            action = int(actions[local_idx])
            old_log_prob = float(old_log_probs[local_idx])
            old_value = float(old_values[local_idx])

            next_obs, reward, terminated, truncated, _ = envs[slot].step(action)
            done = bool(terminated or truncated)
            total_rewards[slot] += float(reward)
            episode_steps[slot].append(
                EpisodeStep(
                    packed_membership_mask=packed_mask,
                    step_index=step_index,
                    action=action,
                    reward=float(reward),
                    done=done,
                    old_log_prob=old_log_prob,
                    old_value=old_value,
                )
            )
            observations[slot] = next_obs
            if not done:
                next_active_slots.append(slot)
        active_slots = next_active_slots

    return [
        EpisodeTrajectory(
            episode_slot=int(slot),
            steps=tuple(episode_steps[slot]),
            total_reward=float(total_rewards[slot]),
        )
        for slot in range(n_envs)
    ]


def _rollout_worker(
    episode_slot: int,
    ctx: EpisodeContext,
    model: ActorCritic,
    device: torch.device,
    seed: int,
) -> EpisodeTrajectory:
    """Run one episode rollout in a worker thread."""
    local_rng = np.random.default_rng(seed)
    env = AddStopCellEnv(ctx)
    traj = run_episode(env=env, model=model, device=device, rng=local_rng)
    return EpisodeTrajectory(
        episode_slot=int(episode_slot),
        steps=traj.steps,
        total_reward=float(traj.total_reward),
    )


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
    global_batch = np.zeros((n, AddStopCellEnv.GLOBAL_FEATURE_DIM), dtype=np.float32)
    action_batch = np.zeros((n, max_actions, AddStopCellEnv.ACTION_FEATURE_DIM), dtype=np.float32)
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

    global_batch = np.zeros((n, AddStopCellEnv.GLOBAL_FEATURE_DIM), dtype=np.float32)
    for i, obs in enumerate(observations):
        global_batch[i] = np.asarray(obs["global_features"], dtype=np.float32)
    return torch.as_tensor(global_batch, device=device, dtype=torch.float32)


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
    model = ActorCritic(
        global_dim=AddStopCellEnv.GLOBAL_FEATURE_DIM,
        action_dim=AddStopCellEnv.ACTION_FEATURE_DIM,
        hidden_dim=int(config.hidden_dim),
    ).to(device)
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


def load_ppo_training_config(config_path: str | Path) -> PPOTrainingConfig:
    """Load and validate PPO YAML config."""
    path = Path(config_path)
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ConfigError("config root must be a mapping")

    run = _as_dict(raw.get("run"), "run")
    inputs = _as_dict(raw.get("inputs"), "inputs")
    ppo = _as_dict(raw.get("ppo"), "ppo")
    reward = _as_dict(raw.get("reward"), "reward")
    stopping = _as_dict(raw.get("stopping"), "stopping")

    run_name = str(run.get("name", "hd_cell_ppo")).strip()
    if not run_name:
        raise ConfigError("run.name must be non-empty")
    output_root = Path(str(run.get("output_root", "runs"))).expanduser().resolve()
    seed_raw = run.get("seed")
    seed = None if seed_raw is None else int(seed_raw)
    device = str(run.get("device", "auto")).strip().lower()
    if device not in {"auto", "cpu", "cuda"}:
        raise ConfigError("run.device must be one of: auto, cpu, cuda")
    batch_cells = int(run.get("batch_cells", 100))
    if batch_cells <= 0:
        raise ConfigError("run.batch_cells must be > 0")
    rollout_mode = str(run.get("rollout_mode", "vectorized")).strip().lower()
    if rollout_mode not in {"vectorized", "legacy"}:
        raise ConfigError("run.rollout_mode must be one of: vectorized, legacy")
    n_rollout_workers = int(run.get("n_rollout_workers", 1))
    if n_rollout_workers <= 0:
        raise ConfigError("run.n_rollout_workers must be > 0")
    max_updates = int(run.get("max_updates", 1000))
    if max_updates <= 0:
        raise ConfigError("run.max_updates must be > 0")
    max_steps_raw = run.get("max_steps_per_episode")
    max_steps_per_episode = None if max_steps_raw is None else int(max_steps_raw)
    if max_steps_per_episode is not None and max_steps_per_episode <= 0:
        raise ConfigError("run.max_steps_per_episode must be > 0 when provided")
    training_mode = str(run.get("training_mode", "ppo")).strip().lower()
    if training_mode not in {"ppo", "full_grpo"}:
        raise ConfigError("run.training_mode must be one of: ppo, full_grpo")

    episodes_index_path = Path(str(_require(inputs, "episodes_index_path", "inputs"))).expanduser().resolve()
    reference_cfg = _as_dict(_require(inputs, "reference", "inputs"), "inputs.reference")
    reference_path = Path(str(_require(reference_cfg, "path", "inputs.reference"))).expanduser().resolve()
    reference_format = _normalize_format(str(reference_cfg.get("format", "auto")), reference_path)
    reference_array_key = str(reference_cfg.get("array_key", "reference_counts"))
    reference_genes_key = str(reference_cfg.get("genes_key", "genes"))

    nuclei_cfg = _as_dict(_require(inputs, "nuclei", "inputs"), "inputs.nuclei")
    nuclei_path = Path(str(_require(nuclei_cfg, "path", "inputs.nuclei"))).expanduser().resolve()
    nuclei_format = _normalize_format(str(nuclei_cfg.get("format", "auto")), nuclei_path)
    nuclei_columns = _default_nuclei_center_columns(_as_dict(nuclei_cfg.get("columns", {}), "inputs.nuclei.columns"))

    expression_cache_size_raw = inputs.get("expression_cache_size", None)
    expression_cache_size = None if expression_cache_size_raw is None else int(expression_cache_size_raw)
    if expression_cache_size is not None and expression_cache_size < 0:
        raise ConfigError("inputs.expression_cache_size must be >= 0 when provided")

    gamma = float(ppo.get("gamma", 0.99))
    if not (0.0 < gamma <= 1.0):
        raise ConfigError("ppo.gamma must be in (0, 1]")
    gae_lambda = float(ppo.get("gae_lambda", 0.95))
    if not (0.0 <= gae_lambda <= 1.0):
        raise ConfigError("ppo.gae_lambda must be in [0, 1]")
    normalize_returns_per_episode = bool(ppo.get("normalize_returns_per_episode", True))
    normalize_advantages = bool(ppo.get("normalize_advantages", True))
    eps_clip = float(ppo.get("eps_clip", 0.2))
    if eps_clip <= 0:
        raise ConfigError("ppo.eps_clip must be > 0")
    ppo_epochs = int(ppo.get("ppo_epochs", 4))
    if ppo_epochs <= 0:
        raise ConfigError("ppo.ppo_epochs must be > 0")
    minibatch_size = int(ppo.get("minibatch_size", 256))
    if minibatch_size <= 0:
        raise ConfigError("ppo.minibatch_size must be > 0")
    learning_rate = float(ppo.get("learning_rate", 3e-4))
    if learning_rate <= 0:
        raise ConfigError("ppo.learning_rate must be > 0")
    weight_decay = float(ppo.get("weight_decay", 0.0))
    if weight_decay < 0:
        raise ConfigError("ppo.weight_decay must be >= 0")
    vf_coef = float(ppo.get("vf_coef", 0.5))
    if vf_coef < 0:
        raise ConfigError("ppo.vf_coef must be >= 0")
    ent_coef = float(ppo.get("ent_coef", 0.01))
    if ent_coef < 0:
        raise ConfigError("ppo.ent_coef must be >= 0")
    max_grad_norm = float(ppo.get("max_grad_norm", 1.0))
    if max_grad_norm <= 0:
        raise ConfigError("ppo.max_grad_norm must be > 0")
    hidden_dim = int(ppo.get("hidden_dim", 128))
    if hidden_dim <= 0:
        raise ConfigError("ppo.hidden_dim must be > 0")
    target_kl_raw = ppo.get("target_kl", None)
    target_kl = None if target_kl_raw is None else float(target_kl_raw)
    if target_kl is not None and target_kl <= 0:
        raise ConfigError("ppo.target_kl must be > 0 when provided")

    group_relative = _as_dict(raw.get("group_relative", {}), "group_relative")
    group_relative_enabled = bool(group_relative.get("enabled", False))
    group_relative_group_size = int(group_relative.get("group_size", 4))
    if group_relative_group_size <= 1:
        raise ConfigError("group_relative.group_size must be > 1")
    group_relative_mix_alpha = float(group_relative.get("mix_alpha", 0.3))
    if not (0.0 <= group_relative_mix_alpha <= 1.0):
        raise ConfigError("group_relative.mix_alpha must be in [0, 1]")
    group_relative_norm_epsilon = float(group_relative.get("norm_epsilon", 1.0e-6))
    if group_relative_norm_epsilon <= 0:
        raise ConfigError("group_relative.norm_epsilon must be > 0")
    group_relative_score = str(group_relative.get("score", "episode_total_reward")).strip().lower()
    if group_relative_score != "episode_total_reward":
        raise ConfigError("group_relative.score must be 'episode_total_reward'")
    if (group_relative_enabled or training_mode == "full_grpo") and batch_cells % group_relative_group_size != 0:
        raise ConfigError("run.batch_cells must be divisible by group_relative.group_size when enabled")
    if training_mode == "full_grpo" and not group_relative_enabled:
        raise ConfigError("group_relative.enabled must be true when run.training_mode is full_grpo")

    full_grpo = _as_dict(raw.get("full_grpo", {}), "full_grpo")
    full_grpo_reward_weight = float(full_grpo.get("reward_weight", 1.0))
    full_grpo_size_weight = float(full_grpo.get("size_weight", 0.8))
    full_grpo_stop_weight = float(full_grpo.get("stop_weight", 0.4))
    full_grpo_compact_weight = float(full_grpo.get("compact_weight", 0.2))
    full_grpo_small_over_weight = float(full_grpo.get("small_over_weight", 1.2))
    full_grpo_large_under_weight = float(full_grpo.get("large_under_weight", 1.2))
    full_grpo_small_seed_max = int(full_grpo.get("small_seed_max", 8))
    full_grpo_large_seed_min = int(full_grpo.get("large_seed_min", 17))
    if full_grpo_small_seed_max <= 0:
        raise ConfigError("full_grpo.small_seed_max must be > 0")
    if full_grpo_large_seed_min <= full_grpo_small_seed_max:
        raise ConfigError("full_grpo.large_seed_min must be > full_grpo.small_seed_max")
    for name, val in (
        ("reward_weight", full_grpo_reward_weight),
        ("size_weight", full_grpo_size_weight),
        ("stop_weight", full_grpo_stop_weight),
        ("compact_weight", full_grpo_compact_weight),
        ("small_over_weight", full_grpo_small_over_weight),
        ("large_under_weight", full_grpo_large_under_weight),
    ):
        if val < 0:
            raise ConfigError(f"full_grpo.{name} must be >= 0")
    targets = _as_dict(full_grpo.get("targets", {}), "full_grpo.targets")
    small_target = _parse_grpo_target_interval(targets.get("small", [1.4, 2.4]), "full_grpo.targets.small")
    medium_target = _parse_grpo_target_interval(targets.get("medium", [1.6, 2.8]), "full_grpo.targets.medium")
    large_target = _parse_grpo_target_interval(targets.get("large", [1.8, 3.4]), "full_grpo.targets.large")

    epsilon = float(reward.get("epsilon", 1e-8))
    if epsilon < 0:
        raise ConfigError("reward.epsilon must be >= 0")
    r_max_um = float(reward.get("r_max_um", 80.0))
    if r_max_um <= 0:
        raise ConfigError("reward.r_max_um must be > 0")
    w1 = float(reward.get("w1", 1.0))
    w2 = float(reward.get("w2", 1.0))
    w3 = float(reward.get("w3", 1.0))
    w4 = float(reward.get("w4", 0.0))
    stop_lambda = float(reward.get("stop_lambda", 1.0))
    stop_stat = str(reward.get("stop_stat", "max")).strip().lower()
    if stop_stat not in {"max", "topk_mean"}:
        raise ConfigError("reward.stop_stat must be 'max' or 'topk_mean'")
    stop_top_k = int(reward.get("stop_top_k", 3))
    if stop_top_k <= 0:
        raise ConfigError("reward.stop_top_k must be > 0")
    expression_confidence_pseudocount = float(reward.get("expression_confidence_pseudocount", 5.0))
    if expression_confidence_pseudocount < 0:
        raise ConfigError("reward.expression_confidence_pseudocount must be >= 0")
    for name, val in (("w1", w1), ("w2", w2), ("w3", w3), ("stop_lambda", stop_lambda)):
        if val <= 0:
            raise ConfigError(f"reward.{name} must be > 0")
    if w4 < 0:
        raise ConfigError("reward.w4 must be >= 0")
    normalize_expression_zscore = bool(reward.get("normalize_expression_zscore", False))
    zscore_delta = float(reward.get("zscore_delta", 1e-8))
    if zscore_delta <= 0:
        raise ConfigError("reward.zscore_delta must be > 0")

    moving_avg_window = int(stopping.get("moving_avg_window", 20))
    if moving_avg_window <= 0:
        raise ConfigError("stopping.moving_avg_window must be > 0")
    min_improvement = float(stopping.get("min_improvement", 0.001))
    if min_improvement < 0:
        raise ConfigError("stopping.min_improvement must be >= 0")
    patience = int(stopping.get("patience", 20))
    if patience <= 0:
        raise ConfigError("stopping.patience must be > 0")

    return PPOTrainingConfig(
        run_name=run_name,
        output_root=output_root,
        seed=seed,
        device=device,
        batch_cells=batch_cells,
        rollout_mode=rollout_mode,
        n_rollout_workers=n_rollout_workers,
        max_updates=max_updates,
        max_steps_per_episode=max_steps_per_episode,
        training_mode=training_mode,
        episodes_index_path=episodes_index_path,
        reference_path=reference_path,
        reference_format=reference_format,
        reference_array_key=reference_array_key,
        reference_genes_key=reference_genes_key,
        nuclei_path=nuclei_path,
        nuclei_format=nuclei_format,
        nuclei_columns=nuclei_columns,
        expression_cache_size=expression_cache_size,
        gamma=gamma,
        gae_lambda=gae_lambda,
        normalize_returns_per_episode=normalize_returns_per_episode,
        normalize_advantages=normalize_advantages,
        eps_clip=eps_clip,
        ppo_epochs=ppo_epochs,
        minibatch_size=minibatch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        vf_coef=vf_coef,
        ent_coef=ent_coef,
        max_grad_norm=max_grad_norm,
        hidden_dim=hidden_dim,
        target_kl=target_kl,
        group_relative_enabled=group_relative_enabled,
        group_relative_group_size=group_relative_group_size,
        group_relative_mix_alpha=group_relative_mix_alpha,
        group_relative_norm_epsilon=group_relative_norm_epsilon,
        group_relative_score=group_relative_score,
        full_grpo_reward_weight=full_grpo_reward_weight,
        full_grpo_size_weight=full_grpo_size_weight,
        full_grpo_stop_weight=full_grpo_stop_weight,
        full_grpo_compact_weight=full_grpo_compact_weight,
        full_grpo_small_over_weight=full_grpo_small_over_weight,
        full_grpo_large_under_weight=full_grpo_large_under_weight,
        full_grpo_small_seed_max=full_grpo_small_seed_max,
        full_grpo_large_seed_min=full_grpo_large_seed_min,
        full_grpo_small_target_min=small_target[0],
        full_grpo_small_target_max=small_target[1],
        full_grpo_medium_target_min=medium_target[0],
        full_grpo_medium_target_max=medium_target[1],
        full_grpo_large_target_min=large_target[0],
        full_grpo_large_target_max=large_target[1],
        epsilon=epsilon,
        r_max_um=r_max_um,
        w1=w1,
        w2=w2,
        w3=w3,
        w4=w4,
        stop_lambda=stop_lambda,
        stop_stat=stop_stat,
        stop_top_k=stop_top_k,
        expression_confidence_pseudocount=expression_confidence_pseudocount,
        normalize_expression_zscore=normalize_expression_zscore,
        zscore_delta=zscore_delta,
        moving_avg_window=moving_avg_window,
        min_improvement=min_improvement,
        patience=patience,
    )


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
) -> list[RolloutTransition]:
    out: list[RolloutTransition] = []
    mode = str(training_mode).strip().lower()
    if mode == "full_grpo":
        if episode_contexts is None or config is None:
            raise ValueError("full_grpo rollout buffer requires episode_contexts and config")
        group_relative_bonus = _compute_full_grpo_episode_advantages(
            trajectories,
            episode_contexts,
            group_size=int(group_relative_group_size),
            norm_epsilon=float(group_relative_norm_epsilon),
            config=config,
        )
    else:
        group_relative_bonus = (
            _compute_group_relative_episode_advantages(
                trajectories,
                group_size=int(group_relative_group_size),
                norm_epsilon=float(group_relative_norm_epsilon),
                score=str(group_relative_score),
            )
            if group_relative_enabled and trajectories
            else None
        )
    for traj_idx, traj in enumerate(trajectories):
        rewards = np.asarray([s.reward for s in traj.steps], dtype=np.float64)
        values = np.asarray([s.old_value for s in traj.steps], dtype=np.float64)
        dones = np.asarray([s.done for s in traj.steps], dtype=bool)
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
        for i, step in enumerate(traj.steps):
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
            )
            for i, t in enumerate(out)
        ]
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
            assigned_ll_mean=np.zeros((0,), dtype=np.float32),
            assigned_ll_max=np.zeros((0,), dtype=np.float32),
            actions=np.zeros((0,), dtype=np.int64),
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
    assigned_ll_mean = np.zeros((n,), dtype=np.float32)
    assigned_ll_max = np.zeros((n,), dtype=np.float32)
    actions = np.zeros((n,), dtype=np.int64)
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
        assigned_ll_mean[i] = np.float32(summary["assigned_ll_mean"])
        assigned_ll_max[i] = np.float32(summary["assigned_ll_max"])
        actions[i] = int(t.action)
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
        assigned_ll_mean=assigned_ll_mean,
        assigned_ll_max=assigned_ll_max,
        actions=actions,
        old_log_probs=old_log_probs,
        returns=returns,
        advantages=advantages,
    )


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
            af_padded[i, 1:ai, 5] = membership.astype(np.float32, copy=False)
            frontier = compute_frontier_eligible_mask(membership, static.context.neighbor_index)
            am_padded[i, 1:ai] = frontier
            candidate_centroid_distance, candidate_compactness_gain = _compute_dynamic_add_action_features(
                ctx=static.context,
                membership_mask=membership,
                frontier_mask=frontier,
            )
            af_padded[i, 1:ai, 11] = candidate_centroid_distance
            af_padded[i, 1:ai, 12] = candidate_compactness_gain

        assigned_frac_i = float(rollout_cache.assigned_frac[idx])
        step_frac_i = float(rollout_cache.step_frac[idx])
        remaining_frac_i = float(rollout_cache.remaining_frac[idx])
        grow_ratio_scaled_i = float(rollout_cache.grow_ratio_scaled[idx])
        positive_frontier_fraction_i = float(rollout_cache.positive_frontier_fraction[idx])
        centroid_drift_scaled_i = float(rollout_cache.centroid_drift_scaled[idx])
        compactness_proxy_i = float(rollout_cache.compactness_proxy[idx])
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
            ],
            dtype=np.float32,
        )

    return {
        "global_features": torch.as_tensor(g_batch, device=device, dtype=torch.float32),
        "action_features": torch.as_tensor(af_padded, device=device, dtype=torch.float32),
        "action_mask": torch.as_tensor(am_padded, device=device, dtype=torch.bool),
        "actions": torch.as_tensor(rollout_cache.actions[idx_arr], device=device, dtype=torch.int64),
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
    global_batch[:, 0] = rollout_cache.assigned_frac[idx_arr]
    global_batch[:, 1] = rollout_cache.step_frac[idx_arr]
    global_batch[:, 2] = np.fromiter(
        (rollout_cache.episode_static[int(ep)].n_bins_scaled for ep in episode_slot.tolist()),
        dtype=np.float32,
        count=n,
    )
    global_batch[:, 3] = rollout_cache.assigned_ll_mean[idx_arr]
    global_batch[:, 4] = rollout_cache.assigned_ll_max[idx_arr]
    global_batch[:, 5] = rollout_cache.remaining_frac[idx_arr]
    global_batch[:, 6] = np.fromiter(
        (rollout_cache.episode_static[int(ep)].seed_size_scaled for ep in episode_slot.tolist()),
        dtype=np.float32,
        count=n,
    )
    global_batch[:, 7] = rollout_cache.grow_ratio_scaled[idx_arr]
    global_batch[:, 8] = rollout_cache.positive_frontier_fraction[idx_arr]
    global_batch[:, 9] = rollout_cache.centroid_drift_scaled[idx_arr]
    global_batch[:, 10] = rollout_cache.compactness_proxy[idx_arr]
    return global_batch, episode_slot


def _evaluate_minibatch_from_cache_grouped(
    *,
    model: ActorCritic,
    rollout_cache: RolloutFeatureCache,
    indices: np.ndarray,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Evaluate one PPO minibatch with exact reconstructed dynamic action features."""
    batch = _collate_minibatch_from_cache(indices=indices, rollout_cache=rollout_cache, device=device)
    dist, values = model(batch["global_features"], batch["action_features"], batch["action_mask"])
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
    summary = _compute_state_summary_from_mask(ctx=ctx, membership_mask=mask, step_index=step_index)

    action_features = static_template.copy()
    action_mask = np.zeros(ctx.n_bins + 1, dtype=bool)
    action_mask[0] = True
    if ctx.n_bins > 0:
        action_features[1:, 5] = mask.astype(np.float32, copy=False)
        action_mask[1:] = compute_frontier_eligible_mask(mask, ctx.neighbor_index)
        candidate_centroid_distance, candidate_compactness_gain = _compute_dynamic_add_action_features(
            ctx=ctx,
            membership_mask=mask,
            frontier_mask=action_mask[1:],
        )
        action_features[1:, 11] = candidate_centroid_distance
        action_features[1:, 12] = candidate_compactness_gain
    action_features[0, 1] = np.float32(summary["assigned_frac"])
    action_features[0, 2] = np.float32(summary["step_frac"])
    action_features[0, 4] = np.float32(summary["assigned_ll_mean"])
    action_features[0, 5] = np.float32(summary["remaining_frac"])
    action_features[0, 7] = np.float32(summary["grow_ratio_scaled"])
    action_features[0, 8] = np.float32(summary["positive_frontier_fraction"])
    action_features[0, 9] = np.float32(summary["centroid_drift_scaled"])
    action_features[0, 10] = np.float32(summary["compactness_proxy"])

    global_features = np.asarray(
        [
            summary["assigned_frac"],
            summary["step_frac"],
            n_bins_scaled,
            summary["assigned_ll_mean"],
            summary["assigned_ll_max"],
            summary["remaining_frac"],
            seed_size_scaled,
            summary["grow_ratio_scaled"],
            summary["positive_frontier_fraction"],
            summary["centroid_drift_scaled"],
            summary["compactness_proxy"],
        ],
        dtype=np.float32,
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

    Dynamic fields are filled per state:
    - STOP row columns [1, 2, 4, 5, 7, 8, 9, 10]
    - ADD rows columns [5, 11, 12]
    """
    n_bins = int(ctx.n_bins)
    template = np.zeros((n_bins + 1, AddStopCellEnv.ACTION_FEATURE_DIM), dtype=np.float32)
    template[0, 0] = np.float32(1.0)
    template[0, 3] = np.float32(n_bins_scaled)
    template[0, 6] = np.float32(seed_size_scaled)
    if n_bins > 0:
        template[1:, 0] = np.float32(0.0)
        template[1:, 1] = ctx.ll_mean_z.astype(np.float32, copy=False)
        template[1:, 2] = ctx.ll_max_z.astype(np.float32, copy=False)
        template[1:, 3] = ctx.p_dis.astype(np.float32, copy=False)
        template[1:, 4] = ctx.p_overlap.astype(np.float32, copy=False)
    return template


def _compute_state_summary_from_mask(
    *,
    ctx: EpisodeContext,
    membership_mask: np.ndarray,
    step_index: int,
) -> dict[str, float]:
    """Compute dynamic scalar features for one state."""
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

    if assigned_count > 0:
        m = mask.astype(np.float32, copy=False)
        assigned_ll_mean = float(np.dot(m, ctx.ll_mean_z) / assigned_count)
        assigned_ll_max = float(np.dot(m, ctx.ll_max_z) / assigned_count)
    else:
        assigned_ll_mean = 0.0
        assigned_ll_max = 0.0

    initial_seed_count = int(np.sum(ctx.initial_membership_mask))
    grow_ratio_scaled = _scale_grow_ratio_feature(assigned_count, initial_seed_count)
    positive_frontier_fraction, centroid_drift_scaled, compactness_proxy = _compute_shape_frontier_features(
        ctx=ctx,
        membership_mask=mask,
    )

    return {
        "assigned_frac": assigned_frac,
        "step_frac": step_frac,
        "remaining_frac": remaining_frac,
        "grow_ratio_scaled": grow_ratio_scaled,
        "positive_frontier_fraction": positive_frontier_fraction,
        "centroid_drift_scaled": centroid_drift_scaled,
        "compactness_proxy": compactness_proxy,
        "assigned_ll_mean": assigned_ll_mean,
        "assigned_ll_max": assigned_ll_max,
    }


def _compute_shape_frontier_features(
    *,
    ctx: EpisodeContext,
    membership_mask: np.ndarray,
) -> tuple[float, float, float]:
    """Compute dynamic shape/frontier summaries used by STOP/value features."""
    mask = np.asarray(membership_mask, dtype=np.uint8)
    assigned = mask.astype(bool, copy=False)
    assigned_count = int(np.sum(assigned))

    neighbor_support = compute_neighbor_support_fraction(mask, ctx.neighbor_index)

    if assigned_count > 0:
        compactness_proxy = float(np.mean(neighbor_support[assigned]))
        centroid_xy = np.mean(ctx.candidate_bin_xy_um[assigned], axis=0, dtype=np.float64)
        drift_um = float(np.sqrt(np.sum((centroid_xy - ctx.nucleus_center_xy_um) ** 2)))
        centroid_drift_scaled = float(min(max(drift_um / max(float(ctx.r_max_um), 1.0e-8), 0.0), 1.0))
    else:
        compactness_proxy = 0.0
        centroid_drift_scaled = 0.0

    frontier = compute_frontier_eligible_mask(mask, ctx.neighbor_index)
    if np.any(frontier):
        posterior = _posterior_from_membership_mask(ctx=ctx, membership_mask=mask)
        add_rewards = _add_rewards_from_membership_mask(
            ctx=ctx,
            membership_mask=mask,
            posterior=posterior,
            neighbor_support=neighbor_support,
            frontier_mask=frontier,
        )
        positive_frontier_fraction = float(np.mean(add_rewards[frontier] > 0.0))
    else:
        positive_frontier_fraction = 0.0

    return positive_frontier_fraction, centroid_drift_scaled, compactness_proxy


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
    n_bins = int(ctx.n_bins)
    candidate_to_current_centroid_distance = np.zeros((n_bins,), dtype=np.float32)
    candidate_compactness_gain = np.zeros((n_bins,), dtype=np.float32)
    if n_bins == 0:
        return candidate_to_current_centroid_distance, candidate_compactness_gain

    mask = np.asarray(membership_mask, dtype=np.uint8)
    assigned = mask.astype(bool, copy=False)
    assigned_count = int(np.sum(assigned))
    if frontier_mask is None:
        frontier = compute_frontier_eligible_mask(mask, ctx.neighbor_index)
    else:
        frontier = np.asarray(frontier_mask, dtype=bool)
        if frontier.shape != (n_bins,):
            raise ValueError(f"frontier_mask shape mismatch: expected {(n_bins,)}, got {frontier.shape}")

    if assigned_count > 0:
        current_centroid_xy = np.mean(ctx.candidate_bin_xy_um[assigned], axis=0, dtype=np.float64)
    else:
        current_centroid_xy = np.asarray(ctx.nucleus_center_xy_um, dtype=np.float64)

    if np.any(frontier):
        xy = np.asarray(ctx.candidate_bin_xy_um, dtype=np.float64)
        centroid_dist_um = np.sqrt(np.sum((xy - current_centroid_xy) ** 2, axis=1))
        scaled_dist = centroid_dist_um / max(float(ctx.r_max_um), 1.0e-8)
        candidate_to_current_centroid_distance[frontier] = np.clip(scaled_dist[frontier], 0.0, 1.0).astype(
            np.float32,
            copy=False,
        )

        neighbor_support = compute_neighbor_support_fraction(mask, ctx.neighbor_index)
        if assigned_count > 0:
            current_compactness_sum = float(np.sum(neighbor_support[assigned], dtype=np.float64))
            current_compactness = current_compactness_sum / float(assigned_count)
            new_compactness = (current_compactness_sum + 2.0 * neighbor_support) / float(assigned_count + 1)
            gains = new_compactness - current_compactness
            candidate_compactness_gain[frontier] = gains[frontier].astype(np.float32, copy=False)

    return candidate_to_current_centroid_distance, candidate_compactness_gain


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


def _add_rewards_from_membership_mask(
    *,
    ctx: EpisodeContext,
    membership_mask: np.ndarray,
    posterior: np.ndarray,
    neighbor_support: np.ndarray,
    frontier_mask: np.ndarray,
) -> np.ndarray:
    """Recompute current ADD rewards from one state summary path."""
    del membership_mask
    r_expr = (ctx.ll @ posterior) * ctx.expression_confidence
    if ctx.normalize_expression_zscore:
        if np.any(frontier_mask):
            expr_frontier = r_expr[frontier_mask]
            mu = float(np.mean(expr_frontier))
            sigma = float(np.std(expr_frontier, ddof=0))
            expr_term = (r_expr - mu) / (sigma + float(ctx.zscore_delta))
        else:
            expr_term = np.zeros_like(r_expr, dtype=np.float32)
    else:
        expr_term = r_expr
    return ctx.w1 * expr_term - ctx.base_penalty + ctx.w4 * neighbor_support


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


def _pack_mask(mask: np.ndarray) -> np.ndarray:
    bits = np.asarray(mask, dtype=np.uint8)
    return np.packbits(bits, bitorder="little")


def _unpack_mask(packed: np.ndarray, n_bits: int) -> np.ndarray:
    out = np.unpackbits(np.asarray(packed, dtype=np.uint8), count=int(n_bits), bitorder="little")
    return out.astype(np.uint8, copy=False)


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


def _load_reference_counts(path: Path, reference_format: str, array_key: str) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"reference file not found: {path}")
    if reference_format == "npy":
        arr = np.load(path)
        return _validate_reference_matrix(arr)
    if reference_format == "npz":
        with np.load(path) as data:
            if array_key not in data:
                raise ConfigError(f"reference array key {array_key!r} is not present in {path}")
            arr = data[array_key]
        return _validate_reference_matrix(arr)

    table = _load_table(path, reference_format)
    numeric_cols = [c for c in table.columns if pd.api.types.is_numeric_dtype(table[c])]
    if not numeric_cols:
        raise ValueError(f"reference table {path} has no numeric columns to form C[K,G]")
    arr = table.loc[:, numeric_cols].to_numpy(dtype=np.float64, copy=True)
    return _validate_reference_matrix(arr)


def _validate_reference_matrix(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("reference matrix must have shape (K, G)")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError("reference matrix must have positive shape in both dimensions")
    if not np.isfinite(arr).all():
        raise ValueError("reference matrix contains non-finite values")
    if (arr < 0).any():
        raise ValueError("reference matrix must be non-negative")
    return arr


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


def _normalize_format(raw: str, path: Path) -> str:
    value = str(raw).strip().lower()
    if value == "auto":
        name = path.name.lower()
        if name.endswith(".parquet"):
            return "parquet"
        if name.endswith(".csv") or name.endswith(".csv.gz"):
            return "csv"
        if name.endswith(".tsv") or name.endswith(".tsv.gz"):
            return "tsv"
        if name.endswith(".npy"):
            return "npy"
        if name.endswith(".npz"):
            return "npz"
        raise ConfigError(f"could not infer format from path: {path}")
    if value not in {"parquet", "csv", "tsv", "npy", "npz"}:
        raise ConfigError(f"unsupported format: {value!r}")
    return value


def _load_table(path: Path, table_format: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"table file not found: {path}")
    if table_format == "parquet":
        return pd.read_parquet(path)
    if table_format == "csv":
        return pd.read_csv(path)
    if table_format == "tsv":
        return pd.read_csv(path, sep="\t")
    raise ConfigError(f"unsupported table format for loader: {table_format!r}")


def _default_nuclei_center_columns(overrides: dict[str, Any]) -> dict[str, str]:
    cols = {
        "cell_id": "cell_id",
        "center_x_um": "center_x_um",
        "center_y_um": "center_y_um",
    }
    cols.update(overrides)
    for key in ("cell_id", "center_x_um", "center_y_um"):
        if cols.get(key) is None:
            raise ConfigError(f"inputs.nuclei.columns.{key} must not be null")
        cols[key] = str(cols[key])
    return cols


def _save_checkpoint(
    *,
    path: Path,
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    update_index: int,
    best_moving_avg_reward: float | None,
    config: PPOTrainingConfig,
) -> None:
    payload = {
        "update_index": int(update_index),
        "best_moving_avg_reward": best_moving_avg_reward,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config.to_serializable_dict(),
    }
    torch.save(payload, path)


def _build_metadata(run_dir: Path, seed: int | None) -> dict[str, Any]:
    now_utc, now_local = _now_utc_and_local()
    return {
        "created_at_utc": now_utc.isoformat(),
        "created_at_local": now_local.isoformat(),
        "local_timezone": _LOCAL_TIMEZONE_NAME,
        "run_dir": str(run_dir),
        "python_version": sys.version,
        "platform": platform.platform(),
        "seed": seed,
        "git_commit": _git_commit_hash(),
    }


def _git_commit_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def _slugify(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip()).strip("_")
    return value.lower() or "run"


def _as_dict(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"{name} must be a mapping")
    return value


def _require(mapping: dict[str, Any], key: str, section: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"missing required key {key!r} in {section}")
    return mapping[key]


def _parse_grpo_target_interval(value: Any, section: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ConfigError(f"{section} must be a two-value [min, max] list")
    lo = float(value[0])
    hi = float(value[1])
    if lo <= 0 or hi <= lo:
        raise ConfigError(f"{section} must satisfy 0 < min < max")
    return lo, hi


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _append_step_log(path: Path, event: str, payload: dict[str, Any]) -> None:
    now_utc, now_local = _now_utc_and_local()
    row = {
        "timestamp_utc": now_utc.isoformat(),
        "timestamp_local": now_local.isoformat(),
        "event": event,
        "payload": payload,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row))
        handle.write("\n")


def _now_utc_and_local() -> tuple[dt.datetime, dt.datetime]:
    """Return current UTC and America/Chicago timestamps."""
    now_utc = dt.datetime.now(dt.timezone.utc)
    return now_utc, now_utc.astimezone(_LOCAL_TIMEZONE)
