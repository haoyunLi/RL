"""PPO/GRPO training config parsing and planner constants."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PLANNER_MODES = ("stop", "compact", "balanced", "explore")
PLANNER_MODE_STOP = 0
PLANNER_MODE_COMPACT = 1
PLANNER_MODE_BALANCED = 2
PLANNER_MODE_EXPLORE = 3
_PLANNER_MODE_TO_INDEX = {name: idx for idx, name in enumerate(PLANNER_MODES)}
_PLANNER_LOGIT_BIAS_FEATURES = ("neighbor_support", "compactness_gain", "centroid_distance", "expression")
_DEFAULT_PLANNER_LOGIT_BIAS = {
    "compact": {
        "neighbor_support": 0.8,
        "compactness_gain": 0.5,
        "centroid_distance": -0.3,
        "expression": 0.2,
    },
    "balanced": {
        "neighbor_support": 0.2,
        "compactness_gain": 0.1,
        "centroid_distance": -0.1,
        "expression": 0.2,
    },
    "explore": {
        "neighbor_support": 0.0,
        "compactness_gain": 0.0,
        "centroid_distance": -0.1,
        "expression": 0.5,
    },
    "stop": {
        "neighbor_support": 0.0,
        "compactness_gain": 0.0,
        "centroid_distance": 0.0,
        "expression": 0.0,
    },
}


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

    planner_enabled: bool
    planner_interval: int
    planner_cot_weight: float
    planner_entropy_coef: float
    planner_modes: tuple[str, ...]
    planner_logit_bias: dict[str, dict[str, float]]

    full_grpo_reward_weight: float
    full_grpo_evidence_growth_weight: float
    full_grpo_stop_weight: float
    full_grpo_overgrowth_weight: float
    full_grpo_compact_weight: float
    full_grpo_compact_streak_weight: float
    full_grpo_explore_weight: float
    full_grpo_tau_frontier: float
    full_grpo_frontier_temp: float

    epsilon: float
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
            "planner": {
                "enabled": self.planner_enabled,
                "interval": self.planner_interval,
                "modes": list(self.planner_modes),
                "cot_weight": self.planner_cot_weight,
                "entropy_coef": self.planner_entropy_coef,
                "logit_bias": self.planner_logit_bias,
            },
            "full_grpo": {
                "reward_weight": self.full_grpo_reward_weight,
                "evidence_growth_weight": self.full_grpo_evidence_growth_weight,
                "stop_weight": self.full_grpo_stop_weight,
                "overgrowth_weight": self.full_grpo_overgrowth_weight,
                "compact_weight": self.full_grpo_compact_weight,
                "compact_streak_weight": self.full_grpo_compact_streak_weight,
                "explore_weight": self.full_grpo_explore_weight,
                "tau_frontier": self.full_grpo_tau_frontier,
                "frontier_temp": self.full_grpo_frontier_temp,
            },
            "reward": {
                "epsilon": self.epsilon,
                "r_max_um": self.r_max_um,
                "w1": self.w1,
                "w2": self.w2,
                "w3": self.w3,
                "w4": self.w4,
                "w5": self.w5,
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

    planner = _as_dict(raw.get("planner", {}), "planner")
    planner_enabled = bool(planner.get("enabled", False))
    planner_interval = int(planner.get("interval", 4))
    if planner_interval <= 0:
        raise ConfigError("planner.interval must be > 0")
    planner_cot_weight = float(planner.get("cot_weight", 0.4))
    if planner_cot_weight < 0:
        raise ConfigError("planner.cot_weight must be >= 0")
    planner_entropy_coef = float(planner.get("entropy_coef", 0.02))
    if planner_entropy_coef < 0:
        raise ConfigError("planner.entropy_coef must be >= 0")
    planner_modes_raw = planner.get("modes", list(PLANNER_MODES))
    if not isinstance(planner_modes_raw, (list, tuple)):
        raise ConfigError("planner.modes must be a list")
    planner_modes = tuple(str(x).strip().lower() for x in planner_modes_raw)
    if planner_modes != PLANNER_MODES:
        raise ConfigError(f"planner.modes must be exactly {list(PLANNER_MODES)!r} for this implementation")
    if planner_enabled and training_mode != "full_grpo":
        raise ConfigError("planner.enabled requires run.training_mode: full_grpo")
    planner_logit_bias = _parse_planner_logit_bias(planner.get("logit_bias", {}))

    full_grpo = _as_dict(raw.get("full_grpo", {}), "full_grpo")
    full_grpo_reward_weight = float(full_grpo.get("reward_weight", 1.0))
    full_grpo_evidence_growth_weight = float(full_grpo.get("evidence_growth_weight", 0.7))
    full_grpo_stop_weight = float(full_grpo.get("stop_weight", 0.5))
    full_grpo_overgrowth_weight = float(full_grpo.get("overgrowth_weight", 0.8))
    full_grpo_compact_weight = float(full_grpo.get("compact_weight", 0.25))
    full_grpo_compact_streak_weight = float(full_grpo.get("compact_streak_weight", 0.25))
    full_grpo_explore_weight = float(full_grpo.get("explore_weight", 0.4))
    full_grpo_tau_frontier = float(full_grpo.get("tau_frontier", 0.0))
    full_grpo_frontier_temp = float(full_grpo.get("frontier_temp", 0.2))
    for name, val in (
        ("reward_weight", full_grpo_reward_weight),
        ("evidence_growth_weight", full_grpo_evidence_growth_weight),
        ("stop_weight", full_grpo_stop_weight),
        ("overgrowth_weight", full_grpo_overgrowth_weight),
        ("compact_weight", full_grpo_compact_weight),
        ("compact_streak_weight", full_grpo_compact_streak_weight),
        ("explore_weight", full_grpo_explore_weight),
    ):
        if val < 0:
            raise ConfigError(f"full_grpo.{name} must be >= 0")
    if full_grpo_frontier_temp <= 0:
        raise ConfigError("full_grpo.frontier_temp must be > 0")

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
    w5 = float(reward.get("w5", 0.0))
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
    if w5 < 0:
        raise ConfigError("reward.w5 must be >= 0")
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
        planner_enabled=planner_enabled,
        planner_interval=planner_interval,
        planner_cot_weight=planner_cot_weight,
        planner_entropy_coef=planner_entropy_coef,
        planner_modes=planner_modes,
        planner_logit_bias=planner_logit_bias,
        full_grpo_reward_weight=full_grpo_reward_weight,
        full_grpo_evidence_growth_weight=full_grpo_evidence_growth_weight,
        full_grpo_stop_weight=full_grpo_stop_weight,
        full_grpo_overgrowth_weight=full_grpo_overgrowth_weight,
        full_grpo_compact_weight=full_grpo_compact_weight,
        full_grpo_compact_streak_weight=full_grpo_compact_streak_weight,
        full_grpo_explore_weight=full_grpo_explore_weight,
        full_grpo_tau_frontier=full_grpo_tau_frontier,
        full_grpo_frontier_temp=full_grpo_frontier_temp,
        epsilon=epsilon,
        r_max_um=r_max_um,
        w1=w1,
        w2=w2,
        w3=w3,
        w4=w4,
        w5=w5,
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


def _parse_planner_logit_bias(value: Any) -> dict[str, dict[str, float]]:
    raw = _as_dict(value, "planner.logit_bias")
    parsed: dict[str, dict[str, float]] = {
        mode: dict(weights) for mode, weights in _DEFAULT_PLANNER_LOGIT_BIAS.items()
    }
    for mode_name, weights_raw in raw.items():
        mode = str(mode_name).strip().lower()
        if mode not in _PLANNER_MODE_TO_INDEX:
            raise ConfigError(f"unsupported planner.logit_bias mode: {mode_name!r}")
        weights = _as_dict(weights_raw, f"planner.logit_bias.{mode}")
        for feature_name, feature_value in weights.items():
            feature = str(feature_name).strip().lower()
            if feature not in _PLANNER_LOGIT_BIAS_FEATURES:
                raise ConfigError(f"unsupported planner.logit_bias.{mode} feature: {feature_name!r}")
            parsed[mode][feature] = float(feature_value)
    return parsed
