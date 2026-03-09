"""Reproducible grid search for reward weights using a greedy add-or-stop baseline."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import datetime as dt
import itertools
import json
import multiprocessing as mp
from pathlib import Path
import platform
import re
import subprocess
import sys
from typing import Any

import h5py
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import yaml

from .reward import PosteriorAddBinReward, compute_bin_log_likelihood_by_type, compute_reference_distribution


class ConfigError(ValueError):
    """Raised when reward-grid-search config is invalid."""


_REWARD_GRID_WORKER_CONTEXT: dict[str, Any] | None = None


@dataclass(frozen=True)
class GridAxis:
    """Inclusive numeric axis specification for one weight parameter."""

    start: float
    stop: float
    step: float

    def values(self) -> tuple[float, ...]:
        """Return rounded inclusive values from start to stop."""
        if self.step <= 0:
            raise ConfigError("grid step must be > 0")
        if self.start <= 0 or self.stop <= 0:
            raise ConfigError("grid values must be > 0")
        if self.stop < self.start:
            raise ConfigError("grid stop must be >= start")

        scale = 1000000
        start_i = int(round(self.start * scale))
        stop_i = int(round(self.stop * scale))
        step_i = int(round(self.step * scale))
        if step_i <= 0:
            raise ConfigError("grid step is too small after rounding")

        values: list[float] = []
        current = start_i
        while current <= stop_i:
            values.append(round(current / scale, 6))
            current += step_i

        if not values:
            raise ConfigError("grid axis produced no values")
        return tuple(values)


@dataclass(frozen=True)
class RewardGridSearchConfig:
    """Resolved config for one reward-weight search run."""

    run_name: str
    output_root: Path
    seed: int | None
    n_workers: int
    episodes_index_path: Path
    reference_path: Path
    reference_format: str
    reference_array_key: str
    reference_genes_key: str
    reference_cell_type_column: str | None
    reference_gene_mode: str
    reference_gene_prefix: str | None
    reference_gene_columns: tuple[str, ...]
    nuclei_path: Path
    nuclei_format: str
    nuclei_columns: dict[str, str]
    max_episodes: int | None
    objective: str
    epsilon: float
    r_max_um: float
    normalize_expression_zscore: bool
    zscore_delta: float
    w1_axis: GridAxis
    w2_axis: GridAxis
    w3_axis: GridAxis
    stop_lambda_axis: GridAxis

    def to_serializable_dict(self) -> dict[str, Any]:
        """Return config as plain Python types for YAML/JSON output."""
        return {
            "run": {
                "name": self.run_name,
                "output_root": str(self.output_root),
                "seed": self.seed,
                "max_episodes": self.max_episodes,
                "n_workers": self.n_workers,
            },
            "inputs": {
                "episodes_index_path": str(self.episodes_index_path),
                "reference": {
                    "path": str(self.reference_path),
                    "format": self.reference_format,
                    "array_key": self.reference_array_key,
                    "genes_key": self.reference_genes_key,
                    "cell_type_column": self.reference_cell_type_column,
                    "gene_mode": self.reference_gene_mode,
                    "gene_prefix": self.reference_gene_prefix,
                    "gene_columns": list(self.reference_gene_columns),
                },
                "nuclei": {
                    "path": str(self.nuclei_path),
                    "format": self.nuclei_format,
                    "columns": self.nuclei_columns,
                },
            },
            "reward": {
                "objective": self.objective,
                "epsilon": self.epsilon,
                "r_max_um": self.r_max_um,
                "normalize_expression_zscore": self.normalize_expression_zscore,
                "zscore_delta": self.zscore_delta,
            },
            "search": {
                "w1": _axis_to_dict(self.w1_axis),
                "w2": _axis_to_dict(self.w2_axis),
                "w3": _axis_to_dict(self.w3_axis),
                "stop_lambda": _axis_to_dict(self.stop_lambda_axis),
            },
        }


@dataclass(frozen=True)
class PreparedEpisodeRewardData:
    """Episode payload with static reward terms precomputed once per run."""

    cell_id: str
    candidate_bin_ids: tuple[str, ...]
    candidate_bin_xy_um: np.ndarray
    nucleus_center_xy_um: np.ndarray
    precomputed_ll: np.ndarray
    precomputed_d_other_um: np.ndarray


@dataclass(frozen=True)
class NucleiSpatialIndex:
    """Fast nearest-neighbor lookup over nuclei centers."""

    centers_xy_um: np.ndarray
    cell_id_to_index: dict[str, int]
    tree: cKDTree


@dataclass(frozen=True)
class RewardGridPreparedContext:
    """Reusable in-memory context for repeated grid runs with same data inputs."""

    reference_counts: np.ndarray
    episodes: list[PreparedEpisodeRewardData]


@dataclass(frozen=True)
class GreedyEpisodeMetrics:
    """Metrics from greedy rollout on one episode for one weight setting."""

    total_return: float
    assigned_bins: int
    n_add_actions: int
    stop_reward: float
    final_best_add: float


@dataclass(frozen=True)
class RewardGridSearchResult:
    """Aggregated metrics for one weight combination."""

    w1: float
    w2: float
    w3: float
    stop_lambda: float
    n_episodes: int
    mean_return: float
    mean_assigned_bins: float
    mean_add_actions: float
    mean_stop_reward: float
    mean_final_best_add: float

    def objective_value(self, objective: str) -> float:
        """Return comparable scalar used for best-weight selection."""
        if objective == "mean_return":
            return self.mean_return
        raise ConfigError(f"unsupported objective: {objective!r}")

    def to_row(self, objective: str) -> dict[str, Any]:
        """Return CSV/JSON row for this result."""
        return {
            "w1": self.w1,
            "w2": self.w2,
            "w3": self.w3,
            "stop_lambda": self.stop_lambda,
            "n_episodes": self.n_episodes,
            "mean_return": self.mean_return,
            "mean_assigned_bins": self.mean_assigned_bins,
            "mean_add_actions": self.mean_add_actions,
            "mean_stop_reward": self.mean_stop_reward,
            "mean_final_best_add": self.mean_final_best_add,
            "objective": objective,
            "objective_value": self.objective_value(objective),
        }


def load_reward_grid_search_config(config_path: str | Path) -> RewardGridSearchConfig:
    """Load and validate YAML config for reward-weight search."""
    path = Path(config_path)
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if not isinstance(raw, dict):
        raise ConfigError("config root must be a mapping")

    run = _as_dict(raw.get("run"), "run")
    inputs = _as_dict(raw.get("inputs"), "inputs")
    reward_cfg = _as_dict(raw.get("reward"), "reward")
    search = _as_dict(raw.get("search"), "search")

    run_name = str(run.get("name", "reward_grid_search"))
    if not run_name.strip():
        raise ConfigError("run.name must be a non-empty string")

    output_root = Path(str(run.get("output_root", "runs"))).expanduser().resolve()
    seed = run.get("seed")
    if seed is not None:
        seed = int(seed)
    n_workers = int(run.get("n_workers", 1))
    if n_workers <= 0:
        raise ConfigError("run.n_workers must be > 0")

    max_episodes = run.get("max_episodes")
    if max_episodes is not None:
        max_episodes = int(max_episodes)
        if max_episodes <= 0:
            raise ConfigError("run.max_episodes must be > 0 when provided")

    episodes_index_path = Path(str(_require(inputs, "episodes_index_path", "inputs"))).expanduser().resolve()

    reference_cfg = _as_dict(_require(inputs, "reference", "inputs"), "inputs.reference")
    reference_path = Path(str(_require(reference_cfg, "path", "inputs.reference"))).expanduser().resolve()
    reference_format = _normalize_format(str(reference_cfg.get("format", "auto")), reference_path)
    reference_array_key = str(reference_cfg.get("array_key", "reference_counts"))
    reference_genes_key = str(reference_cfg.get("genes_key", "genes"))
    reference_cell_type_column_raw = reference_cfg.get("cell_type_column", None)
    reference_cell_type_column = None if reference_cell_type_column_raw is None else str(reference_cell_type_column_raw)
    reference_gene_mode = str(reference_cfg.get("gene_mode", "all"))
    if reference_gene_mode not in {"all", "prefix", "list"}:
        raise ConfigError("inputs.reference.gene_mode must be 'all', 'prefix', or 'list'")
    reference_gene_prefix_raw = reference_cfg.get("gene_prefix", None)
    reference_gene_prefix = None if reference_gene_prefix_raw is None else str(reference_gene_prefix_raw)
    reference_gene_columns_raw = reference_cfg.get("gene_columns", [])
    if reference_gene_columns_raw is None:
        reference_gene_columns_raw = []
    if not isinstance(reference_gene_columns_raw, list):
        raise ConfigError("inputs.reference.gene_columns must be a list")
    reference_gene_columns = tuple(str(col) for col in reference_gene_columns_raw)

    nuclei_cfg = _as_dict(_require(inputs, "nuclei", "inputs"), "inputs.nuclei")
    nuclei_path = Path(str(_require(nuclei_cfg, "path", "inputs.nuclei"))).expanduser().resolve()
    nuclei_format = _normalize_format(str(nuclei_cfg.get("format", "auto")), nuclei_path)
    nuclei_columns = _default_nuclei_center_columns(_as_dict(nuclei_cfg.get("columns", {}), "inputs.nuclei.columns"))

    objective = str(reward_cfg.get("objective", "mean_return"))
    if objective not in {"mean_return"}:
        raise ConfigError("reward.objective must be 'mean_return'")

    epsilon = float(reward_cfg.get("epsilon", 1e-8))
    if epsilon < 0:
        raise ConfigError("reward.epsilon must be >= 0")

    r_max_um = float(reward_cfg.get("r_max_um", 80.0))
    if r_max_um <= 0:
        raise ConfigError("reward.r_max_um must be > 0")

    normalize_expression_zscore = bool(reward_cfg.get("normalize_expression_zscore", False))
    zscore_delta = float(reward_cfg.get("zscore_delta", 1e-8))
    if zscore_delta <= 0:
        raise ConfigError("reward.zscore_delta must be > 0")

    w1_axis = _load_axis(_require(search, "w1", "search"), "search.w1")
    w2_axis = _load_axis(_require(search, "w2", "search"), "search.w2")
    w3_axis = _load_axis(_require(search, "w3", "search"), "search.w3")
    stop_lambda_axis = _load_axis(_require(search, "stop_lambda", "search"), "search.stop_lambda")

    return RewardGridSearchConfig(
        run_name=run_name,
        output_root=output_root,
        seed=seed,
        n_workers=n_workers,
        episodes_index_path=episodes_index_path,
        reference_path=reference_path,
        reference_format=reference_format,
        reference_array_key=reference_array_key,
        reference_genes_key=reference_genes_key,
        reference_cell_type_column=reference_cell_type_column,
        reference_gene_mode=reference_gene_mode,
        reference_gene_prefix=reference_gene_prefix,
        reference_gene_columns=reference_gene_columns,
        nuclei_path=nuclei_path,
        nuclei_format=nuclei_format,
        nuclei_columns=nuclei_columns,
        max_episodes=max_episodes,
        objective=objective,
        epsilon=epsilon,
        r_max_um=r_max_um,
        normalize_expression_zscore=normalize_expression_zscore,
        zscore_delta=zscore_delta,
        w1_axis=w1_axis,
        w2_axis=w2_axis,
        w3_axis=w3_axis,
        stop_lambda_axis=stop_lambda_axis,
    )


def prepare_reward_grid_context(config: RewardGridSearchConfig) -> RewardGridPreparedContext:
    """Load reference + episode artifacts once and precompute static reward terms."""
    rng = np.random.default_rng(config.seed)
    reference_counts = _load_reference_counts(config)

    nuclei_df = _load_table(config.nuclei_path, config.nuclei_format)
    nuclei_centers = _build_nuclei_centers(df=nuclei_df, columns=config.nuclei_columns)
    nuclei_spatial_index = _build_nuclei_spatial_index(nuclei_centers)

    episodes = _load_prepared_episode_reward_data(
        episodes_index_path=config.episodes_index_path,
        nuclei_centers_by_cell=nuclei_centers,
        nuclei_spatial_index=nuclei_spatial_index,
        rng=rng,
        max_episodes=config.max_episodes,
        reference_counts=reference_counts,
        epsilon=config.epsilon,
        reference_path=config.reference_path,
        reference_format=config.reference_format,
        reference_genes_key=config.reference_genes_key,
    )
    if not episodes:
        raise ValueError("no episodes available for reward grid search")

    return RewardGridPreparedContext(reference_counts=reference_counts, episodes=episodes)


def run_reward_grid_search(
    config: RewardGridSearchConfig,
    prepared_context: RewardGridPreparedContext | None = None,
) -> Path:
    """Run grid search over reward weights and return output run directory."""
    config.output_root.mkdir(parents=True, exist_ok=True)

    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = config.output_root / f"{_slugify(config.run_name)}_{timestamp}"
    run_dir.mkdir(parents=False, exist_ok=False)

    config_dir = run_dir / "config"
    logs_dir = run_dir / "logs"
    config_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    steps_log_path = logs_dir / "steps.jsonl"
    _append_step_log(
        steps_log_path,
        event="run_start",
        payload={"run_name": config.run_name, "seed": config.seed},
    )

    if prepared_context is None:
        prepared_context = prepare_reward_grid_context(config)
        _append_step_log(
            steps_log_path,
            event="prepared_context_created",
            payload={
                "n_episodes": int(len(prepared_context.episodes)),
                "n_cell_types": int(prepared_context.reference_counts.shape[0]),
                "n_genes": int(prepared_context.reference_counts.shape[1]),
            },
        )
    else:
        _append_step_log(
            steps_log_path,
            event="prepared_context_reused",
            payload={
                "n_episodes": int(len(prepared_context.episodes)),
                "n_cell_types": int(prepared_context.reference_counts.shape[0]),
                "n_genes": int(prepared_context.reference_counts.shape[1]),
            },
        )

    reference_counts = prepared_context.reference_counts
    prepared_episodes = prepared_context.episodes

    grid_values = {
        "w1": config.w1_axis.values(),
        "w2": config.w2_axis.values(),
        "w3": config.w3_axis.values(),
        "stop_lambda": config.stop_lambda_axis.values(),
    }
    n_combinations = (
        len(grid_values["w1"])
        * len(grid_values["w2"])
        * len(grid_values["w3"])
        * len(grid_values["stop_lambda"])
    )
    _append_step_log(
        steps_log_path,
        event="grid_built",
        payload={
            "n_combinations": int(n_combinations),
            "n_workers": int(config.n_workers),
            "w1_values": list(grid_values["w1"]),
            "w2_values": list(grid_values["w2"]),
            "w3_values": list(grid_values["w3"]),
            "stop_lambda_values": list(grid_values["stop_lambda"]),
        },
    )

    combo_specs = [
        (combo_index, float(w1), float(w2), float(w3), float(stop_lambda))
        for combo_index, (w1, w2, w3, stop_lambda) in enumerate(
            itertools.product(
                grid_values["w1"],
                grid_values["w2"],
                grid_values["w3"],
                grid_values["stop_lambda"],
            ),
            start=1,
        )
    ]

    results: list[RewardGridSearchResult] = []
    use_parallel = config.n_workers > 1 and len(combo_specs) > 1
    if use_parallel:
        try:
            mp_ctx = mp.get_context("fork")
        except ValueError:
            use_parallel = False
            _append_step_log(
                steps_log_path,
                event="parallel_fallback",
                payload={"reason": "fork_start_method_unavailable"},
            )

    if use_parallel:
        global _REWARD_GRID_WORKER_CONTEXT
        _REWARD_GRID_WORKER_CONTEXT = {
            "episodes": prepared_episodes,
            "reference_counts": reference_counts,
            "epsilon": float(config.epsilon),
            "r_max_um": float(config.r_max_um),
            "normalize_expression_zscore": bool(config.normalize_expression_zscore),
            "zscore_delta": float(config.zscore_delta),
        }
        try:
            try:
                with mp_ctx.Pool(processes=int(config.n_workers)) as pool:
                    for combo_index, result in pool.imap_unordered(_evaluate_weight_combination_worker, combo_specs):
                        results.append(result)
                        _append_step_log(
                            steps_log_path,
                            event="combination_evaluated",
                            payload={
                                "combo_index": int(combo_index),
                                "n_combinations": int(n_combinations),
                                **result.to_row(config.objective),
                            },
                        )
            except (PermissionError, OSError) as exc:
                use_parallel = False
                _append_step_log(
                    steps_log_path,
                    event="parallel_fallback",
                    payload={"reason": f"pool_creation_failed: {type(exc).__name__}", "detail": str(exc)},
                )
        finally:
            _REWARD_GRID_WORKER_CONTEXT = None

    if not use_parallel:
        for combo_index, w1, w2, w3, stop_lambda in combo_specs:
            result = _evaluate_weight_combination(
                episodes=prepared_episodes,
                reference_counts=reference_counts,
                epsilon=config.epsilon,
                r_max_um=config.r_max_um,
                w1=w1,
                w2=w2,
                w3=w3,
                stop_lambda=stop_lambda,
                normalize_expression_zscore=config.normalize_expression_zscore,
                zscore_delta=config.zscore_delta,
            )
            results.append(result)
            _append_step_log(
                steps_log_path,
                event="combination_evaluated",
                payload={
                    "combo_index": int(combo_index),
                    "n_combinations": int(n_combinations),
                    **result.to_row(config.objective),
                },
            )

    best_result = max(results, key=lambda row: row.objective_value(config.objective))

    results_df = pd.DataFrame([row.to_row(config.objective) for row in results])
    results_df = results_df.sort_values(by="objective_value", ascending=False).reset_index(drop=True)
    results_df.to_csv(run_dir / "results.csv", index=False)

    best_payload = best_result.to_row(config.objective)
    _write_yaml(config_dir / "best_weights.yaml", best_payload)
    _write_yaml(config_dir / "config_resolved.yaml", config.to_serializable_dict())
    _write_json(config_dir / "metadata.json", _build_metadata(run_dir=run_dir, seed=config.seed))

    summary = {
        "objective": config.objective,
        "n_episodes": int(len(prepared_episodes)),
        "n_combinations": int(n_combinations),
        "n_workers": int(config.n_workers),
        "best": best_payload,
    }
    _write_json(run_dir / "summary.json", summary)

    _append_step_log(
        steps_log_path,
        event="run_complete",
        payload={
            "n_episodes": int(len(prepared_episodes)),
            "n_combinations": int(n_combinations),
            "best_objective_value": float(best_result.objective_value(config.objective)),
        },
    )

    return run_dir


def run_reward_grid_search_from_config(config_path: str | Path) -> Path:
    """Convenience wrapper: load config, execute grid search, and return run dir."""
    config = load_reward_grid_search_config(config_path)
    return run_reward_grid_search(config)


def _evaluate_weight_combination(
    episodes: list[PreparedEpisodeRewardData],
    reference_counts: np.ndarray,
    epsilon: float,
    r_max_um: float,
    w1: float,
    w2: float,
    w3: float,
    stop_lambda: float,
    normalize_expression_zscore: bool,
    zscore_delta: float,
) -> RewardGridSearchResult:
    totals: list[GreedyEpisodeMetrics] = []

    for episode in episodes:
        reward_fn = PosteriorAddBinReward(
            reference_counts=reference_counts,
            candidate_bin_ids=list(episode.candidate_bin_ids),
            candidate_expression=None,
            candidate_bin_xy_um=episode.candidate_bin_xy_um,
            nucleus_center_xy_um=episode.nucleus_center_xy_um,
            other_nuclei_center_xy_um=None,
            epsilon=epsilon,
            r_max_um=r_max_um,
            w1=w1,
            w2=w2,
            w3=w3,
            stop_lambda=stop_lambda,
            normalize_expression_zscore=normalize_expression_zscore,
            zscore_delta=zscore_delta,
            precomputed_ll=episode.precomputed_ll,
            precomputed_d_other_um=episode.precomputed_d_other_um,
        )
        totals.append(_run_greedy_episode(reward_fn))

    total_return = np.asarray([m.total_return for m in totals], dtype=np.float64)
    assigned_bins = np.asarray([m.assigned_bins for m in totals], dtype=np.float64)
    add_actions = np.asarray([m.n_add_actions for m in totals], dtype=np.float64)
    stop_rewards = np.asarray([m.stop_reward for m in totals], dtype=np.float64)
    final_best_add = np.asarray([m.final_best_add for m in totals], dtype=np.float64)

    return RewardGridSearchResult(
        w1=float(w1),
        w2=float(w2),
        w3=float(w3),
        stop_lambda=float(stop_lambda),
        n_episodes=len(totals),
        mean_return=float(total_return.mean()),
        mean_assigned_bins=float(assigned_bins.mean()),
        mean_add_actions=float(add_actions.mean()),
        mean_stop_reward=float(stop_rewards.mean()),
        mean_final_best_add=float(final_best_add.mean()),
    )


def _evaluate_weight_combination_worker(
    combo_spec: tuple[int, float, float, float, float],
) -> tuple[int, RewardGridSearchResult]:
    """Process-pool worker wrapper for one weight combination."""
    if _REWARD_GRID_WORKER_CONTEXT is None:
        raise RuntimeError("reward grid worker context is not initialized")

    combo_index, w1, w2, w3, stop_lambda = combo_spec
    ctx = _REWARD_GRID_WORKER_CONTEXT
    result = _evaluate_weight_combination(
        episodes=ctx["episodes"],
        reference_counts=ctx["reference_counts"],
        epsilon=float(ctx["epsilon"]),
        r_max_um=float(ctx["r_max_um"]),
        w1=float(w1),
        w2=float(w2),
        w3=float(w3),
        stop_lambda=float(stop_lambda),
        normalize_expression_zscore=bool(ctx["normalize_expression_zscore"]),
        zscore_delta=float(ctx["zscore_delta"]),
    )
    return int(combo_index), result


def _run_greedy_episode(reward_fn: PosteriorAddBinReward) -> GreedyEpisodeMetrics:
    """Roll out the simple greedy add-or-stop baseline for one episode."""
    membership_mask = np.zeros(reward_fn.n_candidate_bins, dtype=np.int8)
    total_return = 0.0
    n_add_actions = 0

    while True:
        eligible = membership_mask == 0
        if not np.any(eligible):
            stop_reward = float(reward_fn.stop_reward(membership_mask))
            total_return += stop_reward
            return GreedyEpisodeMetrics(
                total_return=total_return,
                assigned_bins=int(membership_mask.sum()),
                n_add_actions=n_add_actions,
                stop_reward=stop_reward,
                final_best_add=0.0,
            )

        r_add = reward_fn.add_reward_per_bin(membership_mask)
        eligible_idx = np.flatnonzero(eligible)
        best_pos = int(np.argmax(r_add[eligible_idx]))
        best_idx = int(eligible_idx[best_pos])
        best_add = float(r_add[best_idx])

        if best_add <= 0.0:
            stop_reward = float(reward_fn.stop_reward(membership_mask))
            total_return += stop_reward
            return GreedyEpisodeMetrics(
                total_return=total_return,
                assigned_bins=int(membership_mask.sum()),
                n_add_actions=n_add_actions,
                stop_reward=stop_reward,
                final_best_add=best_add,
            )

        membership_mask[best_idx] = 1
        total_return += best_add
        n_add_actions += 1


def _load_reference_counts(config: RewardGridSearchConfig) -> np.ndarray:
    path = config.reference_path

    if config.reference_format == "npy":
        matrix = np.load(path)
        return _validate_reference_matrix(matrix)

    if config.reference_format == "npz":
        with np.load(path) as data:
            if config.reference_array_key not in data:
                raise ConfigError(
                    f"inputs.reference.array_key {config.reference_array_key!r} is not present in {path}"
                )
            matrix = data[config.reference_array_key]
        return _validate_reference_matrix(matrix)

    table = _load_table(path, config.reference_format)
    gene_columns = _resolve_reference_gene_columns(
        df=table,
        gene_mode=config.reference_gene_mode,
        gene_prefix=config.reference_gene_prefix,
        configured_columns=config.reference_gene_columns,
        cell_type_column=config.reference_cell_type_column,
    )
    if config.reference_cell_type_column is not None and config.reference_cell_type_column not in table.columns:
        raise ValueError(
            f"reference cell_type column {config.reference_cell_type_column!r} is not present in reference table"
        )

    matrix = table.loc[:, list(gene_columns)].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float64, copy=True)
    return _validate_reference_matrix(matrix)


def _validate_reference_matrix(matrix: np.ndarray) -> np.ndarray:
    counts = np.asarray(matrix, dtype=np.float64)
    if counts.ndim != 2:
        raise ValueError("reference counts must have shape (K, G)")
    if counts.shape[0] == 0 or counts.shape[1] == 0:
        raise ValueError("reference counts must have positive shape in both dimensions")
    if not np.isfinite(counts).all():
        raise ValueError("reference counts contain non-finite values")
    if (counts < 0).any():
        raise ValueError("reference counts must be non-negative")
    return counts


def _resolve_reference_gene_columns(
    df: pd.DataFrame,
    gene_mode: str,
    gene_prefix: str | None,
    configured_columns: tuple[str, ...],
    cell_type_column: str | None,
) -> tuple[str, ...]:
    if gene_mode == "all":
        columns = [col for col in df.columns if col != cell_type_column]
        if not columns:
            raise ConfigError("reference table does not contain any gene columns")
        return tuple(str(col) for col in columns)

    if gene_mode == "prefix":
        if gene_prefix is None or not gene_prefix:
            raise ConfigError("inputs.reference.gene_prefix must be set when gene_mode='prefix'")
        columns = [col for col in df.columns if str(col).startswith(gene_prefix)]
        if not columns:
            raise ConfigError(f"no reference gene columns found with prefix {gene_prefix!r}")
        return tuple(str(col) for col in columns)

    if gene_mode == "list":
        if not configured_columns:
            raise ConfigError("inputs.reference.gene_columns must be non-empty when gene_mode='list'")
        missing = [col for col in configured_columns if col not in df.columns]
        if missing:
            raise ConfigError(f"reference gene columns missing in table: {missing}")
        return configured_columns

    raise ConfigError(f"unsupported reference gene_mode: {gene_mode!r}")


def _load_prepared_episode_reward_data(
    episodes_index_path: Path,
    nuclei_centers_by_cell: dict[str, np.ndarray],
    nuclei_spatial_index: NucleiSpatialIndex,
    rng: np.random.Generator,
    max_episodes: int | None,
    reference_counts: np.ndarray,
    epsilon: float,
    reference_path: Path,
    reference_format: str,
    reference_genes_key: str,
) -> list[PreparedEpisodeRewardData]:
    if not episodes_index_path.exists():
        raise FileNotFoundError(f"episodes index file not found: {episodes_index_path}")

    index_df = pd.read_csv(episodes_index_path)
    required_cols = ["cell_id", "artifact_path"]
    missing = [col for col in required_cols if col not in index_df.columns]
    if missing:
        raise ValueError(f"missing columns in episodes index: {missing}")

    if max_episodes is not None and len(index_df) > max_episodes:
        keep = np.sort(rng.choice(len(index_df), size=max_episodes, replace=False))
        index_df = index_df.iloc[keep].reset_index(drop=True)

    theta = compute_reference_distribution(reference_counts=reference_counts, epsilon=epsilon)
    log_theta = np.log(theta)

    expression_ctx = _load_episode_build_expression_context(episodes_index_path)
    matrix_h5_path = None if expression_ctx is None else expression_ctx["matrix_h5_path"]
    expression_cache_size = 20000 if expression_ctx is None else int(expression_ctx["cache_size"])
    expression_loader: _MatrixOnDemandExpressionLoader | None = None
    if matrix_h5_path is not None and reference_format == "npz":
        expression_loader = _MatrixOnDemandExpressionLoader(
            matrix_h5_path=matrix_h5_path,
            reference_npz_path=reference_path,
            reference_genes_key=reference_genes_key,
            cache_size=expression_cache_size,
        )

    prepared: list[PreparedEpisodeRewardData] = []
    try:
        for row in index_df.itertuples(index=False):
            cell_id = str(row.cell_id)
            artifact_path = Path(str(row.artifact_path)).expanduser().resolve()
            if cell_id not in nuclei_centers_by_cell:
                raise ValueError(f"cell_id {cell_id!r} from episodes index is missing from nuclei table")
            episode = _load_one_episode_artifact(
                artifact_path=artifact_path,
                cell_id=cell_id,
                expression_loader=expression_loader,
                theta=theta,
                log_theta=log_theta,
                nuclei_spatial_index=nuclei_spatial_index,
            )
            if episode is None:
                continue
            prepared.append(episode)
    finally:
        if expression_loader is not None:
            expression_loader.close()

    return prepared


def _build_nuclei_spatial_index(nuclei_centers_by_cell: dict[str, np.ndarray]) -> NucleiSpatialIndex:
    """Build KD-tree over nucleus centers for nearest-other distance queries."""
    if not nuclei_centers_by_cell:
        raise ValueError("nuclei centers table is empty")

    cell_ids = list(nuclei_centers_by_cell.keys())
    centers = np.vstack([np.asarray(nuclei_centers_by_cell[cell_id], dtype=np.float64) for cell_id in cell_ids])
    if centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError("nuclei center table must have shape (N, 2)")
    if not np.isfinite(centers).all():
        raise ValueError("nuclei centers contain non-finite values")

    return NucleiSpatialIndex(
        centers_xy_um=centers,
        cell_id_to_index={cell_id: idx for idx, cell_id in enumerate(cell_ids)},
        tree=cKDTree(centers),
    )


def _nearest_other_nucleus_distances(
    candidate_bin_xy_um: np.ndarray,
    cell_id: str,
    nuclei_spatial_index: NucleiSpatialIndex,
) -> np.ndarray:
    """Return exact distance from each candidate bin to nearest other nucleus center."""
    bin_xy = np.asarray(candidate_bin_xy_um, dtype=np.float64)
    if bin_xy.ndim != 2 or bin_xy.shape[1] != 2:
        raise ValueError("candidate_bin_xy_um must have shape (B, 2)")
    if bin_xy.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64)

    if cell_id not in nuclei_spatial_index.cell_id_to_index:
        raise ValueError(f"cell_id {cell_id!r} not found in nuclei spatial index")

    n_nuclei = int(nuclei_spatial_index.centers_xy_um.shape[0])
    if n_nuclei <= 1:
        return np.full(bin_xy.shape[0], np.inf, dtype=np.float64)

    own_idx = int(nuclei_spatial_index.cell_id_to_index[cell_id])
    query_k = int(min(8, n_nuclei))
    dists, nn_idx = nuclei_spatial_index.tree.query(bin_xy, k=query_k)

    if query_k == 1:
        dists = np.asarray(dists, dtype=np.float64)[:, None]
        nn_idx = np.asarray(nn_idx, dtype=np.int64)[:, None]
    else:
        dists = np.asarray(dists, dtype=np.float64)
        nn_idx = np.asarray(nn_idx, dtype=np.int64)

    out = np.full(bin_xy.shape[0], np.inf, dtype=np.float64)
    unresolved = np.ones(bin_xy.shape[0], dtype=bool)
    for rank in range(query_k):
        choose = unresolved & (nn_idx[:, rank] != own_idx)
        if np.any(choose):
            out[choose] = dists[choose, rank]
            unresolved[choose] = False
        if not np.any(unresolved):
            break

    # Rare fallback path for degenerate duplicate-center cases.
    if np.any(unresolved):
        other_centers = np.delete(nuclei_spatial_index.centers_xy_um, own_idx, axis=0)
        deltas = bin_xy[unresolved, None, :] - other_centers[None, :, :]
        brute = np.sqrt(np.sum(deltas * deltas, axis=2))
        out[unresolved] = np.min(brute, axis=1)

    return out


def _load_one_episode_artifact(
    artifact_path: Path,
    cell_id: str,
    expression_loader: "_MatrixOnDemandExpressionLoader | None",
    theta: np.ndarray,
    log_theta: np.ndarray,
    nuclei_spatial_index: NucleiSpatialIndex,
) -> PreparedEpisodeRewardData | None:
    if not artifact_path.exists():
        raise FileNotFoundError(f"episode artifact not found: {artifact_path}")

    with np.load(artifact_path, allow_pickle=True) as data:
        candidate_bin_ids = tuple(str(x) for x in np.asarray(data["candidate_bin_ids"], dtype=object).tolist())
        candidate_bin_xy_um = np.asarray(data["candidate_bin_xy_um"], dtype=np.float64)
        nucleus_center_xy_um = np.asarray(data["nucleus_center_xy_um"], dtype=np.float64)
        if "candidate_expression" in data:
            candidate_expression = np.asarray(data["candidate_expression"], dtype=np.float64)
            ll = compute_bin_log_likelihood_by_type(bin_counts=candidate_expression, theta=theta)
        elif "candidate_matrix_col_index" in data:
            if expression_loader is None:
                raise ValueError(
                    f"artifact {artifact_path} stores candidate_matrix_col_index but no matrix loader is available. "
                    "Ensure episodes_index.csv is from a run with config/config_resolved.yaml and reference format is npz."
                )
            col_index = np.asarray(data["candidate_matrix_col_index"], dtype=np.int64)
            ll = expression_loader.compute_ll_for_columns(col_index=col_index, log_theta=log_theta)
        else:
            raise ValueError(
                f"artifact {artifact_path} must contain either candidate_expression or candidate_matrix_col_index"
            )

    if ll.ndim != 2:
        raise ValueError(f"precomputed ll in {artifact_path} must have shape (B, K)")
    if candidate_bin_xy_um.shape != (ll.shape[0], 2):
        raise ValueError(f"candidate_bin_xy_um in {artifact_path} must have shape (B, 2)")
    if len(candidate_bin_ids) != ll.shape[0]:
        raise ValueError(f"candidate_bin_ids length mismatch in {artifact_path}")
    if nucleus_center_xy_um.shape != (2,):
        raise ValueError(f"nucleus_center_xy_um in {artifact_path} must have shape (2,)")
    if ll.shape[0] == 0 or ll.shape[1] == 0:
        return None

    d_other = _nearest_other_nucleus_distances(
        candidate_bin_xy_um=candidate_bin_xy_um,
        cell_id=cell_id,
        nuclei_spatial_index=nuclei_spatial_index,
    )

    return PreparedEpisodeRewardData(
        cell_id=cell_id,
        candidate_bin_ids=candidate_bin_ids,
        candidate_bin_xy_um=candidate_bin_xy_um,
        nucleus_center_xy_um=nucleus_center_xy_um,
        precomputed_ll=ll,
        precomputed_d_other_um=d_other,
    )


class _MatrixOnDemandExpressionLoader:
    """Load selected-gene expression vectors for matrix column indices from 10x H5."""

    def __init__(
        self,
        matrix_h5_path: Path,
        reference_npz_path: Path,
        reference_genes_key: str,
        cache_size: int = 20000,
    ) -> None:
        if not matrix_h5_path.exists():
            raise FileNotFoundError(f"matrix H5 file not found: {matrix_h5_path}")
        if not reference_npz_path.exists():
            raise FileNotFoundError(f"reference NPZ file not found: {reference_npz_path}")
        if cache_size < 0:
            raise ValueError("cache_size must be >= 0")

        self._h5 = h5py.File(matrix_h5_path, "r")
        if "matrix" not in self._h5:
            raise ValueError(f"H5 file does not contain 'matrix' group: {matrix_h5_path}")
        mg = self._h5["matrix"]
        for key in ("data", "indices", "indptr", "shape", "features"):
            if key not in mg:
                raise ValueError(f"H5 matrix group missing key: matrix/{key}")
        fg = mg["features"]
        if "name" not in fg:
            raise ValueError("H5 matrix/features missing 'name' dataset")

        shape = tuple(int(v) for v in mg["shape"][:].tolist())
        if len(shape) != 2:
            raise ValueError(f"matrix/shape in {matrix_h5_path} is invalid: {shape}")
        n_features, n_cols = int(shape[0]), int(shape[1])
        feature_names = np.asarray([x.decode("utf-8") for x in fg["name"][:]], dtype="U")
        selected_feature_indices = _resolve_matrix_feature_indices_from_reference(
            feature_names=feature_names,
            reference_npz_path=reference_npz_path,
            reference_genes_key=reference_genes_key,
        )
        if selected_feature_indices.size == 0:
            raise ValueError("selected zero genes for on-demand expression loader")
        if selected_feature_indices.max() >= n_features:
            raise ValueError("selected feature index exceeds matrix row count")

        self._n_cols = int(n_cols)
        self._expression_dim = int(selected_feature_indices.size)
        self._data_ds = mg["data"]
        self._indices_ds = mg["indices"]
        self._indptr = np.asarray(mg["indptr"][:], dtype=np.int64)
        if self._indptr.shape[0] != self._n_cols + 1:
            raise ValueError("matrix indptr length mismatch with matrix column count")

        feature_lookup = np.full(n_features, -1, dtype=np.int32)
        feature_lookup[selected_feature_indices] = np.arange(self._expression_dim, dtype=np.int32)
        self._feature_lookup = feature_lookup
        self._cache_size = int(cache_size)
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()

    @property
    def expression_dim(self) -> int:
        return self._expression_dim

    def close(self) -> None:
        try:
            self._h5.close()
        except Exception:
            pass

    def load_columns(self, col_indices: np.ndarray) -> np.ndarray:
        col_indices = np.asarray(col_indices, dtype=np.int64)
        if col_indices.ndim != 1:
            raise ValueError("candidate_matrix_col_index must be a 1D array")
        if col_indices.size == 0:
            return np.zeros((0, self._expression_dim), dtype=np.float64)
        if (col_indices < 0).any():
            raise ValueError("candidate_matrix_col_index contains negative values")
        if (col_indices >= self._n_cols).any():
            bad = int(col_indices[col_indices >= self._n_cols][0])
            raise ValueError(
                f"candidate_matrix_col_index contains value {bad} outside matrix range [0, {self._n_cols})"
            )

        out = np.zeros((col_indices.size, self._expression_dim), dtype=np.float64)
        for i, col in enumerate(col_indices.tolist()):
            out[i] = self._load_one_column(int(col))
        return out

    def compute_ll_for_columns(self, col_index: np.ndarray, log_theta: np.ndarray) -> np.ndarray:
        """Compute LL[B, K] directly from sparse matrix columns without dense BxG arrays."""
        cols = np.asarray(col_index, dtype=np.int64)
        if cols.ndim != 1:
            raise ValueError("candidate_matrix_col_index must be a 1D array")
        if cols.size == 0:
            return np.zeros((0, int(np.asarray(log_theta).shape[0])), dtype=np.float64)
        if (cols < 0).any():
            raise ValueError("candidate_matrix_col_index contains negative values")
        if (cols >= self._n_cols).any():
            bad = int(cols[cols >= self._n_cols][0])
            raise ValueError(
                f"candidate_matrix_col_index contains value {bad} outside matrix range [0, {self._n_cols})"
            )

        lt = np.asarray(log_theta, dtype=np.float64)
        if lt.ndim != 2:
            raise ValueError("log_theta must have shape (K, G)")
        if lt.shape[1] != self._expression_dim:
            raise ValueError(
                "log_theta gene dimension mismatch: %d != %d" % (lt.shape[1], self._expression_dim)
            )

        out = np.zeros((cols.size, lt.shape[0]), dtype=np.float64)
        for i, col in enumerate(cols.tolist()):
            start = int(self._indptr[col])
            end = int(self._indptr[col + 1])
            if end <= start:
                continue

            col_feature_idx = np.asarray(self._indices_ds[start:end], dtype=np.int64)
            col_values = np.asarray(self._data_ds[start:end], dtype=np.float64)
            selected_pos = self._feature_lookup[col_feature_idx]
            keep = selected_pos >= 0
            if not np.any(keep):
                continue

            pos = selected_pos[keep].astype(np.int64, copy=False)
            vals = col_values[keep]
            nb = float(np.sum(vals))
            if nb <= 0:
                continue

            # Weighted average log-likelihood over selected genes for this one bin.
            weighted = np.sum(lt[:, pos] * vals[None, :], axis=1)
            out[i, :] = weighted / nb

        return out

    def _load_one_column(self, col_index: int) -> np.ndarray:
        cached = self._cache.get(col_index)
        if cached is not None:
            self._cache.move_to_end(col_index)
            return cached

        start = int(self._indptr[col_index])
        end = int(self._indptr[col_index + 1])
        expr = np.zeros(self._expression_dim, dtype=np.float64)
        if end > start:
            col_feature_idx = np.asarray(self._indices_ds[start:end], dtype=np.int64)
            col_values = np.asarray(self._data_ds[start:end], dtype=np.float64)
            selected_pos = self._feature_lookup[col_feature_idx]
            keep = selected_pos >= 0
            if keep.any():
                expr[selected_pos[keep]] = col_values[keep]

        if self._cache_size > 0:
            self._cache[col_index] = expr
            self._cache.move_to_end(col_index)
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return expr


def _resolve_matrix_feature_indices_from_reference(
    feature_names: np.ndarray,
    reference_npz_path: Path,
    reference_genes_key: str,
) -> np.ndarray:
    with np.load(reference_npz_path) as data:
        if reference_genes_key not in data:
            raise ConfigError(
                f"inputs.reference.genes_key {reference_genes_key!r} is not present in {reference_npz_path}"
            )
        ordered_genes = data[reference_genes_key].astype(str)

    first_index_by_name: dict[str, int] = {}
    for i, name in enumerate(feature_names.astype(str)):
        if name not in first_index_by_name:
            first_index_by_name[name] = i

    missing = [g for g in ordered_genes if g not in first_index_by_name]
    if missing:
        preview = missing[:5]
        raise ValueError(
            f"{len(missing)} reference genes are missing in matrix features; first missing: {preview}"
        )
    return np.asarray([first_index_by_name[g] for g in ordered_genes], dtype=np.int64)


def _load_episode_build_expression_context(
    episodes_index_path: Path,
) -> dict[str, Any] | None:
    run_dir = episodes_index_path.parent
    cfg_path = run_dir / "config" / "config_resolved.yaml"
    if not cfg_path.exists():
        return None
    with cfg_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        return None
    inputs = raw.get("inputs")
    if not isinstance(inputs, dict):
        return None
    expression = inputs.get("expression")
    if not isinstance(expression, dict):
        return None
    if str(expression.get("mode", "")).strip() != "matrix_h5":
        return None
    matrix_value = expression.get("matrix_h5_path", None)
    if matrix_value is None:
        return None
    return {
        "matrix_h5_path": Path(str(matrix_value)).expanduser().resolve(),
        "cache_size": int(expression.get("cache_size", 20000)),
    }


def _build_nuclei_centers(df: pd.DataFrame, columns: dict[str, str]) -> dict[str, np.ndarray]:
    required = [columns["cell_id"], columns["center_x_um"], columns["center_y_um"]]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"missing columns in nuclei table: {missing}")

    cell_ids = df[columns["cell_id"]].astype(str)
    if cell_ids.isna().any():
        raise ValueError("nuclei cell_id column contains missing values")
    if cell_ids.duplicated().any():
        dup_value = str(cell_ids.loc[cell_ids.duplicated()].iloc[0])
        raise ValueError(f"duplicate cell_id found in nuclei table: {dup_value!r}")

    center_x = pd.to_numeric(df[columns["center_x_um"]], errors="raise").to_numpy(dtype=np.float64)
    center_y = pd.to_numeric(df[columns["center_y_um"]], errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(center_x).all() or not np.isfinite(center_y).all():
        raise ValueError("nuclei center coordinates must be finite")

    centers: dict[str, np.ndarray] = {}
    for idx, cell_id in enumerate(cell_ids.to_numpy()):
        centers[str(cell_id)] = np.asarray([center_x[idx], center_y[idx]], dtype=np.float64)
    return centers


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


def _load_axis(value: Any, name: str) -> GridAxis:
    mapping = _as_dict(value, name)
    return GridAxis(
        start=float(_require(mapping, "start", name)),
        stop=float(_require(mapping, "stop", name)),
        step=float(_require(mapping, "step", name)),
    )


def _axis_to_dict(axis: GridAxis) -> dict[str, float]:
    return {"start": axis.start, "stop": axis.stop, "step": axis.step}


def _as_dict(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"{name} must be a mapping")
    return value


def _require(mapping: dict[str, Any], key: str, section: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"missing required key '{key}' in section '{section}'")
    return mapping[key]


def _normalize_format(raw_format: str, path: Path) -> str:
    value = raw_format.strip().lower()
    if value == "auto":
        suffix = path.suffix.lower()
        if suffix in {".csv"}:
            return "csv"
        if suffix in {".tsv", ".txt"}:
            return "tsv"
        if suffix in {".parquet", ".pq"}:
            return "parquet"
        if suffix in {".npy"}:
            return "npy"
        if suffix in {".npz"}:
            return "npz"
        raise ConfigError(f"cannot infer format from extension for file: {path}")

    if value not in {"csv", "tsv", "parquet", "npy", "npz"}:
        raise ConfigError(f"unsupported format: {raw_format!r}")
    return value


def _load_table(path: Path, table_format: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"table file not found: {path}")

    if table_format == "csv":
        return pd.read_csv(path)
    if table_format == "tsv":
        return pd.read_csv(path, sep="\t")
    if table_format == "parquet":
        return pd.read_parquet(path)

    raise ConfigError(f"unsupported table format in loader: {table_format!r}")


def _build_metadata(run_dir: Path, seed: int | None) -> dict[str, Any]:
    return {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "seed": seed,
        "python_version": sys.version,
        "platform": platform.platform(),
        "git_commit": _try_git_commit(),
    }


def _try_git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return completed.stdout.strip() or None
    except Exception:
        return None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _append_step_log(path: Path, event: str, payload: dict[str, Any]) -> None:
    entry = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "event": event,
        "payload": payload,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=False))
        handle.write("\n")


def _slugify(text: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    normalized = normalized.strip("_")
    return normalized or "item"
