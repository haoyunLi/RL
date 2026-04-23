"""Reproducible grid search for reward weights using a greedy add-or-stop baseline."""

from __future__ import annotations

from collections import OrderedDict, defaultdict
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
from zoneinfo import ZoneInfo

import h5py
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import yaml

from .matrix_io import resolve_matrix_csc_h5_path
from .reward import (
    PosteriorAddBinReward,
    compute_bin_log_likelihood_by_type,
    compute_reference_distribution,
)


class ConfigError(ValueError):
    """Raised when reward-grid-search config is invalid."""


_REWARD_GRID_WORKER_CONTEXT: dict[str, Any] | None = None
_ARTIFACT_SHARD_CACHE: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()
_ARTIFACT_SHARD_CACHE_SIZE = 8
_LOCAL_TIMEZONE = ZoneInfo("America/Chicago")
_LOCAL_TIMEZONE_NAME = "America/Chicago"


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
        if self.start < 0 or self.stop < 0:
            raise ConfigError("grid values must be >= 0")
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
class SupervisedRewardConfig:
    """Config for GT-shape-supervised weight tuning over a selected cohort."""

    eval_csv_path: Path
    gt_cell_bins_path: Path
    selection_mode: str
    require_match_method: str | None
    overgrowth_ratio_threshold: float
    max_selected_episodes: int | None
    target_size_ratio: float
    oversize_penalty_weight: float
    size_ratio_filter_min: float | None
    size_ratio_filter_max: float | None

    def to_serializable_dict(self) -> dict[str, Any]:
        return {
            "eval_csv_path": str(self.eval_csv_path),
            "gt_cell_bins_path": str(self.gt_cell_bins_path),
            "selection_mode": self.selection_mode,
            "require_match_method": self.require_match_method,
            "overgrowth_ratio_threshold": self.overgrowth_ratio_threshold,
            "max_selected_episodes": self.max_selected_episodes,
            "target_size_ratio": self.target_size_ratio,
            "oversize_penalty_weight": self.oversize_penalty_weight,
            "size_ratio_filter_min": self.size_ratio_filter_min,
            "size_ratio_filter_max": self.size_ratio_filter_max,
        }


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
    expression_confidence_pseudocount: float
    normalize_expression_zscore: bool
    zscore_delta: float
    w4: float
    w1_axis: GridAxis
    w2_axis: GridAxis
    w3_axis: GridAxis
    w4_axis: GridAxis
    stop_lambda_axis: GridAxis
    stop_stat: str
    stop_top_k: int
    supervised: SupervisedRewardConfig | None

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
                "expression_confidence_pseudocount": self.expression_confidence_pseudocount,
                "normalize_expression_zscore": self.normalize_expression_zscore,
                "zscore_delta": self.zscore_delta,
                "w4": self.w4,
                "stop_stat": self.stop_stat,
                "stop_top_k": self.stop_top_k,
            },
            "search": {
                "w1": _axis_to_dict(self.w1_axis),
                "w2": _axis_to_dict(self.w2_axis),
                "w3": _axis_to_dict(self.w3_axis),
                "w4": _axis_to_dict(self.w4_axis),
                "stop_lambda": _axis_to_dict(self.stop_lambda_axis),
            },
            "supervised": None if self.supervised is None else self.supervised.to_serializable_dict(),
        }


@dataclass(frozen=True)
class PreparedEpisodeRewardData:
    """Episode payload with static reward terms precomputed once per run."""

    cell_id: str
    candidate_bin_ids: tuple[str, ...]
    initial_membership_mask: np.ndarray
    candidate_bin_xy_um: np.ndarray
    nucleus_center_xy_um: np.ndarray
    bin_count_totals: np.ndarray
    precomputed_ll: np.ndarray
    precomputed_d_other_um: np.ndarray
    matched_gt_cell_id: str | None = None
    gt_candidate_mask: np.ndarray | None = None
    gt_candidate_bin_count: int = 0
    gt_full_bin_count: int = 0
    eval_size_ratio: float | None = None


@dataclass(frozen=True)
class EpisodeArtifactLocator:
    """Resolved storage reference for one episode artifact."""

    path: Path
    member_index: int | None


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
    preparation_summary: dict[str, Any] | None = None


@dataclass(frozen=True)
class SupervisedEpisodeTarget:
    """GT target metadata for one selected episode cell."""

    episode_cell_id: str
    matched_gt_cell_id: str
    gt_full_bin_count: int
    eval_assigned_bin_count: int
    eval_size_ratio: float


@dataclass(frozen=True)
class SupervisedSelectionBundle:
    """Selected overgrowth tuning cohort from a prior evaluation run."""

    targets_by_cell_id: dict[str, SupervisedEpisodeTarget]
    summary: dict[str, Any]


@dataclass(frozen=True)
class GreedyEpisodeMetrics:
    """Metrics from greedy rollout on one episode for one weight setting."""

    total_return: float
    assigned_bins: int
    n_add_actions: int
    stop_reward: float
    final_best_add: float
    final_membership_mask: np.ndarray


@dataclass(frozen=True)
class RewardGridSearchResult:
    """Aggregated metrics for one weight combination."""

    w1: float
    w2: float
    w3: float
    w4: float
    stop_lambda: float
    n_episodes: int
    mean_return: float
    mean_assigned_bins: float
    mean_add_actions: float
    mean_stop_reward: float
    mean_final_best_add: float
    mean_gt_iou: float | None = None
    mean_gt_dice: float | None = None
    mean_gt_precision: float | None = None
    mean_gt_recall: float | None = None
    mean_gt_size_ratio: float | None = None
    mean_oversize_excess: float | None = None
    mean_oversize_penalty: float | None = None
    gt_shape_score: float | None = None

    def objective_value(self, objective: str) -> float:
        """Return comparable scalar used for best-weight selection."""
        if objective == "mean_return":
            return self.mean_return
        if objective == "gt_shape":
            if self.gt_shape_score is None:
                raise ConfigError("gt_shape objective requested but supervised metrics are missing")
            return self.gt_shape_score
        raise ConfigError(f"unsupported objective: {objective!r}")

    def to_row(self, objective: str) -> dict[str, Any]:
        """Return CSV/JSON row for this result."""
        return {
            "w1": self.w1,
            "w2": self.w2,
            "w3": self.w3,
            "w4": self.w4,
            "stop_lambda": self.stop_lambda,
            "n_episodes": self.n_episodes,
            "mean_return": self.mean_return,
            "mean_assigned_bins": self.mean_assigned_bins,
            "mean_add_actions": self.mean_add_actions,
            "mean_stop_reward": self.mean_stop_reward,
            "mean_final_best_add": self.mean_final_best_add,
            "mean_gt_iou": self.mean_gt_iou,
            "mean_gt_dice": self.mean_gt_dice,
            "mean_gt_precision": self.mean_gt_precision,
            "mean_gt_recall": self.mean_gt_recall,
            "mean_gt_size_ratio": self.mean_gt_size_ratio,
            "mean_oversize_excess": self.mean_oversize_excess,
            "mean_oversize_penalty": self.mean_oversize_penalty,
            "gt_shape_score": self.gt_shape_score,
            "objective": objective,
            "objective_value": self.objective_value(objective),
        }


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
    if re.fullmatch(r"[+-]?\d+\.0+", text):
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
    """Seed greedy reward evaluation with all confident nuclear bins for this cell."""
    if not candidate_bin_ids:
        return np.zeros((0,), dtype=np.int8)
    normalized_cell_id = _normalize_cell_id(cell_id)
    return np.asarray(
        [
            1 if normalized_cell_id is not None and nuclear_barcode_to_cell.get(str(bin_id)) == normalized_cell_id else 0
            for bin_id in candidate_bin_ids
        ],
        dtype=np.int8,
    )


def _load_supervised_selection(
    supervised: SupervisedRewardConfig,
    rng: np.random.Generator,
) -> SupervisedSelectionBundle:
    """Load GT-supervised cohort from a prior evaluation CSV and keep overgrowth cases only."""
    if not supervised.eval_csv_path.exists():
        raise FileNotFoundError(f"supervised eval CSV not found: {supervised.eval_csv_path}")

    eval_df = pd.read_csv(supervised.eval_csv_path)
    required = ["cell_id", "matched_gt_cell_id", "gt_cell_bin_count", "n_assigned_bins"]
    missing = [col for col in required if col not in eval_df.columns]
    if missing:
        raise ValueError(f"supervised eval CSV missing required columns: {missing}")

    work = eval_df.copy()
    work["episode_cell_id"] = work["cell_id"].map(_normalize_cell_id)
    work["matched_gt_cell_id_norm"] = work["matched_gt_cell_id"].map(_normalize_cell_id)
    work["gt_cell_bin_count"] = pd.to_numeric(work["gt_cell_bin_count"], errors="coerce")
    work["n_assigned_bins"] = pd.to_numeric(work["n_assigned_bins"], errors="coerce")
    work = work.loc[
        work["episode_cell_id"].notna()
        & work["matched_gt_cell_id_norm"].notna()
        & work["gt_cell_bin_count"].notna()
        & work["n_assigned_bins"].notna()
        & (work["gt_cell_bin_count"] > 0)
    ].copy()
    if supervised.require_match_method is not None and "match_method" in work.columns:
        work = work.loc[work["match_method"].astype(str) == supervised.require_match_method].copy()
    work["size_ratio"] = work["n_assigned_bins"] / work["gt_cell_bin_count"]
    work = work.loc[work["size_ratio"] >= float(supervised.overgrowth_ratio_threshold)].copy()
    if work.empty:
        raise ValueError(
            "supervised overgrowth cohort is empty; adjust eval CSV or overgrowth_ratio_threshold"
        )

    work = work.drop_duplicates(subset=["episode_cell_id"], keep="first").reset_index(drop=True)
    n_candidates = int(len(work))
    if supervised.max_selected_episodes is not None and len(work) > supervised.max_selected_episodes:
        keep = np.sort(rng.choice(len(work), size=supervised.max_selected_episodes, replace=False))
        work = work.iloc[keep].reset_index(drop=True)

    targets: dict[str, SupervisedEpisodeTarget] = {}
    for row in work.itertuples(index=False):
        episode_cell_id = str(row.episode_cell_id)
        targets[episode_cell_id] = SupervisedEpisodeTarget(
            episode_cell_id=episode_cell_id,
            matched_gt_cell_id=str(row.matched_gt_cell_id_norm),
            gt_full_bin_count=int(row.gt_cell_bin_count),
            eval_assigned_bin_count=int(row.n_assigned_bins),
            eval_size_ratio=float(row.size_ratio),
        )

    summary = {
        "selection_mode": supervised.selection_mode,
        "require_match_method": supervised.require_match_method,
        "overgrowth_ratio_threshold": float(supervised.overgrowth_ratio_threshold),
        "target_size_ratio": float(supervised.target_size_ratio),
        "oversize_penalty_weight": float(supervised.oversize_penalty_weight),
        "eval_rows": int(len(eval_df)),
        "eligible_overgrowth_rows": n_candidates,
        "selected_episode_count": int(len(targets)),
        "selected_mean_eval_size_ratio": float(work["size_ratio"].mean()),
        "selected_median_eval_size_ratio": float(work["size_ratio"].median()),
    }
    return SupervisedSelectionBundle(targets_by_cell_id=targets, summary=summary)


def _load_gt_cell_barcode_lookup(
    *,
    gt_cell_bins_path: Path,
    gt_cell_ids: set[str],
) -> dict[str, set[str]]:
    """Load GT outer-cell barcode sets for only the requested GT cell IDs."""
    if not gt_cell_bins_path.exists():
        raise FileNotFoundError(f"GT cell bins file not found: {gt_cell_bins_path}")
    if not gt_cell_ids:
        return {}

    out: dict[str, set[str]] = defaultdict(set)
    suffixes = "".join(gt_cell_bins_path.suffixes).lower()
    if suffixes.endswith(".parquet") or gt_cell_bins_path.suffix.lower() in {".parquet", ".pq"}:
        table = pd.read_parquet(gt_cell_bins_path, columns=["cell_id", "barcode"])
        table["cell_id_norm"] = table["cell_id"].map(_normalize_cell_id)
        table = table.loc[table["cell_id_norm"].isin(gt_cell_ids), ["cell_id_norm", "barcode"]].copy()
        for gt_cell_id, group in table.groupby("cell_id_norm", sort=False):
            out[str(gt_cell_id)].update(group["barcode"].astype(str).tolist())
        return dict(out)

    for chunk in pd.read_csv(gt_cell_bins_path, usecols=["cell_id", "barcode"], chunksize=2_000_000):
        chunk["cell_id_norm"] = chunk["cell_id"].map(_normalize_cell_id)
        chunk = chunk.loc[chunk["cell_id_norm"].isin(gt_cell_ids), ["cell_id_norm", "barcode"]]
        if chunk.empty:
            continue
        for gt_cell_id, group in chunk.groupby("cell_id_norm", sort=False):
            out[str(gt_cell_id)].update(group["barcode"].astype(str).tolist())
    return dict(out)


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
    supervised_cfg = _as_dict(raw.get("supervised"), "supervised")

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
    if objective not in {"mean_return", "gt_shape"}:
        raise ConfigError("reward.objective must be 'mean_return' or 'gt_shape'")

    epsilon = float(reward_cfg.get("epsilon", 1e-8))
    if epsilon < 0:
        raise ConfigError("reward.epsilon must be >= 0")

    r_max_um = float(reward_cfg.get("r_max_um", 80.0))
    if r_max_um <= 0:
        raise ConfigError("reward.r_max_um must be > 0")

    expression_confidence_pseudocount = float(reward_cfg.get("expression_confidence_pseudocount", 5.0))
    if expression_confidence_pseudocount < 0:
        raise ConfigError("reward.expression_confidence_pseudocount must be >= 0")

    normalize_expression_zscore = bool(reward_cfg.get("normalize_expression_zscore", False))
    zscore_delta = float(reward_cfg.get("zscore_delta", 1e-8))
    if zscore_delta <= 0:
        raise ConfigError("reward.zscore_delta must be > 0")
    w4 = float(reward_cfg.get("w4", 0.0))
    if w4 < 0:
        raise ConfigError("reward.w4 must be >= 0")
    stop_stat = str(reward_cfg.get("stop_stat", "max")).strip().lower()
    if stop_stat not in {"max", "topk_mean"}:
        raise ConfigError("reward.stop_stat must be 'max' or 'topk_mean'")
    stop_top_k = int(reward_cfg.get("stop_top_k", 3))
    if stop_top_k <= 0:
        raise ConfigError("reward.stop_top_k must be > 0")

    w1_axis = _load_axis(_require(search, "w1", "search"), "search.w1")
    w2_axis = _load_axis(_require(search, "w2", "search"), "search.w2")
    w3_axis = _load_axis(_require(search, "w3", "search"), "search.w3")
    if "w4" in search:
        w4_axis = _load_axis(search["w4"], "search.w4")
    else:
        w4_axis = GridAxis(start=float(w4), stop=float(w4), step=1.0)
    stop_lambda_axis = _load_axis(_require(search, "stop_lambda", "search"), "search.stop_lambda")
    _validate_axis_bounds(w1_axis, "search.w1", allow_zero=False)
    _validate_axis_bounds(w2_axis, "search.w2", allow_zero=False)
    _validate_axis_bounds(w3_axis, "search.w3", allow_zero=False)
    _validate_axis_bounds(w4_axis, "search.w4", allow_zero=True)
    _validate_axis_bounds(stop_lambda_axis, "search.stop_lambda", allow_zero=False)

    supervised: SupervisedRewardConfig | None = None
    if objective == "gt_shape":
        eval_csv_path = Path(str(_require(supervised_cfg, "eval_csv_path", "supervised"))).expanduser().resolve()
        gt_cell_bins_path = Path(str(_require(supervised_cfg, "gt_cell_bins_path", "supervised"))).expanduser().resolve()
        selection_mode = str(supervised_cfg.get("selection_mode", "overgrowth_only")).strip() or "overgrowth_only"
        if selection_mode != "overgrowth_only":
            raise ConfigError("supervised.selection_mode must be 'overgrowth_only'")
        require_match_method_raw = supervised_cfg.get("require_match_method", "nuclear_overlap")
        require_match_method = (
            None if require_match_method_raw in {None, ""} else str(require_match_method_raw).strip()
        )
        overgrowth_ratio_threshold = float(supervised_cfg.get("overgrowth_ratio_threshold", 1.4))
        if overgrowth_ratio_threshold <= 0:
            raise ConfigError("supervised.overgrowth_ratio_threshold must be > 0")
        max_selected_episodes = supervised_cfg.get("max_selected_episodes", None)
        if max_selected_episodes is not None:
            max_selected_episodes = int(max_selected_episodes)
            if max_selected_episodes <= 0:
                raise ConfigError("supervised.max_selected_episodes must be > 0 when provided")
        target_size_ratio = float(supervised_cfg.get("target_size_ratio", 1.15))
        if target_size_ratio <= 0:
            raise ConfigError("supervised.target_size_ratio must be > 0")
        oversize_penalty_weight = float(supervised_cfg.get("oversize_penalty_weight", 1.0))
        if oversize_penalty_weight < 0:
            raise ConfigError("supervised.oversize_penalty_weight must be >= 0")
        size_ratio_filter_min_raw = supervised_cfg.get("size_ratio_filter_min", 1.0)
        size_ratio_filter_max_raw = supervised_cfg.get("size_ratio_filter_max", 1.15)
        size_ratio_filter_min = None if size_ratio_filter_min_raw is None else float(size_ratio_filter_min_raw)
        size_ratio_filter_max = None if size_ratio_filter_max_raw is None else float(size_ratio_filter_max_raw)
        if size_ratio_filter_min is not None and size_ratio_filter_min <= 0:
            raise ConfigError("supervised.size_ratio_filter_min must be > 0 when provided")
        if size_ratio_filter_max is not None and size_ratio_filter_max <= 0:
            raise ConfigError("supervised.size_ratio_filter_max must be > 0 when provided")
        if (
            size_ratio_filter_min is not None
            and size_ratio_filter_max is not None
            and size_ratio_filter_max <= size_ratio_filter_min
        ):
            raise ConfigError("supervised.size_ratio_filter_max must be > supervised.size_ratio_filter_min")
        supervised = SupervisedRewardConfig(
            eval_csv_path=eval_csv_path,
            gt_cell_bins_path=gt_cell_bins_path,
            selection_mode=selection_mode,
            require_match_method=require_match_method,
            overgrowth_ratio_threshold=overgrowth_ratio_threshold,
            max_selected_episodes=max_selected_episodes,
            target_size_ratio=target_size_ratio,
            oversize_penalty_weight=oversize_penalty_weight,
            size_ratio_filter_min=size_ratio_filter_min,
            size_ratio_filter_max=size_ratio_filter_max,
        )

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
        expression_confidence_pseudocount=expression_confidence_pseudocount,
        normalize_expression_zscore=normalize_expression_zscore,
        zscore_delta=zscore_delta,
        w4=w4,
        w1_axis=w1_axis,
        w2_axis=w2_axis,
        w3_axis=w3_axis,
        w4_axis=w4_axis,
        stop_lambda_axis=stop_lambda_axis,
        stop_stat=stop_stat,
        stop_top_k=stop_top_k,
        supervised=supervised,
    )


def prepare_reward_grid_context(config: RewardGridSearchConfig) -> RewardGridPreparedContext:
    """Load reference + episode artifacts once and precompute static reward terms."""
    rng = np.random.default_rng(config.seed)
    reference_counts = _load_reference_counts(config)
    preparation_summary: dict[str, Any] | None = None

    supervised_selection: SupervisedSelectionBundle | None = None
    gt_barcodes_by_cell: dict[str, set[str]] | None = None
    if config.supervised is not None:
        supervised_selection = _load_supervised_selection(config.supervised, rng)
        gt_barcodes_by_cell = _load_gt_cell_barcode_lookup(
            gt_cell_bins_path=config.supervised.gt_cell_bins_path,
            gt_cell_ids={target.matched_gt_cell_id for target in supervised_selection.targets_by_cell_id.values()},
        )
        preparation_summary = dict(supervised_selection.summary)
        preparation_summary["loaded_gt_cell_count"] = int(len(gt_barcodes_by_cell))

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
        supervised_targets=None if supervised_selection is None else supervised_selection.targets_by_cell_id,
        gt_barcodes_by_cell=gt_barcodes_by_cell,
    )
    if not episodes:
        raise ValueError("no episodes available for reward grid search")

    if preparation_summary is not None:
        preparation_summary["prepared_episode_count"] = int(len(episodes))
    return RewardGridPreparedContext(
        reference_counts=reference_counts,
        episodes=episodes,
        preparation_summary=preparation_summary,
    )


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
        payload = {
            "n_episodes": int(len(prepared_context.episodes)),
            "n_cell_types": int(prepared_context.reference_counts.shape[0]),
            "n_genes": int(prepared_context.reference_counts.shape[1]),
        }
        if prepared_context.preparation_summary is not None:
            payload["preparation_summary"] = prepared_context.preparation_summary
        _append_step_log(steps_log_path, event="prepared_context_created", payload=payload)
    else:
        payload = {
            "n_episodes": int(len(prepared_context.episodes)),
            "n_cell_types": int(prepared_context.reference_counts.shape[0]),
            "n_genes": int(prepared_context.reference_counts.shape[1]),
        }
        if prepared_context.preparation_summary is not None:
            payload["preparation_summary"] = prepared_context.preparation_summary
        _append_step_log(steps_log_path, event="prepared_context_reused", payload=payload)

    reference_counts = prepared_context.reference_counts
    prepared_episodes = prepared_context.episodes

    grid_values = {
        "w1": config.w1_axis.values(),
        "w2": config.w2_axis.values(),
        "w3": config.w3_axis.values(),
        "w4": config.w4_axis.values(),
        "stop_lambda": config.stop_lambda_axis.values(),
    }
    n_combinations = (
        len(grid_values["w1"])
        * len(grid_values["w2"])
        * len(grid_values["w3"])
        * len(grid_values["w4"])
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
            "w4_values": list(grid_values["w4"]),
            "stop_lambda_values": list(grid_values["stop_lambda"]),
        },
    )

    combo_specs = [
        (combo_index, float(w1), float(w2), float(w3), float(w4), float(stop_lambda))
        for combo_index, (w1, w2, w3, w4, stop_lambda) in enumerate(
            itertools.product(
                grid_values["w1"],
                grid_values["w2"],
                grid_values["w3"],
                grid_values["w4"],
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
            "expression_confidence_pseudocount": float(config.expression_confidence_pseudocount),
            "stop_stat": str(config.stop_stat),
            "stop_top_k": int(config.stop_top_k),
            "normalize_expression_zscore": bool(config.normalize_expression_zscore),
            "zscore_delta": float(config.zscore_delta),
            "objective": str(config.objective),
            "supervised_target_size_ratio": None if config.supervised is None else float(config.supervised.target_size_ratio),
            "supervised_oversize_penalty_weight": None if config.supervised is None else float(config.supervised.oversize_penalty_weight),
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
        for combo_index, w1, w2, w3, w4, stop_lambda in combo_specs:
            result = _evaluate_weight_combination(
                episodes=prepared_episodes,
                reference_counts=reference_counts,
                epsilon=config.epsilon,
                r_max_um=config.r_max_um,
                expression_confidence_pseudocount=config.expression_confidence_pseudocount,
                w1=w1,
                w2=w2,
                w3=w3,
                w4=w4,
                stop_lambda=stop_lambda,
                stop_stat=config.stop_stat,
                stop_top_k=config.stop_top_k,
                normalize_expression_zscore=config.normalize_expression_zscore,
                zscore_delta=config.zscore_delta,
                objective=config.objective,
                supervised_target_size_ratio=None if config.supervised is None else config.supervised.target_size_ratio,
                supervised_oversize_penalty_weight=None if config.supervised is None else config.supervised.oversize_penalty_weight,
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

    results_df = pd.DataFrame([row.to_row(config.objective) for row in results])
    results_df = results_df.sort_values(by="objective_value", ascending=False).reset_index(drop=True)
    results_df.to_csv(run_dir / "results.csv", index=False)

    selection_df = results_df
    ratio_filter_payload: dict[str, Any] | None = None
    if config.supervised is not None and "mean_gt_size_ratio" in results_df.columns:
        size_ratio_filter_min = config.supervised.size_ratio_filter_min
        size_ratio_filter_max = config.supervised.size_ratio_filter_max
        if size_ratio_filter_min is not None or size_ratio_filter_max is not None:
            keep = pd.Series(True, index=results_df.index)
            if size_ratio_filter_min is not None:
                keep = keep & (results_df["mean_gt_size_ratio"] > float(size_ratio_filter_min))
            if size_ratio_filter_max is not None:
                keep = keep & (results_df["mean_gt_size_ratio"] <= float(size_ratio_filter_max))
            filtered_df = results_df.loc[keep].copy().reset_index(drop=True)
            filtered_df.to_csv(run_dir / "results_size_ratio_filtered.csv", index=False)
            ratio_filter_payload = {
                "size_ratio_filter_min": size_ratio_filter_min,
                "size_ratio_filter_max": size_ratio_filter_max,
                "n_input_rows": int(len(results_df)),
                "n_filtered_rows": int(len(filtered_df)),
            }
            _append_step_log(
                steps_log_path,
                event="size_ratio_filter_applied",
                payload=ratio_filter_payload,
            )
            if filtered_df.empty:
                raise ValueError(
                    "no weight combinations passed supervised size-ratio filter; "
                    "relax supervised.size_ratio_filter_min/max"
                )
            selection_df = filtered_df

    best_row = selection_df.iloc[0].to_dict()
    best_result = max(results, key=lambda row: row.objective_value(config.objective))
    if ratio_filter_payload is not None:
        # Ensure best_result matches the post-filter best row, not the global max over all rows.
        best_result = next(
            row
            for row in results
            if np.isclose(row.w1, float(best_row["w1"]))
            and np.isclose(row.w2, float(best_row["w2"]))
            and np.isclose(row.w3, float(best_row["w3"]))
            and np.isclose(row.w4, float(best_row["w4"]))
            and np.isclose(row.stop_lambda, float(best_row["stop_lambda"]))
        )

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
    if prepared_context.preparation_summary is not None:
        summary["preparation_summary"] = prepared_context.preparation_summary
    if ratio_filter_payload is not None:
        summary["size_ratio_filter"] = ratio_filter_payload
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
    expression_confidence_pseudocount: float,
    w1: float,
    w2: float,
    w3: float,
    w4: float,
    stop_lambda: float,
    stop_stat: str,
    stop_top_k: int,
    normalize_expression_zscore: bool,
    zscore_delta: float,
    objective: str,
    supervised_target_size_ratio: float | None,
    supervised_oversize_penalty_weight: float | None,
) -> RewardGridSearchResult:
    totals: list[GreedyEpisodeMetrics] = []
    gt_iou: list[float] = []
    gt_dice: list[float] = []
    gt_precision: list[float] = []
    gt_recall: list[float] = []
    gt_size_ratio: list[float] = []
    oversize_excess: list[float] = []
    oversize_penalty: list[float] = []

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
            w4=w4,
            stop_lambda=stop_lambda,
            stop_stat=stop_stat,
            stop_top_k=stop_top_k,
            expression_confidence_pseudocount=expression_confidence_pseudocount,
            normalize_expression_zscore=normalize_expression_zscore,
            zscore_delta=zscore_delta,
            precomputed_bin_count_totals=episode.bin_count_totals,
            precomputed_ll=episode.precomputed_ll,
            precomputed_d_other_um=episode.precomputed_d_other_um,
        )
        metrics = _run_greedy_episode(reward_fn, episode.initial_membership_mask)
        totals.append(metrics)
        if episode.gt_candidate_mask is not None:
            pred_mask = np.asarray(metrics.final_membership_mask, dtype=bool)
            gt_mask = np.asarray(episode.gt_candidate_mask, dtype=bool)
            pred_count = int(pred_mask.sum())
            gt_count = int(episode.gt_candidate_bin_count)
            inter = int(np.logical_and(pred_mask, gt_mask).sum())
            union = int(np.logical_or(pred_mask, gt_mask).sum())
            gt_iou.append(0.0 if union <= 0 else float(inter / union))
            denom = pred_count + gt_count
            gt_dice.append(0.0 if denom <= 0 else float((2.0 * inter) / denom))
            gt_precision.append(0.0 if pred_count <= 0 else float(inter / pred_count))
            gt_recall.append(0.0 if gt_count <= 0 else float(inter / gt_count))
            size_ratio = 0.0 if gt_count <= 0 else float(pred_count / gt_count)
            gt_size_ratio.append(size_ratio)
            target_ratio = 1.0 if supervised_target_size_ratio is None else float(supervised_target_size_ratio)
            excess = max(size_ratio - target_ratio, 0.0)
            oversize_excess.append(float(excess))
            oversize_penalty.append(float(excess * excess))

    total_return = np.asarray([m.total_return for m in totals], dtype=np.float64)
    assigned_bins = np.asarray([m.assigned_bins for m in totals], dtype=np.float64)
    add_actions = np.asarray([m.n_add_actions for m in totals], dtype=np.float64)
    stop_rewards = np.asarray([m.stop_reward for m in totals], dtype=np.float64)
    final_best_add = np.asarray([m.final_best_add for m in totals], dtype=np.float64)

    mean_gt_iou = None
    mean_gt_dice = None
    mean_gt_precision = None
    mean_gt_recall = None
    mean_gt_size_ratio = None
    mean_oversize_excess = None
    mean_oversize_penalty = None
    gt_shape_score = None
    if gt_dice:
        mean_gt_iou = float(np.mean(np.asarray(gt_iou, dtype=np.float64)))
        mean_gt_dice = float(np.mean(np.asarray(gt_dice, dtype=np.float64)))
        mean_gt_precision = float(np.mean(np.asarray(gt_precision, dtype=np.float64)))
        mean_gt_recall = float(np.mean(np.asarray(gt_recall, dtype=np.float64)))
        mean_gt_size_ratio = float(np.mean(np.asarray(gt_size_ratio, dtype=np.float64)))
        mean_oversize_excess = float(np.mean(np.asarray(oversize_excess, dtype=np.float64)))
        mean_oversize_penalty = float(np.mean(np.asarray(oversize_penalty, dtype=np.float64)))
        penalty_weight = 0.0 if supervised_oversize_penalty_weight is None else float(supervised_oversize_penalty_weight)
        gt_shape_score = float(mean_gt_dice - penalty_weight * mean_oversize_penalty)
    if objective == "gt_shape" and gt_shape_score is None:
        raise ValueError("gt_shape objective requested but no GT-supervised episodes were available")

    return RewardGridSearchResult(
        w1=float(w1),
        w2=float(w2),
        w3=float(w3),
        w4=float(w4),
        stop_lambda=float(stop_lambda),
        n_episodes=len(totals),
        mean_return=float(total_return.mean()),
        mean_assigned_bins=float(assigned_bins.mean()),
        mean_add_actions=float(add_actions.mean()),
        mean_stop_reward=float(stop_rewards.mean()),
        mean_final_best_add=float(final_best_add.mean()),
        mean_gt_iou=mean_gt_iou,
        mean_gt_dice=mean_gt_dice,
        mean_gt_precision=mean_gt_precision,
        mean_gt_recall=mean_gt_recall,
        mean_gt_size_ratio=mean_gt_size_ratio,
        mean_oversize_excess=mean_oversize_excess,
        mean_oversize_penalty=mean_oversize_penalty,
        gt_shape_score=gt_shape_score,
    )


def _evaluate_weight_combination_worker(
    combo_spec: tuple[int, float, float, float, float, float],
) -> tuple[int, RewardGridSearchResult]:
    """Process-pool worker wrapper for one weight combination."""
    if _REWARD_GRID_WORKER_CONTEXT is None:
        raise RuntimeError("reward grid worker context is not initialized")

    combo_index, w1, w2, w3, w4, stop_lambda = combo_spec
    ctx = _REWARD_GRID_WORKER_CONTEXT
    result = _evaluate_weight_combination(
        episodes=ctx["episodes"],
        reference_counts=ctx["reference_counts"],
        epsilon=float(ctx["epsilon"]),
        r_max_um=float(ctx["r_max_um"]),
        expression_confidence_pseudocount=float(ctx["expression_confidence_pseudocount"]),
        w1=float(w1),
        w2=float(w2),
        w3=float(w3),
        w4=float(w4),
        stop_lambda=float(stop_lambda),
        stop_stat=str(ctx["stop_stat"]),
        stop_top_k=int(ctx["stop_top_k"]),
        normalize_expression_zscore=bool(ctx["normalize_expression_zscore"]),
        zscore_delta=float(ctx["zscore_delta"]),
        objective=str(ctx["objective"]),
        supervised_target_size_ratio=ctx["supervised_target_size_ratio"],
        supervised_oversize_penalty_weight=ctx["supervised_oversize_penalty_weight"],
    )
    return int(combo_index), result


def _run_greedy_episode(
    reward_fn: PosteriorAddBinReward,
    initial_membership_mask: np.ndarray,
) -> GreedyEpisodeMetrics:
    """Roll out the simple greedy add-or-stop baseline for one episode."""
    membership_mask = np.asarray(initial_membership_mask, dtype=np.int8).copy()
    if membership_mask.shape != (reward_fn.n_candidate_bins,):
        raise ValueError(
            f"initial_membership_mask must have shape ({reward_fn.n_candidate_bins},), got {membership_mask.shape}"
        )
    total_return = 0.0
    n_add_actions = 0

    while True:
        eligible = reward_fn.frontier_add_mask(membership_mask)
        if not np.any(eligible):
            stop_reward = float(reward_fn.stop_reward(membership_mask))
            total_return += stop_reward
            return GreedyEpisodeMetrics(
                total_return=total_return,
                assigned_bins=int(membership_mask.sum()),
                n_add_actions=n_add_actions,
                stop_reward=stop_reward,
                final_best_add=0.0,
                final_membership_mask=membership_mask.copy(),
            )

        r_add = reward_fn.add_reward_per_bin(membership_mask)
        eligible_idx = np.flatnonzero(eligible)
        best_pos = int(np.argmax(r_add[eligible_idx]))
        best_idx = int(eligible_idx[best_pos])
        best_add = float(r_add[best_idx])
        stop_delta = float(reward_fn.stop_delta(membership_mask))

        if stop_delta <= 0.0:
            stop_reward = float(reward_fn.stop_reward(membership_mask))
            total_return += stop_reward
            return GreedyEpisodeMetrics(
                total_return=total_return,
                assigned_bins=int(membership_mask.sum()),
                n_add_actions=n_add_actions,
                stop_reward=stop_reward,
                final_best_add=best_add,
                final_membership_mask=membership_mask.copy(),
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
    supervised_targets: dict[str, SupervisedEpisodeTarget] | None,
    gt_barcodes_by_cell: dict[str, set[str]] | None,
) -> list[PreparedEpisodeRewardData]:
    if not episodes_index_path.exists():
        raise FileNotFoundError(f"episodes index file not found: {episodes_index_path}")

    index_df = pd.read_csv(episodes_index_path)
    required_cols = ["cell_id", "artifact_path"]
    missing = [col for col in required_cols if col not in index_df.columns]
    if missing:
        raise ValueError(f"missing columns in episodes index: {missing}")
    index_df["cell_id_norm"] = index_df["cell_id"].map(_normalize_cell_id)
    if supervised_targets is not None:
        index_df = index_df.loc[index_df["cell_id_norm"].isin(supervised_targets)].reset_index(drop=True)
        if index_df.empty:
            raise ValueError("no overlap between episodes_index.csv and supervised overgrowth cohort")

    if max_episodes is not None and len(index_df) > max_episodes:
        keep = np.sort(rng.choice(len(index_df), size=max_episodes, replace=False))
        index_df = index_df.iloc[keep].reset_index(drop=True)

    theta = compute_reference_distribution(reference_counts=reference_counts, epsilon=epsilon)
    log_theta = np.log(theta)

    expression_ctx = _load_episode_build_expression_context(episodes_index_path)
    matrix_path = None if expression_ctx is None else expression_ctx["matrix_path"]
    expression_cache_size = 20000 if expression_ctx is None else int(expression_ctx["cache_size"])
    bins_path = None if expression_ctx is None else expression_ctx.get("bins_path")
    expression_loader: _MatrixOnDemandExpressionLoader | None = None
    nuclear_barcode_to_cell: dict[str, str] = {}
    if matrix_path is not None and reference_format == "npz":
        expression_loader = _MatrixOnDemandExpressionLoader(
            matrix_path=matrix_path,
            reference_npz_path=reference_path,
            reference_genes_key=reference_genes_key,
            cache_size=expression_cache_size,
        )
        if bins_path is not None:
            nuclear_barcode_to_cell = _load_nuclear_barcode_assignment_lookup(Path(bins_path))

    prepared: list[PreparedEpisodeRewardData] = []
    try:
        for row in index_df.itertuples(index=False):
            cell_id_norm = _normalize_cell_id(getattr(row, "cell_id_norm", row.cell_id))
            cell_id_raw = str(row.cell_id)
            cell_id = cell_id_raw if cell_id_raw in nuclei_centers_by_cell else (cell_id_norm or cell_id_raw)
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
            initial_membership_mask = _build_initial_membership_mask(
                candidate_bin_ids=episode.candidate_bin_ids,
                cell_id=cell_id,
                nuclear_barcode_to_cell=nuclear_barcode_to_cell,
            )
            matched_gt_cell_id: str | None = None
            gt_candidate_mask: np.ndarray | None = None
            gt_candidate_bin_count = 0
            gt_full_bin_count = 0
            eval_size_ratio: float | None = None
            if supervised_targets is not None:
                if cell_id_norm is None:
                    continue
                target = supervised_targets.get(cell_id_norm)
                if target is None:
                    continue
                matched_gt_cell_id = str(target.matched_gt_cell_id)
                gt_full_bin_count = int(target.gt_full_bin_count)
                eval_size_ratio = float(target.eval_size_ratio)
                gt_barcodes = None if gt_barcodes_by_cell is None else gt_barcodes_by_cell.get(matched_gt_cell_id)
                if not gt_barcodes:
                    continue
                gt_candidate_mask = np.asarray(
                    [str(bin_id) in gt_barcodes for bin_id in episode.candidate_bin_ids],
                    dtype=np.int8,
                )
                gt_candidate_bin_count = int(gt_candidate_mask.sum())
                if gt_candidate_bin_count <= 0:
                    continue
            prepared.append(
                PreparedEpisodeRewardData(
                    cell_id=episode.cell_id,
                    candidate_bin_ids=episode.candidate_bin_ids,
                    initial_membership_mask=initial_membership_mask,
                    candidate_bin_xy_um=episode.candidate_bin_xy_um,
                    nucleus_center_xy_um=episode.nucleus_center_xy_um,
                    bin_count_totals=episode.bin_count_totals,
                    precomputed_ll=episode.precomputed_ll,
                    precomputed_d_other_um=episode.precomputed_d_other_um,
                    matched_gt_cell_id=matched_gt_cell_id,
                    gt_candidate_mask=gt_candidate_mask,
                    gt_candidate_bin_count=gt_candidate_bin_count,
                    gt_full_bin_count=gt_full_bin_count,
                    eval_size_ratio=eval_size_ratio,
                )
            )
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
    artifact_path: str | Path,
    cell_id: str,
    expression_loader: "_MatrixOnDemandExpressionLoader | None",
    theta: np.ndarray,
    log_theta: np.ndarray,
    nuclei_spatial_index: NucleiSpatialIndex,
    include_candidate_bin_ids: bool = True,
) -> PreparedEpisodeRewardData | None:
    locator = _parse_episode_artifact_locator(artifact_path)
    if not locator.path.exists():
        raise FileNotFoundError(f"episode artifact not found: {locator.path}")

    if locator.member_index is None:
        candidate_bin_ids, candidate_bin_xy_um, nucleus_center_xy_um, candidate_expression, col_index = _load_legacy_episode_artifact_payload(
            artifact_path=locator.path,
            include_candidate_bin_ids=include_candidate_bin_ids,
        )
    else:
        candidate_bin_ids, candidate_bin_xy_um, nucleus_center_xy_um, candidate_expression, col_index = _load_sharded_episode_artifact_payload(
            locator=locator,
            cell_id=cell_id,
            include_candidate_bin_ids=include_candidate_bin_ids,
        )

    if candidate_expression is not None:
        expr = np.asarray(candidate_expression, dtype=np.float64)
        ll = compute_bin_log_likelihood_by_type(bin_counts=expr, theta=theta)
        bin_count_totals = np.sum(expr, axis=1, dtype=np.float64)
    elif col_index is not None:
        if expression_loader is None:
            raise ValueError(
                f"artifact {locator.path} stores candidate_matrix_col_index but no matrix loader is available. "
                "Ensure episodes_index.csv is from a run with config/config_resolved.yaml and reference format is npz."
            )
        ll, bin_count_totals = expression_loader.compute_ll_and_bin_counts_for_columns(
            col_index=col_index,
            log_theta=log_theta,
        )
    else:
        raise ValueError(
            f"artifact {locator.path} must contain either candidate_expression or candidate_matrix_col_index"
        )

    if ll.ndim != 2:
        raise ValueError(f"precomputed ll in {artifact_path} must have shape (B, K)")
    if candidate_bin_xy_um.shape != (ll.shape[0], 2):
        raise ValueError(f"candidate_bin_xy_um in {artifact_path} must have shape (B, 2)")
    if include_candidate_bin_ids and len(candidate_bin_ids) != ll.shape[0]:
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
        initial_membership_mask=np.zeros((ll.shape[0],), dtype=np.int8),
        candidate_bin_xy_um=candidate_bin_xy_um,
        nucleus_center_xy_um=nucleus_center_xy_um,
        bin_count_totals=np.asarray(bin_count_totals, dtype=np.float64),
        precomputed_ll=ll,
        precomputed_d_other_um=d_other,
    )


def _parse_episode_artifact_locator(artifact_path: str | Path) -> EpisodeArtifactLocator:
    raw = str(artifact_path)
    if "::" in raw:
        path_str, member_str = raw.rsplit("::", 1)
        try:
            member_index = int(member_str)
        except ValueError as exc:
            raise ValueError(f"invalid artifact locator member index in {raw!r}") from exc
        return EpisodeArtifactLocator(
            path=Path(path_str).expanduser().resolve(),
            member_index=member_index,
        )
    return EpisodeArtifactLocator(
        path=Path(raw).expanduser().resolve(),
        member_index=None,
    )


def _load_legacy_episode_artifact_payload(
    *,
    artifact_path: Path,
    include_candidate_bin_ids: bool,
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    with np.load(artifact_path, allow_pickle=True) as data:
        candidate_bin_ids: tuple[str, ...]
        if include_candidate_bin_ids:
            candidate_bin_ids = tuple(str(x) for x in np.asarray(data["candidate_bin_ids"], dtype=object).tolist())
        else:
            candidate_bin_ids = tuple()
        candidate_bin_xy_um = np.asarray(data["candidate_bin_xy_um"], dtype=np.float64)
        nucleus_center_xy_um = np.asarray(data["nucleus_center_xy_um"], dtype=np.float64)
        candidate_expression = None
        col_index = None
        if "candidate_expression" in data:
            candidate_expression = np.asarray(data["candidate_expression"], dtype=np.float64)
        elif "candidate_matrix_col_index" in data:
            col_index = np.asarray(data["candidate_matrix_col_index"], dtype=np.int64)
    return candidate_bin_ids, candidate_bin_xy_um, nucleus_center_xy_um, candidate_expression, col_index


def _load_sharded_episode_artifact_payload(
    *,
    locator: EpisodeArtifactLocator,
    cell_id: str,
    include_candidate_bin_ids: bool,
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    shard = _open_episode_artifact_shard(locator.path)
    member_index = int(locator.member_index)
    n_members = int(shard["nucleus_center_xy_um"].shape[0])
    if member_index < 0 or member_index >= n_members:
        raise IndexError(
            f"artifact locator member index {member_index} is outside shard range [0, {n_members}) for {locator.path}"
        )

    row_splits = np.asarray(shard["candidate_row_splits"], dtype=np.int64)
    start = int(row_splits[member_index])
    end = int(row_splits[member_index + 1])
    candidate_bin_xy_um = np.asarray(shard["candidate_bin_xy_um"][start:end], dtype=np.float64)
    nucleus_center_xy_um = np.asarray(shard["nucleus_center_xy_um"][member_index], dtype=np.float64)

    candidate_bin_ids: tuple[str, ...]
    candidate_ids_arr = shard.get("candidate_bin_ids")
    if include_candidate_bin_ids and candidate_ids_arr is not None:
        candidate_bin_ids = tuple(str(x) for x in np.asarray(candidate_ids_arr[start:end]).tolist())
    elif include_candidate_bin_ids:
        candidate_bin_ids = tuple(f"{cell_id}::bin::{i}" for i in range(end - start))
    else:
        candidate_bin_ids = tuple()

    candidate_expression = None
    if "candidate_expression" in shard:
        candidate_expression = np.asarray(shard["candidate_expression"][start:end], dtype=np.float64)

    col_index = None
    if "candidate_matrix_col_index" in shard:
        col_index = np.asarray(shard["candidate_matrix_col_index"][start:end], dtype=np.int64)

    return candidate_bin_ids, candidate_bin_xy_um, nucleus_center_xy_um, candidate_expression, col_index


def _open_episode_artifact_shard(shard_path: Path) -> dict[str, np.ndarray]:
    key = str(shard_path)
    cached = _ARTIFACT_SHARD_CACHE.get(key)
    if cached is not None:
        _ARTIFACT_SHARD_CACHE.move_to_end(key)
        return cached

    if not shard_path.is_dir():
        raise FileNotFoundError(f"episode artifact shard directory not found: {shard_path}")

    required = {
        "nucleus_center_xy_um": "nucleus_center_xy_um.npy",
        "candidate_row_splits": "candidate_row_splits.npy",
        "candidate_bin_xy_um": "candidate_bin_xy_um.npy",
    }
    optional = {
        "candidate_bin_ids": "candidate_bin_ids.npy",
        "candidate_expression": "candidate_expression.npy",
        "candidate_matrix_col_index": "candidate_matrix_col_index.npy",
    }

    shard: dict[str, np.ndarray] = {}
    for name, filename in required.items():
        file_path = shard_path / filename
        if not file_path.exists():
            raise FileNotFoundError(f"artifact shard is missing required array: {file_path}")
        shard[name] = np.load(file_path, mmap_mode="r", allow_pickle=False)

    for name, filename in optional.items():
        file_path = shard_path / filename
        if file_path.exists():
            shard[name] = np.load(file_path, mmap_mode="r", allow_pickle=False)

    _ARTIFACT_SHARD_CACHE[key] = shard
    _ARTIFACT_SHARD_CACHE.move_to_end(key)
    while len(_ARTIFACT_SHARD_CACHE) > _ARTIFACT_SHARD_CACHE_SIZE:
        _ARTIFACT_SHARD_CACHE.popitem(last=False)
    return shard


class _MatrixOnDemandExpressionLoader:
    """Load selected-gene expression vectors for matrix column indices from 10x H5."""

    def __init__(
        self,
        matrix_path: Path,
        reference_npz_path: Path,
        reference_genes_key: str,
        cache_size: int = 20000,
    ) -> None:
        if not matrix_path.exists():
            raise FileNotFoundError(f"matrix source not found: {matrix_path}")
        if not reference_npz_path.exists():
            raise FileNotFoundError(f"reference NPZ file not found: {reference_npz_path}")
        if cache_size < 0:
            raise ValueError("cache_size must be >= 0")

        resolved_matrix_h5_path = resolve_matrix_csc_h5_path(matrix_path)
        self._h5 = h5py.File(resolved_matrix_h5_path, "r")
        if "matrix" not in self._h5:
            raise ValueError(f"H5 file does not contain 'matrix' group: {resolved_matrix_h5_path}")
        mg = self._h5["matrix"]
        for key in ("data", "indices", "indptr", "shape", "features"):
            if key not in mg:
                raise ValueError(f"H5 matrix group missing key: matrix/{key}")
        fg = mg["features"]
        if "name" not in fg:
            raise ValueError("H5 matrix/features missing 'name' dataset")

        shape = tuple(int(v) for v in mg["shape"][:].tolist())
        if len(shape) != 2:
            raise ValueError(f"matrix/shape in {resolved_matrix_h5_path} is invalid: {shape}")
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
        ll, _ = self.compute_ll_and_bin_counts_for_columns(col_index=col_index, log_theta=log_theta)
        return ll

    def compute_ll_and_bin_counts_for_columns(
        self,
        col_index: np.ndarray,
        log_theta: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute LL[B, K] and total selected-gene counts per bin from sparse columns."""
        cols = np.asarray(col_index, dtype=np.int64)
        if cols.ndim != 1:
            raise ValueError("candidate_matrix_col_index must be a 1D array")
        if cols.size == 0:
            return (
                np.zeros((0, int(np.asarray(log_theta).shape[0])), dtype=np.float64),
                np.zeros((0,), dtype=np.float64),
            )
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
        bin_count_totals = np.zeros((cols.size,), dtype=np.float64)
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
            bin_count_totals[i] = nb
            if nb <= 0:
                continue

            # Weighted average log-likelihood over selected genes for this one bin.
            weighted = np.sum(lt[:, pos] * vals[None, :], axis=1)
            out[i, :] = weighted / nb

        return out, bin_count_totals

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
    matrix_value = expression.get("matrix_path", expression.get("matrix_h5_path", None))
    if matrix_value is None:
        return None
    bins_value = inputs.get("bins_path")
    return {
        "matrix_path": Path(str(matrix_value)).expanduser().resolve(),
        "cache_size": int(expression.get("cache_size", 20000)),
        "bins_path": None if bins_value is None else Path(str(bins_value)).expanduser().resolve(),
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


def _validate_axis_bounds(axis: GridAxis, name: str, *, allow_zero: bool) -> None:
    if allow_zero:
        if axis.start < 0 or axis.stop < 0:
            raise ConfigError(f"{name} values must be >= 0")
        return
    if axis.start <= 0 or axis.stop <= 0:
        raise ConfigError(f"{name} values must be > 0")


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
    now_utc, now_local = _now_utc_and_local()
    return {
        "timestamp_utc": now_utc.isoformat(),
        "timestamp_local": now_local.isoformat(),
        "local_timezone": _LOCAL_TIMEZONE_NAME,
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
    now_utc, now_local = _now_utc_and_local()
    entry = {
        "timestamp_utc": now_utc.isoformat(),
        "timestamp_local": now_local.isoformat(),
        "event": event,
        "payload": payload,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=False))
        handle.write("\n")


def _now_utc_and_local() -> tuple[dt.datetime, dt.datetime]:
    """Return current UTC and America/Chicago timestamps."""
    now_utc = dt.datetime.now(dt.timezone.utc)
    return now_utc, now_utc.astimezone(_LOCAL_TIMEZONE)


def _slugify(text: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    normalized = normalized.strip("_")
    return normalized or "item"
