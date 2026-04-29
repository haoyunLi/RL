"""Interactive PPO debugging helpers for single-cell, single-step inspection."""

from __future__ import annotations

from dataclasses import dataclass, replace
import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from hd_cell_rl.ppo_training import (
    ActorCritic,
    AddStopCellEnv,
    EpisodeDataset,
    EpisodeContext,
    PPOTrainingConfig,
    _add_rewards_from_membership_mask,
    _compute_dynamic_add_action_features,
    _compute_state_summary_from_mask,
    _expression_reward_terms_per_bin,
    _normalize_cell_id,
    _observation_to_tensors,
    _posterior_from_membership_mask,
    _resolve_device,
    _load_table,
    load_ppo_training_config,
)
from hd_cell_rl.reward_grid_search import (
    _load_legacy_episode_artifact_payload,
    _load_sharded_episode_artifact_payload,
    _parse_episode_artifact_locator,
)
from hd_cell_rl.reward import (
    compute_frontier_eligible_mask,
    compute_neighbor_support_fraction,
    compute_stop_delta,
)


GLOBAL_FEATURE_NAMES: tuple[str, ...] = (
    "assigned_frac",
    "step_frac",
    "n_bins_scaled",
    "assigned_ll_mean",
    "assigned_ll_max",
    "remaining_frac",
    "seed_size_scaled",
    "grow_ratio_scaled",
    "positive_frontier_fraction",
    "centroid_drift_scaled",
    "compactness_proxy",
    "frontier_add_reward_topk_mean",
    "frontier_add_reward_mean",
    "frontier_add_reward_std",
    "frontier_add_reward_max",
    "seed_compactness",
    "seed_radius_p90_scaled",
    "seed_aspect_ratio_scaled",
)

_ACTION_FEATURE_TAIL: tuple[str, ...] = (
    "candidate_to_current_centroid_distance",
    "candidate_compactness_gain",
    "dx_from_nucleus_scaled",
    "dy_from_nucleus_scaled",
    "dx_from_current_centroid_scaled",
    "dy_from_current_centroid_scaled",
    "radial_alignment_with_centroid_drift",
    "candidate_add_reward_total",
    "candidate_expr_term",
    "candidate_base_penalty",
    "candidate_neighbor_support",
    "candidate_expression_confidence",
    "candidate_count_scaled",
    "frontier_add_reward_topk_mean",
    "frontier_add_reward_mean",
    "frontier_add_reward_std",
    "frontier_add_reward_max",
    "seed_compactness",
    "seed_radius_p90_scaled",
    "seed_aspect_ratio_scaled",
    "ll_margin_z",
    "ll_entropy_scaled",
    "candidate_seed_neighbor_support",
    "candidate_dist_to_seed_centroid_scaled",
    "candidate_expr_raw",
    "candidate_expr_weighted",
    "candidate_distance_penalty",
    "candidate_overlap_penalty",
)

ACTION_FEATURE_NAMES: tuple[str, ...] = (
    "is_stop_action",
    "col1",
    "col2",
    "col3",
    "col4",
    "col5",
    "col6",
    "col7",
    "col8",
    "col9",
    "col10",
    *_ACTION_FEATURE_TAIL,
)

STOP_ACTION_FEATURE_LABELS: tuple[str, ...] = (
    "is_stop_action",
    "assigned_frac",
    "step_frac",
    "n_bins_scaled",
    "assigned_ll_mean",
    "remaining_frac",
    "seed_size_scaled",
    "grow_ratio_scaled",
    "positive_frontier_fraction",
    "centroid_drift_scaled",
    "compactness_proxy",
    *_ACTION_FEATURE_TAIL,
)

ADD_ACTION_FEATURE_LABELS: tuple[str, ...] = (
    "is_stop_action",
    "ll_mean_z",
    "ll_max_z",
    "p_dis",
    "p_overlap",
    "is_assigned",
    "seed_size_scaled_or_zero",
    "grow_ratio_scaled_or_zero",
    "positive_frontier_fraction_or_zero",
    "centroid_drift_scaled_or_zero",
    "compactness_proxy_or_zero",
    *_ACTION_FEATURE_TAIL,
)


@dataclass(frozen=True)
class PPODebugSummary:
    """Resolved run metadata loaded from a PPO evaluation directory."""

    eval_run_dir: Path
    checkpoint_path: Path
    config_path: Path
    summary: dict[str, Any]
    step_traces_dir: Path | None = None


@dataclass(frozen=True)
class StepDebugState:
    """One decision-time state before the greedy policy takes its action."""

    cell_id: str
    step_index: int
    membership_mask: np.ndarray
    frontier_mask: np.ndarray
    action_mask: np.ndarray
    global_features: np.ndarray
    action_features: np.ndarray
    posterior: np.ndarray
    value_estimate: float
    raw_policy_logits: np.ndarray
    masked_action_probabilities: np.ndarray
    stop_delta: float
    stop_reward: float
    stop_probability: float
    stop_logit: float
    expr_frontier_mean: float
    expr_frontier_std: float
    state_summary: dict[str, float]
    chosen_action: int
    chosen_action_probability: float
    chosen_action_logit: float
    chosen_reward: float
    chosen_barcode: str | None
    terminated_after_action: bool
    truncated_after_action: bool
    n_assigned_bins_after: int
    bin_table: pd.DataFrame


@dataclass(frozen=True)
class EpisodeDebugTrace:
    """Full greedy replay trace for one cell."""

    cell_id: str
    episode_metrics: dict[str, Any]
    candidate_bin_ids: tuple[str, ...]
    candidate_bin_xy_um: np.ndarray
    nucleus_center_xy_um: np.ndarray
    gt_cell_xy_um: np.ndarray | None
    final_membership_mask: np.ndarray
    step_states: tuple[StepDebugState, ...]
    total_reward_replayed: float
    n_steps_replayed: int
    n_assigned_bins_replayed: int
    replay_matches_eval: bool
    replay_source: str
    step_trace_path: Path | None = None


def _load_debug_summary(
    *,
    ppo_eval_run_dir: Path,
    checkpoint_path: str | Path | None,
) -> PPODebugSummary:
    run_dir = Path(ppo_eval_run_dir).expanduser().resolve()
    summary_path = run_dir / "summary.json"
    config_path = run_dir / "config_used.yaml"
    if not summary_path.exists():
        raise FileNotFoundError(f"PPO eval summary not found: {summary_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"PPO eval config_used.yaml not found: {config_path}")

    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    if not isinstance(summary, dict):
        raise ValueError(f"invalid summary JSON in {summary_path}")

    resolved_checkpoint: Path | None = None
    if checkpoint_path is not None:
        resolved_checkpoint = Path(checkpoint_path).expanduser().resolve()
    else:
        summary_ckpt = summary.get("checkpoint_path")
        if summary_ckpt:
            resolved_checkpoint = Path(str(summary_ckpt)).expanduser().resolve()
    if resolved_checkpoint is None:
        raise ValueError(
            "checkpoint path could not be resolved. Pass --checkpoint or use a PPO eval run with summary.json['checkpoint_path']."
        )
    if not resolved_checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {resolved_checkpoint}")

    step_traces_dir = None
    step_traces_dir_raw = summary.get("step_traces_dir")
    if step_traces_dir_raw:
        step_traces_dir_candidate = Path(str(step_traces_dir_raw)).expanduser().resolve()
        if step_traces_dir_candidate.exists():
            step_traces_dir = step_traces_dir_candidate

    policy_mode = str(summary.get("policy_mode", ""))
    if policy_mode and policy_mode != "greedy" and step_traces_dir is None:
        raise ValueError(
            "debug app needs saved step traces for non-greedy eval runs, but summary.json does not provide a valid step_traces_dir"
        )

    return PPODebugSummary(
        eval_run_dir=run_dir,
        checkpoint_path=resolved_checkpoint,
        config_path=config_path,
        summary=summary,
        step_traces_dir=step_traces_dir,
    )


def _normalize_eval_cell_ids(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["cell_id"] = out["cell_id"].map(_normalize_cell_id)
    out = out.loc[out["cell_id"].notna()].copy()
    out["cell_id"] = out["cell_id"].astype(str)
    return out


def _build_cell_artifact_map(episodes_index_path: Path) -> dict[str, Path]:
    index_df = pd.read_csv(episodes_index_path, usecols=["cell_id", "artifact_path"])
    index_df["cell_id"] = index_df["cell_id"].map(_normalize_cell_id)
    index_df = index_df.loc[index_df["cell_id"].notna()].copy()
    if index_df.empty:
        return {}
    index_df["cell_id"] = index_df["cell_id"].astype(str)
    out: dict[str, Path] = {}
    for row in index_df.itertuples(index=False):
        cell_id = str(row.cell_id)
        if cell_id not in out:
            out[cell_id] = Path(str(row.artifact_path)).expanduser().resolve()
    return out


def _load_gt_cell_xy_for_one_cell(csv_path: Path, matched_gt_cell_id: str) -> np.ndarray | None:
    x_values: list[float] = []
    y_values: list[float] = []
    for chunk in pd.read_csv(
        csv_path,
        usecols=["cell_id", "x_um", "y_um"],
        compression="infer",
        chunksize=1_000_000,
    ):
        chunk = chunk.dropna(subset=["cell_id", "x_um", "y_um"]).copy()
        chunk["cell_id"] = chunk["cell_id"].map(_normalize_cell_id)
        chunk = chunk.loc[chunk["cell_id"] == matched_gt_cell_id].copy()
        if chunk.empty:
            continue
        x_values.extend(pd.to_numeric(chunk["x_um"], errors="coerce").dropna().astype(float).tolist())
        y_values.extend(pd.to_numeric(chunk["y_um"], errors="coerce").dropna().astype(float).tolist())
    if not x_values or not y_values or len(x_values) != len(y_values):
        return None
    return np.column_stack(
        (
            np.asarray(x_values, dtype=np.float32),
            np.asarray(y_values, dtype=np.float32),
        )
    )


def _load_saved_action_trace(trace_path: Path) -> tuple[dict[str, Any], ...]:
    if not trace_path.exists():
        raise FileNotFoundError(f"saved step trace not found: {trace_path}")
    if trace_path.suffix == ".gz":
        with gzip.open(trace_path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        with trace_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"invalid step trace payload in {trace_path}")
    raw_steps = payload.get("action_trace")
    if not isinstance(raw_steps, list):
        raise ValueError(f"step trace missing action_trace list: {trace_path}")
    out: list[dict[str, Any]] = []
    for item in raw_steps:
        if not isinstance(item, dict):
            raise ValueError(f"invalid step trace entry in {trace_path}")
        out.append(dict(item))
    return tuple(out)


def _estimate_grid_step(coords: np.ndarray) -> float:
    vals = np.unique(np.asarray(coords, dtype=np.float64))
    if vals.size <= 1:
        return 2.0
    diffs = np.diff(np.sort(vals))
    diffs = diffs[np.isfinite(diffs) & (diffs > 1.0e-6)]
    if diffs.size == 0:
        return 2.0
    return float(np.median(diffs))


def build_gt_contour_grid(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Convert GT occupied bins into a padded binary grid for plotly contour display."""
    pts = np.asarray(xy, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] == 0:
        return None

    step_x = _estimate_grid_step(pts[:, 0])
    step_y = _estimate_grid_step(pts[:, 1])
    x0 = float(np.min(pts[:, 0]))
    y0 = float(np.min(pts[:, 1]))

    ix = np.rint((pts[:, 0] - x0) / step_x).astype(np.int64)
    iy = np.rint((pts[:, 1] - y0) / step_y).astype(np.int64)
    if ix.size == 0 or iy.size == 0:
        return None

    min_ix = int(ix.min())
    max_ix = int(ix.max())
    min_iy = int(iy.min())
    max_iy = int(iy.max())
    width = max_ix - min_ix + 1
    height = max_iy - min_iy + 1
    if width <= 0 or height <= 0:
        return None

    grid = np.zeros((height + 2, width + 2), dtype=np.uint8)
    gx = (ix - min_ix + 1).astype(np.int64, copy=False)
    gy = (iy - min_iy + 1).astype(np.int64, copy=False)
    grid[gy, gx] = 1

    x_coords = x0 + ((np.arange(width + 2, dtype=np.float64) + min_ix - 1.0) * step_x)
    y_coords = y0 + ((np.arange(height + 2, dtype=np.float64) + min_iy - 1.0) * step_y)
    return x_coords, y_coords, grid


def build_gt_outline_segments(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Build explicit GT boundary line segments around occupied bins.

    This is more reliable than plotly contour rendering for sparse square-bin masks.
    Returned arrays include ``nan`` separators between independent segments.
    """
    pts = np.asarray(xy, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] == 0:
        return None

    step_x = _estimate_grid_step(pts[:, 0])
    step_y = _estimate_grid_step(pts[:, 1])
    x0 = float(np.min(pts[:, 0]))
    y0 = float(np.min(pts[:, 1]))

    ix = np.rint((pts[:, 0] - x0) / step_x).astype(np.int64)
    iy = np.rint((pts[:, 1] - y0) / step_y).astype(np.int64)
    occupied = {(int(xi), int(yi)) for xi, yi in zip(ix, iy, strict=False)}
    if not occupied:
        return None

    x_segments: list[float] = []
    y_segments: list[float] = []
    half_x = 0.5 * float(step_x)
    half_y = 0.5 * float(step_y)

    def add_segment(x1: float, y1: float, x2: float, y2: float) -> None:
        x_segments.extend([x1, x2, np.nan])
        y_segments.extend([y1, y2, np.nan])

    for gx, gy in occupied:
        cx = x0 + float(gx) * float(step_x)
        cy = y0 + float(gy) * float(step_y)

        if (gx - 1, gy) not in occupied:
            add_segment(cx - half_x, cy - half_y, cx - half_x, cy + half_y)
        if (gx + 1, gy) not in occupied:
            add_segment(cx + half_x, cy - half_y, cx + half_x, cy + half_y)
        if (gx, gy - 1) not in occupied:
            add_segment(cx - half_x, cy - half_y, cx + half_x, cy - half_y)
        if (gx, gy + 1) not in occupied:
            add_segment(cx - half_x, cy + half_y, cx + half_x, cy + half_y)

    if not x_segments:
        return None
    return np.asarray(x_segments, dtype=np.float64), np.asarray(y_segments, dtype=np.float64)


def build_gt_boundary_bin_centers(xy: np.ndarray) -> np.ndarray | None:
    """Return GT boundary bin centers for high-visibility overlay markers."""
    pts = np.asarray(xy, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] == 0:
        return None

    step_x = _estimate_grid_step(pts[:, 0])
    step_y = _estimate_grid_step(pts[:, 1])
    x0 = float(np.min(pts[:, 0]))
    y0 = float(np.min(pts[:, 1]))

    ix = np.rint((pts[:, 0] - x0) / step_x).astype(np.int64)
    iy = np.rint((pts[:, 1] - y0) / step_y).astype(np.int64)
    occupied = {(int(xi), int(yi)) for xi, yi in zip(ix, iy, strict=False)}
    if not occupied:
        return None

    boundary_xy: list[tuple[float, float]] = []
    for gx, gy in occupied:
        if (
            (gx - 1, gy) not in occupied
            or (gx + 1, gy) not in occupied
            or (gx, gy - 1) not in occupied
            or (gx, gy + 1) not in occupied
        ):
            cx = x0 + float(gx) * float(step_x)
            cy = y0 + float(gy) * float(step_y)
            boundary_xy.append((cx, cy))

    if not boundary_xy:
        return None
    return np.asarray(boundary_xy, dtype=np.float64)


class PPODebugSession:
    """Lazy-loaded state for interactive PPO inspection."""

    def __init__(
        self,
        *,
        debug_summary: PPODebugSummary,
        config: PPOTrainingConfig,
        device: torch.device,
        dataset: EpisodeDataset,
        model: ActorCritic,
        per_episode_df: pd.DataFrame,
        cell_to_artifact_path: dict[str, Path],
    ) -> None:
        self.debug_summary = debug_summary
        self.config = config
        self.device = device
        self.dataset = dataset
        self.model = model
        self.per_episode_df = per_episode_df
        self._cell_row_map = {
            str(row["cell_id"]): row.to_dict()
            for _, row in per_episode_df.iterrows()
        }
        self._cell_ids = tuple(self._cell_row_map.keys())
        self._cell_to_artifact_path = dict(cell_to_artifact_path)
        self._trace_cache: dict[str, EpisodeDebugTrace] = {}
        self._gt_xy_cache: dict[str, np.ndarray | None] = {}
        self._candidate_expression_cache: dict[str, np.ndarray] = {}
        self._reference_gene_names = _load_reference_gene_names(
            config=self.config,
            n_genes=int(getattr(self.dataset, "_reference_counts").shape[1]),
        )

        gt_cell_bins_path = self.debug_summary.summary.get("gt_cell_bins_path")
        self.gt_cell_bins_path = None if not gt_cell_bins_path else Path(str(gt_cell_bins_path)).expanduser().resolve()

    @classmethod
    def from_eval_run(
        cls,
        *,
        ppo_eval_run_dir: str | Path,
        checkpoint_path: str | Path | None = None,
        device_name: str = "auto",
    ) -> "PPODebugSession":
        debug_summary = _load_debug_summary(
            ppo_eval_run_dir=Path(ppo_eval_run_dir),
            checkpoint_path=checkpoint_path,
        )
        config = load_ppo_training_config(debug_summary.config_path)
        payload = torch.load(debug_summary.checkpoint_path, map_location="cpu")
        if not isinstance(payload, dict) or "model_state_dict" not in payload:
            raise ValueError(f"invalid checkpoint payload in {debug_summary.checkpoint_path}")

        resolved_device = _resolve_device(str(device_name))
        model = ActorCritic(
            global_dim=AddStopCellEnv.GLOBAL_FEATURE_DIM,
            action_dim=AddStopCellEnv.ACTION_FEATURE_DIM,
            hidden_dim=int(config.hidden_dim),
        ).to(resolved_device)
        model.load_state_dict(payload["model_state_dict"])
        model.eval()

        rng = np.random.default_rng(int(config.seed or 0))
        dataset = EpisodeDataset(config=config, rng=rng)
        per_episode_csv = debug_summary.eval_run_dir / "per_episode.csv"
        if not per_episode_csv.exists():
            raise FileNotFoundError(f"PPO eval per_episode.csv not found: {per_episode_csv}")
        per_episode_df = _normalize_eval_cell_ids(pd.read_csv(per_episode_csv))

        cell_to_artifact_path = _build_cell_artifact_map(config.episodes_index_path)
        missing = [cell_id for cell_id in per_episode_df["cell_id"].astype(str).tolist() if cell_id not in cell_to_artifact_path]
        if missing:
            raise KeyError(
                f"{len(missing)} eval cells are missing from episodes_index.csv; first few: {missing[:5]}"
            )

        return cls(
            debug_summary=debug_summary,
            config=config,
            device=resolved_device,
            dataset=dataset,
            model=model,
            per_episode_df=per_episode_df,
            cell_to_artifact_path=cell_to_artifact_path,
        )

    @property
    def cell_ids(self) -> tuple[str, ...]:
        return self._cell_ids

    def close(self) -> None:
        self.dataset.close()

    def get_episode_metrics(self, cell_id: str) -> dict[str, Any]:
        normalized = _normalize_cell_id(cell_id)
        if normalized is None or normalized not in self._cell_row_map:
            raise KeyError(f"cell_id not found in eval run: {cell_id}")
        return dict(self._cell_row_map[normalized])

    def get_gt_cell_xy(self, matched_gt_cell_id: str | None) -> np.ndarray | None:
        normalized = _normalize_cell_id(matched_gt_cell_id)
        if normalized is None or self.gt_cell_bins_path is None:
            return None
        if normalized not in self._gt_xy_cache:
            self._gt_xy_cache[normalized] = _load_gt_cell_xy_for_one_cell(self.gt_cell_bins_path, normalized)
        xy = self._gt_xy_cache[normalized]
        return None if xy is None else np.asarray(xy, dtype=np.float32)

    def get_trace(self, cell_id: str) -> EpisodeDebugTrace:
        normalized = _normalize_cell_id(cell_id)
        if normalized is None or normalized not in self._cell_row_map:
            raise KeyError(f"cell_id not found in eval run: {cell_id}")
        if normalized not in self._trace_cache:
            self._trace_cache[normalized] = self._build_trace(normalized)
        return self._trace_cache[normalized]

    @property
    def reference_gene_names(self) -> tuple[str, ...]:
        return self._reference_gene_names

    def get_candidate_expression_matrix(self, cell_id: str) -> np.ndarray:
        normalized = _normalize_cell_id(cell_id)
        if normalized is None or normalized not in self._cell_row_map:
            raise KeyError(f"cell_id not found in eval run: {cell_id}")
        if normalized in self._candidate_expression_cache:
            return self._candidate_expression_cache[normalized]

        artifact_path = self._cell_to_artifact_path[normalized]
        locator = _parse_episode_artifact_locator(artifact_path)
        if locator.member_index is None:
            _, _, _, candidate_expression, col_index = _load_legacy_episode_artifact_payload(
                artifact_path=locator.path,
                include_candidate_bin_ids=False,
            )
        else:
            _, _, _, candidate_expression, col_index = _load_sharded_episode_artifact_payload(
                locator=locator,
                cell_id=normalized,
                include_candidate_bin_ids=False,
            )

        if candidate_expression is not None:
            expr = np.asarray(candidate_expression, dtype=np.float32)
        elif col_index is not None:
            expression_loader = getattr(self.dataset, "_expression_loader", None)
            if expression_loader is None:
                raise RuntimeError(
                    "episode artifacts store candidate_matrix_col_index, but no expression loader is available in the current debug session"
                )
            expr = np.asarray(
                expression_loader.load_columns(np.asarray(col_index, dtype=np.int64)),
                dtype=np.float32,
            )
        else:
            raise RuntimeError(f"episode artifact for cell_id={normalized} has no expression payload")

        if expr.ndim != 2:
            raise ValueError(f"candidate expression for cell_id={normalized} must have shape (B, G)")
        if expr.shape[1] != len(self._reference_gene_names):
            raise ValueError(
                f"candidate expression gene dimension mismatch for cell_id={normalized}: {expr.shape[1]} != {len(self._reference_gene_names)}"
            )
        self._candidate_expression_cache[normalized] = expr
        return expr

    def get_bin_expression_table(
        self,
        *,
        cell_id: str,
        bin_idx: int,
        top_k: int = 25,
    ) -> tuple[pd.DataFrame, dict[str, float]]:
        expr = self.get_candidate_expression_matrix(cell_id)
        if bin_idx < 0 or bin_idx >= expr.shape[0]:
            raise IndexError(f"bin_idx out of range for cell_id={cell_id}: {bin_idx}")
        values = np.asarray(expr[int(bin_idx)], dtype=np.float32)
        gene_names = np.asarray(self._reference_gene_names, dtype=object)
        order = np.argsort(-values, kind="stable")
        if top_k > 0:
            order = order[: int(top_k)]
        table = pd.DataFrame(
            {
                "gene": gene_names[order],
                "expression": values[order].astype(np.float32, copy=False),
            }
        )
        summary = {
            "sum": float(np.sum(values, dtype=np.float64)),
            "max": float(np.max(values)) if values.size > 0 else 0.0,
            "nonzero_genes": float(np.count_nonzero(values)),
            "n_genes": float(values.shape[0]),
        }
        return table, summary

    def _resolve_saved_step_trace_path(self, metrics: dict[str, Any]) -> Path | None:
        raw = metrics.get("step_trace_path")
        if raw is None or pd.isna(raw):
            return None
        rel = Path(str(raw))
        if rel.is_absolute():
            return rel if rel.exists() else None
        candidate = self.debug_summary.eval_run_dir / rel
        return candidate if candidate.exists() else None

    def _build_trace(self, cell_id: str) -> EpisodeDebugTrace:
        metrics = self.get_episode_metrics(cell_id)
        artifact_path = self._cell_to_artifact_path[cell_id]
        context = self.dataset.load_episode_context(
            cell_id=cell_id,
            artifact_path=artifact_path,
            max_steps_per_episode=self.config.max_steps_per_episode,
            include_candidate_bin_ids=True,
        )
        if context is None:
            raise RuntimeError(f"failed to load episode context for cell_id={cell_id}")

        gt_xy = self.get_gt_cell_xy(metrics.get("matched_gt_cell_id"))
        saved_step_trace_path = self._resolve_saved_step_trace_path(metrics)
        saved_action_trace = None if saved_step_trace_path is None else _load_saved_action_trace(saved_step_trace_path)
        env = AddStopCellEnv(context)
        obs, _ = env.reset()

        step_states: list[StepDebugState] = []
        total_reward = 0.0
        final_membership_mask = np.asarray(obs["membership_mask"], dtype=np.uint8).copy()
        final_assigned = int(np.sum(final_membership_mask))
        replay_source = "saved_trace" if saved_action_trace is not None else "greedy_policy"
        trace_step_idx = 0

        while True:
            forced_action = None
            if saved_action_trace is not None:
                if trace_step_idx >= len(saved_action_trace):
                    raise RuntimeError(
                        f"saved action trace ended early for cell_id={cell_id}: replay needed more than {len(saved_action_trace)} steps"
                    )
                forced_action = int(saved_action_trace[trace_step_idx]["action"])

            step_state = self._build_step_state(context=context, obs=obs, forced_action=forced_action)
            if not bool(step_state.action_mask[step_state.chosen_action]):
                raise RuntimeError(
                    f"invalid replay action for cell_id={cell_id} step={step_state.step_index}: action {step_state.chosen_action} is masked out"
                )

            next_obs, reward, terminated, truncated, info = env.step(step_state.chosen_action)
            total_reward += float(reward)
            step_state = replace(
                step_state,
                terminated_after_action=bool(terminated),
                truncated_after_action=bool(truncated),
                n_assigned_bins_after=int(info.get("n_assigned_bins", 0)),
            )
            step_states.append(step_state)
            final_membership_mask = np.asarray(next_obs["membership_mask"], dtype=np.uint8).copy()
            final_assigned = int(info.get("n_assigned_bins", 0))
            obs = next_obs
            trace_step_idx += 1
            if terminated or truncated:
                break

        total_reward_eval = float(metrics.get("total_reward", np.nan))
        n_steps_eval = int(metrics.get("n_steps", -1))
        n_assigned_eval = int(metrics.get("n_assigned_bins", -1))
        reward_matches = np.isfinite(total_reward_eval) and abs(total_reward - total_reward_eval) <= 1.0e-5
        replay_matches_eval = (
            reward_matches
            and int(len(step_states)) == n_steps_eval
            and int(final_assigned) == n_assigned_eval
        )

        return EpisodeDebugTrace(
            cell_id=cell_id,
            episode_metrics=metrics,
            candidate_bin_ids=tuple(str(x) for x in context.candidate_bin_ids),
            candidate_bin_xy_um=np.asarray(context.candidate_bin_xy_um, dtype=np.float32),
            nucleus_center_xy_um=np.asarray(context.nucleus_center_xy_um, dtype=np.float32),
            gt_cell_xy_um=None if gt_xy is None else np.asarray(gt_xy, dtype=np.float32),
            final_membership_mask=final_membership_mask,
            step_states=tuple(step_states),
            total_reward_replayed=float(total_reward),
            n_steps_replayed=int(len(step_states)),
            n_assigned_bins_replayed=int(final_assigned),
            replay_matches_eval=bool(replay_matches_eval),
            replay_source=replay_source,
            step_trace_path=saved_step_trace_path,
        )

    def _build_step_state(
        self,
        *,
        context: EpisodeContext,
        obs: dict[str, Any],
        forced_action: int | None = None,
    ) -> StepDebugState:
        membership_mask = np.asarray(obs["membership_mask"], dtype=np.uint8).copy()
        frontier_mask = compute_frontier_eligible_mask(membership_mask, context.neighbor_index)
        action_mask = np.asarray(obs["action_mask"], dtype=bool).copy()
        global_features = np.asarray(obs["global_features"], dtype=np.float32).copy()
        action_features = np.asarray(obs["action_features"], dtype=np.float32).copy()
        step_index = int(obs["step_index"])

        posterior = _posterior_from_membership_mask(ctx=context, membership_mask=membership_mask)
        neighbor_support = compute_neighbor_support_fraction(membership_mask, context.neighbor_index).astype(
            np.float32,
            copy=False,
        )
        r_expr_raw, expr_term, expr_old_raw, expr_old_term = _expression_reward_terms_per_bin(
            ctx=context,
            membership_mask=membership_mask,
            posterior=posterior,
            frontier_mask=frontier_mask,
        )
        if bool(context.normalize_expression_zscore) and np.any(frontier_mask):
            expr_frontier = r_expr_raw[frontier_mask]
            expr_frontier_mean = float(np.mean(expr_frontier))
            expr_frontier_std = float(np.std(expr_frontier, ddof=0))
        elif bool(context.normalize_expression_zscore):
            expr_frontier_mean = 0.0
            expr_frontier_std = 0.0
        else:
            expr_frontier_mean = float(np.mean(r_expr_raw[frontier_mask])) if np.any(frontier_mask) else 0.0
            expr_frontier_std = float(np.std(r_expr_raw[frontier_mask], ddof=0)) if np.any(frontier_mask) else 0.0

        expr_weighted = (float(context.w1) * expr_term).astype(np.float32, copy=False)
        expr_old_weighted = (float(context.w5) * expr_old_term).astype(np.float32, copy=False)
        distance_penalty = (float(context.w2) * context.p_dis).astype(np.float32, copy=False)
        overlap_penalty = (float(context.w3) * context.p_overlap).astype(np.float32, copy=False)
        base_penalty = context.base_penalty.astype(np.float32, copy=False)
        neighbor_bonus = (float(context.w4) * neighbor_support).astype(np.float32, copy=False)
        add_rewards = _add_rewards_from_membership_mask(
            ctx=context,
            membership_mask=membership_mask,
            posterior=posterior,
            neighbor_support=neighbor_support,
            frontier_mask=frontier_mask,
        ).astype(np.float32, copy=False)
        candidate_centroid_distance, candidate_compactness_gain = _compute_dynamic_add_action_features(
            ctx=context,
            membership_mask=membership_mask,
            frontier_mask=frontier_mask,
        )
        state_summary = _compute_state_summary_from_mask(
            ctx=context,
            membership_mask=membership_mask,
            step_index=step_index,
        )
        if action_features.shape[1] > 30:
            state_summary["seed_compactness"] = float(action_features[0, 28])
            state_summary["seed_radius_p90_scaled"] = float(action_features[0, 29])
            state_summary["seed_aspect_ratio_scaled"] = float(action_features[0, 30])
        stop_delta = float(
            compute_stop_delta(
                add_rewards,
                frontier_mask,
                stop_stat=str(context.stop_stat),
                stop_top_k=int(context.stop_top_k),
            )
        )
        stop_reward = float(-float(context.stop_lambda) * stop_delta)

        global_t, action_t, mask_t = _observation_to_tensors(obs, device=self.device)
        with torch.inference_mode():
            action_latent = self.model.encode_action_features(action_t)
            raw_logits = self.model.policy_logits_from_action_latent(action_latent).squeeze(0).detach().cpu().numpy()
            dist, value = self.model(global_t, action_t, mask_t)
            masked_action_probabilities = dist.probs.squeeze(0).detach().cpu().numpy()
        raw_logits = np.asarray(raw_logits, dtype=np.float32)
        masked_action_probabilities = np.asarray(masked_action_probabilities, dtype=np.float32)
        if forced_action is None:
            chosen_action = int(np.argmax(masked_action_probabilities))
        else:
            chosen_action = int(forced_action)
            if chosen_action < 0 or chosen_action >= masked_action_probabilities.shape[0]:
                raise ValueError(
                    f"forced_action out of range for cell_id={context.cell_id} step={step_index}: {chosen_action}"
                )
        chosen_action_probability = float(masked_action_probabilities[chosen_action])
        chosen_action_logit = float(raw_logits[chosen_action])
        chosen_reward = float(stop_reward if chosen_action == 0 else add_rewards[chosen_action - 1])
        chosen_barcode = None if chosen_action == 0 else str(context.candidate_bin_ids[chosen_action - 1])

        n_bins = int(context.n_bins)
        reward_rank = np.full((n_bins,), np.nan, dtype=np.float32)
        probability_rank = np.full((n_bins,), np.nan, dtype=np.float32)
        if np.any(frontier_mask):
            frontier_indices = np.flatnonzero(frontier_mask)
            reward_order = frontier_indices[np.argsort(-add_rewards[frontier_mask], kind="stable")]
            probability_order = frontier_indices[
                np.argsort(-masked_action_probabilities[1:][frontier_mask], kind="stable")
            ]
            reward_rank[reward_order] = np.arange(1, reward_order.shape[0] + 1, dtype=np.float32)
            probability_rank[probability_order] = np.arange(1, probability_order.shape[0] + 1, dtype=np.float32)

        bin_xy = np.asarray(context.candidate_bin_xy_um, dtype=np.float32)
        bin_table = pd.DataFrame(
            {
                "bin_idx": np.arange(n_bins, dtype=np.int32),
                "action_idx": np.arange(1, n_bins + 1, dtype=np.int32),
                "barcode": np.asarray(context.candidate_bin_ids, dtype=object),
                "x_um": bin_xy[:, 0].astype(np.float32, copy=False),
                "y_um": bin_xy[:, 1].astype(np.float32, copy=False),
                "is_assigned": membership_mask.astype(bool, copy=False),
                "is_frontier": frontier_mask.astype(bool, copy=False),
                "policy_logit": raw_logits[1:].astype(np.float32, copy=False),
                "policy_probability": masked_action_probabilities[1:].astype(np.float32, copy=False),
                "reward_total": add_rewards.astype(np.float32, copy=False),
                "expr_raw": r_expr_raw.astype(np.float32, copy=False),
                "expr_confidence": context.expression_confidence.astype(np.float32, copy=False),
                "expr_term": expr_term.astype(np.float32, copy=False),
                "expr_weighted": expr_weighted.astype(np.float32, copy=False),
                "expr_old_raw": expr_old_raw.astype(np.float32, copy=False),
                "expr_old_term": expr_old_term.astype(np.float32, copy=False),
                "expr_old_weighted": expr_old_weighted.astype(np.float32, copy=False),
                "distance_penalty": distance_penalty.astype(np.float32, copy=False),
                "overlap_penalty": overlap_penalty.astype(np.float32, copy=False),
                "base_penalty": base_penalty.astype(np.float32, copy=False),
                "neighbor_support": neighbor_support.astype(np.float32, copy=False),
                "neighbor_bonus": neighbor_bonus.astype(np.float32, copy=False),
                "candidate_to_current_centroid_distance": candidate_centroid_distance.astype(np.float32, copy=False),
                "candidate_compactness_gain": candidate_compactness_gain.astype(np.float32, copy=False),
                "reward_rank": reward_rank.astype(np.float32, copy=False),
                "probability_rank": probability_rank.astype(np.float32, copy=False),
                "is_chosen_action": np.arange(1, n_bins + 1, dtype=np.int32) == int(chosen_action),
            }
        )

        return StepDebugState(
            cell_id=str(context.cell_id),
            step_index=int(step_index),
            membership_mask=membership_mask,
            frontier_mask=frontier_mask.astype(bool, copy=False),
            action_mask=action_mask.astype(bool, copy=False),
            global_features=global_features,
            action_features=action_features,
            posterior=np.asarray(posterior, dtype=np.float32),
            value_estimate=float(value.item()),
            raw_policy_logits=raw_logits,
            masked_action_probabilities=masked_action_probabilities,
            stop_delta=float(stop_delta),
            stop_reward=float(stop_reward),
            stop_probability=float(masked_action_probabilities[0]),
            stop_logit=float(raw_logits[0]),
            expr_frontier_mean=float(expr_frontier_mean),
            expr_frontier_std=float(expr_frontier_std),
            state_summary={str(k): float(v) for k, v in state_summary.items()},
            chosen_action=int(chosen_action),
            chosen_action_probability=float(chosen_action_probability),
            chosen_action_logit=float(chosen_action_logit),
            chosen_reward=float(chosen_reward),
            chosen_barcode=chosen_barcode,
            terminated_after_action=False,
            truncated_after_action=False,
            n_assigned_bins_after=int(np.sum(membership_mask)),
            bin_table=bin_table,
        )


def _load_reference_gene_names(config: PPOTrainingConfig, n_genes: int) -> tuple[str, ...]:
    path = Path(config.reference_path)
    fmt = str(config.reference_format)
    if fmt == "npz":
        with np.load(path) as data:
            if config.reference_genes_key in data:
                arr = np.asarray(data[config.reference_genes_key]).astype(str)
                if arr.ndim == 1 and arr.shape[0] == n_genes:
                    return tuple(str(x) for x in arr.tolist())
    elif fmt in {"csv", "tsv", "parquet"}:
        table = _load_table(path, fmt)
        numeric_cols = [str(c) for c in table.columns if pd.api.types.is_numeric_dtype(table[c])]
        if len(numeric_cols) == n_genes:
            return tuple(numeric_cols)
    return tuple(f"gene_{i}" for i in range(int(n_genes)))
