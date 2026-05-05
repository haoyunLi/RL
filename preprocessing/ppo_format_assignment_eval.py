from __future__ import annotations

import datetime as dt
import h5py
import json
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hd_cell_rl.matrix_io import resolve_matrix_csc_h5_path
from hd_cell_rl.reward import compute_reference_distribution
from hd_cell_rl.reward_grid_search import (
    _MatrixOnDemandExpressionLoader,
    _build_nuclei_centers,
    _build_nuclei_spatial_index,
    _default_nuclei_center_columns,
    _load_episode_build_expression_context,
    _load_one_episode_artifact,
)
from preprocessing.ppo_eval_plots import save_overlay_plots, save_summary_plots
from preprocessing.ppo_eval_metrics import (
    build_episode_nuclear_barcode_map,
    collect_gt_nuclear_candidates,
    compute_spatial_overlap_metrics,
    load_episode_build_bins_path,
    load_gt_bins_for_cells,
    match_episode_cells_by_nuclear_overlap,
    normalize_cell_id,
)

logger = logging.getLogger(__name__)

_LOCAL_TIMEZONE = ZoneInfo("America/Chicago")
_LOCAL_TIMEZONE_NAME = "America/Chicago"


def add_ppo_format_assignment_eval_args(
    parser: Any,
    *,
    default_eval_run_name: str,
    method_label: str,
) -> None:
    """Add the shared PPO-format evaluation CLI arguments used by method runners."""
    parser.add_argument(
        "--ppo_eval_run_dir",
        type=str,
        default=None,
        help=(
            "Optional PPO evaluation run directory containing per_episode.csv and config_used.yaml. "
            f"If provided, evaluate {method_label} on the same episode cell set."
        ),
    )
    parser.add_argument(
        "--eval_run_name",
        type=str,
        default=default_eval_run_name,
        help=f"Run-name prefix for PPO-format {method_label} evaluation outputs.",
    )
    parser.add_argument(
        "--eval_output_root",
        type=str,
        default="runs",
        help=f"Root directory for PPO-format {method_label} evaluation outputs.",
    )
    parser.add_argument(
        "--overlay_max_cells",
        type=int,
        default=300,
        help="Maximum number of overlay PNGs to write for PPO-format evaluation.",
    )
    parser.add_argument(
        "--overlay_selection",
        type=str,
        choices=("first", "random", "best_iou", "worst_iou"),
        default="first",
        help="How to choose per-cell overlays for PPO-format evaluation.",
    )
    parser.add_argument(
        "--eval_seed",
        type=int,
        default=7,
        help="Random seed for PPO-format evaluation overlay selection.",
    )
    parser.add_argument(
        "--pred_min_nuclear_overlap_frac",
        type=float,
        default=0.3,
        help=f"Minimum episode nuclear-bin fraction needed to accept a {method_label} prediction match.",
    )
    parser.add_argument(
        "--pred_min_nuclear_overlap_bins",
        type=int,
        default=2,
        help=f"Minimum overlapping nuclear bins needed to accept a {method_label} prediction match.",
    )
    parser.add_argument(
        "--gt_cell_bins_path",
        type=str,
        default=None,
        help="Optional GT full-cell bin table used for PPO-format IoU/Dice evaluation.",
    )
    parser.add_argument(
        "--gt_nuclear_bins_path",
        type=str,
        default=None,
        help="Optional GT nuclear-bin table used for PPO-format GT matching.",
    )
    parser.add_argument(
        "--gt_cell_assignments_csv",
        type=str,
        default=None,
        help="Optional pseudo-data cell_id to sc_cell_barcode mapping for gene correlation.",
    )
    parser.add_argument(
        "--gt_sc_expression_h5",
        type=str,
        default=None,
        help="Optional ground-truth single-cell expression H5 for gene correlation.",
    )
    parser.add_argument(
        "--gt_min_nuclear_overlap_frac",
        type=float,
        default=0.3,
        help="Minimum episode nuclear-bin fraction needed to accept a GT match.",
    )
    parser.add_argument(
        "--gt_min_nuclear_overlap_bins",
        type=int,
        default=2,
        help="Minimum overlapping nuclear bins needed to accept a GT match.",
    )


def validate_ppo_format_assignment_eval_args(args: Any) -> None:
    """Validate shared PPO-format evaluation CLI arguments."""
    if int(args.overlay_max_cells) < 0:
        raise ValueError("--overlay_max_cells must be >= 0")
    if not (0.0 <= float(args.pred_min_nuclear_overlap_frac) <= 1.0):
        raise ValueError("--pred_min_nuclear_overlap_frac must be in [0, 1]")
    if not (0.0 <= float(args.gt_min_nuclear_overlap_frac) <= 1.0):
        raise ValueError("--gt_min_nuclear_overlap_frac must be in [0, 1]")
    if int(args.pred_min_nuclear_overlap_bins) < 0:
        raise ValueError("--pred_min_nuclear_overlap_bins must be >= 0")
    if int(args.gt_min_nuclear_overlap_bins) < 0:
        raise ValueError("--gt_min_nuclear_overlap_bins must be >= 0")


def _decode_h5_strings(values: np.ndarray) -> list[str]:
    out: list[str] = []
    for value in values:
        if isinstance(value, bytes):
            out.append(value.decode("utf-8"))
        else:
            out.append(str(value))
    return out


def _load_10x_feature_names(matrix_path: Path) -> list[str]:
    resolved_path = resolve_matrix_csc_h5_path(matrix_path)
    with h5py.File(resolved_path, "r") as h5:
        return _decode_h5_strings(h5["matrix/features/name"][:])


def _common_genes_in_spatial_order(spatial_matrix_path: Path, sc_expression_h5: Path) -> list[str]:
    spatial_genes = _load_10x_feature_names(spatial_matrix_path)
    sc_genes = set(_load_10x_feature_names(sc_expression_h5))
    common: list[str] = []
    seen: set[str] = set()
    for gene in spatial_genes:
        gene = str(gene)
        if gene in sc_genes and gene not in seen:
            common.append(gene)
            seen.add(gene)
    return common


def _load_gt_cell_assignment_map(path: Path) -> dict[str, dict[str, str]]:
    df = pd.read_csv(path)
    required = {"cell_id", "sc_cell_barcode"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"GT cell assignment CSV missing columns: {sorted(missing)}")
    df["cell_id"] = df["cell_id"].map(normalize_cell_id)
    df = df.loc[df["cell_id"].notna()].copy()
    mapping: dict[str, dict[str, str]] = {}
    for row in df.itertuples(index=False):
        cell_id = str(getattr(row, "cell_id"))
        entry = {"sc_cell_barcode": str(getattr(row, "sc_cell_barcode"))}
        if "cell_type" in df.columns:
            entry["cell_type"] = str(getattr(row, "cell_type"))
        mapping[cell_id] = entry
    return mapping


def _safe_gene_spearman(pred_expr: np.ndarray, gt_expr: np.ndarray) -> float:
    pred = np.asarray(pred_expr, dtype=np.float64)
    gt = np.asarray(gt_expr, dtype=np.float64)
    if pred.shape != gt.shape or pred.size < 2:
        return np.nan
    if float(np.sum(pred)) <= 0.0 or float(np.sum(gt)) <= 0.0:
        return np.nan
    if float(np.std(pred)) <= 0.0 or float(np.std(gt)) <= 0.0:
        return np.nan
    corr, _ = spearmanr(pred, gt)
    return float(corr) if np.isfinite(corr) else np.nan


def annotate_records_with_gene_correlation(
    *,
    records: list[Any],
    episodes_index_path: Path,
    gt_cell_assignments_csv: Path | None,
    gt_sc_expression_h5: Path | None,
) -> list[Any]:
    """Attach per-cell gene correlation against pseudo-data source single cells."""
    if gt_cell_assignments_csv is None or gt_sc_expression_h5 is None:
        return records
    if not records:
        return records
    if not gt_cell_assignments_csv.exists():
        raise FileNotFoundError(f"GT cell assignment CSV not found: {gt_cell_assignments_csv}")
    if not gt_sc_expression_h5.exists():
        raise FileNotFoundError(f"GT single-cell expression H5 not found: {gt_sc_expression_h5}")

    expression_context = _load_episode_build_expression_context(episodes_index_path)
    if expression_context is None:
        raise ValueError(f"could not resolve episode-build expression context from {episodes_index_path}")
    spatial_matrix_path = Path(str(expression_context["matrix_path"])).expanduser().resolve()

    common_genes = _common_genes_in_spatial_order(spatial_matrix_path, gt_sc_expression_h5)
    if len(common_genes) < 10:
        logger.warning("Skipping gene correlation; only %d common genes found", len(common_genes))
        return records

    gt_assignment = _load_gt_cell_assignment_map(gt_cell_assignments_csv)
    spatial_reader = _TenXColumnExpressionReader(spatial_matrix_path, common_genes)
    sc_reader = _TenXColumnExpressionReader(gt_sc_expression_h5, common_genes)
    updated: list[Any] = []
    try:
        for rec in records:
            metrics = dict(rec.metrics)
            matched_gt_cell_id = normalize_cell_id(metrics.get("matched_gt_cell_id"))
            gt_meta = gt_assignment.get(str(matched_gt_cell_id)) if matched_gt_cell_id is not None else None
            assigned_mask = np.asarray(rec.final_membership_mask, dtype=np.uint8) == 1
            pred_barcodes = [
                str(barcode)
                for i, barcode in enumerate(rec.candidate_bin_ids)
                if i < assigned_mask.shape[0] and bool(assigned_mask[i])
            ]

            gene_metrics: dict[str, Any] = {
                "gene_common_count": int(len(common_genes)),
                "gt_sc_barcode": None,
                "gt_cell_type": None,
                "gene_spearman_r": np.nan,
                "gene_rmse": np.nan,
                "pred_gene_total_counts": np.nan,
                "gt_gene_total_counts": np.nan,
                "pred_genes_detected": np.nan,
                "gt_genes_detected": np.nan,
            }

            if gt_meta is not None and pred_barcodes:
                sc_barcode = str(gt_meta["sc_cell_barcode"])
                pred_expr = spatial_reader.sum_barcodes(pred_barcodes)
                gt_expr = sc_reader.sum_barcodes([sc_barcode])
                gene_metrics.update(
                    {
                        "gt_sc_barcode": sc_barcode,
                        "gt_cell_type": gt_meta.get("cell_type"),
                        "gene_spearman_r": _safe_gene_spearman(pred_expr, gt_expr),
                        "gene_rmse": float(np.sqrt(np.mean((pred_expr - gt_expr) ** 2))),
                        "pred_gene_total_counts": float(np.sum(pred_expr)),
                        "gt_gene_total_counts": float(np.sum(gt_expr)),
                        "pred_genes_detected": int(np.count_nonzero(pred_expr > 0)),
                        "gt_genes_detected": int(np.count_nonzero(gt_expr > 0)),
                    }
                )

            updated.append(replace(rec, metrics={**metrics, **gene_metrics}))
    finally:
        spatial_reader.close()
        sc_reader.close()

    return updated


@dataclass(frozen=True)
class PredictionEvalRecord:
    metrics: dict[str, Any]
    candidate_bin_ids: tuple[str, ...]
    final_membership_mask: np.ndarray
    candidate_bin_xy_um: np.ndarray
    nucleus_center_xy_um: np.ndarray
    gt_cell_xy_um: np.ndarray | None = None
    gt_nuclear_xy_um: np.ndarray | None = None


@dataclass(frozen=True)
class EpisodeGeometry:
    cell_id: str
    candidate_bin_ids: tuple[str, ...]
    candidate_bin_xy_um: np.ndarray
    nucleus_center_xy_um: np.ndarray


class _TenXColumnExpressionReader:
    """Read selected-gene 10x CSC columns by barcode and sum them."""

    def __init__(self, matrix_path: Path, selected_gene_names: list[str]) -> None:
        resolved_path = resolve_matrix_csc_h5_path(matrix_path)
        self._h5 = h5py.File(resolved_path, "r")
        matrix = self._h5["matrix"]
        feature_names = _decode_h5_strings(matrix["features"]["name"][:])
        barcodes = _decode_h5_strings(matrix["barcodes"][:])
        shape = tuple(int(x) for x in matrix["shape"][:].tolist())
        self._n_features = int(shape[0])
        self._n_cols = int(shape[1])
        self._data = matrix["data"]
        self._indices = matrix["indices"]
        self._indptr = np.asarray(matrix["indptr"][:], dtype=np.int64)
        self._barcode_to_col = {str(barcode): i for i, barcode in enumerate(barcodes)}

        first_idx: dict[str, int] = {}
        for idx, name in enumerate(feature_names):
            first_idx.setdefault(str(name), int(idx))
        selected_indices = [first_idx[g] for g in selected_gene_names if g in first_idx]
        if len(selected_indices) != len(selected_gene_names):
            missing = sorted(set(selected_gene_names) - set(first_idx))
            raise ValueError(f"selected genes missing from matrix {resolved_path}: {missing[:5]}")

        lookup = np.full(self._n_features, -1, dtype=np.int32)
        lookup[np.asarray(selected_indices, dtype=np.int64)] = np.arange(len(selected_indices), dtype=np.int32)
        self._feature_lookup = lookup
        self.expression_dim = int(len(selected_indices))

    def close(self) -> None:
        try:
            self._h5.close()
        except Exception:
            pass

    def sum_barcodes(self, barcodes: list[str] | tuple[str, ...] | set[str]) -> np.ndarray:
        out = np.zeros((self.expression_dim,), dtype=np.float64)
        for barcode in barcodes:
            col = self._barcode_to_col.get(str(barcode))
            if col is None:
                continue
            start = int(self._indptr[col])
            end = int(self._indptr[col + 1])
            if end <= start:
                continue
            feature_idx = np.asarray(self._indices[start:end], dtype=np.int64)
            values = np.asarray(self._data[start:end], dtype=np.float64)
            selected_pos = self._feature_lookup[feature_idx]
            keep = selected_pos >= 0
            if np.any(keep):
                np.add.at(out, selected_pos[keep].astype(np.int64, copy=False), values[keep])
        return out


def _now_utc_and_local() -> tuple[dt.datetime, dt.datetime]:
    now_utc = dt.datetime.now(dt.timezone.utc)
    return now_utc, now_utc.astimezone(_LOCAL_TIMEZONE)


def coerce_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    text = series.astype("string").str.strip().str.lower()
    return text.isin({"1", "true", "t", "yes", "y"})


def _load_yaml_dict(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"expected YAML mapping in {path}")
    return raw


def _read_table(path: Path, fmt: str) -> pd.DataFrame:
    fmt_norm = str(fmt).strip().lower()
    if fmt_norm == "auto":
        suffix = "".join(path.suffixes).lower()
        if suffix.endswith(".parquet"):
            fmt_norm = "parquet"
        elif suffix.endswith(".csv.gz") or suffix.endswith(".csv"):
            fmt_norm = "csv"
        elif suffix.endswith(".tsv.gz") or suffix.endswith(".tsv"):
            fmt_norm = "tsv"
        else:
            raise ValueError(f"could not infer table format from path: {path}")
    if fmt_norm == "parquet":
        return pd.read_parquet(path)
    if fmt_norm == "csv":
        return pd.read_csv(path)
    if fmt_norm == "tsv":
        return pd.read_csv(path, sep="\t")
    raise ValueError(f"unsupported table format: {fmt}")


def _load_reference_theta(reference_cfg: dict[str, Any], *, epsilon: float) -> tuple[np.ndarray, np.ndarray]:
    ref_path = Path(str(reference_cfg["path"])).expanduser().resolve()
    ref_format = str(reference_cfg.get("format", "auto")).strip().lower()
    if ref_format == "auto":
        ref_format = "npz" if ref_path.suffix.lower() == ".npz" else ref_format
    if ref_format != "npz":
        raise ValueError(f"PPO-format assignment evaluation currently supports NPZ reference only, got: {ref_format}")
    array_key = str(reference_cfg.get("array_key", "reference_counts"))
    with np.load(ref_path, allow_pickle=False) as data:
        if array_key not in data:
            raise KeyError(f"reference array key {array_key!r} not found in {ref_path}")
        reference_counts = np.asarray(data[array_key], dtype=np.float64)
    theta = compute_reference_distribution(reference_counts=reference_counts, epsilon=float(epsilon))
    return theta, np.log(theta)


def load_eval_cell_ids(per_episode_csv: Path) -> list[str]:
    df = pd.read_csv(per_episode_csv, usecols=["cell_id"])
    ordered: list[str] = []
    seen: set[str] = set()
    for raw in df["cell_id"].tolist():
        cell_id = normalize_cell_id(raw)
        if cell_id is None or cell_id in seen:
            continue
        ordered.append(cell_id)
        seen.add(cell_id)
    return ordered


def load_episode_geometries(
    *,
    eval_config_path: Path,
    target_cell_ids: list[str],
) -> tuple[Path, dict[str, EpisodeGeometry]]:
    raw_cfg = _load_yaml_dict(eval_config_path)
    inputs = raw_cfg.get("inputs")
    reward_cfg = raw_cfg.get("reward")
    if not isinstance(inputs, dict) or not isinstance(reward_cfg, dict):
        raise ValueError(f"invalid PPO config structure in {eval_config_path}")

    episodes_index_path = Path(str(inputs["episodes_index_path"])).expanduser().resolve()
    reference_cfg = inputs.get("reference")
    nuclei_cfg = inputs.get("nuclei")
    if not isinstance(reference_cfg, dict) or not isinstance(nuclei_cfg, dict):
        raise ValueError(f"missing inputs.reference or inputs.nuclei in {eval_config_path}")

    expression_context = _load_episode_build_expression_context(episodes_index_path)
    if expression_context is None:
        raise ValueError(f"could not resolve matrix_h5 episode-build expression context from {episodes_index_path}")

    epsilon = float(reward_cfg.get("epsilon", 1.0e-8))
    theta, log_theta = _load_reference_theta(reference_cfg, epsilon=epsilon)

    nuclei_path = Path(str(nuclei_cfg["path"])).expanduser().resolve()
    nuclei_format = str(nuclei_cfg.get("format", "auto"))
    nuclei_columns = _default_nuclei_center_columns(dict(nuclei_cfg.get("columns", {})))
    nuclei_df = _read_table(nuclei_path, nuclei_format)
    nuclei_centers = _build_nuclei_centers(nuclei_df, nuclei_columns)
    nuclei_spatial_index = _build_nuclei_spatial_index(nuclei_centers)

    cache_size = inputs.get("expression_cache_size", None)
    if cache_size is None:
        cache_size = expression_context.get("cache_size", 20000)

    episodes_df = pd.read_csv(episodes_index_path, usecols=["cell_id", "artifact_path"])
    episodes_df["cell_id"] = episodes_df["cell_id"].map(normalize_cell_id)
    episodes_df = episodes_df.loc[episodes_df["cell_id"].notna()].copy()
    artifact_by_cell = {
        str(cell_id): str(artifact_path)
        for cell_id, artifact_path in zip(episodes_df["cell_id"].tolist(), episodes_df["artifact_path"].tolist())
    }

    geometries: dict[str, EpisodeGeometry] = {}
    expression_loader = _MatrixOnDemandExpressionLoader(
        matrix_path=Path(str(expression_context["matrix_path"])).expanduser().resolve(),
        reference_npz_path=Path(str(reference_cfg["path"])).expanduser().resolve(),
        reference_genes_key=str(reference_cfg.get("genes_key", "genes")),
        cache_size=int(cache_size),
    )
    try:
        for cell_id in target_cell_ids:
            artifact_path = artifact_by_cell.get(cell_id)
            if artifact_path is None:
                logger.warning("Cell %s not found in episodes_index.csv; skipping in assignment evaluation", cell_id)
                continue
            prepared = _load_one_episode_artifact(
                artifact_path=artifact_path,
                cell_id=cell_id,
                expression_loader=expression_loader,
                theta=theta,
                log_theta=log_theta,
                nuclei_spatial_index=nuclei_spatial_index,
                include_candidate_bin_ids=True,
            )
            if prepared is None:
                logger.warning("Cell %s episode artifact yielded no candidate bins; skipping", cell_id)
                continue
            geometries[cell_id] = EpisodeGeometry(
                cell_id=str(cell_id),
                candidate_bin_ids=tuple(str(x) for x in prepared.candidate_bin_ids),
                candidate_bin_xy_um=np.asarray(prepared.candidate_bin_xy_um, dtype=np.float32),
                nucleus_center_xy_um=np.asarray(prepared.nucleus_center_xy_um, dtype=np.float32),
            )
    finally:
        expression_loader.close()

    return episodes_index_path, geometries


def collect_prediction_nuclear_candidates(
    *,
    assignments_csv: Path,
    episode_nuclear_barcodes: set[str],
) -> dict[str, set[str]]:
    barcode_to_pred_cells: dict[str, set[str]] = defaultdict(set)
    usecols = ["cell_id", "barcode", "is_nuclear"]
    for chunk in pd.read_csv(assignments_csv, usecols=usecols, chunksize=1_000_000):
        chunk = chunk.dropna(subset=["cell_id", "barcode"]).copy()
        chunk["cell_id"] = chunk["cell_id"].map(normalize_cell_id)
        chunk = chunk.loc[chunk["cell_id"].notna()].copy()
        if chunk.empty:
            continue
        chunk["barcode"] = chunk["barcode"].astype(str)
        chunk["is_nuclear"] = coerce_bool_series(chunk["is_nuclear"]).fillna(False)
        chunk = chunk.loc[chunk["is_nuclear"]].copy()
        if chunk.empty:
            continue
        overlap = chunk.loc[chunk["barcode"].isin(episode_nuclear_barcodes), ["barcode", "cell_id"]]
        for barcode, group in overlap.groupby("barcode", sort=False):
            barcode_to_pred_cells[str(barcode)].update(group["cell_id"].astype(str).tolist())
    return dict(barcode_to_pred_cells)


def match_episode_cells_to_predictions(
    *,
    episode_nuclear_by_cell: dict[str, set[str]],
    barcode_to_pred_cells: dict[str, set[str]],
    min_overlap_frac: float,
    min_overlap_bins: int,
    pred_match_method: str,
) -> tuple[dict[str, dict[str, Any]], set[str]]:
    raw, matched_ids = match_episode_cells_by_nuclear_overlap(
        episode_nuclear_by_cell=episode_nuclear_by_cell,
        barcode_to_target_cells=barcode_to_pred_cells,
        min_overlap_frac=min_overlap_frac,
        min_overlap_bins=min_overlap_bins,
        match_method=pred_match_method,
    )
    converted = {
        cell_id: {
            "matched_pred_cell_id": meta.get("matched_cell_id"),
            "pred_match_method": meta.get("match_method", "unmatched"),
            "pred_nuclear_overlap_bins": int(meta.get("nuclear_overlap_bins", 0)),
            "pred_nuclear_overlap_frac_episode": float(meta.get("nuclear_overlap_frac_episode", np.nan)),
        }
        for cell_id, meta in raw.items()
    }
    return converted, matched_ids


def load_assignment_barcodes_for_cells(assignments_csv: Path, matched_cell_ids: set[str]) -> dict[str, set[str]]:
    if not matched_cell_ids:
        return {}
    out: dict[str, set[str]] = defaultdict(set)
    usecols = ["cell_id", "barcode"]
    for chunk in pd.read_csv(assignments_csv, usecols=usecols, chunksize=1_000_000):
        chunk = chunk.dropna(subset=["cell_id", "barcode"]).copy()
        chunk["cell_id"] = chunk["cell_id"].map(normalize_cell_id)
        chunk = chunk.loc[chunk["cell_id"].isin(matched_cell_ids)].copy()
        if chunk.empty:
            continue
        chunk["barcode"] = chunk["barcode"].astype(str)
        for pred_cell_id, group in chunk.groupby("cell_id", sort=False):
            out[str(pred_cell_id)].update(group["barcode"].astype(str).tolist())
    return dict(out)


def annotate_records_with_ground_truth(
    *,
    records: list[PredictionEvalRecord],
    episodes_index_path: Path,
    gt_cell_bins_path: Path,
    gt_nuclear_bins_path: Path,
    min_overlap_frac: float,
    min_overlap_bins: int,
) -> list[PredictionEvalRecord]:
    if not records:
        return records

    bins_path = load_episode_build_bins_path(episodes_index_path)
    if bins_path is None:
        raise FileNotFoundError(
            f"could not resolve episode-build bins_path from {episodes_index_path.parent / 'config' / 'config_resolved.yaml'}"
        )

    episode_cell_ids = {str(rec.metrics["cell_id"]) for rec in records}
    episode_nuclear_by_cell = build_episode_nuclear_barcode_map(bins_path=bins_path, target_cell_ids=episode_cell_ids)
    episode_nuclear_union = set().union(*episode_nuclear_by_cell.values()) if episode_nuclear_by_cell else set()
    barcode_to_gt_cells = collect_gt_nuclear_candidates(
        gt_nuclear_bins_path=gt_nuclear_bins_path,
        episode_nuclear_barcodes=episode_nuclear_union,
    )
    gt_meta_by_cell, matched_gt_ids = match_episode_cells_to_predictions(
        episode_nuclear_by_cell=episode_nuclear_by_cell,
        barcode_to_pred_cells=barcode_to_gt_cells,
        min_overlap_frac=min_overlap_frac,
        min_overlap_bins=min_overlap_bins,
        pred_match_method="nuclear_overlap",
    )
    gt_barcodes_by_cell, gt_xy_by_cell = load_gt_bins_for_cells(csv_path=gt_cell_bins_path, matched_cell_ids=matched_gt_ids)
    gt_nuclear_barcodes_by_cell, gt_nuclear_xy_by_cell = load_gt_bins_for_cells(
        csv_path=gt_nuclear_bins_path,
        matched_cell_ids=matched_gt_ids,
    )

    updated: list[PredictionEvalRecord] = []
    for rec in records:
        cell_id = str(rec.metrics["cell_id"])
        meta = gt_meta_by_cell.get(cell_id, {})
        matched_gt_cell_id = meta.get("matched_pred_cell_id")
        assigned_barcodes = {
            str(barcode)
            for barcode, chosen in zip(rec.candidate_bin_ids, np.asarray(rec.final_membership_mask, dtype=np.uint8))
            if int(chosen) == 1
        }
        gt_barcodes = gt_barcodes_by_cell.get(str(matched_gt_cell_id), set()) if matched_gt_cell_id is not None else set()
        overlap = compute_spatial_overlap_metrics(assigned_barcodes, gt_barcodes) if matched_gt_cell_id is not None else None
        gt_cell_xy = gt_xy_by_cell.get(str(matched_gt_cell_id)) if matched_gt_cell_id is not None else None
        gt_nuclear_xy = gt_nuclear_xy_by_cell.get(str(matched_gt_cell_id)) if matched_gt_cell_id is not None else None
        updated.append(
            replace(
                rec,
                metrics={
                    **rec.metrics,
                    "matched_gt_cell_id": matched_gt_cell_id,
                    "match_method": meta.get("pred_match_method", "unmatched"),
                    "gt_nuclear_overlap_bins": int(meta.get("pred_nuclear_overlap_bins", 0)),
                    "gt_nuclear_overlap_frac_episode": float(meta.get("pred_nuclear_overlap_frac_episode", np.nan)),
                    "pred_iou": np.nan if overlap is None else overlap["iou"],
                    "pred_dice": np.nan if overlap is None else overlap["dice"],
                    "pred_precision": np.nan if overlap is None else overlap["precision"],
                    "pred_recall": np.nan if overlap is None else overlap["recall"],
                    "pred_f1": np.nan if overlap is None else overlap["f1"],
                    "gt_assigned_intersection": 0 if overlap is None else int(overlap["intersection"]),
                    "gt_assigned_union": 0 if overlap is None else int(overlap["union"]),
                    "pred_n_bins": int(len(assigned_barcodes)) if overlap is None else int(overlap["pred_n_bins"]),
                    "gt_n_bins": int(len(gt_barcodes)) if overlap is None else int(overlap["gt_n_bins"]),
                },
                gt_cell_xy_um=None if gt_cell_xy is None else np.asarray(gt_cell_xy, dtype=np.float32),
                gt_nuclear_xy_um=None if gt_nuclear_xy is None else np.asarray(gt_nuclear_xy, dtype=np.float32),
            )
        )
    return updated


def _add_numeric_summary(summary: dict[str, Any], df: pd.DataFrame, column: str, prefix: str = "") -> None:
    if column not in df.columns:
        return
    values = pd.to_numeric(df[column], errors="coerce")
    values = values[np.isfinite(values.to_numpy(dtype=np.float64))]
    if len(values) == 0:
        return
    key = f"{prefix}{column}" if prefix else column
    summary[f"mean_{key}"] = float(values.mean())
    summary[f"median_{key}"] = float(values.median())
    summary[f"n_valid_{key}"] = int(len(values))


def run_ppo_format_assignment_evaluation(
    *,
    assignments_csv: Path,
    method_name: str,
    method_label: str,
    nuclear_source: str,
    external_nuclear_bins_path: Path | None,
    args: Any,
    pipeline_config: dict[str, Any] | None = None,
) -> Path:
    eval_run_dir = Path(str(args.ppo_eval_run_dir)).expanduser().resolve()
    if not eval_run_dir.exists():
        raise FileNotFoundError(f"PPO eval run dir not found: {eval_run_dir}")

    per_episode_csv = eval_run_dir / "per_episode.csv"
    eval_config_path = eval_run_dir / "config_used.yaml"
    if not per_episode_csv.exists():
        raise FileNotFoundError(f"PPO eval per_episode.csv not found: {per_episode_csv}")
    if not eval_config_path.exists():
        raise FileNotFoundError(f"PPO eval config_used.yaml not found: {eval_config_path}")

    gt_cell_bins_path = None if getattr(args, "gt_cell_bins_path", None) is None else Path(str(args.gt_cell_bins_path)).expanduser().resolve()
    gt_nuclear_bins_path = None if getattr(args, "gt_nuclear_bins_path", None) is None else Path(str(args.gt_nuclear_bins_path)).expanduser().resolve()
    if (gt_cell_bins_path is None) ^ (gt_nuclear_bins_path is None):
        raise ValueError("gt_cell_bins_path and gt_nuclear_bins_path must be provided together")

    target_cell_ids = load_eval_cell_ids(per_episode_csv)
    if not target_cell_ids:
        raise RuntimeError(f"no cell_id values found in {per_episode_csv}")

    episodes_index_path, geometries = load_episode_geometries(
        eval_config_path=eval_config_path,
        target_cell_ids=target_cell_ids,
    )
    target_cell_ids = [cell_id for cell_id in target_cell_ids if cell_id in geometries]
    if not target_cell_ids:
        raise RuntimeError("no overlapping episode cells were loaded for PPO-format assignment evaluation")

    bins_path = load_episode_build_bins_path(episodes_index_path)
    if bins_path is None:
        raise FileNotFoundError(
            f"could not resolve episode-build bins_path from {episodes_index_path.parent / 'config' / 'config_resolved.yaml'}"
        )
    episode_nuclear_by_cell = build_episode_nuclear_barcode_map(
        bins_path=bins_path,
        target_cell_ids=set(target_cell_ids),
    )
    episode_nuclear_union = set().union(*episode_nuclear_by_cell.values()) if episode_nuclear_by_cell else set()
    barcode_to_pred_cells = collect_prediction_nuclear_candidates(
        assignments_csv=assignments_csv,
        episode_nuclear_barcodes=episode_nuclear_union,
    )
    pred_meta_by_cell, matched_pred_ids = match_episode_cells_to_predictions(
        episode_nuclear_by_cell=episode_nuclear_by_cell,
        barcode_to_pred_cells=barcode_to_pred_cells,
        min_overlap_frac=float(args.pred_min_nuclear_overlap_frac),
        min_overlap_bins=int(args.pred_min_nuclear_overlap_bins),
        pred_match_method=f"{method_name}_nuclear_overlap",
    )
    pred_barcodes_by_cell = load_assignment_barcodes_for_cells(assignments_csv, matched_pred_ids)

    records: list[PredictionEvalRecord] = []
    for cell_id in target_cell_ids:
        geom = geometries[cell_id]
        meta = pred_meta_by_cell.get(cell_id, {})
        matched_pred_cell_id = meta.get("matched_pred_cell_id")
        pred_bars = pred_barcodes_by_cell.get(str(matched_pred_cell_id), set()) if matched_pred_cell_id is not None else set()
        membership_mask = np.asarray([str(bin_id) in pred_bars for bin_id in geom.candidate_bin_ids], dtype=np.uint8)
        records.append(
            PredictionEvalRecord(
                metrics={
                    "cell_id": str(cell_id),
                    "matched_pred_cell_id": matched_pred_cell_id,
                    "pred_match_method": meta.get("pred_match_method", "unmatched"),
                    "pred_nuclear_overlap_bins": int(meta.get("pred_nuclear_overlap_bins", 0)),
                    "pred_nuclear_overlap_frac_episode": float(meta.get("pred_nuclear_overlap_frac_episode", np.nan)),
                    "n_assigned_bins": int(membership_mask.sum()),
                    "n_candidate_bins": int(len(geom.candidate_bin_ids)),
                    "terminated": True,
                    "truncated": False,
                },
                candidate_bin_ids=geom.candidate_bin_ids,
                final_membership_mask=membership_mask,
                candidate_bin_xy_um=np.asarray(geom.candidate_bin_xy_um, dtype=np.float32),
                nucleus_center_xy_um=np.asarray(geom.nucleus_center_xy_um, dtype=np.float32),
            )
        )

    gt_enabled = gt_cell_bins_path is not None and gt_nuclear_bins_path is not None
    if gt_enabled:
        if not gt_cell_bins_path.exists():
            raise FileNotFoundError(f"GT cell bins file not found: {gt_cell_bins_path}")
        if not gt_nuclear_bins_path.exists():
            raise FileNotFoundError(f"GT nuclear bins file not found: {gt_nuclear_bins_path}")
        records = annotate_records_with_ground_truth(
            records=records,
            episodes_index_path=episodes_index_path,
            gt_cell_bins_path=gt_cell_bins_path,
            gt_nuclear_bins_path=gt_nuclear_bins_path,
            min_overlap_frac=float(args.gt_min_nuclear_overlap_frac),
            min_overlap_bins=int(args.gt_min_nuclear_overlap_bins),
        )

    gt_cell_assignments_csv = (
        None
        if getattr(args, "gt_cell_assignments_csv", None) is None
        else Path(str(args.gt_cell_assignments_csv)).expanduser().resolve()
    )
    gt_sc_expression_h5 = (
        None
        if getattr(args, "gt_sc_expression_h5", None) is None
        else Path(str(args.gt_sc_expression_h5)).expanduser().resolve()
    )
    if (gt_cell_assignments_csv is None) ^ (gt_sc_expression_h5 is None):
        raise ValueError("gt_cell_assignments_csv and gt_sc_expression_h5 must be provided together")
    if gt_cell_assignments_csv is not None and gt_sc_expression_h5 is not None:
        records = annotate_records_with_gene_correlation(
            records=records,
            episodes_index_path=episodes_index_path,
            gt_cell_assignments_csv=gt_cell_assignments_csv,
            gt_sc_expression_h5=gt_sc_expression_h5,
        )

    out_root = Path(str(args.eval_output_root)).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    now_utc, now_local = _now_utc_and_local()
    ts = now_utc.strftime("%Y%m%dT%H%M%SZ")
    run_dir = out_root / f"{args.eval_run_name}_{ts}"
    run_dir.mkdir(parents=False, exist_ok=False)

    df = pd.DataFrame([rec.metrics for rec in records])
    df.to_csv(run_dir / "per_episode.csv", index=False)
    summary_plots = save_summary_plots(df=df, run_dir=run_dir, method_label=method_label)
    overlay_plots = save_overlay_plots(
        records=records,
        df=df,
        run_dir=run_dir,
        max_cells=int(args.overlay_max_cells),
        selection=str(args.overlay_selection),
        seed=int(args.eval_seed),
        method_label=method_label,
    )

    assignments_summary_key = f"{method_name}_assignments_path"
    summary = {
        "method": method_name,
        "nuclear_source": nuclear_source,
        "external_nuclear_bins_path": None if external_nuclear_bins_path is None else str(external_nuclear_bins_path),
        assignments_summary_key: str(assignments_csv),
        "source_ppo_eval_run_dir": str(eval_run_dir),
        "source_ppo_eval_csv": str(per_episode_csv),
        "episodes_index_path": str(episodes_index_path),
        "gt_enabled": bool(gt_enabled),
        "gt_cell_bins_path": None if gt_cell_bins_path is None else str(gt_cell_bins_path),
        "gt_nuclear_bins_path": None if gt_nuclear_bins_path is None else str(gt_nuclear_bins_path),
        "gt_cell_assignments_csv": None if gt_cell_assignments_csv is None else str(gt_cell_assignments_csv),
        "gt_sc_expression_h5": None if gt_sc_expression_h5 is None else str(gt_sc_expression_h5),
        "gt_match_mode": "nuclear_overlap",
        "pred_match_mode": f"{method_name}_nuclear_overlap",
        "evaluation_timestamp_utc": now_utc.isoformat(),
        "evaluation_timestamp_local": now_local.isoformat(),
        "local_timezone": _LOCAL_TIMEZONE_NAME,
        "n_episodes_evaluated": int(len(df)),
        "mean_n_assigned_bins": float(df["n_assigned_bins"].mean()),
        "median_n_assigned_bins": float(df["n_assigned_bins"].median()),
        "terminated_fraction": float(df["terminated"].mean()),
        "truncated_fraction": float(df["truncated"].mean()),
        "n_summary_plots": int(len(summary_plots)),
        "n_overlay_plots": int(len(overlay_plots)),
    }
    matched_pred = None
    matched_gt = None
    if "matched_pred_cell_id" in df.columns:
        matched_pred = df["matched_pred_cell_id"].astype("string").notna()
        summary["matched_pred_fraction"] = float(matched_pred.mean())
        summary["n_matched_pred"] = int(matched_pred.sum())
    if "matched_gt_cell_id" in df.columns:
        matched_gt = df["matched_gt_cell_id"].astype("string").notna()
        summary["matched_gt_fraction"] = float(matched_gt.mean())
        summary["n_matched_gt"] = int(matched_gt.sum())
    if matched_pred is not None and matched_gt is not None:
        n_gt_matched = int(matched_gt.sum())
        if n_gt_matched > 0:
            summary["matched_pred_fraction_among_gt_matched"] = float((matched_pred & matched_gt).sum() / n_gt_matched)
    for metric_col in ("pred_iou", "pred_dice", "pred_precision", "pred_recall", "pred_f1", "gene_spearman_r", "gene_rmse"):
        _add_numeric_summary(summary, df, metric_col)
    if "pred_iou" in df.columns:
        iou_series = pd.to_numeric(df["pred_iou"], errors="coerce")
        if matched_pred is not None:
            matched_pred_iou = iou_series.loc[matched_pred]
            matched_pred_iou = matched_pred_iou[np.isfinite(matched_pred_iou.to_numpy(dtype=np.float64))]
            if len(matched_pred_iou) > 0:
                summary["n_matched_pred_with_valid_iou"] = int(len(matched_pred_iou))
                summary["matched_pred_only_mean_pred_iou"] = float(matched_pred_iou.mean())
                summary["matched_pred_only_median_pred_iou"] = float(matched_pred_iou.median())
    if "pred_dice" in df.columns:
        dice_series = pd.to_numeric(df["pred_dice"], errors="coerce")
        if matched_pred is not None:
            matched_pred_dice = dice_series.loc[matched_pred]
            matched_pred_dice = matched_pred_dice[np.isfinite(matched_pred_dice.to_numpy(dtype=np.float64))]
            if len(matched_pred_dice) > 0:
                summary["n_matched_pred_with_valid_dice"] = int(len(matched_pred_dice))
                summary["matched_pred_only_mean_pred_dice"] = float(matched_pred_dice.mean())
                summary["matched_pred_only_median_pred_dice"] = float(matched_pred_dice.median())
    if "pred_match_method" in df.columns:
        summary["pred_match_method_counts"] = {
            str(k): int(v) for k, v in df["pred_match_method"].fillna("unmatched").value_counts(dropna=False).items()
        }
    if "match_method" in df.columns:
        summary["match_method_counts"] = {
            str(k): int(v) for k, v in df["match_method"].fillna("unmatched").value_counts(dropna=False).items()
        }

    with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=False)
        handle.write("\n")

    pipeline_key = f"{method_name}_pipeline"
    config_payload = {
        pipeline_key: dict(pipeline_config or {}),
        "ppo_format_eval": {
            "ppo_eval_run_dir": str(eval_run_dir),
            "overlay_max_cells": int(args.overlay_max_cells),
            "overlay_selection": str(args.overlay_selection),
            "eval_seed": int(args.eval_seed),
            "pred_min_nuclear_overlap_frac": float(args.pred_min_nuclear_overlap_frac),
            "pred_min_nuclear_overlap_bins": int(args.pred_min_nuclear_overlap_bins),
            "gt_cell_bins_path": None if gt_cell_bins_path is None else str(gt_cell_bins_path),
            "gt_nuclear_bins_path": None if gt_nuclear_bins_path is None else str(gt_nuclear_bins_path),
            "gt_cell_assignments_csv": None if gt_cell_assignments_csv is None else str(gt_cell_assignments_csv),
            "gt_sc_expression_h5": None if gt_sc_expression_h5 is None else str(gt_sc_expression_h5),
            "gt_min_nuclear_overlap_frac": float(args.gt_min_nuclear_overlap_frac),
            "gt_min_nuclear_overlap_bins": int(args.gt_min_nuclear_overlap_bins),
        },
    }
    with (run_dir / "config_used.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config_payload, handle, sort_keys=False)

    logger.info("PPO-format %s evaluation complete: %s", method_name, run_dir)
    logger.info("PPO-format %s summary: %s", method_name, summary)
    return run_dir
