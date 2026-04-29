#!/usr/bin/env python
"""Run STCS with PPO-aligned nuclear seeds and PPO-format evaluation."""

from __future__ import annotations

import argparse
import gc
import datetime as dt
import json
import logging
import sys
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import scanpy as sc
import yaml
from scipy import sparse

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from preprocessing.ppo_format_assignment_eval import (
    coerce_bool_series,
    load_eval_cell_ids,
    normalize_cell_id,
    run_ppo_format_assignment_evaluation,
)

logger = logging.getLogger(__name__)
_LOCAL_TIMEZONE = ZoneInfo("America/Chicago")
_CANDIDATE_SHARD_CACHE: dict[str, dict[str, np.ndarray]] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run STCS on pseudo Visium HD using PPO-aligned nuclei.")
    parser.add_argument("--pseudo_hd_dir", required=True, help="Pseudo Visium HD directory.")
    parser.add_argument("--output_dir", required=True, help="Output directory for STCS results.")
    parser.add_argument("--dataset_name", default="human_colorectal", help="Dataset prefix.")
    parser.add_argument("--stcs_repo_dir", default="external/STCS", help="Path to cloned YangLabRutgers/STCS repo.")
    parser.add_argument("--full_res_image_path", default=None, help="Optional image path stored in adata.uns['spatial'].")
    parser.add_argument("--external_nuclear_bins_path", required=True, help="PPO-aligned nuclear-bin CSV/CSV.GZ.")
    parser.add_argument("--search_radius", type=int, default=5, help="STCS candidate search radius S.")
    parser.add_argument("--lambda_spatial", type=float, default=0.5, help="STCS spatial weight lambda L.")
    parser.add_argument("--target_sum", type=float, default=1.0e4, help="Scanpy normalize_total target sum.")
    parser.add_argument("--n_top_genes", type=int, default=5000, help="Highly-variable genes for pseudobulk PCA.")
    parser.add_argument("--min_cells", type=int, default=3, help="Minimum cells for gene filtering.")
    parser.add_argument("--pseudobulk_mode", choices=("mean", "sum"), default="mean", help="STCS pseudobulk aggregation mode.")
    parser.add_argument("--normalize_distances", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force_nuclear_assignments", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--restrict_to_eval_cells", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--eval_context_radius_bins", type=int, default=None)
    parser.add_argument("--ppo_eval_run_dir", default=None, help="PPO eval run dir for same-cell evaluation.")
    parser.add_argument("--eval_run_name", default="human_colorectal_stcs_eval", help="Evaluation run-name prefix.")
    parser.add_argument("--eval_output_root", default="runs", help="Evaluation output root.")
    parser.add_argument("--overlay_max_cells", type=int, default=300)
    parser.add_argument("--overlay_selection", choices=("first", "random", "best_iou", "worst_iou"), default="first")
    parser.add_argument("--eval_seed", type=int, default=7)
    parser.add_argument("--pred_min_nuclear_overlap_frac", type=float, default=0.3)
    parser.add_argument("--pred_min_nuclear_overlap_bins", type=int, default=2)
    parser.add_argument("--gt_cell_bins_path", default=None)
    parser.add_argument("--gt_nuclear_bins_path", default=None)
    parser.add_argument("--gt_cell_assignments_csv", default=None)
    parser.add_argument("--gt_sc_expression_h5", default=None)
    parser.add_argument("--gt_min_nuclear_overlap_frac", type=float, default=0.3)
    parser.add_argument("--gt_min_nuclear_overlap_bins", type=int, default=2)
    return parser.parse_args()


def _configure_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(output_dir / "stcs_analysis.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def _load_pseudo_visium_mtx(pseudo_hd_dir: Path, full_res_image_path: Path | None) -> Any:
    mtx_dir = pseudo_hd_dir / "filtered_feature_bc_matrix"
    spatial_dir = pseudo_hd_dir / "spatial"
    positions_path = spatial_dir / "tissue_positions.parquet"
    scalefactors_path = spatial_dir / "scalefactors_json.json"
    if not mtx_dir.exists():
        raise FileNotFoundError(f"filtered_feature_bc_matrix directory not found: {mtx_dir}")
    if not positions_path.exists():
        raise FileNotFoundError(f"tissue_positions.parquet not found: {positions_path}")

    logger.info("Loading expression matrix from %s", mtx_dir)
    adata = sc.read_10x_mtx(str(mtx_dir), var_names="gene_symbols", cache=False)
    adata.var_names_make_unique()

    positions = pd.read_parquet(positions_path)
    required = {"barcode", "in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"}
    missing = required.difference(positions.columns)
    if missing:
        raise ValueError(f"tissue_positions.parquet missing columns: {sorted(missing)}")

    common = adata.obs_names.intersection(pd.Index(positions["barcode"].astype(str)))
    if common.empty:
        raise RuntimeError("no overlapping barcodes between matrix and tissue_positions")
    adata = adata[common].copy()
    positions = positions.set_index("barcode").loc[common]

    adata.obs["in_tissue"] = positions["in_tissue"].to_numpy()
    adata.obs["array_row"] = positions["array_row"].to_numpy(dtype=np.int64)
    adata.obs["array_col"] = positions["array_col"].to_numpy(dtype=np.int64)
    adata.obs["pxl_row_in_fullres"] = positions["pxl_row_in_fullres"].to_numpy(dtype=np.float64)
    adata.obs["pxl_col_in_fullres"] = positions["pxl_col_in_fullres"].to_numpy(dtype=np.float64)
    # STCS downstream assignment uses array_row/array_col for candidate search.
    # Keep obsm['spatial'] in full-resolution pixel coordinates for compatibility.
    adata.obsm["spatial"] = positions[["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy(dtype=np.float64)

    scalefactors: dict[str, Any] = {}
    if scalefactors_path.exists():
        with scalefactors_path.open("r", encoding="utf-8") as handle:
            scalefactors = json.load(handle)
    adata.uns["spatial"] = {
        "VisiumHD": {
            "scalefactors": scalefactors,
            "metadata": {
                "source_image_path": None if full_res_image_path is None else str(full_res_image_path),
            },
        }
    }
    logger.info("Loaded pseudo Visium HD: %d bins, %d genes", adata.n_obs, adata.n_vars)
    return adata


def _load_external_nuclear_labels(external_nuclear_bins_path: Path) -> tuple[pd.DataFrame, dict[int, str]]:
    header = pd.read_csv(external_nuclear_bins_path, nrows=0, compression="infer")
    usecols = ["barcode", "cell_id", "is_nuclear"]
    for col in ("array_row", "array_col"):
        if col in header.columns:
            usecols.append(col)
    df = pd.read_csv(external_nuclear_bins_path, usecols=usecols, compression="infer")
    df["barcode"] = df["barcode"].astype(str)
    df["cell_id"] = df["cell_id"].map(normalize_cell_id)
    df["is_nuclear"] = coerce_bool_series(df["is_nuclear"]).fillna(False)
    df = df.loc[df["is_nuclear"] & df["cell_id"].notna(), ["barcode", "cell_id"]].drop_duplicates()
    if df.empty:
        raise RuntimeError(f"no nuclear rows loaded from {external_nuclear_bins_path}")

    cell_ids = sorted(df["cell_id"].astype(str).unique(), key=lambda x: (not x.isdigit(), x))
    if all(str(cid).isdigit() and int(cid) > 0 for cid in cell_ids):
        cell_to_label = {cid: int(cid) for cid in cell_ids}
    else:
        cell_to_label = {cid: i + 1 for i, cid in enumerate(cell_ids)}
    label_to_cell = {label: cell for cell, label in cell_to_label.items()}
    df["label_id"] = df["cell_id"].map(cell_to_label).astype(np.int64)
    for col in ("array_row", "array_col"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    logger.info("Loaded %d nuclear bin rows across %d cells", len(df), len(label_to_cell))
    return df, label_to_cell


def _load_yaml_dict(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"expected YAML mapping in {path}")
    return raw


def _load_candidate_ids_from_artifact(artifact_path: str | Path) -> tuple[str, ...]:
    raw = str(artifact_path)
    if "::" not in raw:
        path = Path(raw).expanduser().resolve()
        with np.load(path, allow_pickle=False) as data:
            if "candidate_bin_ids" not in data:
                raise KeyError(f"candidate_bin_ids not found in {path}")
            return tuple(str(x) for x in np.asarray(data["candidate_bin_ids"]).tolist())

    shard_path_raw, member_raw = raw.rsplit("::", 1)
    shard_path = Path(shard_path_raw).expanduser().resolve()
    member_index = int(member_raw)
    key = str(shard_path)
    shard = _CANDIDATE_SHARD_CACHE.get(key)
    if shard is None:
        shard = {
            "candidate_row_splits": np.load(shard_path / "candidate_row_splits.npy", mmap_mode="r", allow_pickle=False),
            "candidate_bin_ids": np.load(shard_path / "candidate_bin_ids.npy", mmap_mode="r", allow_pickle=False),
        }
        _CANDIDATE_SHARD_CACHE[key] = shard
    row_splits = np.asarray(shard["candidate_row_splits"], dtype=np.int64)
    start = int(row_splits[member_index])
    end = int(row_splits[member_index + 1])
    return tuple(str(x) for x in np.asarray(shard["candidate_bin_ids"][start:end]).tolist())


def _load_eval_candidate_barcodes(args: argparse.Namespace) -> tuple[list[str], set[str]]:
    if args.ppo_eval_run_dir is None:
        raise ValueError("--restrict_to_eval_cells requires --ppo_eval_run_dir")
    eval_run_dir = Path(str(args.ppo_eval_run_dir)).expanduser().resolve()
    per_episode_csv = eval_run_dir / "per_episode.csv"
    eval_config_path = eval_run_dir / "config_used.yaml"
    if not per_episode_csv.exists():
        raise FileNotFoundError(f"PPO eval per_episode.csv not found: {per_episode_csv}")
    if not eval_config_path.exists():
        raise FileNotFoundError(f"PPO eval config_used.yaml not found: {eval_config_path}")

    target_cell_ids = load_eval_cell_ids(per_episode_csv)
    raw_cfg = _load_yaml_dict(eval_config_path)
    inputs = raw_cfg.get("inputs")
    if not isinstance(inputs, dict) or "episodes_index_path" not in inputs:
        raise ValueError(f"missing inputs.episodes_index_path in {eval_config_path}")
    episodes_index_path = Path(str(inputs["episodes_index_path"])).expanduser().resolve()

    episodes_df = pd.read_csv(episodes_index_path, usecols=["cell_id", "artifact_path"])
    episodes_df["cell_id"] = episodes_df["cell_id"].map(normalize_cell_id)
    target_set = set(target_cell_ids)
    episodes_df = episodes_df.loc[episodes_df["cell_id"].isin(target_set)].copy()
    artifact_by_cell = {
        str(cell_id): str(artifact_path)
        for cell_id, artifact_path in zip(episodes_df["cell_id"].tolist(), episodes_df["artifact_path"].tolist())
    }
    candidate_barcodes: set[str] = set()
    missing_cells = 0
    for cell_id in target_cell_ids:
        artifact_path = artifact_by_cell.get(str(cell_id))
        if artifact_path is None:
            missing_cells += 1
            continue
        candidate_barcodes.update(_load_candidate_ids_from_artifact(artifact_path))
    if not candidate_barcodes:
        raise RuntimeError(f"no candidate barcodes loaded from PPO eval run: {eval_run_dir}")
    if missing_cells:
        logger.warning("Skipped %d eval cells missing from episodes_index.csv", missing_cells)
    return target_cell_ids, candidate_barcodes


def _restrict_to_eval_context(
    *,
    adata: Any,
    nuclear_df: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[Any, pd.DataFrame, dict[str, Any]]:
    from scipy.spatial import cKDTree

    target_cell_ids, candidate_barcodes = _load_eval_candidate_barcodes(args)
    target_cell_set = set(target_cell_ids)
    obs_names = pd.Index(adata.obs_names.astype(str))
    candidate_barcodes = set(obs_names.intersection(pd.Index(sorted(candidate_barcodes))).astype(str))
    if not candidate_barcodes:
        raise RuntimeError("no PPO eval candidate barcodes overlap the STCS AnnData matrix")

    nuclear_df = nuclear_df.copy()
    for col in ("array_row", "array_col"):
        if col not in nuclear_df.columns:
            nuclear_df[col] = np.nan
    missing_coords = nuclear_df["array_row"].isna() | nuclear_df["array_col"].isna()
    if bool(missing_coords.any()):
        coord_lookup = adata.obs[["array_row", "array_col"]].copy()
        fill = nuclear_df.loc[missing_coords, ["barcode"]].join(coord_lookup, on="barcode")
        nuclear_df.loc[missing_coords, "array_row"] = fill["array_row"].to_numpy()
        nuclear_df.loc[missing_coords, "array_col"] = fill["array_col"].to_numpy()

    radius = args.eval_context_radius_bins
    if radius is None:
        radius = int(args.search_radius)
    radius = max(0, int(radius))

    included_cell_ids = set(target_cell_set)
    candidate_index = pd.Index(sorted(candidate_barcodes))
    candidate_coords = adata.obs.loc[candidate_index, ["array_row", "array_col"]].to_numpy(dtype=np.float64)
    coord_mask = nuclear_df["array_row"].notna() & nuclear_df["array_col"].notna()
    if radius > 0 and candidate_coords.size > 0 and bool(coord_mask.any()):
        tree = cKDTree(candidate_coords)
        nuclear_coords = nuclear_df.loc[coord_mask, ["array_row", "array_col"]].to_numpy(dtype=np.float64)
        distances, _ = tree.query(nuclear_coords, k=1, distance_upper_bound=float(radius))
        nearby_cells = nuclear_df.loc[coord_mask].loc[np.isfinite(distances), "cell_id"].astype(str).tolist()
        included_cell_ids.update(nearby_cells)

    restricted_nuclear_df = nuclear_df.loc[nuclear_df["cell_id"].astype(str).isin(included_cell_ids)].copy()
    restricted_barcodes = set(candidate_barcodes)
    restricted_barcodes.update(restricted_nuclear_df["barcode"].astype(str).tolist())
    keep_barcodes = obs_names.intersection(pd.Index(sorted(restricted_barcodes)))
    restricted_adata = adata[keep_barcodes].copy()
    restricted_nuclear_df = restricted_nuclear_df.loc[
        restricted_nuclear_df["barcode"].astype(str).isin(set(restricted_adata.obs_names.astype(str)))
    ].copy()

    summary = {
        "enabled": True,
        "n_eval_cells": int(len(target_cell_ids)),
        "n_included_cells": int(len(included_cell_ids)),
        "n_candidate_barcodes": int(len(candidate_barcodes)),
        "n_restricted_barcodes": int(restricted_adata.n_obs),
        "n_restricted_nuclear_rows": int(len(restricted_nuclear_df)),
        "eval_context_radius_bins": int(radius),
    }
    logger.info("Restricted STCS to PPO eval context: %s", summary)
    return restricted_adata, restricted_nuclear_df, summary


def _build_stcs_object(*, stcs_repo_dir: Path, adata: Any, pseudo_hd_dir: Path, full_res_image_path: Path | None) -> Any:
    stcs_module_dir = stcs_repo_dir / "STCS"
    if not (stcs_module_dir / "STCS_main.py").exists():
        raise FileNotFoundError(
            f"STCS_main.py not found under {stcs_module_dir}. Run jobs/setup_stcs_env.sbatch first "
            "or set --stcs_repo_dir to a cloned YangLabRutgers/STCS repo."
        )
    if str(stcs_module_dir) not in sys.path:
        sys.path.insert(0, str(stcs_module_dir))
    from STCS_main import STCS  # type: ignore

    raw_adata = adata.copy()
    in_tissue = raw_adata.obs["in_tissue"].astype(bool).to_numpy()
    tissue_adata = raw_adata[in_tissue].copy()

    stcs = STCS.__new__(STCS)
    stcs.Folder_path = str(pseudo_hd_dir)
    stcs.full_res_image_path = None if full_res_image_path is None else str(full_res_image_path)
    stcs.sc_ref = None
    stcs.model_path = None
    stcs.cropped = False
    stcs.crop_coords = None
    stcs.raw_adata = raw_adata
    stcs.adata = tissue_adata
    stcs._barcode_data_path = None
    stcs._cell_data_path = None
    stcs._dc_pseudobulk_data_path = None
    stcs._dc_assignment_pseudobulk_data_path = None
    stcs._celltypist_results_path = None
    logger.info("Constructed STCS object: %d in-tissue bins, %d genes", stcs.adata.n_obs, stcs.adata.n_vars)
    return stcs


def _inject_external_nuclei(stcs: Any, nuclear_df: pd.DataFrame, output_dir: Path) -> None:
    # STCS later looks up pseudobulk ids as "123.0"; float labels keep that
    # official naming convention after labels_he.astype(str).
    labels = pd.Series(0.0, index=stcs.adata.obs_names, dtype=np.float64)
    barcode_to_label = nuclear_df.drop_duplicates("barcode").set_index("barcode")["label_id"]
    overlap = labels.index.intersection(barcode_to_label.index)
    labels.loc[overlap] = barcode_to_label.loc[overlap].astype(np.float64).to_numpy()
    stcs.adata.obs["labels_he"] = labels.astype(np.float64)
    stcs.raw_adata.obs["labels_he"] = 0.0
    raw_overlap = stcs.raw_adata.obs_names.intersection(labels.index)
    stcs.raw_adata.obs.loc[raw_overlap, "labels_he"] = labels.loc[raw_overlap].astype(np.float64).to_numpy()

    stardist_dir = output_dir / "stardist"
    stardist_dir.mkdir(parents=True, exist_ok=True)
    barcode_path = stardist_dir / "stardist_barcode_outputs.h5ad"
    stcs.adata.write_h5ad(barcode_path)
    stcs._barcode_data_path = str(barcode_path)
    logger.info("Injected external labels_he for %d overlapping nuclear bins", int((labels > 0).sum()))


def _create_pseudobulk_from_external_nuclei(stcs: Any, output_dir: Path, mode: str) -> None:
    pseudobulk_dir = output_dir / "pseudobulk"
    pseudobulk_dir.mkdir(parents=True, exist_ok=True)
    labels = pd.to_numeric(stcs.adata.obs["labels_he"], errors="coerce").fillna(0.0)
    nuclei_only = stcs.adata[labels.to_numpy(dtype=np.float64) > 0.0].copy()
    if nuclei_only.n_obs == 0:
        raise RuntimeError("no positive external nuclear labels available for STCS pseudobulk")
    if sparse.issparse(nuclei_only.X):
        nuclei_only.X = nuclei_only.X.tocsr(copy=True)
    else:
        nuclei_only.X = sparse.csr_matrix(nuclei_only.X)
    pseudobulk_data = stcs._create_pseudobulk(  # noqa: SLF001 - official STCS helper is the stable path here.
        adata=nuclei_only,
        mode=mode,
        cell_key="labels_he",
    )
    pseudobulk_file_path = pseudobulk_dir / "direct_pseudobulk.h5ad"
    pseudobulk_data.write_h5ad(pseudobulk_file_path)
    stcs._dc_pseudobulk_data_path = str(pseudobulk_file_path)
    logger.info("Created external-nuclei pseudobulk: %d cells -> %s", pseudobulk_data.n_obs, pseudobulk_file_path)


def _configure_stcs_runtime_globals(args: argparse.Namespace) -> None:
    stcs_main = sys.modules.get("STCS_main")
    if stcs_main is None:
        raise RuntimeError("STCS_main module is not loaded")
    runtime_values = {
        "min_cells": int(args.min_cells),
        "target_sum": float(args.target_sum),
        "n_top_genes": int(args.n_top_genes),
    }
    for name, value in runtime_values.items():
        if hasattr(stcs_main, name):
            setattr(stcs_main, name, value)
    logger.info("Configured STCS runtime globals: %s", runtime_values)


def _force_nuclear_assignments(stcs: Any) -> None:
    labels = pd.to_numeric(stcs.adata.obs["labels_he"], errors="coerce").fillna(0).astype(np.int64)
    mask = labels > 0
    if "assigned_cell_id" not in stcs.adata.obs.columns:
        stcs.adata.obs["assigned_cell_id"] = None
    stcs.adata.obs.loc[mask, "assigned_cell_id"] = labels.loc[mask].astype(str)
    logger.info("Forced %d nuclear seed bins to their external nucleus labels", int(mask.sum()))


def _save_assignments(
    *,
    stcs: Any,
    output_dir: Path,
    dataset_name: str,
    nuclear_df: pd.DataFrame,
    label_to_cell: dict[int, str],
) -> tuple[Path, Path]:
    obs = stcs.adata.obs.copy()
    if "assigned_cell_id" not in obs.columns:
        raise RuntimeError("STCS did not create adata.obs['assigned_cell_id']")
    obs["barcode"] = obs.index.astype(str)
    obs["assigned_label_id"] = pd.to_numeric(obs["assigned_cell_id"], errors="coerce")
    obs = obs.loc[obs["assigned_label_id"].notna()].copy()
    obs["assigned_label_id"] = obs["assigned_label_id"].astype(np.int64)
    obs = obs.loc[obs["assigned_label_id"] > 0].copy()
    obs["cell_id"] = obs["assigned_label_id"].map(label_to_cell)
    obs = obs.loc[obs["cell_id"].notna()].copy()

    nuclear_lookup = {
        (str(row.barcode), str(row.cell_id))
        for row in nuclear_df[["barcode", "cell_id"]].drop_duplicates().itertuples(index=False)
    }
    obs["is_nuclear"] = [
        (str(barcode), str(cell_id)) in nuclear_lookup
        for barcode, cell_id in zip(obs["barcode"].tolist(), obs["cell_id"].tolist())
    ]

    assignments = pd.DataFrame(
        {
            "cell_id": obs["cell_id"].astype(str),
            "barcode": obs["barcode"].astype(str),
            "is_nuclear": obs["is_nuclear"].astype(bool),
            "array_row": obs["array_row"].to_numpy(dtype=np.int64),
            "array_col": obs["array_col"].to_numpy(dtype=np.int64),
            "x_um": obs["array_col"].to_numpy(dtype=np.float64) * 2.0,
            "y_um": obs["array_row"].to_numpy(dtype=np.float64) * 2.0,
            "assigned_label_id": obs["assigned_label_id"].to_numpy(dtype=np.int64),
        }
    ).sort_values(["cell_id", "barcode"], kind="stable")
    assignments_path = output_dir / f"{dataset_name}_stcs_assignments.csv"
    assignments.to_csv(assignments_path, index=False)

    summary = assignments.groupby("cell_id", sort=False).agg(
        n_bins=("barcode", "size"),
        n_nuclear_bins=("is_nuclear", "sum"),
    ).reset_index()
    summary["n_cytoplasm_bins"] = summary["n_bins"] - summary["n_nuclear_bins"]
    summary_path = output_dir / f"{dataset_name}_stcs_cell_summary.csv"
    summary.to_csv(summary_path, index=False)

    logger.info("Saved %d STCS assigned bins to %s", len(assignments), assignments_path)
    logger.info("Saved %d STCS cell summaries to %s", len(summary), summary_path)
    return assignments_path, summary_path


def main() -> None:
    args = parse_args()
    if args.overlay_max_cells < 0:
        raise ValueError("--overlay_max_cells must be >= 0")
    if not (0.0 <= args.pred_min_nuclear_overlap_frac <= 1.0):
        raise ValueError("--pred_min_nuclear_overlap_frac must be in [0, 1]")
    if not (0.0 <= args.gt_min_nuclear_overlap_frac <= 1.0):
        raise ValueError("--gt_min_nuclear_overlap_frac must be in [0, 1]")

    output_dir = Path(args.output_dir).expanduser().resolve()
    _configure_logging(output_dir)
    pseudo_hd_dir = Path(args.pseudo_hd_dir).expanduser().resolve()
    stcs_repo_dir = Path(args.stcs_repo_dir).expanduser().resolve()
    full_res_image_path = None if args.full_res_image_path is None else Path(args.full_res_image_path).expanduser().resolve()
    external_nuclear_bins_path = Path(args.external_nuclear_bins_path).expanduser().resolve()

    logger.info("=" * 80)
    logger.info("Starting STCS analysis with external PPO-aligned nuclei")
    logger.info("pseudo_hd_dir: %s", pseudo_hd_dir)
    logger.info("stcs_repo_dir: %s", stcs_repo_dir)
    logger.info("output_dir: %s", output_dir)
    logger.info("external_nuclear_bins_path: %s", external_nuclear_bins_path)

    adata = _load_pseudo_visium_mtx(pseudo_hd_dir, full_res_image_path)
    nuclear_df, label_to_cell = _load_external_nuclear_labels(external_nuclear_bins_path)
    restriction_summary: dict[str, Any] = {"enabled": False}
    if bool(args.restrict_to_eval_cells):
        adata, nuclear_df, restriction_summary = _restrict_to_eval_context(
            adata=adata,
            nuclear_df=nuclear_df,
            args=args,
        )
        gc.collect()
    stcs = _build_stcs_object(
        stcs_repo_dir=stcs_repo_dir,
        adata=adata,
        pseudo_hd_dir=pseudo_hd_dir,
        full_res_image_path=full_res_image_path,
    )
    _configure_stcs_runtime_globals(args)
    _inject_external_nuclei(stcs, nuclear_df, output_dir)

    _create_pseudobulk_from_external_nuclei(stcs, output_dir, mode=str(args.pseudobulk_mode))
    stcs.run_assignment(
        str(output_dir),
        min_cells=int(args.min_cells),
        L=float(args.lambda_spatial),
        target_sum=float(args.target_sum),
        top_genes=int(args.n_top_genes),
        search_radius=int(args.search_radius),
        use_sc_ref=False,
        normalize_distances=bool(args.normalize_distances),
    )
    if args.force_nuclear_assignments:
        _force_nuclear_assignments(stcs)

    assignments_path, cell_summary_path = _save_assignments(
        stcs=stcs,
        output_dir=output_dir,
        dataset_name=str(args.dataset_name),
        nuclear_df=nuclear_df,
        label_to_cell=label_to_cell,
    )
    stcs_h5ad_path = output_dir / f"{args.dataset_name}_stcs_spatial_adata.h5ad"
    stcs.adata.write_h5ad(stcs_h5ad_path)

    now_utc = dt.datetime.now(dt.timezone.utc)
    pipeline_summary = {
        "method": "stcs",
        "nuclear_source": "external_ppo_aligned",
        "created_utc": now_utc.isoformat(),
        "created_local": now_utc.astimezone(_LOCAL_TIMEZONE).isoformat(),
        "pseudo_hd_dir": str(pseudo_hd_dir),
        "stcs_repo_dir": str(stcs_repo_dir),
        "output_dir": str(output_dir),
        "dataset_name": str(args.dataset_name),
        "external_nuclear_bins_path": str(external_nuclear_bins_path),
        "search_radius": int(args.search_radius),
        "lambda_spatial": float(args.lambda_spatial),
        "target_sum": float(args.target_sum),
        "n_top_genes": int(args.n_top_genes),
        "min_cells": int(args.min_cells),
        "pseudobulk_mode": str(args.pseudobulk_mode),
        "normalize_distances": bool(args.normalize_distances),
        "force_nuclear_assignments": bool(args.force_nuclear_assignments),
        "restrict_to_eval_cells": bool(args.restrict_to_eval_cells),
        "eval_context_restriction": restriction_summary,
        "stcs_assignments_path": str(assignments_path),
        "stcs_cell_summary_path": str(cell_summary_path),
        "stcs_spatial_h5ad_path": str(stcs_h5ad_path),
    }
    pipeline_summary_path = output_dir / f"{args.dataset_name}_stcs_pipeline_summary.json"
    with pipeline_summary_path.open("w", encoding="utf-8") as handle:
        json.dump(pipeline_summary, handle, indent=2, sort_keys=False)
        handle.write("\n")
    logger.info("Pipeline summary saved to %s", pipeline_summary_path)

    if args.ppo_eval_run_dir is not None:
        logger.info("Running PPO-format STCS evaluation...")
        eval_run_dir = run_ppo_format_assignment_evaluation(
            assignments_csv=assignments_path,
            method_name="stcs",
            method_label="STCS",
            nuclear_source="external_ppo_aligned",
            external_nuclear_bins_path=external_nuclear_bins_path,
            args=args,
            pipeline_config=pipeline_summary,
        )
        logger.info("PPO-format STCS evaluation complete: %s", eval_run_dir)

    logger.info("=" * 80)
    logger.info("STCS analysis complete")
    logger.info("Assignments: %s", assignments_path)
    logger.info("Summary: %s", cell_summary_path)
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
