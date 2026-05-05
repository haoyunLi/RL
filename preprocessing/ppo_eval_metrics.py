from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def normalize_cell_id(value: Any) -> str | None:
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


def load_episode_build_bins_path(episodes_index_path: Path) -> Path | None:
    """Read bins_path from the episode-build resolved config."""
    cfg_path = episodes_index_path.parent / "config" / "config_resolved.yaml"
    if not cfg_path.exists():
        return None
    with cfg_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        return None
    inputs = raw.get("inputs")
    if not isinstance(inputs, dict):
        return None
    bins_value = inputs.get("bins_path")
    if bins_value is None:
        return None
    return Path(str(bins_value)).expanduser().resolve()


def build_episode_nuclear_barcode_map(bins_path: Path, target_cell_ids: set[str]) -> dict[str, set[str]]:
    """Map evaluated episode cell_id to its episode-build nuclear seed barcodes."""
    if not bins_path.exists():
        raise FileNotFoundError(f"episode-build bins metadata not found: {bins_path}")

    df = pd.read_parquet(
        bins_path,
        columns=["barcode", "dominant_cell_id", "has_nuclear_annotation"],
    )
    df = df.loc[df["has_nuclear_annotation"].fillna(False).astype(bool)].copy()
    dominant = pd.to_numeric(df["dominant_cell_id"], errors="coerce")
    keep = dominant.notna()
    if not np.any(keep.to_numpy(dtype=bool, copy=False)):
        return {}

    out_df = pd.DataFrame(
        {
            "cell_id": dominant.loc[keep].astype(np.int64).astype(str),
            "barcode": df.loc[keep, "barcode"].astype(str),
        }
    )
    out_df = out_df.loc[out_df["cell_id"].isin(target_cell_ids)].copy()
    if out_df.empty:
        return {}

    mapping: dict[str, set[str]] = {}
    for cell_id, group in out_df.groupby("cell_id", sort=False):
        mapping[str(cell_id)] = set(group["barcode"].astype(str).tolist())
    return mapping


def collect_gt_nuclear_candidates(
    *,
    gt_nuclear_bins_path: Path,
    episode_nuclear_barcodes: set[str],
) -> dict[str, set[str]]:
    """Collect barcode-to-GT-cell links for episode nuclear barcodes."""
    barcode_to_gt_cells: dict[str, set[str]] = defaultdict(set)
    usecols = ["cell_id", "barcode"]
    for chunk in pd.read_csv(
        gt_nuclear_bins_path,
        usecols=usecols,
        compression="infer",
        chunksize=1_000_000,
    ):
        chunk = chunk.dropna(subset=["cell_id", "barcode"]).copy()
        chunk["cell_id"] = chunk["cell_id"].map(normalize_cell_id)
        chunk = chunk.loc[chunk["cell_id"].notna()].copy()
        if chunk.empty:
            continue
        chunk["barcode"] = chunk["barcode"].astype(str)

        overlap = chunk.loc[chunk["barcode"].isin(episode_nuclear_barcodes), ["barcode", "cell_id"]]
        for barcode, group in overlap.groupby("barcode", sort=False):
            barcode_to_gt_cells[str(barcode)].update(group["cell_id"].astype(str).tolist())

    return dict(barcode_to_gt_cells)


def load_gt_bins_for_cells(
    *,
    csv_path: Path,
    matched_cell_ids: set[str],
) -> tuple[dict[str, set[str]], dict[str, np.ndarray]]:
    """Load GT barcode sets and XY arrays only for requested GT cells."""
    if not matched_cell_ids:
        return {}, {}

    barcode_map: dict[str, list[str]] = defaultdict(list)
    xy_x: dict[str, list[float]] = defaultdict(list)
    xy_y: dict[str, list[float]] = defaultdict(list)
    usecols = ["cell_id", "barcode", "x_um", "y_um"]
    for chunk in pd.read_csv(
        csv_path,
        usecols=usecols,
        compression="infer",
        chunksize=1_000_000,
    ):
        chunk = chunk.dropna(subset=["cell_id", "barcode", "x_um", "y_um"]).copy()
        chunk["cell_id"] = chunk["cell_id"].map(normalize_cell_id)
        chunk = chunk.loc[chunk["cell_id"].isin(matched_cell_ids)].copy()
        if chunk.empty:
            continue

        chunk["barcode"] = chunk["barcode"].astype(str)
        chunk["x_um"] = pd.to_numeric(chunk["x_um"], errors="coerce")
        chunk["y_um"] = pd.to_numeric(chunk["y_um"], errors="coerce")
        chunk = chunk.dropna(subset=["x_um", "y_um"])
        if chunk.empty:
            continue

        for cell_id, group in chunk.groupby("cell_id", sort=False):
            barcode_map[str(cell_id)].extend(group["barcode"].astype(str).tolist())
            xy_x[str(cell_id)].extend(group["x_um"].astype(float).tolist())
            xy_y[str(cell_id)].extend(group["y_um"].astype(float).tolist())

    barcode_sets = {cell_id: set(values) for cell_id, values in barcode_map.items()}
    xy_map = {
        cell_id: np.column_stack(
            (
                np.asarray(xy_x[cell_id], dtype=np.float32),
                np.asarray(xy_y[cell_id], dtype=np.float32),
            )
        )
        for cell_id in barcode_map
    }
    return barcode_sets, xy_map


def match_episode_cells_by_nuclear_overlap(
    *,
    episode_nuclear_by_cell: dict[str, set[str]],
    barcode_to_target_cells: dict[str, set[str]],
    min_overlap_frac: float,
    min_overlap_bins: int,
    match_method: str,
) -> tuple[dict[str, dict[str, Any]], set[str]]:
    """Match each episode cell to one target cell by maximum nuclear barcode overlap."""
    provisional: dict[str, dict[str, Any]] = {}
    matched_ids: set[str] = set()
    for cell_id, ep_nuclear in episode_nuclear_by_cell.items():
        matched_cell_id: str | None = None
        overlap_count = 0
        overlap_frac_episode = np.nan
        row_match_method = "unmatched"
        if ep_nuclear:
            counts: Counter[str] = Counter()
            for barcode in ep_nuclear:
                for target_cell_id in barcode_to_target_cells.get(barcode, ()):
                    counts[str(target_cell_id)] += 1
            if counts:
                best_cell_id, best_count = max(counts.items(), key=lambda kv: (kv[1], kv[0]))
                best_frac = float(best_count / len(ep_nuclear))
                if int(best_count) >= int(min_overlap_bins) and best_frac >= float(min_overlap_frac):
                    matched_cell_id = str(best_cell_id)
                    overlap_count = int(best_count)
                    overlap_frac_episode = best_frac
                    row_match_method = str(match_method)
                    matched_ids.add(matched_cell_id)
        provisional[str(cell_id)] = {
            "matched_cell_id": matched_cell_id,
            "match_method": row_match_method,
            "nuclear_overlap_bins": int(overlap_count),
            "nuclear_overlap_frac_episode": float(overlap_frac_episode),
        }
    return provisional, matched_ids


def compute_spatial_overlap_metrics(pred_barcodes: set[str], gt_barcodes: set[str]) -> dict[str, Any]:
    """Compute whole-cell overlap metrics from predicted and GT barcode sets."""
    pred = {str(x) for x in pred_barcodes}
    gt = {str(x) for x in gt_barcodes}
    intersection = int(len(pred & gt))
    union = int(len(pred | gt))
    iou = float(intersection / union) if union > 0 else np.nan
    denom = int(len(pred) + len(gt))
    dice = float((2.0 * intersection) / denom) if denom > 0 else np.nan
    precision = float(intersection / len(pred)) if len(pred) > 0 else 0.0
    recall = float(intersection / len(gt)) if len(gt) > 0 else 0.0
    f1 = (
        float((2.0 * precision * recall) / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )
    return {
        "intersection": intersection,
        "union": union,
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pred_n_bins": int(len(pred)),
        "gt_n_bins": int(len(gt)),
    }
