#!/usr/bin/env python
"""Evaluate a trained PPO checkpoint on episode data."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
import datetime as dt
import gzip
import json
from pathlib import Path
import random
import re
import sys
import tempfile
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import torch
import yaml

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hd_cell_rl.ppo_training import (
    ActorCritic,
    AddStopCellEnv,
    EpisodeDataset,
    PPOTrainingConfig,
    _observation_to_tensors,
    load_ppo_training_config,
)
from preprocessing.ppo_format_assignment_eval import (
    _add_numeric_summary,
    annotate_records_with_gene_correlation,
)

_LOCAL_TIMEZONE = ZoneInfo("America/Chicago")
_LOCAL_TIMEZONE_NAME = "America/Chicago"


@dataclass(frozen=True)
class EpisodeEvalRecord:
    """One evaluated episode plus final state geometry for overlays."""

    metrics: dict[str, Any]
    candidate_bin_ids: tuple[str, ...]
    final_membership_mask: np.ndarray
    candidate_bin_xy_um: np.ndarray
    nucleus_center_xy_um: np.ndarray
    action_trace: tuple[dict[str, Any], ...] = ()
    gt_cell_xy_um: np.ndarray | None = None
    gt_nuclear_xy_um: np.ndarray | None = None


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate PPO checkpoint on HD cell episodes")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional PPO config YAML. If omitted, uses config stored in checkpoint.",
    )
    parser.add_argument(
        "--episodes-index-path",
        type=str,
        default=None,
        help="Optional override for episodes_index.csv path.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=200,
        help="Number of episodes to evaluate (default: 200).",
    )
    parser.add_argument(
        "--policy-mode",
        type=str,
        choices=("greedy", "sample"),
        default="greedy",
        help="Action selection mode during evaluation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device for policy inference.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Sampling/evaluation seed.")
    parser.add_argument("--run-name", type=str, default="ppo_checkpoint_eval", help="Output run name prefix.")
    parser.add_argument("--output-root", type=str, default="runs", help="Root directory for evaluation outputs.")
    parser.add_argument(
        "--overlay-max-cells",
        type=int,
        default=12,
        help="Number of per-cell overlay PNGs to write (0 disables overlays).",
    )
    parser.add_argument(
        "--overlay-selection",
        type=str,
        choices=("top_reward", "random", "first"),
        default="top_reward",
        help="How to choose cells for overlay plots.",
    )
    parser.add_argument(
        "--gt-cell-bins-path",
        type=str,
        default=None,
        help="Optional GT full-cell bin table (.csv/.csv.gz) for overlay + IoU/Dice evaluation.",
    )
    parser.add_argument(
        "--gt-nuclear-bins-path",
        type=str,
        default=None,
        help="Optional GT nuclear-bin table (.csv/.csv.gz) used to match each episode to a GT cell.",
    )
    parser.add_argument(
        "--gt-cell-assignments-csv",
        type=str,
        default=None,
        help="Optional pseudo-data cell_id -> sc_cell_barcode mapping for gene correlation.",
    )
    parser.add_argument(
        "--gt-sc-expression-h5",
        type=str,
        default=None,
        help="Optional ground-truth single-cell expression H5 for gene correlation.",
    )
    parser.add_argument(
        "--gt-min-nuclear-overlap-frac",
        type=float,
        default=0.3,
        help="Minimum fraction of episode nucleus bins needed to accept a nucleus-overlap GT match.",
    )
    parser.add_argument(
        "--gt-min-nuclear-overlap-bins",
        type=int,
        default=2,
        help="Minimum count of overlapping nucleus bins needed to accept a nucleus-overlap GT match.",
    )
    return parser


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("requested --device cuda, but CUDA is not available")
        return torch.device("cuda")
    return torch.device("cpu")


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_config_from_checkpoint_payload(payload: dict[str, Any]) -> PPOTrainingConfig:
    cfg_dict = payload.get("config")
    if not isinstance(cfg_dict, dict):
        raise ValueError("checkpoint does not contain a valid 'config' dictionary")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as tmp:
        yaml.safe_dump(cfg_dict, tmp, sort_keys=False)
        tmp_path = Path(tmp.name)
    try:
        return load_ppo_training_config(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _run_single_episode(
    *,
    context,
    model: ActorCritic,
    device: torch.device,
    policy_mode: str,
    rng: np.random.Generator,
) -> EpisodeEvalRecord:
    env = AddStopCellEnv(context)
    obs, _ = env.reset()

    total_reward = 0.0
    n_steps = 0
    final_info: dict[str, Any] | None = None
    terminated = False
    truncated = False
    action_trace: list[dict[str, Any]] = []

    while True:
        step_index = int(obs["step_index"])
        g_t, a_t, m_t = _observation_to_tensors(obs, device=device)
        with torch.inference_mode():
            dist, _ = model(g_t, a_t, m_t)
            probs = dist.probs.squeeze(0).detach().cpu().numpy()
        if policy_mode == "greedy":
            action = int(np.argmax(probs))
        else:
            prob_sum = float(np.sum(probs))
            if prob_sum <= 0.0 or not np.isfinite(prob_sum):
                raise RuntimeError("non-finite policy probabilities during evaluation")
            action = int(rng.choice(probs.shape[0], p=probs / prob_sum))

        obs, reward, term, trunc, info = env.step(action)
        total_reward += float(reward)
        n_steps += 1
        final_info = info
        terminated = bool(term)
        truncated = bool(trunc)
        chosen_barcode = None if action == 0 else str(context.candidate_bin_ids[action - 1])
        action_trace.append(
            {
                "step_index": int(step_index),
                "action": int(action),
                "action_probability": float(probs[action]),
                "reward": float(reward),
                "chosen_barcode": chosen_barcode,
                "terminated_after_action": bool(term),
                "truncated_after_action": bool(trunc),
                "n_assigned_bins_after": int(info.get("n_assigned_bins", 0)),
            }
        )
        if term or trunc:
            break

    if final_info is None:
        raise RuntimeError("episode produced no final info")

    final_membership_mask = np.asarray(obs["membership_mask"], dtype=np.uint8).copy()
    metrics = {
        "cell_id": str(context.cell_id),
        "total_reward": float(total_reward),
        "n_steps": int(n_steps),
        "n_assigned_bins": int(final_info.get("n_assigned_bins", 0)),
        "n_candidate_bins": int(final_info.get("n_candidate_bins", 0)),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
    }
    return EpisodeEvalRecord(
        metrics=metrics,
        candidate_bin_ids=tuple(str(x) for x in context.candidate_bin_ids),
        final_membership_mask=final_membership_mask,
        candidate_bin_xy_um=np.asarray(context.candidate_bin_xy_um, dtype=np.float32),
        nucleus_center_xy_um=np.asarray(context.nucleus_center_xy_um, dtype=np.float32),
        action_trace=tuple(action_trace),
    )


def _slug(value: str) -> str:
    out = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(value)).strip("_")
    return out or "cell"


def _now_utc_and_local() -> tuple[dt.datetime, dt.datetime]:
    """Return current UTC and America/Chicago timestamps."""
    now_utc = dt.datetime.now(dt.timezone.utc)
    return now_utc, now_utc.astimezone(_LOCAL_TIMEZONE)


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


def _load_episode_build_bins_path(episodes_index_path: Path) -> Path | None:
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


def _build_episode_nuclear_barcode_map(bins_path: Path, target_cell_ids: set[str]) -> dict[str, set[str]]:
    """Map each evaluated episode cell_id to its nucleus seed barcodes from episode-build bins metadata."""
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


def _collect_gt_nuclear_candidates(
    *,
    gt_nuclear_bins_path: Path,
    episode_nuclear_barcodes: set[str],
) -> dict[str, set[str]]:
    """Collect barcode->GT-cell links for episode nucleus bars."""
    barcode_to_gt_cells: dict[str, set[str]] = defaultdict(set)

    usecols = ["cell_id", "barcode"]
    for chunk in pd.read_csv(
        gt_nuclear_bins_path,
        usecols=usecols,
        compression="infer",
        chunksize=1_000_000,
    ):
        chunk = chunk.dropna(subset=["cell_id", "barcode"]).copy()
        chunk["cell_id"] = chunk["cell_id"].map(_normalize_cell_id)
        chunk = chunk.loc[chunk["cell_id"].notna()].copy()
        if chunk.empty:
            continue
        chunk["barcode"] = chunk["barcode"].astype(str)

        overlap = chunk.loc[chunk["barcode"].isin(episode_nuclear_barcodes), ["barcode", "cell_id"]]
        for barcode, group in overlap.groupby("barcode", sort=False):
            barcode_to_gt_cells[str(barcode)].update(group["cell_id"].astype(str).tolist())

    return dict(barcode_to_gt_cells)


def _load_gt_bins_for_cells(
    *,
    csv_path: Path,
    matched_cell_ids: set[str],
) -> tuple[dict[str, set[str]], dict[str, np.ndarray]]:
    """Load GT bin barcode sets and XY arrays only for matched GT cells."""
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
        chunk["cell_id"] = chunk["cell_id"].map(_normalize_cell_id)
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


def _annotate_records_with_ground_truth(
    *,
    records: list[EpisodeEvalRecord],
    episodes_index_path: Path,
    gt_cell_bins_path: Path,
    gt_nuclear_bins_path: Path,
    min_overlap_frac: float,
    min_overlap_bins: int,
) -> list[EpisodeEvalRecord]:
    """Match each episode to a GT cell and attach GT metrics/geometry."""
    if not records:
        return records

    bins_path = _load_episode_build_bins_path(episodes_index_path)
    if bins_path is None:
        raise FileNotFoundError(
            f"could not resolve episode-build bins_path from {episodes_index_path.parent / 'config' / 'config_resolved.yaml'}"
        )

    episode_cell_ids = {str(rec.metrics["cell_id"]) for rec in records}
    episode_nuclear_by_cell = _build_episode_nuclear_barcode_map(bins_path=bins_path, target_cell_ids=episode_cell_ids)
    episode_nuclear_union = set().union(*episode_nuclear_by_cell.values()) if episode_nuclear_by_cell else set()

    barcode_to_gt_cells = _collect_gt_nuclear_candidates(
        gt_nuclear_bins_path=gt_nuclear_bins_path,
        episode_nuclear_barcodes=episode_nuclear_union,
    )

    matched_gt_ids: set[str] = set()
    provisional: dict[str, dict[str, Any]] = {}
    for rec in records:
        cell_id = str(rec.metrics["cell_id"])
        ep_nuclear = episode_nuclear_by_cell.get(cell_id, set())

        match_method = "unmatched"
        matched_gt_cell_id: str | None = None
        overlap_count = 0
        overlap_frac_episode = np.nan

        if ep_nuclear:
            counts: Counter[str] = Counter()
            for barcode in ep_nuclear:
                for gt_cell_id in barcode_to_gt_cells.get(barcode, ()):
                    counts[str(gt_cell_id)] += 1
            if counts:
                best_gt_cell_id, best_count = max(counts.items(), key=lambda kv: (kv[1], kv[0]))
                best_frac = float(best_count / len(ep_nuclear))
                if int(best_count) >= int(min_overlap_bins) and best_frac >= float(min_overlap_frac):
                    matched_gt_cell_id = str(best_gt_cell_id)
                    match_method = "nuclear_overlap"
                    overlap_count = int(best_count)
                    overlap_frac_episode = best_frac

        if matched_gt_cell_id is not None:
            matched_gt_ids.add(matched_gt_cell_id)

        provisional[cell_id] = {
            "matched_gt_cell_id": matched_gt_cell_id,
            "match_method": match_method,
            "episode_nuclear_bin_count": int(len(ep_nuclear)),
            "nuclear_overlap_bins": int(overlap_count),
            "nuclear_overlap_frac_episode": float(overlap_frac_episode),
        }

    gt_nuclear_by_cell, gt_nuclear_xy_by_cell = _load_gt_bins_for_cells(
        csv_path=gt_nuclear_bins_path,
        matched_cell_ids=matched_gt_ids,
    )
    gt_cell_by_cell, gt_cell_xy_by_cell = _load_gt_bins_for_cells(
        csv_path=gt_cell_bins_path,
        matched_cell_ids=matched_gt_ids,
    )

    updated: list[EpisodeEvalRecord] = []
    for rec in records:
        cell_id = str(rec.metrics["cell_id"])
        meta = dict(provisional[cell_id])
        matched_gt_cell_id = meta["matched_gt_cell_id"]

        gt_nuclear = set()
        gt_cell = set()
        gt_nuclear_xy = None
        gt_cell_xy = None
        overlap_frac_gt = np.nan
        pred_iou = np.nan
        pred_dice = np.nan
        pred_precision = np.nan
        pred_recall = np.nan
        pred_intersection_bins = 0

        if matched_gt_cell_id is not None:
            gt_nuclear = gt_nuclear_by_cell.get(matched_gt_cell_id, set())
            gt_cell = gt_cell_by_cell.get(matched_gt_cell_id, set())
            gt_nuclear_xy = gt_nuclear_xy_by_cell.get(matched_gt_cell_id)
            gt_cell_xy = gt_cell_xy_by_cell.get(matched_gt_cell_id)

            overlap_bins = int(meta["nuclear_overlap_bins"])
            if gt_nuclear:
                overlap_frac_gt = float(overlap_bins / len(gt_nuclear))

            assigned_mask = np.asarray(rec.final_membership_mask, dtype=np.uint8) == 1
            pred_bars = {
                str(barcode)
                for i, barcode in enumerate(rec.candidate_bin_ids)
                if i < assigned_mask.shape[0] and assigned_mask[i]
            }
            if gt_cell:
                pred_intersection_bins = int(len(pred_bars & gt_cell))
                union = int(len(pred_bars | gt_cell))
                if union > 0:
                    pred_iou = float(pred_intersection_bins / union)
                denom = int(len(pred_bars) + len(gt_cell))
                if denom > 0:
                    pred_dice = float((2.0 * pred_intersection_bins) / denom)
                pred_precision = float(pred_intersection_bins / len(pred_bars)) if len(pred_bars) > 0 else 0.0
                pred_recall = float(pred_intersection_bins / len(gt_cell))
        pred_f1 = (
            float((2 * pred_precision * pred_recall) / (pred_precision + pred_recall))
            if np.isfinite(pred_precision) and np.isfinite(pred_recall) and (pred_precision + pred_recall) > 0
            else np.nan
        )

        meta.update(
            {
                "gt_nuclear_bin_count": int(len(gt_nuclear)),
                "gt_cell_bin_count": int(len(gt_cell)),
                "nuclear_overlap_frac_gt": float(overlap_frac_gt),
                "pred_gt_intersection_bins": int(pred_intersection_bins),
                "pred_iou": float(pred_iou),
                "pred_dice": float(pred_dice),
                "pred_precision": float(pred_precision),
                "pred_recall": float(pred_recall),
                "pred_f1": float(pred_f1),
            }
        )

        updated.append(
            replace(
                rec,
                metrics={**rec.metrics, **meta},
                gt_cell_xy_um=None if gt_cell_xy is None else np.asarray(gt_cell_xy, dtype=np.float32),
                gt_nuclear_xy_um=None if gt_nuclear_xy is None else np.asarray(gt_nuclear_xy, dtype=np.float32),
            )
        )
    return updated


def _save_summary_plots(df: pd.DataFrame, run_dir: Path) -> list[str]:
    """Write aggregate evaluation plots."""
    if not HAS_MATPLOTLIB:
        return []

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.hist(df["total_reward"].to_numpy(dtype=np.float64), bins=40, color="#2A9D8F", alpha=0.9)
    ax.set_title("Episode Total Reward Distribution")
    ax.set_xlabel("Total Reward")
    ax.set_ylabel("Episode Count")
    fig.tight_layout()
    p = plots_dir / "reward_hist.png"
    fig.savefig(p, dpi=180)
    plt.close(fig)
    saved.append(str(p))

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.hist(df["n_assigned_bins"].to_numpy(dtype=np.float64), bins=40, color="#E76F51", alpha=0.9)
    ax.set_title("Assigned Bins Distribution")
    ax.set_xlabel("Assigned Bins")
    ax.set_ylabel("Episode Count")
    fig.tight_layout()
    p = plots_dir / "assigned_bins_hist.png"
    fig.savefig(p, dpi=180)
    plt.close(fig)
    saved.append(str(p))

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.scatter(
        df["n_assigned_bins"].to_numpy(dtype=np.float64),
        df["total_reward"].to_numpy(dtype=np.float64),
        s=12,
        alpha=0.5,
        color="#264653",
        linewidths=0.0,
    )
    ax.set_title("Reward vs Assigned Bins")
    ax.set_xlabel("Assigned Bins")
    ax.set_ylabel("Total Reward")
    fig.tight_layout()
    p = plots_dir / "reward_vs_assigned_bins.png"
    fig.savefig(p, dpi=180)
    plt.close(fig)
    saved.append(str(p))

    if "pred_iou" in df.columns:
        iou = df["pred_iou"].to_numpy(dtype=np.float64)
        iou = iou[np.isfinite(iou)]
        if iou.size > 0:
            fig, ax = plt.subplots(figsize=(7.5, 5.0))
            ax.hist(iou, bins=30, color="#457B9D", alpha=0.9)
            ax.set_title("Predicted vs GT IoU Distribution")
            ax.set_xlabel("IoU")
            ax.set_ylabel("Episode Count")
            fig.tight_layout()
            p = plots_dir / "pred_gt_iou_hist.png"
            fig.savefig(p, dpi=180)
            plt.close(fig)
            saved.append(str(p))

            fig, ax = plt.subplots(figsize=(7.5, 5.0))
            iou_sorted = np.sort(iou)
            cdf_y = np.arange(1, iou_sorted.size + 1, dtype=np.float64) / float(iou_sorted.size)
            ax.plot(iou_sorted, cdf_y, color="#2A6F97", linewidth=2.0)
            ax.set_title("Predicted vs GT IoU CDF")
            ax.set_xlabel("IoU")
            ax.set_ylabel("Cumulative Fraction")
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
            ax.grid(True, alpha=0.25, linewidth=0.6)
            fig.tight_layout()
            p = plots_dir / "pred_gt_iou_cdf.png"
            fig.savefig(p, dpi=180)
            plt.close(fig)
            saved.append(str(p))

    if "pred_dice" in df.columns:
        dice = df["pred_dice"].to_numpy(dtype=np.float64)
        dice = dice[np.isfinite(dice)]
        if dice.size > 0:
            fig, ax = plt.subplots(figsize=(7.5, 5.0))
            ax.hist(dice, bins=30, color="#1D3557", alpha=0.9)
            ax.set_title("Predicted vs GT Dice Distribution")
            ax.set_xlabel("Dice")
            ax.set_ylabel("Episode Count")
            fig.tight_layout()
            p = plots_dir / "pred_gt_dice_hist.png"
            fig.savefig(p, dpi=180)
            plt.close(fig)
            saved.append(str(p))

    return saved


def _choose_overlay_indices(df: pd.DataFrame, n_pick: int, selection: str, seed: int) -> list[int]:
    if n_pick <= 0 or len(df) == 0:
        return []
    n_pick = min(n_pick, len(df))
    if selection == "first":
        return list(range(n_pick))
    if selection == "top_reward":
        order = np.argsort(-df["total_reward"].to_numpy(dtype=np.float64))
        return [int(i) for i in order[:n_pick]]
    rng = np.random.default_rng(seed)
    choices = rng.choice(len(df), size=n_pick, replace=False)
    return [int(i) for i in np.asarray(choices, dtype=np.int64)]


def _estimate_grid_step(coords: np.ndarray) -> float:
    """Estimate bin-center spacing from one axis of GT bin coordinates."""
    vals = np.unique(np.asarray(coords, dtype=np.float64))
    if vals.size <= 1:
        return 2.0
    diffs = np.diff(np.sort(vals))
    diffs = diffs[np.isfinite(diffs) & (diffs > 1.0e-6)]
    if diffs.size == 0:
        return 2.0
    return float(np.median(diffs))


def _build_gt_contour_grid(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Convert GT occupied bins into a padded binary grid for connected contour drawing."""
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


def _finite_float_or_none(value: Any) -> float | None:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    return val if np.isfinite(val) else None


def _format_overlay_title(row: dict[str, Any]) -> str:
    lines = [
        f"cell={row['cell_id']}  reward={row['total_reward']:.2f}  "
        f"assigned={row['n_assigned_bins']}/{row['n_candidate_bins']}",
        f"GT match={row.get('match_method', 'none')}",
    ]

    iou = _finite_float_or_none(row.get("pred_iou", np.nan))
    dice = _finite_float_or_none(row.get("pred_dice", np.nan))
    precision = _finite_float_or_none(row.get("pred_precision", np.nan))
    recall = _finite_float_or_none(row.get("pred_recall", np.nan))
    metric_parts: list[str] = []
    if iou is not None:
        metric_parts.append(f"IoU={iou:.3f}")
    if dice is not None:
        metric_parts.append(f"Dice={dice:.3f}")
    if precision is not None:
        metric_parts.append(f"P={precision:.3f}")
    if recall is not None:
        metric_parts.append(f"R={recall:.3f}")
    if metric_parts:
        lines.append("  ".join(metric_parts))

    gene = _finite_float_or_none(row.get("gene_spearman_r", np.nan))
    if gene is not None:
        lines.append(f"Gene Spearman={gene:.3f}")
    return "\n".join(lines)


def _save_overlay_plots(
    *,
    records: list[EpisodeEvalRecord],
    df: pd.DataFrame,
    run_dir: Path,
    max_cells: int,
    selection: str,
    seed: int,
) -> list[str]:
    """Write per-cell spatial overlay plots."""
    if not HAS_MATPLOTLIB or max_cells <= 0 or not records:
        return []

    overlays_dir = run_dir / "overlays"
    overlays_dir.mkdir(parents=True, exist_ok=True)

    indices = _choose_overlay_indices(df=df, n_pick=max_cells, selection=selection, seed=seed)
    saved: list[str] = []
    for idx in indices:
        rec = records[idx]
        row = rec.metrics
        xy = np.asarray(rec.candidate_bin_xy_um, dtype=np.float32)
        if xy.ndim != 2 or xy.shape[1] != 2 or xy.shape[0] == 0:
            continue
        assigned = np.asarray(rec.final_membership_mask, dtype=np.uint8) == 1
        if assigned.shape[0] != xy.shape[0]:
            continue

        fig, ax = plt.subplots(figsize=(6.2, 6.2))
        ax.scatter(
            xy[:, 0],
            xy[:, 1],
            s=3,
            c="#D0D4DB",
            alpha=0.22,
            linewidths=0.0,
            zorder=1,
            label="candidate bins",
        )
        gt_cell_xy = None if rec.gt_cell_xy_um is None else np.asarray(rec.gt_cell_xy_um, dtype=np.float32)
        if gt_cell_xy is not None and gt_cell_xy.ndim == 2 and gt_cell_xy.shape[1] == 2 and gt_cell_xy.shape[0] > 0:
            contour_grid = _build_gt_contour_grid(gt_cell_xy)
            if contour_grid is not None:
                x_coords, y_coords, grid = contour_grid
                ax.contour(
                    x_coords,
                    y_coords,
                    grid,
                    levels=[0.5],
                    colors=["#1D3557"],
                    linewidths=3.0,
                    alpha=1.0,
                    zorder=4,
                )
                ax.plot(
                    [],
                    [],
                    color="#1D3557",
                    linewidth=3.0,
                    alpha=1.0,
                    label="GT cell outline",
                )
        if np.any(assigned):
            ax.scatter(
                xy[assigned, 0],
                xy[assigned, 1],
                s=8,
                c="#E63946",
                alpha=0.95,
                linewidths=0.0,
                zorder=3,
                label="assigned bins",
            )

        center = np.asarray(rec.nucleus_center_xy_um, dtype=np.float32)
        if center.shape == (2,):
            ax.scatter(
                [float(center[0])],
                [float(center[1])],
                s=90,
                marker="x",
                c="#1D3557",
                linewidths=2.0,
                zorder=5,
                label="nucleus center",
            )

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x (um)")
        ax.set_ylabel("y (um)")
        ax.set_title(_format_overlay_title(row), fontsize=10.5, pad=10)
        ax.legend(loc="best", fontsize=8, frameon=False)
        fig.tight_layout()

        out = overlays_dir / f"overlay_{idx:04d}_{_slug(str(row['cell_id']))}.png"
        fig.savefig(out, dpi=180, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(out))

    return saved


def _write_step_traces(
    *,
    records: list[EpisodeEvalRecord],
    run_dir: Path,
) -> list[EpisodeEvalRecord]:
    """Write one compressed per-episode action trace for exact replay."""
    if not records:
        return records

    traces_dir = run_dir / "step_traces"
    traces_dir.mkdir(parents=True, exist_ok=True)

    updated: list[EpisodeEvalRecord] = []
    for idx, rec in enumerate(records):
        trace_relpath = Path("step_traces") / f"trace_{idx:04d}_{_slug(str(rec.metrics['cell_id']))}.json.gz"
        trace_abspath = run_dir / trace_relpath
        payload = {
            "cell_id": str(rec.metrics["cell_id"]),
            "candidate_bin_count": int(len(rec.candidate_bin_ids)),
            "n_steps": int(len(rec.action_trace)),
            "action_trace": list(rec.action_trace),
        }
        with gzip.open(trace_abspath, "wt", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=False)
            handle.write("\n")
        updated.append(
            replace(
                rec,
                metrics={**rec.metrics, "step_trace_path": str(trace_relpath)},
            )
        )
    return updated


def main() -> None:
    args = _build_arg_parser().parse_args()
    if args.max_episodes <= 0:
        raise ValueError("--max-episodes must be > 0")
    if args.overlay_max_cells < 0:
        raise ValueError("--overlay-max-cells must be >= 0")
    if args.gt_min_nuclear_overlap_frac < 0 or args.gt_min_nuclear_overlap_frac > 1:
        raise ValueError("--gt-min-nuclear-overlap-frac must be in [0, 1]")
    if args.gt_min_nuclear_overlap_bins < 0:
        raise ValueError("--gt-min-nuclear-overlap-bins must be >= 0")
    if (args.gt_cell_bins_path is None) ^ (args.gt_nuclear_bins_path is None):
        raise ValueError("--gt-cell-bins-path and --gt-nuclear-bins-path must be provided together")
    if (args.gt_cell_assignments_csv is None) ^ (args.gt_sc_expression_h5 is None):
        raise ValueError("--gt-cell-assignments-csv and --gt-sc-expression-h5 must be provided together")

    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    payload = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("invalid checkpoint payload")
    if "model_state_dict" not in payload:
        raise ValueError("checkpoint missing 'model_state_dict'")

    if args.config is not None:
        config = load_ppo_training_config(args.config)
    else:
        config = _load_config_from_checkpoint_payload(payload)

    if args.episodes_index_path is not None:
        config = replace(config, episodes_index_path=Path(args.episodes_index_path).expanduser().resolve())

    _set_seeds(int(args.seed))
    rng = np.random.default_rng(int(args.seed))
    device = _resolve_device(args.device)

    out_root = Path(args.output_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    now_utc, now_local = _now_utc_and_local()
    ts = now_utc.strftime("%Y%m%dT%H%M%SZ")
    run_dir = out_root / f"{args.run_name}_{ts}"
    run_dir.mkdir(parents=False, exist_ok=False)

    model = ActorCritic(
        global_dim=AddStopCellEnv.GLOBAL_FEATURE_DIM,
        action_dim=AddStopCellEnv.ACTION_FEATURE_DIM,
        hidden_dim=int(config.hidden_dim),
    ).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    dataset = EpisodeDataset(config=config, rng=rng)
    try:
        n_eval = min(int(args.max_episodes), int(dataset.n_cells))
        rows = dataset.sample_rows(n_eval)
        episode_records: list[EpisodeEvalRecord] = []
        for i, row in enumerate(rows.itertuples(index=False), start=1):
            cell_id = str(row.cell_id)
            artifact_path = Path(str(row.artifact_path)).expanduser().resolve()
            context = dataset.load_episode_context(
                cell_id=cell_id,
                artifact_path=artifact_path,
                max_steps_per_episode=config.max_steps_per_episode,
                include_candidate_bin_ids=True,
            )
            if context is None:
                continue
            episode_records.append(
                _run_single_episode(
                    context=context,
                    model=model,
                    device=device,
                    policy_mode=str(args.policy_mode),
                    rng=rng,
                )
            )
            if i % 10 == 0 or i == n_eval:
                print(f"Evaluation progress: {i}/{n_eval} episodes")
    finally:
        dataset.close()

    if not episode_records:
        raise RuntimeError("no valid episodes were evaluated")

    gt_cell_bins_path = None if args.gt_cell_bins_path is None else Path(args.gt_cell_bins_path).expanduser().resolve()
    gt_nuclear_bins_path = None if args.gt_nuclear_bins_path is None else Path(args.gt_nuclear_bins_path).expanduser().resolve()
    gt_enabled = gt_cell_bins_path is not None and gt_nuclear_bins_path is not None
    if gt_enabled:
        if not gt_cell_bins_path.exists():
            raise FileNotFoundError(f"GT cell bins file not found: {gt_cell_bins_path}")
        if not gt_nuclear_bins_path.exists():
            raise FileNotFoundError(f"GT nuclear bins file not found: {gt_nuclear_bins_path}")
        episode_records = _annotate_records_with_ground_truth(
            records=episode_records,
            episodes_index_path=config.episodes_index_path,
            gt_cell_bins_path=gt_cell_bins_path,
            gt_nuclear_bins_path=gt_nuclear_bins_path,
            min_overlap_frac=float(args.gt_min_nuclear_overlap_frac),
            min_overlap_bins=int(args.gt_min_nuclear_overlap_bins),
        )

    gt_cell_assignments_csv = None if args.gt_cell_assignments_csv is None else Path(args.gt_cell_assignments_csv).expanduser().resolve()
    gt_sc_expression_h5 = None if args.gt_sc_expression_h5 is None else Path(args.gt_sc_expression_h5).expanduser().resolve()
    if gt_cell_assignments_csv is not None and gt_sc_expression_h5 is not None:
        episode_records = annotate_records_with_gene_correlation(
            records=episode_records,
            episodes_index_path=config.episodes_index_path,
            gt_cell_assignments_csv=gt_cell_assignments_csv,
            gt_sc_expression_h5=gt_sc_expression_h5,
        )

    episode_records = _write_step_traces(records=episode_records, run_dir=run_dir)

    results = [rec.metrics for rec in episode_records]
    df = pd.DataFrame(results)
    df.to_csv(run_dir / "per_episode.csv", index=False)

    summary_plots = _save_summary_plots(df=df, run_dir=run_dir)
    overlay_plots = _save_overlay_plots(
        records=episode_records,
        df=df,
        run_dir=run_dir,
        max_cells=int(args.overlay_max_cells),
        selection=str(args.overlay_selection),
        seed=int(args.seed),
    )

    summary = {
        "checkpoint_path": str(ckpt_path),
        "episodes_index_path": str(config.episodes_index_path),
        "device": str(device),
        "policy_mode": str(args.policy_mode),
        "seed": int(args.seed),
        "gt_enabled": bool(gt_enabled),
        "gt_cell_bins_path": None if gt_cell_bins_path is None else str(gt_cell_bins_path),
        "gt_nuclear_bins_path": None if gt_nuclear_bins_path is None else str(gt_nuclear_bins_path),
        "gt_cell_assignments_csv": None if gt_cell_assignments_csv is None else str(gt_cell_assignments_csv),
        "gt_sc_expression_h5": None if gt_sc_expression_h5 is None else str(gt_sc_expression_h5),
        "gt_match_mode": "nuclear_overlap",
        "evaluation_timestamp_utc": now_utc.isoformat(),
        "evaluation_timestamp_local": now_local.isoformat(),
        "local_timezone": _LOCAL_TIMEZONE_NAME,
        "n_episodes_evaluated": int(len(df)),
        "mean_total_reward": float(df["total_reward"].mean()),
        "median_total_reward": float(df["total_reward"].median()),
        "mean_n_assigned_bins": float(df["n_assigned_bins"].mean()),
        "median_n_assigned_bins": float(df["n_assigned_bins"].median()),
        "mean_n_steps": float(df["n_steps"].mean()),
        "terminated_fraction": float(df["terminated"].mean()),
        "truncated_fraction": float(df["truncated"].mean()),
        "matplotlib_available": bool(HAS_MATPLOTLIB),
        "n_summary_plots": int(len(summary_plots)),
        "n_overlay_plots": int(len(overlay_plots)),
        "step_traces_enabled": True,
        "step_traces_dir": str(run_dir / "step_traces"),
        "n_step_trace_files": int(len(episode_records)),
    }
    if "matched_gt_cell_id" in df.columns:
        matched = df["matched_gt_cell_id"].astype("string").notna()
        summary["matched_gt_fraction"] = float(matched.mean())
        summary["n_matched_gt"] = int(matched.sum())
    for metric_col in ("pred_iou", "pred_dice", "pred_precision", "pred_recall", "pred_f1", "gene_spearman_r", "gene_rmse"):
        _add_numeric_summary(summary, df, metric_col)
    if "match_method" in df.columns:
        summary["match_method_counts"] = {
            str(k): int(v) for k, v in df["match_method"].fillna("unmatched").value_counts(dropna=False).items()
        }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=False)
        handle.write("\n")

    with (run_dir / "config_used.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_serializable_dict(), handle, sort_keys=False)

    print(f"PPO checkpoint evaluation complete: {run_dir}")
    print("Summary:", summary)
    if HAS_MATPLOTLIB:
        print(f"Summary plots: {run_dir / 'plots'}")
        print(f"Overlay plots: {run_dir / 'overlays'}")
    else:
        print("Matplotlib not available; skipped plot generation.")


if __name__ == "__main__":
    main()
