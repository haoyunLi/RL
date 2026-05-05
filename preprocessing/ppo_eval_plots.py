from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False
    plt = None  # type: ignore[assignment]


def _slug(value: str) -> str:
    text = str(value).strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text[:80] if text else "unknown"


def _choose_overlay_indices(df: pd.DataFrame, n_pick: int, selection: str, seed: int) -> list[int]:
    if n_pick <= 0 or len(df) == 0:
        return []
    n_pick = min(int(n_pick), len(df))
    if selection == "first":
        return list(range(n_pick))
    if selection == "top_reward" and "total_reward" in df.columns:
        scores = pd.to_numeric(df["total_reward"], errors="coerce").fillna(-np.inf).to_numpy(dtype=np.float64)
        order = np.argsort(-scores)
        return [int(i) for i in order[:n_pick]]
    if selection == "best_iou" and "pred_iou" in df.columns:
        scores = pd.to_numeric(df["pred_iou"], errors="coerce").fillna(-np.inf).to_numpy(dtype=np.float64)
        order = np.argsort(-scores)
        return [int(i) for i in order[:n_pick]]
    if selection == "worst_iou" and "pred_iou" in df.columns:
        scores = pd.to_numeric(df["pred_iou"], errors="coerce").fillna(np.inf).to_numpy(dtype=np.float64)
        order = np.argsort(scores)
        return [int(i) for i in order[:n_pick]]
    rng = np.random.default_rng(int(seed))
    choices = rng.choice(len(df), size=n_pick, replace=False)
    return [int(i) for i in np.asarray(choices, dtype=np.int64)]


def _estimate_grid_step(coords: np.ndarray) -> float:
    vals = np.unique(np.asarray(coords, dtype=np.float64))
    if vals.size <= 1:
        return 2.0
    diffs = np.diff(np.sort(vals))
    diffs = diffs[np.isfinite(diffs) & (diffs > 1.0e-6)]
    if diffs.size == 0:
        return 2.0
    return float(np.median(diffs))


def _build_gt_contour_grid(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
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


def _format_overlay_title(row: dict[str, Any], method_label: str) -> str:
    if "total_reward" in row:
        reward = _finite_float_or_none(row.get("total_reward", np.nan))
        reward_text = "nan" if reward is None else f"{reward:.2f}"
        lines = [
            f"cell={row['cell_id']}  reward={reward_text}  "
            f"assigned={row['n_assigned_bins']}/{row['n_candidate_bins']}",
            f"GT match={row.get('match_method', 'none')}",
        ]
    else:
        lines = [
            f"cell={row['cell_id']}  {method_label.lower()}={row.get('matched_pred_cell_id', 'unmatched')}  "
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


def save_summary_plots(df: pd.DataFrame, run_dir: Path, *, method_label: str) -> list[str]:
    if not HAS_MATPLOTLIB:
        return []

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []

    if "total_reward" in df.columns:
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

    if "total_reward" in df.columns:
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
        iou = pd.to_numeric(df["pred_iou"], errors="coerce").to_numpy(dtype=np.float64)
        iou = iou[np.isfinite(iou)]
        if iou.size > 0:
            fig, ax = plt.subplots(figsize=(7.5, 5.0))
            ax.hist(iou, bins=30, color="#457B9D", alpha=0.9)
            ax.set_title(f"{method_label} vs GT IoU Distribution")
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
            ax.set_title(f"{method_label} vs GT IoU CDF")
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
        dice = pd.to_numeric(df["pred_dice"], errors="coerce").to_numpy(dtype=np.float64)
        dice = dice[np.isfinite(dice)]
        if dice.size > 0:
            fig, ax = plt.subplots(figsize=(7.5, 5.0))
            ax.hist(dice, bins=30, color="#1D3557", alpha=0.9)
            ax.set_title(f"{method_label} vs GT Dice Distribution")
            ax.set_xlabel("Dice")
            ax.set_ylabel("Episode Count")
            fig.tight_layout()
            p = plots_dir / "pred_gt_dice_hist.png"
            fig.savefig(p, dpi=180)
            plt.close(fig)
            saved.append(str(p))

    return saved


def save_overlay_plots(
    *,
    records: list[Any],
    df: pd.DataFrame,
    run_dir: Path,
    max_cells: int,
    selection: str,
    seed: int,
    method_label: str,
) -> list[str]:
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
                ax.plot([], [], color="#1D3557", linewidth=3.0, alpha=1.0, label="GT cell outline")
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
        ax.set_title(_format_overlay_title(row, method_label), fontsize=10.5, pad=10)
        ax.legend(loc="best", fontsize=8, frameon=False)
        fig.tight_layout()

        out = overlays_dir / f"overlay_{idx:04d}_{_slug(str(row['cell_id']))}.png"
        fig.savefig(out, dpi=180, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(out))

    return saved
