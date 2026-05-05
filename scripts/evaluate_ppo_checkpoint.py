#!/usr/bin/env python
"""Evaluate a trained PPO checkpoint on episode data."""

from __future__ import annotations

import argparse
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


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hd_cell_rl.ppo_checkpoint import load_actor_critic_checkpoint, load_checkpoint_payload
from hd_cell_rl.ppo_training import (
    AddStopCellEnv,
    EpisodeDataset,
    PPOTrainingConfig,
    PLANNER_MODE_BALANCED,
    PLANNER_MODE_STOP,
    _observation_to_tensors,
    _observation_with_compact_streak,
    _observation_without_stop_action,
    _planner_logit_bias_from_action_features,
    load_ppo_training_config,
)
from preprocessing.ppo_format_assignment_eval import (
    _add_numeric_summary,
    annotate_records_with_gene_correlation,
)
from preprocessing.ppo_eval_metrics import (
    build_episode_nuclear_barcode_map,
    collect_gt_nuclear_candidates,
    compute_spatial_overlap_metrics,
    load_episode_build_bins_path,
    load_gt_bins_for_cells,
    match_episode_cells_by_nuclear_overlap,
)
from preprocessing.ppo_eval_plots import HAS_MATPLOTLIB, save_overlay_plots, save_summary_plots

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
    config: PPOTrainingConfig,
) -> EpisodeEvalRecord:
    env = AddStopCellEnv(context)
    obs, _ = env.reset()

    total_reward = 0.0
    n_steps = 0
    final_info: dict[str, Any] | None = None
    terminated = False
    truncated = False
    action_trace: list[dict[str, Any]] = []
    current_mode = PLANNER_MODE_BALANCED
    current_compact_streak = 0
    steps_since_planner = int(config.planner_interval)

    while True:
        step_index = int(obs["step_index"])
        planner_mode_name = None
        planner_mode_prob = None
        if bool(config.planner_enabled) and (not action_trace or steps_since_planner >= int(config.planner_interval)):
            planner_obs = _observation_with_compact_streak(obs, current_compact_streak)
            global_t = torch.as_tensor(
                np.asarray(planner_obs["global_features"], dtype=np.float32),
                device=device,
            ).unsqueeze(0)
            with torch.inference_mode():
                planner_dist = model.planner_distribution(global_t)
                planner_probs = planner_dist.probs.squeeze(0).detach().cpu().numpy()
            if policy_mode == "greedy":
                current_mode = int(np.argmax(planner_probs))
            else:
                prob_sum = float(np.sum(planner_probs))
                if prob_sum <= 0.0 or not np.isfinite(prob_sum):
                    raise RuntimeError("non-finite planner probabilities during evaluation")
                current_mode = int(rng.choice(planner_probs.shape[0], p=planner_probs / prob_sum))
            steps_since_planner = 0
            planner_mode_name = str(config.planner_modes[current_mode])
            planner_mode_prob = float(planner_probs[current_mode])
            current_compact_streak = current_compact_streak + 1 if planner_mode_name == "compact" else 0

        if bool(config.planner_enabled):
            has_add = bool(np.any(np.asarray(obs["action_mask"], dtype=bool)[1:]))
            if current_mode == PLANNER_MODE_STOP or not has_add:
                probs = np.zeros_like(np.asarray(obs["action_mask"], dtype=np.float32), dtype=np.float64)
                probs[0] = 1.0
                action = 0
            else:
                low_obs = _observation_with_compact_streak(
                    _observation_without_stop_action(obs),
                    current_compact_streak,
                )
                g_t, a_t, m_t = _observation_to_tensors(low_obs, device=device)
                mode_t = torch.as_tensor([current_mode], device=device, dtype=torch.long)
                action_bias = _planner_logit_bias_from_action_features(a_t, mode_t, config)
                with torch.inference_mode():
                    dist, _ = model(g_t, a_t, m_t, action_logit_bias=action_bias)
                    probs = dist.probs.squeeze(0).detach().cpu().numpy()
                if policy_mode == "greedy":
                    action = int(np.argmax(probs))
                else:
                    prob_sum = float(np.sum(probs))
                    if prob_sum <= 0.0 or not np.isfinite(prob_sum):
                        raise RuntimeError("non-finite policy probabilities during evaluation")
                    action = int(rng.choice(probs.shape[0], p=probs / prob_sum))
                if action == 0:
                    raise RuntimeError("planner-controlled evaluation selected STOP despite STOP being masked")
        else:
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
                "planner_mode": planner_mode_name if planner_mode_name is not None else (
                    str(config.planner_modes[current_mode]) if bool(config.planner_enabled) else None
                ),
                "planner_mode_probability": planner_mode_prob,
                "compact_streak": int(current_compact_streak),
                "terminated_after_action": bool(term),
                "truncated_after_action": bool(trunc),
                "n_assigned_bins_after": int(info.get("n_assigned_bins", 0)),
            }
        )
        if bool(config.planner_enabled) and action != 0:
            steps_since_planner += 1
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

    matched_raw, matched_gt_ids = match_episode_cells_by_nuclear_overlap(
        episode_nuclear_by_cell=episode_nuclear_by_cell,
        barcode_to_target_cells=barcode_to_gt_cells,
        min_overlap_frac=min_overlap_frac,
        min_overlap_bins=min_overlap_bins,
        match_method="nuclear_overlap",
    )
    provisional: dict[str, dict[str, Any]] = {}
    for rec in records:
        cell_id = str(rec.metrics["cell_id"])
        ep_nuclear = episode_nuclear_by_cell.get(cell_id, set())
        raw = matched_raw.get(cell_id, {})
        provisional[cell_id] = {
            "matched_gt_cell_id": raw.get("matched_cell_id"),
            "match_method": raw.get("match_method", "unmatched"),
            "episode_nuclear_bin_count": int(len(ep_nuclear)),
            "nuclear_overlap_bins": int(raw.get("nuclear_overlap_bins", 0)),
            "nuclear_overlap_frac_episode": float(raw.get("nuclear_overlap_frac_episode", np.nan)),
        }

    gt_nuclear_by_cell, gt_nuclear_xy_by_cell = load_gt_bins_for_cells(
        csv_path=gt_nuclear_bins_path,
        matched_cell_ids=matched_gt_ids,
    )
    gt_cell_by_cell, gt_cell_xy_by_cell = load_gt_bins_for_cells(
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
        overlap_metrics = None

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
                overlap_metrics = compute_spatial_overlap_metrics(pred_bars, gt_cell)

        meta.update(
            {
                "gt_nuclear_bin_count": int(len(gt_nuclear)),
                "gt_cell_bin_count": int(len(gt_cell)),
                "nuclear_overlap_frac_gt": float(overlap_frac_gt),
                "pred_gt_intersection_bins": 0 if overlap_metrics is None else int(overlap_metrics["intersection"]),
                "pred_iou": np.nan if overlap_metrics is None else float(overlap_metrics["iou"]),
                "pred_dice": np.nan if overlap_metrics is None else float(overlap_metrics["dice"]),
                "pred_precision": np.nan if overlap_metrics is None else float(overlap_metrics["precision"]),
                "pred_recall": np.nan if overlap_metrics is None else float(overlap_metrics["recall"]),
                "pred_f1": np.nan if overlap_metrics is None else float(overlap_metrics["f1"]),
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

    payload = load_checkpoint_payload(ckpt_path)

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

    model, payload = load_actor_critic_checkpoint(ckpt_path, config, device=device, payload=payload)

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
                    config=config,
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

    summary_plots = save_summary_plots(df=df, run_dir=run_dir, method_label="PPO")
    overlay_plots = save_overlay_plots(
        records=episode_records,
        df=df,
        run_dir=run_dir,
        max_cells=int(args.overlay_max_cells),
        selection=str(args.overlay_selection),
        seed=int(args.seed),
        method_label="PPO",
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
    if bool(config.planner_enabled):
        planner_modes = [
            str(step.get("planner_mode"))
            for rec in episode_records
            for step in rec.action_trace
            if step.get("planner_mode") is not None
        ]
        compact_streaks = [
            int(step.get("compact_streak", 0))
            for rec in episode_records
            for step in rec.action_trace
            if step.get("planner_mode") is not None
        ]
        summary["planner_mode_counts"] = {
            str(k): int(v) for k, v in pd.Series(planner_modes, dtype="string").value_counts().items()
        }
        summary["compact_streak_mean"] = float(np.mean(compact_streaks)) if compact_streaks else 0.0
        summary["compact_streak_max"] = int(max(compact_streaks)) if compact_streaks else 0
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
