#!/usr/bin/env python
"""Select reward weights by behavior filters first, then mean_return ranking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd
import yaml


REQUIRED_COLUMNS = (
    "w1",
    "w2",
    "w3",
    "stop_lambda",
    "mean_return",
    "mean_assigned_bins",
    "mean_add_actions",
    "mean_stop_reward",
    "mean_final_best_add",
)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Filter reward-grid results by behavior constraints first, "
            "then rank by mean_return."
        )
    )
    parser.add_argument(
        "--results-csv",
        type=str,
        required=True,
        help="Path to reward grid-search results.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: <results_dir>/selection)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of top filtered rows to save (default: 20)",
    )
    parser.add_argument(
        "--min-assigned-bins",
        type=float,
        default=1.0,
        help="Require mean_assigned_bins >= this value (default: 1)",
    )
    parser.add_argument(
        "--max-assigned-bins",
        type=float,
        default=None,
        help="Optional cap: require mean_assigned_bins <= this value",
    )
    parser.add_argument(
        "--min-add-actions",
        type=float,
        default=1.0,
        help="Require mean_add_actions >= this value (default: 1)",
    )
    parser.add_argument(
        "--max-add-actions",
        type=float,
        default=None,
        help="Optional cap: require mean_add_actions <= this value",
    )
    parser.add_argument(
        "--final-best-add-min",
        type=float,
        default=-0.25,
        help=(
            "Require mean_final_best_add >= this value. "
            "Higher means stop closer to boundary (default: -0.25)"
        ),
    )
    parser.add_argument(
        "--final-best-add-max",
        type=float,
        default=0.05,
        help="Require mean_final_best_add <= this value (default: 0.05)",
    )
    parser.add_argument(
        "--max-stop-reward",
        type=float,
        default=None,
        help="Optional cap: require mean_stop_reward <= this value",
    )
    return parser


def _validate_input(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"results.csv missing required columns: {missing}")


def _filter_results(
    df: pd.DataFrame,
    min_assigned_bins: float,
    max_assigned_bins: float | None,
    min_add_actions: float,
    max_add_actions: float | None,
    final_best_add_min: float,
    final_best_add_max: float,
    max_stop_reward: float | None,
) -> pd.DataFrame:
    keep = (
        (df["mean_assigned_bins"] >= float(min_assigned_bins))
        & (df["mean_add_actions"] >= float(min_add_actions))
        & (df["mean_final_best_add"] >= float(final_best_add_min))
        & (df["mean_final_best_add"] <= float(final_best_add_max))
    )
    if max_assigned_bins is not None:
        keep = keep & (df["mean_assigned_bins"] <= float(max_assigned_bins))
    if max_add_actions is not None:
        keep = keep & (df["mean_add_actions"] <= float(max_add_actions))
    if max_stop_reward is not None:
        keep = keep & (df["mean_stop_reward"] <= float(max_stop_reward))
    return df.loc[keep].copy()


def main() -> None:
    args = _build_arg_parser().parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k must be > 0")
    if args.min_assigned_bins < 0:
        raise ValueError("--min-assigned-bins must be >= 0")
    if args.min_add_actions < 0:
        raise ValueError("--min-add-actions must be >= 0")
    if args.max_assigned_bins is not None and args.max_assigned_bins < args.min_assigned_bins:
        raise ValueError("--max-assigned-bins must be >= --min-assigned-bins")
    if args.max_add_actions is not None and args.max_add_actions < args.min_add_actions:
        raise ValueError("--max-add-actions must be >= --min-add-actions")
    if args.final_best_add_max < args.final_best_add_min:
        raise ValueError("--final-best-add-max must be >= --final-best-add-min")

    results_csv = Path(args.results_csv).expanduser().resolve()
    if not results_csv.exists():
        raise FileNotFoundError(f"results.csv not found: {results_csv}")

    out_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else (results_csv.parent / "selection").resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(results_csv)
    _validate_input(df)

    filtered = _filter_results(
        df=df,
        min_assigned_bins=float(args.min_assigned_bins),
        max_assigned_bins=None if args.max_assigned_bins is None else float(args.max_assigned_bins),
        min_add_actions=float(args.min_add_actions),
        max_add_actions=None if args.max_add_actions is None else float(args.max_add_actions),
        final_best_add_min=float(args.final_best_add_min),
        final_best_add_max=float(args.final_best_add_max),
        max_stop_reward=None if args.max_stop_reward is None else float(args.max_stop_reward),
    )

    filtered = filtered.sort_values("mean_return", ascending=False).reset_index(drop=True)
    top = filtered.head(int(args.top_k)).copy()

    top_csv = out_dir / "top_filtered_results.csv"
    top.to_csv(top_csv, index=False)

    summary = {
        "input_results_csv": str(results_csv),
        "n_input_rows": int(len(df)),
        "n_filtered_rows": int(len(filtered)),
        "n_saved_rows": int(len(top)),
        "filters": {
            "min_assigned_bins": float(args.min_assigned_bins),
            "max_assigned_bins": None if args.max_assigned_bins is None else float(args.max_assigned_bins),
            "min_add_actions": float(args.min_add_actions),
            "max_add_actions": None if args.max_add_actions is None else float(args.max_add_actions),
            "final_best_add_min": float(args.final_best_add_min),
            "final_best_add_max": float(args.final_best_add_max),
            "max_stop_reward": None if args.max_stop_reward is None else float(args.max_stop_reward),
        },
    }

    if len(top) > 0:
        best = top.iloc[0]
        best_weights = {
            "w1": float(best["w1"]),
            "w2": float(best["w2"]),
            "w3": float(best["w3"]),
            "stop_lambda": float(best["stop_lambda"]),
            "mean_return": float(best["mean_return"]),
            "mean_assigned_bins": float(best["mean_assigned_bins"]),
            "mean_add_actions": float(best["mean_add_actions"]),
            "mean_stop_reward": float(best["mean_stop_reward"]),
            "mean_final_best_add": float(best["mean_final_best_add"]),
        }
        with (out_dir / "best_weights_filtered.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(best_weights, handle, sort_keys=False)
        summary["best"] = best_weights
    else:
        summary["best"] = None

    with (out_dir / "selection_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    print(f"Input rows: {len(df)}")
    print(f"Filtered rows: {len(filtered)}")
    print(f"Saved top rows: {len(top)}")
    print(f"Selection outputs: {out_dir}")
    if len(top) == 0:
        print("No rows passed filters; relax thresholds.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - CLI failure path
        print(f"ERROR: {exc}", file=sys.stderr)
        raise

