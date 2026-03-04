#!/usr/bin/env python
"""Run reproducible reward-weight grid search from a YAML config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hd_cell_rl.reward_grid_search import run_reward_grid_search_from_config


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run reproducible reward-weight grid search")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file (configs/*.yaml)",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()

    run_dir = run_reward_grid_search_from_config(config_path=args.config)

    summary_path = run_dir / "summary.json"
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    print(f"Reward grid search complete: {run_dir}")
    print(
        "Best:",
        {
            "w1": summary["best"]["w1"],
            "w2": summary["best"]["w2"],
            "w3": summary["best"]["w3"],
            "stop_lambda": summary["best"]["stop_lambda"],
            "objective_value": summary["best"]["objective_value"],
        },
    )


if __name__ == "__main__":
    main()
