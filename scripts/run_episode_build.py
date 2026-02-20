#!/usr/bin/env python
"""Build RL episodes from nuclei/bin tables using a reproducible YAML config.

Usage:
    python scripts/run_episode_build.py --config configs/episode_build.template.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


# Ensure local package import works when the script is executed directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hd_cell_rl.episode_build import run_episode_build_from_config


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run reproducible episode-build pipeline")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file (configs/*.yaml)",
    )
    parser.add_argument(
        "--limit-nuclei",
        type=int,
        default=None,
        help="Optional debug limit: randomly sample this many nuclei rows",
    )
    parser.add_argument(
        "--limit-bins",
        type=int,
        default=None,
        help="Optional debug limit: randomly sample this many bins rows",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()

    run_dir = run_episode_build_from_config(
        config_path=args.config,
        limit_nuclei=args.limit_nuclei,
        limit_bins=args.limit_bins,
    )

    summary_path = run_dir / "summary.json"
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    print(f"Episode build complete: {run_dir}")
    print(
        "Summary:",
        {
            "n_episodes": summary["n_episodes"],
            "total_candidate_bins": summary["total_candidate_bins"],
            "empty_episode_count": summary["empty_episode_count"],
            "candidate_bins_mean": summary["candidate_bins_mean"],
        },
    )


if __name__ == "__main__":
    main()
