#!/usr/bin/env python
"""Run PPO policy training from a YAML config."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hd_cell_rl.ppo_training import run_ppo_training_from_config


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run PPO training for HD cell ADD/STOP environment")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to PPO training config YAML",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    result = run_ppo_training_from_config(args.config)
    print(f"PPO training complete: {result.run_dir}")
    print(
        "Summary:",
        {
            "updates_completed": result.updates_completed,
            "stopped_early": result.stopped_early,
            "best_moving_average_reward": result.best_moving_average_reward,
            "best_checkpoint_path": None if result.best_checkpoint_path is None else str(result.best_checkpoint_path),
        },
    )


if __name__ == "__main__":
    main()
