#!/usr/bin/env python
"""Evaluate one fixed (w1, w2, w3, stop_lambda) setting on a chosen episodes index."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hd_cell_rl.reward_grid_search import GridAxis, RewardGridSearchConfig, load_reward_grid_search_config, run_reward_grid_search


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate one fixed reward-weight setting")
    parser.add_argument("--config", type=str, required=True, help="Base reward config YAML")
    parser.add_argument("--episodes-index-path", type=str, required=True, help="Episodes index CSV to evaluate on")
    parser.add_argument("--w1", type=float, required=True)
    parser.add_argument("--w2", type=float, required=True)
    parser.add_argument("--w3", type=float, required=True)
    parser.add_argument("--stop-lambda", type=float, required=True)
    parser.add_argument("--run-name", type=str, default="reward_weights_eval")
    parser.add_argument("--max-episodes", type=int, default=None, help="Optional cap for faster validation")
    parser.add_argument("--n-workers", type=int, default=1, help="Parallel workers for evaluation (default: 1)")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for episode sampling (default: use config seed)",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    if args.w1 <= 0 or args.w2 <= 0 or args.w3 <= 0 or args.stop_lambda <= 0:
        raise ValueError("weights must be > 0")
    if args.n_workers <= 0:
        raise ValueError("--n-workers must be > 0")
    if args.max_episodes is not None and args.max_episodes <= 0:
        raise ValueError("--max-episodes must be > 0")

    base = load_reward_grid_search_config(args.config)
    episodes_index = Path(args.episodes_index_path).expanduser().resolve()
    if not episodes_index.exists():
        raise FileNotFoundError(f"episodes index not found: {episodes_index}")

    config = RewardGridSearchConfig(
        run_name=str(args.run_name).strip() or "reward_weights_eval",
        output_root=base.output_root,
        seed=int(args.seed) if args.seed is not None else base.seed,
        n_workers=int(args.n_workers),
        episodes_index_path=episodes_index,
        reference_path=base.reference_path,
        reference_format=base.reference_format,
        reference_array_key=base.reference_array_key,
        reference_genes_key=base.reference_genes_key,
        reference_cell_type_column=base.reference_cell_type_column,
        reference_gene_mode=base.reference_gene_mode,
        reference_gene_prefix=base.reference_gene_prefix,
        reference_gene_columns=base.reference_gene_columns,
        nuclei_path=base.nuclei_path,
        nuclei_format=base.nuclei_format,
        nuclei_columns=base.nuclei_columns,
        max_episodes=int(args.max_episodes) if args.max_episodes is not None else base.max_episodes,
        objective=base.objective,
        epsilon=base.epsilon,
        r_max_um=base.r_max_um,
        normalize_expression_zscore=base.normalize_expression_zscore,
        zscore_delta=base.zscore_delta,
        w1_axis=GridAxis(start=float(args.w1), stop=float(args.w1), step=1.0),
        w2_axis=GridAxis(start=float(args.w2), stop=float(args.w2), step=1.0),
        w3_axis=GridAxis(start=float(args.w3), stop=float(args.w3), step=1.0),
        stop_lambda_axis=GridAxis(start=float(args.stop_lambda), stop=float(args.stop_lambda), step=1.0),
    )

    run_dir = run_reward_grid_search(config)
    print(f"Fixed-weight evaluation complete: {run_dir}")


if __name__ == "__main__":
    main()
