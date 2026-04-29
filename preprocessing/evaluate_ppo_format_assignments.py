#!/usr/bin/env python
"""Run PPO-format evaluation for an existing cell-assignment CSV."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from preprocessing.ppo_format_assignment_eval import run_ppo_format_assignment_evaluation

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate existing assignments with the PPO-format evaluator.")
    parser.add_argument("--assignments_csv", required=True)
    parser.add_argument("--method_name", required=True)
    parser.add_argument("--method_label", required=True)
    parser.add_argument("--nuclear_source", default="external_ppo_aligned")
    parser.add_argument("--external_nuclear_bins_path", default=None)
    parser.add_argument("--pipeline_config_json", default=None)
    parser.add_argument("--ppo_eval_run_dir", required=True)
    parser.add_argument("--eval_run_name", required=True)
    parser.add_argument("--eval_output_root", default="runs")
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


def _load_pipeline_config(path_value: str | None) -> dict[str, Any]:
    if path_value is None:
        return {}
    path = Path(path_value).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"pipeline config JSON not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()

    assignments_csv = Path(str(args.assignments_csv)).expanduser().resolve()
    if not assignments_csv.exists():
        raise FileNotFoundError(f"assignments CSV not found: {assignments_csv}")

    external_nuclear_bins_path = (
        None
        if args.external_nuclear_bins_path is None
        else Path(str(args.external_nuclear_bins_path)).expanduser().resolve()
    )
    pipeline_config = _load_pipeline_config(args.pipeline_config_json)

    run_dir = run_ppo_format_assignment_evaluation(
        assignments_csv=assignments_csv,
        method_name=str(args.method_name),
        method_label=str(args.method_label),
        nuclear_source=str(args.nuclear_source),
        external_nuclear_bins_path=external_nuclear_bins_path,
        args=args,
        pipeline_config=pipeline_config,
    )
    logger.info("Evaluation complete: %s", run_dir)


if __name__ == "__main__":
    main()
