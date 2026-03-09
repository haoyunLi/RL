#!/usr/bin/env python
"""Run two-stage reward grid search: coarse grid then fine grid around coarse best."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import tempfile
import sys

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hd_cell_rl.reward_grid_search import (
    RewardGridSearchConfig,
    load_reward_grid_search_config,
    prepare_reward_grid_context,
    run_reward_grid_search,
)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run two-stage reward grid search")
    parser.add_argument("--config", type=str, required=True, help="Base YAML config path")
    parser.add_argument(
        "--coarse-max-episodes",
        type=int,
        default=200,
        help="Override for coarse stage run.max_episodes (default: 200)",
    )
    parser.add_argument(
        "--coarse-step-multiplier",
        type=float,
        default=2.0,
        help="Multiply base grid step by this factor for coarse stage (default: 2.0)",
    )
    parser.add_argument(
        "--fine-max-episodes",
        type=int,
        default=None,
        help="Optional override for fine stage run.max_episodes",
    )
    parser.add_argument(
        "--fine-span",
        type=float,
        default=0.1,
        help="Fine stage half-width around coarse best for each weight",
    )
    return parser


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("config root must be a mapping")
    return data


def _write_yaml(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def _axis_value(best_payload: dict, axis_name: str) -> float:
    key = "stop_lambda" if axis_name == "stop_lambda" else axis_name
    return float(best_payload[key])


def _make_fine_axis(
    orig_axis: dict,
    center: float,
    fine_span: float,
) -> dict[str, float]:
    orig_start = float(orig_axis["start"])
    orig_stop = float(orig_axis["stop"])
    orig_step = float(orig_axis["step"])
    if orig_stop < orig_start:
        raise ValueError("search axis stop must be >= start")
    if orig_step <= 0:
        raise ValueError("search axis step must be > 0")

    lo = max(orig_start, center - fine_span)
    hi = min(orig_stop, center + fine_span)
    if hi < lo:
        lo = hi = max(orig_start, min(orig_stop, center))

    return {
        "start": round(lo, 6),
        "stop": round(hi, 6),
        "step": round(orig_step, 6),
    }


def _assert_reusable_context(coarse: RewardGridSearchConfig, fine: RewardGridSearchConfig) -> None:
    """Ensure fine stage can safely reuse coarse prepared context."""
    checks = (
        ("episodes_index_path", coarse.episodes_index_path, fine.episodes_index_path),
        ("reference_path", coarse.reference_path, fine.reference_path),
        ("reference_format", coarse.reference_format, fine.reference_format),
        ("reference_array_key", coarse.reference_array_key, fine.reference_array_key),
        ("reference_genes_key", coarse.reference_genes_key, fine.reference_genes_key),
        ("nuclei_path", coarse.nuclei_path, fine.nuclei_path),
        ("epsilon", coarse.epsilon, fine.epsilon),
        ("max_episodes", coarse.max_episodes, fine.max_episodes),
    )
    for name, left, right in checks:
        if left != right:
            raise ValueError(
                f"cannot reuse prepared context: coarse and fine differ in {name}: {left!r} != {right!r}"
            )


def main() -> None:
    args = _build_arg_parser().parse_args()
    if args.fine_span <= 0:
        raise ValueError("--fine-span must be > 0")
    if args.coarse_max_episodes <= 0:
        raise ValueError("--coarse-max-episodes must be > 0")
    if args.coarse_step_multiplier < 1.0:
        raise ValueError("--coarse-step-multiplier must be >= 1.0")
    if args.fine_max_episodes is not None and args.fine_max_episodes <= 0:
        raise ValueError("--fine-max-episodes must be > 0")

    base_config_path = Path(args.config).expanduser().resolve()
    if not base_config_path.exists():
        raise FileNotFoundError(f"config not found: {base_config_path}")

    base = _load_yaml(base_config_path)
    run_cfg = base.setdefault("run", {})
    if not isinstance(run_cfg, dict):
        raise ValueError("run section must be a mapping")
    base_name = str(run_cfg.get("name", "reward_grid_search")).strip() or "reward_grid_search"

    coarse_cfg = copy.deepcopy(base)
    coarse_cfg.setdefault("run", {})
    coarse_cfg["run"]["name"] = f"{base_name}_coarse"
    coarse_cfg["run"]["max_episodes"] = int(args.coarse_max_episodes)
    coarse_search = coarse_cfg.setdefault("search", {})
    if not isinstance(coarse_search, dict):
        raise ValueError("search section must be a mapping")
    for axis_name in ("w1", "w2", "w3", "stop_lambda"):
        axis = coarse_search.get(axis_name)
        if not isinstance(axis, dict):
            raise ValueError(f"missing search axis in config: {axis_name}")
        base_step = float(axis["step"])
        axis["step"] = round(base_step * float(args.coarse_step_multiplier), 6)

    with tempfile.TemporaryDirectory(prefix="reward_two_stage_") as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        coarse_cfg_path = tmp_dir_path / "coarse.yaml"
        _write_yaml(coarse_cfg_path, coarse_cfg)

        coarse_cfg_resolved = load_reward_grid_search_config(coarse_cfg_path)
        prepared_context = prepare_reward_grid_context(coarse_cfg_resolved)
        coarse_run_dir = run_reward_grid_search(coarse_cfg_resolved, prepared_context=prepared_context)
        coarse_summary_path = coarse_run_dir / "summary.json"
        with coarse_summary_path.open("r", encoding="utf-8") as handle:
            coarse_summary = json.load(handle)

        best = coarse_summary["best"]
        fine_cfg = copy.deepcopy(base)
        fine_cfg.setdefault("run", {})
        fine_cfg["run"]["name"] = f"{base_name}_fine"
        if args.fine_max_episodes is not None:
            fine_cfg["run"]["max_episodes"] = int(args.fine_max_episodes)
        else:
            fine_cfg["run"]["max_episodes"] = int(args.coarse_max_episodes)

        search_cfg = fine_cfg.setdefault("search", {})
        if not isinstance(search_cfg, dict):
            raise ValueError("search section must be a mapping")

        for axis_name in ("w1", "w2", "w3", "stop_lambda"):
            if axis_name not in search_cfg or not isinstance(search_cfg[axis_name], dict):
                raise ValueError(f"missing search axis in config: {axis_name}")
            center = _axis_value(best, axis_name)
            search_cfg[axis_name] = _make_fine_axis(
                orig_axis=search_cfg[axis_name],
                center=center,
                fine_span=float(args.fine_span),
            )

        fine_cfg_path = tmp_dir_path / "fine.yaml"
        _write_yaml(fine_cfg_path, fine_cfg)
        fine_cfg_resolved = load_reward_grid_search_config(fine_cfg_path)
        _assert_reusable_context(coarse_cfg_resolved, fine_cfg_resolved)
        fine_run_dir = run_reward_grid_search(fine_cfg_resolved, prepared_context=prepared_context)

        two_stage_summary = {
            "coarse_run_dir": str(coarse_run_dir),
            "fine_run_dir": str(fine_run_dir),
            "coarse_best": best,
            "fine_span": float(args.fine_span),
        }
        with (fine_run_dir / "two_stage_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(two_stage_summary, handle, indent=2)
            handle.write("\n")

    print(f"Two-stage reward grid search complete: {fine_run_dir}")
    print(
        "Coarse best:",
        {
            "w1": best["w1"],
            "w2": best["w2"],
            "w3": best["w3"],
            "stop_lambda": best["stop_lambda"],
            "objective_value": best["objective_value"],
        },
    )


if __name__ == "__main__":
    main()
