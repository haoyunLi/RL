"""Minimal demo for the HD spatial-cell RL scaffold.

Run:
    python examples/minimal_demo.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

# Ensure package import works when running this script directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hd_cell_rl import (
    Action,
    CellAssignmentEnv,
    EnvironmentConfig,
    NucleusRecord,
    BinRecord,
    build_episodes,
)


def make_demo_data() -> tuple[list[NucleusRecord], list[BinRecord]]:
    """Create one nucleus and a few bins for sanity-check execution."""
    nuclei = [
        NucleusRecord(
            cell_id="cell_001",
            center_x_um=100.0,
            center_y_um=100.0,
            radius_um=10.0,
            cell_type=None,
        )
    ]

    # Bins are intentionally placed so some are candidates and one is outside 80um.
    bins = [
        BinRecord(bin_id="bin_A", x_um=95.0, y_um=100.0, expression=np.array([1.0, 0.1, 2.4])),
        BinRecord(bin_id="bin_B", x_um=120.0, y_um=90.0, expression=np.array([0.3, 5.1, 0.2])),
        BinRecord(bin_id="bin_C", x_um=80.0, y_um=140.0, expression=np.array([2.2, 1.7, 0.0])),
        BinRecord(bin_id="bin_D", x_um=300.0, y_um=300.0, expression=np.array([7.3, 0.0, 0.1])),
    ]

    return nuclei, bins


def main() -> None:
    """Build an episode, run a short action sequence, and print state updates."""
    nuclei, bins = make_demo_data()

    config = EnvironmentConfig(
        max_center_distance_um=80.0,
        radius_band_um=80.0,
        strict_action_validation=True,
        max_steps=12,
    )

    episodes = build_episodes(nuclei=nuclei, bins=bins, config=config)
    env = CellAssignmentEnv(episode_data=episodes[0], config=config)

    observation, info = env.reset()
    print("Initial candidate bins:", observation["candidate_bin_ids"])
    print("Initial belonged bins:", observation["belonged_bin_ids"])
    print("Initial valid action count:", len(info["valid_actions"]))

    scripted_actions = [Action.add("bin_A"), Action.add("bin_B"), Action.remove("bin_B"), Action.stop()]

    for action in scripted_actions:
        obs, reward, terminated, truncated, step_info = env.step(action)
        print(f"Action={action.kind.value:>9} target={action.bin_id} reward={reward:.2f}")
        print("Belonged bins:", obs["belonged_bin_ids"])
        print("Processed mask:", obs["processed_mask"].tolist())

        if terminated or truncated:
            print("Episode ended. terminated=", terminated, "truncated=", truncated)
            break

    state = env.current_state()
    state_dir = REPO_ROOT / "workspace_outputs" / "human_colorectal" / "intermediate"
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / "demo_state_snapshot.npz"
    env.save_state_npz(str(state_path))
    print("Saved typed state snapshot to:", state_path)
    print("State belonged bins:", list(state.belonged_bin_ids))

    print("Final summary:", {
        "assigned": obs["belonged_bin_ids"],
        "steps": obs["step_index"],
        "max_steps": obs["max_steps"],
        "n_candidates": step_info["n_candidate_bins"],
    })


if __name__ == "__main__":
    main()
