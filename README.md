# HD Cell RL Scaffold

This repository now includes a production-style starter framework to model **one nucleus-centered cell-assignment problem per RL episode**.

## What Is Implemented

- Environment around each nucleus (one cell per episode).
- Candidate-bin filtering by geometry with your requested rule:
  - center-distance cap: `r <= 80um` (configurable)
  - optional radial band: `|rc - r| <= 80um` (configurable)
- State fields required now:
  - `membership_mask` (`0/1`, per candidate bin)
  - `belonged_bin_ids`
  - `processed_mask` (`f` state)
  - `candidate_expression`
  - `candidate_bin_xy_um`
  - `nucleus_center_xy_um`, `nucleus_radius_um`
  - `cell_type` placeholder
- Actions implemented now:
  - `ADD_BIN`
  - `REMOVE_BIN`
  - `STOP`
- Update logic after each action (`step`) is implemented.
- Reward and policy are clean plug-in interfaces (placeholder behavior by default).

## Project Structure

- `hd_cell_rl/config.py`
  - Runtime config: distance/radius rules, strict validation, max steps.
- `hd_cell_rl/models.py`
  - Typed data objects (`BinRecord`, `NucleusRecord`, `CellEpisodeData`).
- `hd_cell_rl/actions.py`
  - Action enum and typed action payload (`add/remove/stop`).
- `hd_cell_rl/spatial_index.py`
  - Radius-query index (currently brute-force, easy to replace).
- `hd_cell_rl/builder.py`
  - Converts nucleus+bin tables into per-cell episodes.
- `hd_cell_rl/environment.py`
  - Core `CellAssignmentEnv` with `reset`, `step`, state update, validation.
- `hd_cell_rl/state.py`
  - Typed immutable `CellAssignmentState` snapshot with `save_npz`/`load_npz`.
- `hd_cell_rl/reward.py`
  - Reward interface plus `PosteriorAddBinReward` (theta/LL/posterior + distance/overlap penalties) and `ZeroReward`.
  - Includes optimized single-bin add reward helper (`add_reward_for_bin`) for fast ADD action scoring.
- `hd_cell_rl/policy.py`
  - Policy interface and `RandomPolicy` debugging baseline.
- `hd_cell_rl/episode_build.py`
  - Reproducible episode-build pipeline (config parsing, schema checks, artifact writing).
- `examples/minimal_demo.py`
  - End-to-end runnable demo with synthetic data.
- `tests/test_environment.py`
  - Unit tests for candidate selection and action transitions.
- `scripts/run_episode_build.py`
  - CLI entrypoint to build episodes from `configs/*.yaml`.
- `configs/episode_build.template.yaml`
  - Starter config to run repeatable episode-build jobs.
- `jobs/run_crop_visium_hd.sbatch`
  - Slurm job for Visium HD cropping step.
- `jobs/run_cellpose_sam.sbatch`
  - Slurm job for Cellpose-SAM segmentation step.

## Quick Start

Run demo:

```bash
python examples/minimal_demo.py
```

Run tests:

```bash
python -m unittest tests/test_environment.py
```

Build episodes from config:

```bash
python scripts/run_episode_build.py --config configs/episode_build.template.yaml
```

## State Snapshot Usage

Use explicit state object from the environment:

```python
state = env.current_state()
print(state.membership_mask, state.belonged_bin_ids, state.processed_mask)
env.save_state_npz(\"workspace_outputs/human_colorectal/intermediate/state_snapshot.npz\")
```

## Posterior Reward Usage

```python
import numpy as np
from hd_cell_rl import CellAssignmentEnv, PosteriorAddBinReward

reward_fn = PosteriorAddBinReward(
    reference_counts=C_kg,                        # shape (K, G)
    candidate_bin_ids=[b.bin_id for b in episode.candidate_bins],
    candidate_expression=np.vstack([b.expression for b in episode.candidate_bins]),  # (B, G)
    candidate_bin_xy_um=np.asarray([(b.x_um, b.y_um) for b in episode.candidate_bins]),  # (B, 2)
    nucleus_center_xy_um=np.asarray([episode.nucleus.center_x_um, episode.nucleus.center_y_um]),  # (2,)
    other_nuclei_center_xy_um=other_centers,      # optional shape (N, 2)
    epsilon=1e-8,
    r_max_um=80.0,
    w1=1.0,
    w2=1.0,
    w3=1.0,
    stop_lambda=1.0,
)

env = CellAssignmentEnv(episode_data=episode, reward_fn=reward_fn)
```

## Data Location Convention

- `Human_Colorectal/`: input dataset only (treat as read-only source data)
- `workspace_outputs/human_colorectal/`: reusable generated artifacts (intermediate tables, debug snapshots, etc.)

## Episode Build Outputs

Each runner execution creates a new run folder like `runs/<run_name>_<UTC timestamp>/` with:

- `config/config_resolved.yaml`: exact resolved config used for this run
- `config/metadata.json`: runtime metadata (timestamp, platform, python, git commit if available)
- `logs/steps.jsonl`: step-by-step build log for debugging failures quickly
- `states/*.npz`: per-cell state snapshots (nucleus info + candidate bins + expression)
- `summary.json`: aggregate statistics (episode counts, empty episodes, bin stats)
- `episodes_index.csv`: per-episode table and artifact file paths
