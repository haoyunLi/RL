# HD Cell RL Runbook

This runbook lists the execution order for preprocessing, episode building, reward search, PPO/GRPO training, checkpoint evaluation, and method-baseline comparison.
Each step depends on outputs from earlier steps, so run in order unless you already have the required files.

Current main training path:
- Build episodes once.
- Train with `jobs/run_ppo_training_full_grpo_cpu.sbatch`.
- Evaluate checkpoints with `jobs/run_evaluate_ppo_checkpoint.sbatch`.
- Compare Bin2Cell, SMURF, and STCS using the same PPO-evaluation cell set and the same PPO-aligned nuclear seeds.

Current ADD reward expression mixture:
```text
ADD reward =
  w1 * zscore(new posterior-confidence gain over frontier)
+ w5 * zscore(old bin-posterior compatibility over frontier)
- w2 * distance_penalty
- w3 * overlap_penalty
+ w4 * neighbor_support
```
Default full-GRPO config uses `w1: 0.45` and `w5: 0.10`.

## 1) Crop Visium HD image
This step extracts the tissue region used for all downstream processing.
The output image is the input for segmentation.
```bash
sbatch jobs/run_crop_visium_hd.sbatch
```
Expected output:
- `workspace_outputs/human_colorectal/intermediate/cropped_visium_hd_human_Colorectal.png`

## 2) Run Cellpose nuclear segmentation
This step runs nucleus segmentation on the cropped image and writes cell/nucleus masks and related artifacts.
These segmentation results are later mapped to official 2um bins.
```bash
sbatch jobs/run_cellpose_sam.sbatch
```
Expected output folder:
- `workspace_outputs/human_colorectal/intermediate/cellpose_sam_human_colorectal_output/`

## 3) Map pixel nuclear signal to official `square_002um` bins
This step converts pixel-level nuclear assignments into official Visium HD `square_002um` bin coordinates.
It also writes an overlay for quick visual QC.
```bash
sbatch jobs/run_pixel_to_square_002um_bins.sbatch
```
Expected outputs:
- `workspace_outputs/human_colorectal/intermediate/cropped_visium_hd_human_Colorectal_square_002um_nuclear_bins.csv.gz`
- `workspace_outputs/human_colorectal/intermediate/cropped_visium_hd_human_Colorectal_square_002um_nuclear_bins.summary.json`
- `workspace_outputs/human_colorectal/intermediate/cropped_visium_hd_human_Colorectal_square_002um_nuclear_bins_overlay.png`

## 4) Merge filtered expression metadata + nuclear annotation
This step merges official HD filtered-bin metadata/expression indexing with nuclear-bin annotation into one RL-ready table.
The expression source can be a 10x `.h5` file or a `filtered_feature_bc_matrix/` directory.
The merged table is the main structural input for episode construction.
```bash
sbatch jobs/run_merge_square_002um_expression_with_nuclear.sbatch
```
Expected outputs:
- `workspace_outputs/human_colorectal/intermediate/square_002um_filtered_nuclear/human_colorectal_square_002um_filtered_rl.metadata.parquet`
- `workspace_outputs/human_colorectal/intermediate/square_002um_filtered_nuclear/human_colorectal_square_002um_filtered_rl.selected_features.tsv.gz`
- `workspace_outputs/human_colorectal/intermediate/square_002um_filtered_nuclear/human_colorectal_square_002um_filtered_rl.selected_feature_indices.npy`
- `workspace_outputs/human_colorectal/intermediate/square_002um_filtered_nuclear/human_colorectal_square_002um_filtered_rl.manifest.json`

## 5) Build nuclei table (one row per nucleus)
This step builds one nucleus-center record per cell (`center_x_um`, `center_y_um`) from segmentation outputs.
These centers are used by reward distance and overlap terms.
```bash
sbatch jobs/run_build_nuclei_table.sbatch
```
Expected outputs:
- `workspace_outputs/human_colorectal/intermediate/human_colorectal_nuclei.parquet`
- `workspace_outputs/human_colorectal/intermediate/human_colorectal_nuclei.summary.json`

## 6) Build aligned scRNA reference counts (`C[k,g]`)
This step builds reference counts by cell type on the HD-overlap gene set, which drives posterior-based expression reward.
It also writes the selected HD gene allowlist used for alignment.
```bash
sbatch jobs/run_build_reference_counts_sct.sbatch
```
Expected outputs:
- `workspace_outputs/human_colorectal/intermediate/reference_sct/hd_selected_feature_names_unique.txt`
- `workspace_outputs/human_colorectal/intermediate/reference_sct/reference_counts_sct_tumor_aligned_hd_overlap_unique.npz`
- `workspace_outputs/human_colorectal/intermediate/reference_sct/reference_counts_sct_tumor_aligned_hd_overlap_unique.summary.json`

## 7) Build episodes
This step creates per-cell episode artifacts with candidate bins, geometry, and matrix references used by reward computation.
It is the main preprocessing stage before any grid search or evaluation.

Run episode build only:
```bash
sbatch jobs/run_episode_build.sbatch
```

Optional debug subset:
```bash
EPISODE_LIMIT_NUCLEI=800 sbatch jobs/run_episode_build.sbatch
```

Manual run:
```bash
python scripts/run_episode_build.py --config configs/episode_build.template.yaml
```

## 8) Run reward grid search
This step runs two-stage weight search (coarse then fine) to find candidate reward settings.
It uses your built episodes and current reward definition.

Use an existing episode run directory:
```bash
EP_RUN="runs/human_colorectal_episode_build_20260305T212758Z"
sbatch --export=ALL,EPISODE_RUN_DIR="$EP_RUN" jobs/run_reward_grid_search.sbatch
```

Default two-stage parameters in `jobs/run_reward_grid_search.sbatch`:
- `COARSE_MAX_EPISODES=200`
- `COARSE_STEP_MULTIPLIER=2.0`
- `FINE_MAX_EPISODES=$COARSE_MAX_EPISODES`
- `FINE_SPAN=0.1`

Optional override:
```bash
sbatch --export=ALL,EPISODE_RUN_DIR="$EP_RUN",COARSE_MAX_EPISODES=200,COARSE_STEP_MULTIPLIER=2.0,FINE_MAX_EPISODES=200,FINE_SPAN=0.1 jobs/run_reward_grid_search.sbatch
```

Manual two-stage run (ensure `inputs.episodes_index_path` points to your episode run):
```bash
python scripts/run_reward_grid_search.py --config configs/reward_grid_search.template.yaml
```

## 9) Select filtered best weights from grid-search results
This step removes behaviorally bad weight sets first (too few adds, too many adds, poor stop boundary), then ranks the rest by `mean_return`.
Use this output to choose candidates for fixed-weight evaluation.
```bash
python scripts/select_reward_weights.py \
  --results-csv runs/human_colorectal_reward_grid_search_coarse_20260309T171903Z/results.csv \
  --min-assigned-bins 50 \
  --max-assigned-bins 1500 \
  --min-add-actions 50 \
  --max-add-actions 1500 \
  --final-best-add-min -0.2 \
  --final-best-add-max 0.05 \
  --top-k 20
```

Selection outputs:
- `runs/human_colorectal_reward_grid_search_coarse_20260309T171903Z/selection/best_weights_filtered.yaml`
- `runs/human_colorectal_reward_grid_search_coarse_20260309T171903Z/selection/top_filtered_results.csv`
- `runs/human_colorectal_reward_grid_search_coarse_20260309T171903Z/selection/selection_summary.json`

Optional sbatch:
```bash
sbatch --export=ALL,RESULTS_CSV="runs/human_colorectal_reward_grid_search_coarse_20260309T171903Z/results.csv",MAX_ASSIGNED_BINS=1500,MAX_ADD_ACTIONS=1500 jobs/run_select_reward_weights.sbatch
```

## 10) Evaluate one fixed weight set
This step evaluates one chosen weight set on a random subset of episodes (for example, random 200) using the same reward logic as grid search.
Use different seeds to test stability across different random samples.

Manual run:
```bash
python scripts/evaluate_reward_weights.py \
  --config configs/reward_grid_search.template.yaml \
  --episodes-index-path runs/human_colorectal_episode_build_20260305T212758Z/episodes_index.csv \
  --w1 0.9 --w2 0.7 --w3 0.9 --stop-lambda 0.9 \
  --seed 123 \
  --max-episodes 200 --n-workers 32 \
  --run-name reward_eval_val_best
```

Outputs:
- `runs/reward_eval_val_best_*/results.csv`
- `runs/reward_eval_val_best_*/summary.json`
- `runs/reward_eval_val_best_*/config/config_resolved.yaml`

Optional sbatch:
```bash
sbatch --export=ALL,EPISODE_RUN_DIR="runs/human_colorectal_episode_build_20260305T212758Z",W1=0.9,W2=0.7,W3=0.9,STOP_LAMBDA=0.9,MAX_EPISODES=200,N_WORKERS=32,EVAL_SEED=123,RUN_NAME=reward_eval_val_best jobs/run_evaluate_reward_weights.sbatch
```

## 11) Train PPO / GRPO policy
The current preferred training job is full-GRPO on CPU. It uses `configs/ppo_training.full_grpo.yaml`, keeps the policy input dimensions at `global=11` and `action=13`, and uses the current `w1 + w5` expression reward mixture.

Run with the latest episode build:
```bash
sbatch jobs/run_ppo_training_full_grpo_cpu.sbatch
```

Run with a specific episode build:
```bash
EP_RUN="runs/human_colorectal_episode_build_YYYYMMDDTHHMMSSZ"
sbatch --export=ALL,EPISODE_RUN_DIR="$EP_RUN" jobs/run_ppo_training_full_grpo_cpu.sbatch
```

Useful overrides:
```bash
sbatch --export=ALL,EPISODE_RUN_DIR="$EP_RUN",BATCH_CELLS=200,MAX_UPDATES=600,RUN_NAME=human_colorectal_full_grpo jobs/run_ppo_training_full_grpo_cpu.sbatch
```

Alternative training configs:
- `configs/ppo_training.template.yaml`: PPO config.
- `configs/ppo_training.grpo.yaml`: PPO with optional same-cell group-relative auxiliary.
- `configs/ppo_training.full_grpo.yaml`: current full-GRPO training path.

Expected outputs:
- `runs/human_colorectal_full_grpo_*/checkpoints/best_model.pt`
- `runs/human_colorectal_full_grpo_*/checkpoints/final_model.pt`
- `runs/human_colorectal_full_grpo_*/summary.json`
- `runs/human_colorectal_full_grpo_*/logs/steps.jsonl`

## 12) Evaluate PPO / GRPO checkpoint
This evaluates a trained policy on the episode set, writes per-cell metrics, overlays, IoU distribution plots, and optional gene-correlation metrics against pseudo GT single-cell expression.

Run:
```bash
CHECKPOINT_PATH="runs/human_colorectal_full_grpo_YYYYMMDDTHHMMSSZ/checkpoints/best_model.pt"
sbatch --export=ALL,CHECKPOINT_PATH="$CHECKPOINT_PATH" jobs/run_evaluate_ppo_checkpoint.sbatch
```

Common overrides:
```bash
sbatch --export=ALL,\
CHECKPOINT_PATH="$CHECKPOINT_PATH",\
MAX_EPISODES=300,\
POLICY_MODE=greedy,\
RUN_DEVICE=cpu,\
EVAL_SEED=7,\
OVERLAY_MAX_CELLS=300,\
OVERLAY_SELECTION=top_reward \
jobs/run_evaluate_ppo_checkpoint.sbatch
```

Expected outputs:
- `runs/human_colorectal_ppo_eval_*/summary.json`
- `runs/human_colorectal_ppo_eval_*/per_episode.csv`
- `runs/human_colorectal_ppo_eval_*/overlays/`
- IoU distribution/CDF plots in the evaluation run directory.

Notes:
- Set `CHECKPOINT_PATH` manually; do not rely on implicit checkpoint discovery for final comparisons.
- Use the same `PPO_EVAL_RUN_DIR` from this step when comparing Bin2Cell, SMURF, and STCS so all methods use the same cell set.

## 13) Interactive PPO debug app
Use this after checkpoint evaluation to inspect one cell step-by-step, including assigned/frontier bins, GT outline, policy probability, reward decomposition, and selected-gene expression for clicked bins.

Run on a compute node:
```bash
PPO_EVAL_RUN_DIR="runs/human_colorectal_ppo_eval_YYYYMMDDTHHMMSSZ"
sbatch --export=ALL,PPO_EVAL_RUN_DIR="$PPO_EVAL_RUN_DIR",RUN_DEVICE=cpu jobs/run_ppo_debug_app.sbatch
```

The job prints a LAN URL and an SSH tunnel command. If direct LAN access is blocked, use the printed SSH tunnel and open `http://localhost:<port>`.

## 14) Run Bin2Cell / SMURF / STCS baselines with PPO-aligned nuclei
These method jobs are configured to use the same PPO-aligned nuclear bins instead of each tool's independent nuclear segmentation whenever possible. This keeps method comparison on the same nuclear seed level.

All three method jobs can evaluate against the same PPO checkpoint-evaluation cell set by setting `PPO_EVAL_RUN_DIR`.

### Bin2Cell
```bash
PPO_EVAL_RUN_DIR="runs/human_colorectal_ppo_eval_YYYYMMDDTHHMMSSZ"
sbatch --export=ALL,PPO_EVAL_RUN_DIR="$PPO_EVAL_RUN_DIR" jobs/run_bin2cell.sbatch
```

Key behavior:
- Uses `EXTERNAL_NUCLEAR_BINS_PATH` to inject PPO-aligned `labels_he`.
- Skips Bin2Cell's own H&E nuclear segmentation when external nuclei are provided.
- Still runs Bin2Cell expansion/GEX/combine steps.

Main outputs:
- `workspace_outputs/pseudo_human_colorectal/bin2cell_results_colorectal_0.25/human_colorectal_bin2cell_assignments.csv`
- `runs/human_colorectal_bin2cell_eval_*/summary.json`
- `runs/human_colorectal_bin2cell_eval_*/overlays/`

### SMURF
```bash
PPO_EVAL_RUN_DIR="runs/human_colorectal_ppo_eval_YYYYMMDDTHHMMSSZ"
sbatch --export=ALL,PPO_EVAL_RUN_DIR="$PPO_EVAL_RUN_DIR" jobs/run_smurf.sbatch
```

Key behavior:
- Uses `EXTERNAL_NUCLEAR_BINS_PATH` for PPO-aligned nuclear seeds.
- Writes PPO-format assignment outputs and evaluation summaries.

Main outputs:
- `workspace_outputs/pseudo_human_colorectal/smurf_results_colorectal_0.25/human_colorectal_smurf_assignments.csv`
- `runs/human_colorectal_smurf_eval_*/summary.json`
- `runs/human_colorectal_smurf_eval_*/overlays/`

### STCS
Set up the STCS environment once:
```bash
sbatch jobs/setup_stcs_env.sbatch
```

Then run STCS:
```bash
PPO_EVAL_RUN_DIR="runs/human_colorectal_ppo_eval_YYYYMMDDTHHMMSSZ"
sbatch --export=ALL,PPO_EVAL_RUN_DIR="$PPO_EVAL_RUN_DIR" jobs/run_stcs.sbatch
```

Key behavior:
- Uses `EXTERNAL_NUCLEAR_BINS_PATH` for PPO-aligned nuclear seeds.
- By default uses `RESTRICT_TO_EVAL_CELLS=true` and `EVAL_CONTEXT_RADIUS_BINS=10` to reduce memory while preserving nearby competitor nuclei.

Main outputs:
- `workspace_outputs/pseudo_human_colorectal/stcs_results_colorectal_0.25/human_colorectal_stcs_assignments.csv`
- `runs/human_colorectal_stcs_eval_*/summary.json`
- `runs/human_colorectal_stcs_eval_*/overlays/`

## 15) Re-run method evaluation only
If Bin2Cell, SMURF, or STCS assignments already exist, use this job to re-run only PPO-format evaluation without re-running the full method pipeline.

Run all available methods:
```bash
PPO_EVAL_RUN_DIR="runs/human_colorectal_ppo_eval_YYYYMMDDTHHMMSSZ"
sbatch --export=ALL,PPO_EVAL_RUN_DIR="$PPO_EVAL_RUN_DIR" jobs/run_method_ppo_eval_only.sbatch
```

Run one method only:
```bash
sbatch --export=ALL,\
PPO_EVAL_RUN_DIR="$PPO_EVAL_RUN_DIR",\
RUN_BIN2CELL=true,\
RUN_SMURF=false,\
RUN_STCS=false \
jobs/run_method_ppo_eval_only.sbatch
```

The shared evaluator writes consistent fields across PPO, Bin2Cell, SMURF, and STCS:
- matched-cell count
- IoU = intersection / union
- precision = intersection / predicted bins
- recall = intersection / GT bins
- Dice
- size ratio
- per-cell and summary gene-correlation metrics when GT single-cell expression is provided
- overlay plots with predicted bins and GT outline
