# Human Colorectal Runbook

This runbook lists the execution order for preprocessing, episode building, reward search, weight selection, and fixed-weight evaluation.
Each step depends on outputs from earlier steps, so run in order unless you already have the required files.

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

## 4) Merge official expression metadata + nuclear annotation
This step merges official HD bin metadata/expression indexing with nuclear-bin annotation into one RL-ready table.
The merged table is the main structural input for episode construction.
```bash
sbatch jobs/run_merge_square_002um_expression_with_nuclear.sbatch
```
Expected outputs:
- `workspace_outputs/human_colorectal/intermediate/square_002um_nuclear/human_colorectal_square_002um_rl.metadata.parquet`
- `workspace_outputs/human_colorectal/intermediate/square_002um_nuclear/human_colorectal_square_002um_rl.selected_features.tsv.gz`
- `workspace_outputs/human_colorectal/intermediate/square_002um_nuclear/human_colorectal_square_002um_rl.selected_feature_indices.npy`
- `workspace_outputs/human_colorectal/intermediate/square_002um_nuclear/human_colorectal_square_002um_rl.manifest.json`

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
