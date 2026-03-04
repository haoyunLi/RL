# Human Colorectal Runbook

This is a step-by-step command list from crop image to reference-count generation.

## 0) Go to repo
```bash
cd /taiga/illinois/vetmed/cb/kwang222/Haoyun_Li/RL
```

## 1) Crop Visium HD image
```bash
sbatch jobs/run_crop_visium_hd.sbatch
```
Expected output:
- `workspace_outputs/human_colorectal/intermediate/cropped_visium_hd_human_Colorectal.png`

## 2) Run Cellpose nuclear segmentation
```bash
sbatch jobs/run_cellpose_sam.sbatch
```
Expected output folder:
- `workspace_outputs/human_colorectal/intermediate/cellpose_sam_human_colorectal_output/`

## 3) Map pixel nuclear signal to official `square_002um` bins
```bash
sbatch jobs/run_pixel_to_square_002um_bins.sbatch
```
Expected outputs:
- `workspace_outputs/human_colorectal/intermediate/cropped_visium_hd_human_Colorectal_square_002um_nuclear_bins.csv.gz`
- `workspace_outputs/human_colorectal/intermediate/cropped_visium_hd_human_Colorectal_square_002um_nuclear_bins.summary.json`
- `workspace_outputs/human_colorectal/intermediate/cropped_visium_hd_human_Colorectal_square_002um_nuclear_bins_overlay.png`

## 4) Merge official expression metadata + nuclear annotation
```bash
sbatch jobs/run_merge_square_002um_expression_with_nuclear.sbatch
```
Expected outputs:
- `workspace_outputs/human_colorectal/intermediate/square_002um_nuclear/human_colorectal_square_002um_rl.metadata.parquet`
- `workspace_outputs/human_colorectal/intermediate/square_002um_nuclear/human_colorectal_square_002um_rl.selected_features.tsv.gz`
- `workspace_outputs/human_colorectal/intermediate/square_002um_nuclear/human_colorectal_square_002um_rl.selected_feature_indices.npy`
- `workspace_outputs/human_colorectal/intermediate/square_002um_nuclear/human_colorectal_square_002um_rl.manifest.json`

## 5) Build nuclei table (one row per nucleus)
```bash
sbatch jobs/run_build_nuclei_table.sbatch
```
Expected outputs:
- `workspace_outputs/human_colorectal/intermediate/human_colorectal_nuclei.parquet`
- `workspace_outputs/human_colorectal/intermediate/human_colorectal_nuclei.summary.json`

## 6) Build aligned scRNA reference counts (`C[k,g]`)
This job also generates the HD gene allowlist text and applies it during reference build.
```bash
sbatch jobs/run_build_reference_counts_sct.sbatch
```
Expected outputs:
- `workspace_outputs/human_colorectal/intermediate/reference_sct/hd_selected_feature_names_unique.txt`
- `workspace_outputs/human_colorectal/intermediate/reference_sct/reference_counts_sct_tumor_aligned.npz`
- `workspace_outputs/human_colorectal/intermediate/reference_sct/reference_counts_sct_tumor_aligned.summary.json`

## 7) Next step after preprocessing
Run combined episode-build + reward-grid-search job:
```bash
sbatch jobs/run_episode_build_and_reward_grid_search.sbatch
```

Manual run (same logic):
```bash
python scripts/run_episode_build.py --config configs/episode_build.template.yaml
python scripts/run_reward_grid_search.py --config configs/reward_grid_search.template.yaml
```
