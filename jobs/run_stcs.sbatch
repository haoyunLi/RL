#!/bin/bash
#SBATCH --job-name=stcs
#SBATCH --account=kwang222-ic
#SBATCH --partition=bioshare
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=800G
#SBATCH --output=stcs.out
#SBATCH --error=stcs.err

set -euo pipefail

WORK_DIR="/taiga/illinois/vetmed/cb/kwang222/Haoyun_Li/RL"
cd "$WORK_DIR"

STCS_ENV_DIR="${STCS_ENV_DIR:-$WORK_DIR/stcs_env}"
PYTHON_BIN="$STCS_ENV_DIR/bin/python"
STCS_REPO_DIR="${STCS_REPO_DIR:-$WORK_DIR/external/STCS}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "STCS python not found: $PYTHON_BIN" >&2
  echo "Create it first with: sbatch jobs/setup_stcs_env.sbatch" >&2
  exit 1
fi
if [[ ! -f "$STCS_REPO_DIR/STCS/STCS_main.py" ]]; then
  echo "STCS repo not found: $STCS_REPO_DIR" >&2
  echo "Clone it first with: sbatch jobs/setup_stcs_env.sbatch" >&2
  exit 1
fi

source "$STCS_ENV_DIR/bin/activate"

THREADS="${SLURM_CPUS_PER_TASK:-70}"
export OPENBLAS_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export OMP_NUM_THREADS="$THREADS"
export NUMEXPR_NUM_THREADS="$THREADS"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib_${SLURM_JOB_ID:-$$}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/xdg_cache_${SLURM_JOB_ID:-$$}}"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

PSEUDO_HD_DIR="${PSEUDO_HD_DIR:-colorectal_pseudo_visium_hd_output_full_0.25}"
OUTPUT_DIR="${OUTPUT_DIR:-workspace_outputs/pseudo_human_colorectal/stcs_results_colorectal_0.25}"
DATASET_NAME="${DATASET_NAME:-human_colorectal}"
FULL_RES_IMAGE_PATH="${FULL_RES_IMAGE_PATH:-colorectal_pseudo_visium_hd_output_full_0.25/spatial/tissue_hires_image.png}"
EXTERNAL_NUCLEAR_BINS_PATH="${EXTERNAL_NUCLEAR_BINS_PATH:-workspace_outputs/pseudo_human_colorectal/intermediate/cropped_visium_hd_human_colorectal_pseudo_square_002um_nuclear_bins.csv.gz}"

SEARCH_RADIUS="${SEARCH_RADIUS:-5}"
LAMBDA_SPATIAL="${LAMBDA_SPATIAL:-0.5}"
TARGET_SUM="${TARGET_SUM:-10000}"
N_TOP_GENES="${N_TOP_GENES:-5000}"
MIN_CELLS="${MIN_CELLS:-3}"
PSEUDOBULK_MODE="${PSEUDOBULK_MODE:-mean}"
NORMALIZE_DISTANCES="${NORMALIZE_DISTANCES:-true}"
FORCE_NUCLEAR_ASSIGNMENTS="${FORCE_NUCLEAR_ASSIGNMENTS:-true}"
RESTRICT_TO_EVAL_CELLS="${RESTRICT_TO_EVAL_CELLS:-true}"
EVAL_CONTEXT_RADIUS_BINS="${EVAL_CONTEXT_RADIUS_BINS:-10}"

PPO_EVAL_RUN_DIR="${PPO_EVAL_RUN_DIR:-runs/human_colorectal_ppo_eval_20260424T204220Z}"
EVAL_RUN_NAME="${EVAL_RUN_NAME:-human_colorectal_stcs_eval}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-runs}"
OVERLAY_MAX_CELLS="${OVERLAY_MAX_CELLS:-300}"
OVERLAY_SELECTION="${OVERLAY_SELECTION:-first}"
EVAL_SEED="${EVAL_SEED:-7}"
PRED_MIN_NUCLEAR_OVERLAP_FRAC="${PRED_MIN_NUCLEAR_OVERLAP_FRAC:-0.3}"
PRED_MIN_NUCLEAR_OVERLAP_BINS="${PRED_MIN_NUCLEAR_OVERLAP_BINS:-2}"
GT_CELL_BINS_PATH="${GT_CELL_BINS_PATH:-workspace_outputs/pseudo_human_colorectal/intermediate/cropped_visium_hd_human_colorectal_pseudo_square_002um_cell_bins.csv.gz}"
GT_NUCLEAR_BINS_PATH="${GT_NUCLEAR_BINS_PATH:-workspace_outputs/pseudo_human_colorectal/intermediate/cropped_visium_hd_human_colorectal_pseudo_square_002um_nuclear_bins.csv.gz}"
GT_CELL_ASSIGNMENTS_CSV="${GT_CELL_ASSIGNMENTS_CSV:-colorectal_pseudo_visium_hd_output_full_0.25/ground_truth_cell_assignments.csv}"
GT_SC_EXPRESSION_H5="${GT_SC_EXPRESSION_H5:-colorectal_sc_data/CRC_GSE166555_raw_tumor.h5}"
GT_MIN_NUCLEAR_OVERLAP_FRAC="${GT_MIN_NUCLEAR_OVERLAP_FRAC:-0.3}"
GT_MIN_NUCLEAR_OVERLAP_BINS="${GT_MIN_NUCLEAR_OVERLAP_BINS:-2}"

NORMALIZE_DISTANCES_FLAG="--no-normalize_distances"
if [[ "$NORMALIZE_DISTANCES" == "true" || "$NORMALIZE_DISTANCES" == "1" ]]; then
  NORMALIZE_DISTANCES_FLAG="--normalize_distances"
fi

FORCE_NUCLEAR_ASSIGNMENTS_FLAG="--no-force_nuclear_assignments"
if [[ "$FORCE_NUCLEAR_ASSIGNMENTS" == "true" || "$FORCE_NUCLEAR_ASSIGNMENTS" == "1" ]]; then
  FORCE_NUCLEAR_ASSIGNMENTS_FLAG="--force_nuclear_assignments"
fi

RESTRICT_TO_EVAL_CELLS_FLAG="--no-restrict_to_eval_cells"
if [[ "$RESTRICT_TO_EVAL_CELLS" == "true" || "$RESTRICT_TO_EVAL_CELLS" == "1" ]]; then
  RESTRICT_TO_EVAL_CELLS_FLAG="--restrict_to_eval_cells"
fi

echo "Working directory: $WORK_DIR"
echo "Python: $PYTHON_BIN"
echo "STCS_REPO_DIR: $STCS_REPO_DIR"
echo "PSEUDO_HD_DIR: $PSEUDO_HD_DIR"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "DATASET_NAME: $DATASET_NAME"
echo "FULL_RES_IMAGE_PATH: $FULL_RES_IMAGE_PATH"
echo "EXTERNAL_NUCLEAR_BINS_PATH: $EXTERNAL_NUCLEAR_BINS_PATH"
echo "SEARCH_RADIUS: $SEARCH_RADIUS"
echo "LAMBDA_SPATIAL: $LAMBDA_SPATIAL"
echo "TARGET_SUM: $TARGET_SUM"
echo "N_TOP_GENES: $N_TOP_GENES"
echo "MIN_CELLS: $MIN_CELLS"
echo "PSEUDOBULK_MODE: $PSEUDOBULK_MODE"
echo "NORMALIZE_DISTANCES: $NORMALIZE_DISTANCES"
echo "FORCE_NUCLEAR_ASSIGNMENTS: $FORCE_NUCLEAR_ASSIGNMENTS"
echo "RESTRICT_TO_EVAL_CELLS: $RESTRICT_TO_EVAL_CELLS"
echo "EVAL_CONTEXT_RADIUS_BINS: $EVAL_CONTEXT_RADIUS_BINS"
echo "PPO_EVAL_RUN_DIR: $PPO_EVAL_RUN_DIR"
echo "EVAL_RUN_NAME: $EVAL_RUN_NAME"
echo "EVAL_OUTPUT_ROOT: $EVAL_OUTPUT_ROOT"
echo "OVERLAY_MAX_CELLS: $OVERLAY_MAX_CELLS"
echo "OVERLAY_SELECTION: $OVERLAY_SELECTION"
echo "EVAL_SEED: $EVAL_SEED"
echo "PRED_MIN_NUCLEAR_OVERLAP_FRAC: $PRED_MIN_NUCLEAR_OVERLAP_FRAC"
echo "PRED_MIN_NUCLEAR_OVERLAP_BINS: $PRED_MIN_NUCLEAR_OVERLAP_BINS"
echo "GT_CELL_BINS_PATH: $GT_CELL_BINS_PATH"
echo "GT_NUCLEAR_BINS_PATH: $GT_NUCLEAR_BINS_PATH"
echo "GT_CELL_ASSIGNMENTS_CSV: $GT_CELL_ASSIGNMENTS_CSV"
echo "GT_SC_EXPRESSION_H5: $GT_SC_EXPRESSION_H5"
echo "GT_MIN_NUCLEAR_OVERLAP_FRAC: $GT_MIN_NUCLEAR_OVERLAP_FRAC"
echo "GT_MIN_NUCLEAR_OVERLAP_BINS: $GT_MIN_NUCLEAR_OVERLAP_BINS"

"$PYTHON_BIN" preprocessing/run_stcs_analysis.py \
  --pseudo_hd_dir "$PSEUDO_HD_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --dataset_name "$DATASET_NAME" \
  --stcs_repo_dir "$STCS_REPO_DIR" \
  --full_res_image_path "$FULL_RES_IMAGE_PATH" \
  --external_nuclear_bins_path "$EXTERNAL_NUCLEAR_BINS_PATH" \
  --search_radius "$SEARCH_RADIUS" \
  --lambda_spatial "$LAMBDA_SPATIAL" \
  --target_sum "$TARGET_SUM" \
  --n_top_genes "$N_TOP_GENES" \
  --min_cells "$MIN_CELLS" \
  --pseudobulk_mode "$PSEUDOBULK_MODE" \
  "$NORMALIZE_DISTANCES_FLAG" \
  "$FORCE_NUCLEAR_ASSIGNMENTS_FLAG" \
  "$RESTRICT_TO_EVAL_CELLS_FLAG" \
  --eval_context_radius_bins "$EVAL_CONTEXT_RADIUS_BINS" \
  --ppo_eval_run_dir "$PPO_EVAL_RUN_DIR" \
  --eval_run_name "$EVAL_RUN_NAME" \
  --eval_output_root "$EVAL_OUTPUT_ROOT" \
  --overlay_max_cells "$OVERLAY_MAX_CELLS" \
  --overlay_selection "$OVERLAY_SELECTION" \
  --eval_seed "$EVAL_SEED" \
  --pred_min_nuclear_overlap_frac "$PRED_MIN_NUCLEAR_OVERLAP_FRAC" \
  --pred_min_nuclear_overlap_bins "$PRED_MIN_NUCLEAR_OVERLAP_BINS" \
  --gt_cell_bins_path "$GT_CELL_BINS_PATH" \
  --gt_nuclear_bins_path "$GT_NUCLEAR_BINS_PATH" \
  --gt_cell_assignments_csv "$GT_CELL_ASSIGNMENTS_CSV" \
  --gt_sc_expression_h5 "$GT_SC_EXPRESSION_H5" \
  --gt_min_nuclear_overlap_frac "$GT_MIN_NUCLEAR_OVERLAP_FRAC" \
  --gt_min_nuclear_overlap_bins "$GT_MIN_NUCLEAR_OVERLAP_BINS"

echo ""
echo "STCS analysis completed at $(date)"
