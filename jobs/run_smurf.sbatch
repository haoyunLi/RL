#!/bin/bash
#SBATCH --job-name=smurf
#SBATCH --account=kwang222-ic
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=500G
#SBATCH --output=smurf.out
#SBATCH --error=smurf.err

set -euo pipefail

WORK_DIR="/taiga/illinois/vetmed/cb/kwang222/Haoyun_Li/RL"
PYTHON_BIN="$WORK_DIR/smurf/bin/python"
cd "$WORK_DIR"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "SMURF python not found: $PYTHON_BIN" >&2
  exit 1
fi

module load cuda/12.4
export CUDA_HOME="/sw/apps/cuda/12.4"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/sw/apps/cuda/12.4"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib_${SLURM_JOB_ID:-$$}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/xdg_cache_${SLURM_JOB_ID:-$$}}"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

TISSUE_POSITIONS="${TISSUE_POSITIONS:-colorectal_pseudo_visium_hd_output_full_0.25/spatial/tissue_positions.parquet}"
TISSUE_IMAGE="${TISSUE_IMAGE:-Human_Colorectal/input/Visium_HD_Human_Colon_Cancer_tissue_image.btf}"
PSEUDO_HD_DIR="${PSEUDO_HD_DIR:-colorectal_pseudo_visium_hd_output_full_0.25}"
OUTPUT_DIR="${OUTPUT_DIR:-workspace_outputs/pseudo_human_colorectal/smurf_results_colorectal_0.25}"
DATASET_NAME="${DATASET_NAME:-human_colorectal}"
MICRONS_PER_PIXEL="${MICRONS_PER_PIXEL:-0.2737012522439323}"
EXTERNAL_NUCLEAR_BINS_PATH="${EXTERNAL_NUCLEAR_BINS_PATH:-workspace_outputs/pseudo_human_colorectal/intermediate/cropped_visium_hd_human_colorectal_pseudo_square_002um_nuclear_bins.csv.gz}"

PPO_EVAL_RUN_DIR="${PPO_EVAL_RUN_DIR:-runs/human_colorectal_ppo_eval_20260421T181912Z}"
EVAL_RUN_NAME="${EVAL_RUN_NAME:-human_colorectal_smurf_eval}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-runs}"
OVERLAY_MAX_CELLS="${OVERLAY_MAX_CELLS:-300}"
OVERLAY_SELECTION="${OVERLAY_SELECTION:-first}"
EVAL_SEED="${EVAL_SEED:-7}"
PRED_MIN_NUCLEAR_OVERLAP_FRAC="${PRED_MIN_NUCLEAR_OVERLAP_FRAC:-0.3}"
PRED_MIN_NUCLEAR_OVERLAP_BINS="${PRED_MIN_NUCLEAR_OVERLAP_BINS:-2}"
GT_CELL_BINS_PATH="${GT_CELL_BINS_PATH:-workspace_outputs/pseudo_human_colorectal/intermediate/cropped_visium_hd_human_colorectal_pseudo_square_002um_cell_bins.csv.gz}"
GT_NUCLEAR_BINS_PATH="${GT_NUCLEAR_BINS_PATH:-workspace_outputs/pseudo_human_colorectal/intermediate/cropped_visium_hd_human_colorectal_pseudo_square_002um_nuclear_bins.csv.gz}"
GT_MIN_NUCLEAR_OVERLAP_FRAC="${GT_MIN_NUCLEAR_OVERLAP_FRAC:-0.3}"
GT_MIN_NUCLEAR_OVERLAP_BINS="${GT_MIN_NUCLEAR_OVERLAP_BINS:-2}"

echo "Working directory: $WORK_DIR"
echo "Python: $PYTHON_BIN"
echo "TISSUE_POSITIONS: $TISSUE_POSITIONS"
echo "TISSUE_IMAGE: $TISSUE_IMAGE"
echo "PSEUDO_HD_DIR: $PSEUDO_HD_DIR"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "DATASET_NAME: $DATASET_NAME"
echo "MICRONS_PER_PIXEL: $MICRONS_PER_PIXEL"
echo "EXTERNAL_NUCLEAR_BINS_PATH: $EXTERNAL_NUCLEAR_BINS_PATH"
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
echo "GT_MIN_NUCLEAR_OVERLAP_FRAC: $GT_MIN_NUCLEAR_OVERLAP_FRAC"
echo "GT_MIN_NUCLEAR_OVERLAP_BINS: $GT_MIN_NUCLEAR_OVERLAP_BINS"

"$PYTHON_BIN" preprocessing/SMURF_analyze_spatial.py \
  --tissue_positions "$TISSUE_POSITIONS" \
  --tissue_image "$TISSUE_IMAGE" \
  --pseudo_hd_dir "$PSEUDO_HD_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --dataset_name "$DATASET_NAME" \
  --microns_per_pixel "$MICRONS_PER_PIXEL" \
  --external_nuclear_bins_path "$EXTERNAL_NUCLEAR_BINS_PATH" \
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
  --gt_min_nuclear_overlap_frac "$GT_MIN_NUCLEAR_OVERLAP_FRAC" \
  --gt_min_nuclear_overlap_bins "$GT_MIN_NUCLEAR_OVERLAP_BINS"

echo ""
echo "SMURF analysis completed at $(date)"
