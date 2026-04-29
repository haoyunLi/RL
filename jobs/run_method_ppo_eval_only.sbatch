#!/bin/bash
#SBATCH --job-name=method_eval
#SBATCH --account=kwang222-ic
#SBATCH --partition=bioshare
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=400G
#SBATCH --output=method_eval_only.out
#SBATCH --error=method_eval_only.err

set -euo pipefail

WORK_DIR="/taiga/illinois/vetmed/cb/kwang222/Haoyun_Li/RL"
cd "$WORK_DIR"

PYTHON_BIN="${PYTHON_BIN:-$WORK_DIR/Bin2Cell_Validation/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python3}"
fi

THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OPENBLAS_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export OMP_NUM_THREADS="$THREADS"
export NUMEXPR_NUM_THREADS="$THREADS"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib_${SLURM_JOB_ID:-$$}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/xdg_cache_${SLURM_JOB_ID:-$$}}"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

PPO_EVAL_RUN_DIR="${PPO_EVAL_RUN_DIR:-runs/human_colorectal_ppo_eval_20260424T204220Z}"
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
EXTERNAL_NUCLEAR_BINS_PATH="${EXTERNAL_NUCLEAR_BINS_PATH:-workspace_outputs/pseudo_human_colorectal/intermediate/cropped_visium_hd_human_colorectal_pseudo_square_002um_nuclear_bins.csv.gz}"

RUN_BIN2CELL="${RUN_BIN2CELL:-true}"
RUN_SMURF="${RUN_SMURF:-true}"
RUN_STCS="${RUN_STCS:-true}"
SKIP_MISSING="${SKIP_MISSING:-false}"

BIN2CELL_ASSIGNMENTS="${BIN2CELL_ASSIGNMENTS:-workspace_outputs/pseudo_human_colorectal/bin2cell_results_colorectal_0.25/human_colorectal_bin2cell_assignments.csv}"
BIN2CELL_PIPELINE_JSON="${BIN2CELL_PIPELINE_JSON:-workspace_outputs/pseudo_human_colorectal/bin2cell_results_colorectal_0.25/human_colorectal_bin2cell_pipeline_summary.json}"
SMURF_ASSIGNMENTS="${SMURF_ASSIGNMENTS:-workspace_outputs/pseudo_human_colorectal/smurf_results_colorectal_0.25/human_colorectal_smurf_assignments.csv}"
SMURF_PIPELINE_JSON="${SMURF_PIPELINE_JSON:-workspace_outputs/pseudo_human_colorectal/smurf_results_colorectal_0.25/human_colorectal_smurf_pipeline_summary.json}"
STCS_ASSIGNMENTS="${STCS_ASSIGNMENTS:-workspace_outputs/pseudo_human_colorectal/stcs_results_colorectal_0.25/human_colorectal_stcs_assignments.csv}"
STCS_PIPELINE_JSON="${STCS_PIPELINE_JSON:-workspace_outputs/pseudo_human_colorectal/stcs_results_colorectal_0.25/human_colorectal_stcs_pipeline_summary.json}"

echo "Working directory: $WORK_DIR"
echo "Python: $PYTHON_BIN"
echo "PPO_EVAL_RUN_DIR: $PPO_EVAL_RUN_DIR"
echo "EVAL_OUTPUT_ROOT: $EVAL_OUTPUT_ROOT"
echo "OVERLAY_MAX_CELLS: $OVERLAY_MAX_CELLS"
echo "OVERLAY_SELECTION: $OVERLAY_SELECTION"
echo "EVAL_SEED: $EVAL_SEED"
echo "GT_CELL_BINS_PATH: $GT_CELL_BINS_PATH"
echo "GT_NUCLEAR_BINS_PATH: $GT_NUCLEAR_BINS_PATH"
echo "GT_CELL_ASSIGNMENTS_CSV: $GT_CELL_ASSIGNMENTS_CSV"
echo "GT_SC_EXPRESSION_H5: $GT_SC_EXPRESSION_H5"
echo "EXTERNAL_NUCLEAR_BINS_PATH: $EXTERNAL_NUCLEAR_BINS_PATH"
echo "RUN_BIN2CELL: $RUN_BIN2CELL"
echo "RUN_SMURF: $RUN_SMURF"
echo "RUN_STCS: $RUN_STCS"

run_eval() {
  local method_name="$1"
  local method_label="$2"
  local eval_run_name="$3"
  local assignments_csv="$4"
  local pipeline_json="$5"

  if [[ ! -f "$assignments_csv" ]]; then
    if [[ "$SKIP_MISSING" == "true" || "$SKIP_MISSING" == "1" ]]; then
      echo "Skipping $method_name; assignments CSV not found: $assignments_csv"
      return 0
    fi
    echo "Assignments CSV not found for $method_name: $assignments_csv" >&2
    exit 1
  fi

  local pipeline_args=()
  if [[ -f "$pipeline_json" ]]; then
    pipeline_args=(--pipeline_config_json "$pipeline_json")
  fi

  echo ""
  echo "Running PPO-format eval only for $method_label"
  "$PYTHON_BIN" preprocessing/evaluate_ppo_format_assignments.py \
    --assignments_csv "$assignments_csv" \
    --method_name "$method_name" \
    --method_label "$method_label" \
    --nuclear_source external_ppo_aligned \
    --external_nuclear_bins_path "$EXTERNAL_NUCLEAR_BINS_PATH" \
    "${pipeline_args[@]}" \
    --ppo_eval_run_dir "$PPO_EVAL_RUN_DIR" \
    --eval_run_name "$eval_run_name" \
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
}

if [[ "$RUN_BIN2CELL" == "true" || "$RUN_BIN2CELL" == "1" ]]; then
  run_eval "bin2cell" "Bin2Cell" "human_colorectal_bin2cell_eval_only" "$BIN2CELL_ASSIGNMENTS" "$BIN2CELL_PIPELINE_JSON"
fi

if [[ "$RUN_SMURF" == "true" || "$RUN_SMURF" == "1" ]]; then
  run_eval "smurf" "SMURF" "human_colorectal_smurf_eval_only" "$SMURF_ASSIGNMENTS" "$SMURF_PIPELINE_JSON"
fi

if [[ "$RUN_STCS" == "true" || "$RUN_STCS" == "1" ]]; then
  run_eval "stcs" "STCS" "human_colorectal_stcs_eval_only" "$STCS_ASSIGNMENTS" "$STCS_PIPELINE_JSON"
fi

echo ""
echo "Method PPO-format eval-only jobs completed at $(date)"
