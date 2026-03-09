#!/bin/bash
#SBATCH --job-name=SelectWeights
#SBATCH --account=kwang222-ic
#SBATCH --partition=bioshare
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem=64G
#SBATCH --output=select_reward_weights.out
#SBATCH --error=select_reward_weights.err

set -euo pipefail

module load anaconda3

WORK_DIR="/taiga/illinois/vetmed/cb/kwang222/Haoyun_Li/RL"
cd "$WORK_DIR"

echo "Activating conda environment..."
source /sw/apps/anaconda3/2024.10/etc/profile.d/conda.sh
conda activate "$WORK_DIR/Bin2Cell_Validation"

RESULTS_CSV="${RESULTS_CSV:-}"
if [ -z "$RESULTS_CSV" ]; then
  LATEST_FINE=$(ls -dt runs/human_colorectal_reward_grid_search_fine_*/results.csv 2>/dev/null | head -n 1 || true)
  if [ -n "$LATEST_FINE" ]; then
    RESULTS_CSV="$LATEST_FINE"
  else
    LATEST_COARSE=$(ls -dt runs/human_colorectal_reward_grid_search_coarse_*/results.csv 2>/dev/null | head -n 1 || true)
    RESULTS_CSV="$LATEST_COARSE"
  fi
fi

if [ -z "$RESULTS_CSV" ]; then
  echo "Could not resolve RESULTS_CSV. Set RESULTS_CSV=/path/to/results.csv"
  exit 1
fi
if [ ! -f "$RESULTS_CSV" ]; then
  echo "results.csv not found: $RESULTS_CSV"
  exit 1
fi

TOP_K="${TOP_K:-20}"
MIN_ASSIGNED_BINS="${MIN_ASSIGNED_BINS:-50}"
MAX_ASSIGNED_BINS="${MAX_ASSIGNED_BINS:-1000}"
MIN_ADD_ACTIONS="${MIN_ADD_ACTIONS:-50}"
MAX_ADD_ACTIONS="${MAX_ADD_ACTIONS:-1000}"
FINAL_BEST_ADD_MIN="${FINAL_BEST_ADD_MIN:--0.2}"
FINAL_BEST_ADD_MAX="${FINAL_BEST_ADD_MAX:-0.05}"
MAX_STOP_REWARD="${MAX_STOP_REWARD:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"

echo "Using RESULTS_CSV: $RESULTS_CSV"
echo "Filters: min_assigned=$MIN_ASSIGNED_BINS max_assigned=$MAX_ASSIGNED_BINS min_add=$MIN_ADD_ACTIONS max_add=$MAX_ADD_ACTIONS final_best_add=[${FINAL_BEST_ADD_MIN}, ${FINAL_BEST_ADD_MAX}] top_k=$TOP_K"

CMD=(
  python scripts/select_reward_weights.py
  --results-csv "$RESULTS_CSV"
  --top-k "$TOP_K"
  --min-assigned-bins "$MIN_ASSIGNED_BINS"
  --max-assigned-bins "$MAX_ASSIGNED_BINS"
  --min-add-actions "$MIN_ADD_ACTIONS"
  --max-add-actions "$MAX_ADD_ACTIONS"
  --final-best-add-min "$FINAL_BEST_ADD_MIN"
  --final-best-add-max "$FINAL_BEST_ADD_MAX"
)

if [ -n "$MAX_STOP_REWARD" ]; then
  CMD+=(--max-stop-reward "$MAX_STOP_REWARD")
fi

if [ -n "$OUTPUT_DIR" ]; then
  CMD+=(--output-dir "$OUTPUT_DIR")
fi

"${CMD[@]}"

conda deactivate
