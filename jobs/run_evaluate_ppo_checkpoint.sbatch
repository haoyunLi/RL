#!/bin/bash
#SBATCH --job-name=EvalPPOCkpt
#SBATCH --account=kwang222-ic
#SBATCH --partition=bioshare
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=800G
#SBATCH --output=evaluate_ppo_checkpoint.out
#SBATCH --error=evaluate_ppo_checkpoint.err

set -euo pipefail

module load anaconda3

WORK_DIR="/taiga/illinois/vetmed/cb/kwang222/Haoyun_Li/RL"
cd "$WORK_DIR"

echo "Activating conda environment..."
source /sw/apps/anaconda3/2024.10/etc/profile.d/conda.sh
conda activate "$WORK_DIR/Bin2Cell_Validation"

# Keep CPU math thread usage predictable.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TORCH_NUM_THREADS=32
export TORCH_NUM_INTEROP_THREADS=1
unset KMP_LOGFILE || true
unset LOGFILE || true

CHECKPOINT_PATH="${CHECKPOINT_PATH:-runs/human_colorectal_ppo_20260421T184811Z/checkpoints/best_model.pt}"
PPO_CONFIG="${PPO_CONFIG:-}"
EPISODES_INDEX_PATH="${EPISODES_INDEX_PATH:-}"
MAX_EPISODES="${MAX_EPISODES:-300}"
POLICY_MODE="${POLICY_MODE:-greedy}"
RUN_DEVICE="${RUN_DEVICE:-auto}"
EVAL_SEED="${EVAL_SEED:-7}"
RUN_NAME="${RUN_NAME:-human_colorectal_ppo_eval}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs}"
OVERLAY_MAX_CELLS="${OVERLAY_MAX_CELLS:-300}"
OVERLAY_SELECTION="${OVERLAY_SELECTION:-top_reward}"
GT_CELL_BINS_PATH="${GT_CELL_BINS_PATH:-workspace_outputs/pseudo_human_colorectal/intermediate/cropped_visium_hd_human_colorectal_pseudo_square_002um_cell_bins.csv.gz}"
GT_NUCLEAR_BINS_PATH="${GT_NUCLEAR_BINS_PATH:-workspace_outputs/pseudo_human_colorectal/intermediate/cropped_visium_hd_human_colorectal_pseudo_square_002um_nuclear_bins.csv.gz}"
GT_MIN_NUCLEAR_OVERLAP_FRAC="${GT_MIN_NUCLEAR_OVERLAP_FRAC:-0.3}"
GT_MIN_NUCLEAR_OVERLAP_BINS="${GT_MIN_NUCLEAR_OVERLAP_BINS:-2}"

if [[ -z "$CHECKPOINT_PATH" ]]; then
  echo "CHECKPOINT_PATH is required." >&2
  echo "Example:" >&2
  echo "  sbatch --export=ALL,CHECKPOINT_PATH=runs/human_colorectal_ppo_YYYYMMDDTHHMMSSZ/checkpoints/best_model.pt jobs/run_evaluate_ppo_checkpoint.sbatch" >&2
  exit 1
fi
if [[ ! -f "$CHECKPOINT_PATH" ]]; then
  echo "Checkpoint file not found: $CHECKPOINT_PATH" >&2
  exit 1
fi

echo "CHECKPOINT_PATH: $CHECKPOINT_PATH"
echo "PPO_CONFIG: ${PPO_CONFIG:-<from checkpoint>}"
echo "EPISODES_INDEX_PATH: ${EPISODES_INDEX_PATH:-<from checkpoint config>}"
echo "MAX_EPISODES: $MAX_EPISODES"
echo "POLICY_MODE: $POLICY_MODE"
echo "RUN_DEVICE: $RUN_DEVICE"
echo "EVAL_SEED: $EVAL_SEED"
echo "RUN_NAME: $RUN_NAME"
echo "OUTPUT_ROOT: $OUTPUT_ROOT"
echo "OVERLAY_MAX_CELLS: $OVERLAY_MAX_CELLS"
echo "OVERLAY_SELECTION: $OVERLAY_SELECTION"
echo "GT_CELL_BINS_PATH: $GT_CELL_BINS_PATH"
echo "GT_NUCLEAR_BINS_PATH: $GT_NUCLEAR_BINS_PATH"
echo "GT_MIN_NUCLEAR_OVERLAP_FRAC: $GT_MIN_NUCLEAR_OVERLAP_FRAC"
echo "GT_MIN_NUCLEAR_OVERLAP_BINS: $GT_MIN_NUCLEAR_OVERLAP_BINS"

CMD=(python scripts/evaluate_ppo_checkpoint.py
  --checkpoint "$CHECKPOINT_PATH"
  --max-episodes "$MAX_EPISODES"
  --policy-mode "$POLICY_MODE"
  --device "$RUN_DEVICE"
  --seed "$EVAL_SEED"
  --run-name "$RUN_NAME"
  --output-root "$OUTPUT_ROOT"
  --overlay-max-cells "$OVERLAY_MAX_CELLS"
  --overlay-selection "$OVERLAY_SELECTION"
  --gt-cell-bins-path "$GT_CELL_BINS_PATH"
  --gt-nuclear-bins-path "$GT_NUCLEAR_BINS_PATH"
  --gt-min-nuclear-overlap-frac "$GT_MIN_NUCLEAR_OVERLAP_FRAC"
  --gt-min-nuclear-overlap-bins "$GT_MIN_NUCLEAR_OVERLAP_BINS"
)

if [[ -n "$PPO_CONFIG" ]]; then
  CMD+=(--config "$PPO_CONFIG")
fi
if [[ -n "$EPISODES_INDEX_PATH" ]]; then
  CMD+=(--episodes-index-path "$EPISODES_INDEX_PATH")
fi

"${CMD[@]}"

conda deactivate
