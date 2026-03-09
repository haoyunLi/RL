#!/bin/bash
#SBATCH --job-name=EvalWeights
#SBATCH --account=kwang222-ic
#SBATCH --partition=bioshare
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --mem=900G
#SBATCH --output=evaluate_reward_weights.out
#SBATCH --error=evaluate_reward_weights.err

set -euo pipefail

module load anaconda3

WORK_DIR="/taiga/illinois/vetmed/cb/kwang222/Haoyun_Li/RL"
cd "$WORK_DIR"

echo "Activating conda environment..."
source /sw/apps/anaconda3/2024.10/etc/profile.d/conda.sh
conda activate "$WORK_DIR/Bin2Cell_Validation"

# Avoid thread oversubscription; evaluation parallelism is process-based.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

CONFIG_PATH="${CONFIG_PATH:-configs/reward_grid_search.template.yaml}"
EPISODES_INDEX_PATH="${EPISODES_INDEX_PATH:-}"
EPISODE_RUN_DIR="${EPISODE_RUN_DIR:-}"

if [ -z "$EPISODES_INDEX_PATH" ]; then
  if [ -n "$EPISODE_RUN_DIR" ]; then
    EPISODES_INDEX_PATH="${EPISODE_RUN_DIR}/episodes_index.csv"
  else
    LATEST_EPISODE_RUN=$(ls -dt runs/human_colorectal_episode_build_* 2>/dev/null | head -n 1 || true)
    if [ -n "$LATEST_EPISODE_RUN" ]; then
      EPISODES_INDEX_PATH="${LATEST_EPISODE_RUN}/episodes_index.csv"
    fi
  fi
fi

if [ -z "$EPISODES_INDEX_PATH" ]; then
  echo "Could not resolve EPISODES_INDEX_PATH. Set EPISODES_INDEX_PATH or EPISODE_RUN_DIR."
  exit 1
fi
if [ ! -f "$EPISODES_INDEX_PATH" ]; then
  echo "episodes index file not found: $EPISODES_INDEX_PATH"
  exit 1
fi

W1="${W1:-0.7}"
W2="${W2:-0.7}"
W3="${W3:-0.9}"
STOP_LAMBDA="${STOP_LAMBDA:-0.9}"
MAX_EPISODES="${MAX_EPISODES:-500}"
N_WORKERS="${N_WORKERS:-64}"
RUN_NAME="${RUN_NAME:-reward_eval_val}"
EVAL_SEED="${EVAL_SEED:-}"

if [ -z "$EVAL_SEED" ]; then
  EVAL_SEED="$(date +%s)"
fi

echo "Config: $CONFIG_PATH"
echo "Episodes: $EPISODES_INDEX_PATH"
echo "Weights: w1=$W1 w2=$W2 w3=$W3 stop_lambda=$STOP_LAMBDA"
echo "Eval params: max_episodes=$MAX_EPISODES n_workers=$N_WORKERS run_name=$RUN_NAME seed=$EVAL_SEED"

python scripts/evaluate_reward_weights.py \
  --config "$CONFIG_PATH" \
  --episodes-index-path "$EPISODES_INDEX_PATH" \
  --w1 "$W1" \
  --w2 "$W2" \
  --w3 "$W3" \
  --stop-lambda "$STOP_LAMBDA" \
  --max-episodes "$MAX_EPISODES" \
  --n-workers "$N_WORKERS" \
  --run-name "$RUN_NAME" \
  --seed "$EVAL_SEED"

conda deactivate
