#!/bin/bash
#SBATCH --job-name=RewardGrid
#SBATCH --account=kwang222-ic
#SBATCH --partition=bioshare
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --mem=900G
#SBATCH --output=reward_grid_search.out
#SBATCH --error=reward_grid_search.err

set -euo pipefail

module load anaconda3

WORK_DIR="/taiga/illinois/vetmed/cb/kwang222/Haoyun_Li/RL"
cd "$WORK_DIR"

echo "Activating conda environment..."
source /sw/apps/anaconda3/2024.10/etc/profile.d/conda.sh
conda activate "$WORK_DIR/Bin2Cell_Validation"

# Avoid thread oversubscription; grid search parallelism is process-based.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

REWARD_CONFIG="${REWARD_CONFIG:-configs/reward_grid_search.template.yaml}"
EPISODES_INDEX_PATH="${EPISODES_INDEX_PATH:-}"
EPISODE_RUN_DIR="${EPISODE_RUN_DIR:-}"
SUPERVISED_EVAL_CSV="${SUPERVISED_EVAL_CSV:-}"
PPO_EVAL_RUN_DIR="${PPO_EVAL_RUN_DIR:-}"
GT_CELL_BINS_PATH="${GT_CELL_BINS_PATH:-workspace_outputs/pseudo_human_colorectal/intermediate/cropped_visium_hd_human_colorectal_pseudo_square_002um_cell_bins.csv.gz}"

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
  echo "Could not resolve episodes index path. Set EPISODES_INDEX_PATH or EPISODE_RUN_DIR."
  exit 1
fi
if [ ! -f "$EPISODES_INDEX_PATH" ]; then
  echo "Episodes index file not found: $EPISODES_INDEX_PATH"
  exit 1
fi

echo "Using episodes index: $EPISODES_INDEX_PATH"
if [ -z "$SUPERVISED_EVAL_CSV" ]; then
  if [ -n "$PPO_EVAL_RUN_DIR" ]; then
    SUPERVISED_EVAL_CSV="${PPO_EVAL_RUN_DIR}/per_episode.csv"
  else
    LATEST_PPO_EVAL_RUN=$(ls -dt runs/human_colorectal_ppo_eval_* 2>/dev/null | head -n 1 || true)
    if [ -n "$LATEST_PPO_EVAL_RUN" ]; then
      SUPERVISED_EVAL_CSV="${LATEST_PPO_EVAL_RUN}/per_episode.csv"
    fi
  fi
fi
if [ -n "$SUPERVISED_EVAL_CSV" ]; then
  echo "Using supervised eval CSV: $SUPERVISED_EVAL_CSV"
fi
echo "Using GT cell bins: $GT_CELL_BINS_PATH"

TMP_REWARD_CONFIG=$(mktemp --suffix=.yaml)
python - "$REWARD_CONFIG" "$TMP_REWARD_CONFIG" "$EPISODES_INDEX_PATH" "$SUPERVISED_EVAL_CSV" "$GT_CELL_BINS_PATH" <<'PY'
import sys
from pathlib import Path
import yaml

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
episodes_index = str(Path(sys.argv[3]).resolve())
supervised_eval_csv = sys.argv[4]
gt_cell_bins_path = sys.argv[5]

with src.open("r", encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle)

cfg["inputs"]["episodes_index_path"] = episodes_index
reward = cfg.get("reward", {})
if str(reward.get("objective", "")).strip() == "gt_shape":
    if not supervised_eval_csv:
        raise SystemExit("Could not resolve supervised eval CSV. Set SUPERVISED_EVAL_CSV or PPO_EVAL_RUN_DIR.")
    if not Path(supervised_eval_csv).resolve().is_file():
        raise SystemExit(f"Supervised eval CSV not found: {supervised_eval_csv}")
    supervised = cfg.setdefault("supervised", {})
    supervised["eval_csv_path"] = str(Path(supervised_eval_csv).resolve())
    if gt_cell_bins_path:
        supervised["gt_cell_bins_path"] = str(Path(gt_cell_bins_path).resolve())

with dst.open("w", encoding="utf-8") as handle:
    yaml.safe_dump(cfg, handle, sort_keys=False)
PY

COARSE_MAX_EPISODES="${COARSE_MAX_EPISODES:-200}"
COARSE_STEP_MULTIPLIER="${COARSE_STEP_MULTIPLIER:-2.0}"
FINE_MAX_EPISODES="${FINE_MAX_EPISODES:-$COARSE_MAX_EPISODES}"
FINE_SPAN="${FINE_SPAN:-0.1}"

echo "Running two-stage reward grid search..."
python scripts/run_reward_grid_search.py \
  --config "$TMP_REWARD_CONFIG" \
  --coarse-max-episodes "$COARSE_MAX_EPISODES" \
  --coarse-step-multiplier "$COARSE_STEP_MULTIPLIER" \
  --fine-max-episodes "$FINE_MAX_EPISODES" \
  --fine-span "$FINE_SPAN"

conda deactivate
