#!/bin/bash
#SBATCH --job-name=PPOTrainCPU
#SBATCH --account=kwang222-ic
#SBATCH --partition=bioshare
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --hint=nomultithread
#SBATCH --mem=800G
#SBATCH --output=ppo_training_cpu.out
#SBATCH --error=ppo_training_cpu.err

set -euo pipefail

module load anaconda3

WORK_DIR="/taiga/illinois/vetmed/cb/kwang222/Haoyun_Li/RL"
cd "$WORK_DIR"

echo "Activating conda environment..."
source /sw/apps/anaconda3/2024.10/etc/profile.d/conda.sh
conda activate "$WORK_DIR/Bin2Cell_Validation"

# CPU-thread tuning for PPO-heavy updates.
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32
export OPENBLAS_NUM_THREADS=32
export TORCH_NUM_THREADS=32
export TORCH_NUM_INTEROP_THREADS=1

PPO_CONFIG="${PPO_CONFIG:-configs/ppo_training.template.yaml}"
RUN_DEVICE="${RUN_DEVICE:-cpu}"
# Force vectorized rollout in this job script.
RUN_ROLLOUT_MODE="vectorized"
N_ROLLOUT_WORKERS=1
EPISODES_INDEX_PATH="${EPISODES_INDEX_PATH:-}"
EPISODE_RUN_DIR="${EPISODE_RUN_DIR:-}"
BATCH_CELLS="${BATCH_CELLS:-}"
MAX_UPDATES="${MAX_UPDATES:-}"
RUN_NAME="${RUN_NAME:-human_colorectal_ppo}"

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
  echo "Episodes index file not found: $EPISODES_INDEX_PATH"
  exit 1
fi

echo "PPO_CONFIG: $PPO_CONFIG"
echo "EPISODES_INDEX_PATH: $EPISODES_INDEX_PATH"
echo "RUN_DEVICE: $RUN_DEVICE"
echo "RUN_ROLLOUT_MODE: $RUN_ROLLOUT_MODE"
echo "N_ROLLOUT_WORKERS: $N_ROLLOUT_WORKERS"
echo "TORCH_NUM_THREADS: $TORCH_NUM_THREADS"
echo "TORCH_NUM_INTEROP_THREADS: $TORCH_NUM_INTEROP_THREADS"
echo "BATCH_CELLS: ${BATCH_CELLS:-<from config>}"
echo "MAX_UPDATES: ${MAX_UPDATES:-<from config>}"
echo "RUN_NAME: $RUN_NAME"

TMP_PPO_CONFIG="$(mktemp --suffix=.yaml)"
trap 'rm -f "$TMP_PPO_CONFIG"' EXIT
python - "$PPO_CONFIG" "$TMP_PPO_CONFIG" "$EPISODES_INDEX_PATH" "$RUN_DEVICE" "$RUN_ROLLOUT_MODE" "$N_ROLLOUT_WORKERS" "$BATCH_CELLS" "$MAX_UPDATES" "$RUN_NAME" <<'PY'
import sys
from pathlib import Path
import yaml

src = Path(sys.argv[1]).resolve()
dst = Path(sys.argv[2]).resolve()
episodes_index_path = str(Path(sys.argv[3]).resolve())
run_device = str(sys.argv[4]).strip().lower()
run_rollout_mode = str(sys.argv[5]).strip().lower()
n_rollout_workers = int(sys.argv[6])
batch_cells_raw = str(sys.argv[7]).strip()
max_updates_raw = str(sys.argv[8]).strip()
run_name = str(sys.argv[9]).strip()

with src.open("r", encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle)

run = cfg.setdefault("run", {})
run["device"] = run_device
run["rollout_mode"] = run_rollout_mode
run["n_rollout_workers"] = n_rollout_workers
if batch_cells_raw:
    run["batch_cells"] = int(batch_cells_raw)
if max_updates_raw:
    run["max_updates"] = int(max_updates_raw)
if run_name:
    run["name"] = run_name

inputs = cfg.setdefault("inputs", {})
inputs["episodes_index_path"] = episodes_index_path

with dst.open("w", encoding="utf-8") as handle:
    yaml.safe_dump(cfg, handle, sort_keys=False)
PY

python scripts/run_ppo_training.py --config "$TMP_PPO_CONFIG"

conda deactivate
