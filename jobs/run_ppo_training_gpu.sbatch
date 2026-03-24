#!/bin/bash
#SBATCH --job-name=PPOTrainGPU
#SBATCH --account=kwang222-ic
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:1
#SBATCH --mem=500G
#SBATCH --output=ppo_training_gpu.out
#SBATCH --error=ppo_training_gpu.err

set -euo pipefail

module load anaconda3

WORK_DIR="/taiga/illinois/vetmed/cb/kwang222/Haoyun_Li/RL"
cd "$WORK_DIR"

echo "Activating conda environment..."
source /sw/apps/anaconda3/2024.10/etc/profile.d/conda.sh
conda activate "$WORK_DIR/Bin2Cell_Validation"

# CPU-side thread tuning for data/reward prep while training on GPU.
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export TORCH_NUM_THREADS=16
export TORCH_NUM_INTEROP_THREADS=1

PPO_CONFIG="${PPO_CONFIG:-configs/ppo_training.template.yaml}"
RUN_DEVICE="cuda"
RUN_ROLLOUT_MODE="vectorized"
N_ROLLOUT_WORKERS=1
BATCH_CELLS="${BATCH_CELLS:-100}"
MAX_UPDATES="${MAX_UPDATES:-600}"
RUN_NAME="${RUN_NAME:-human_colorectal_ppo_gpu}"

echo "PPO_CONFIG: $PPO_CONFIG"
echo "RUN_DEVICE: $RUN_DEVICE"
echo "RUN_ROLLOUT_MODE: $RUN_ROLLOUT_MODE"
echo "N_ROLLOUT_WORKERS: $N_ROLLOUT_WORKERS"
echo "TORCH_NUM_THREADS: $TORCH_NUM_THREADS"
echo "TORCH_NUM_INTEROP_THREADS: $TORCH_NUM_INTEROP_THREADS"
echo "BATCH_CELLS: $BATCH_CELLS"
echo "MAX_UPDATES: $MAX_UPDATES"
echo "RUN_NAME: $RUN_NAME"

TMP_PPO_CONFIG="$(mktemp --suffix=.yaml)"
trap 'rm -f "$TMP_PPO_CONFIG"' EXIT
python - "$PPO_CONFIG" "$TMP_PPO_CONFIG" "$RUN_DEVICE" "$RUN_ROLLOUT_MODE" "$N_ROLLOUT_WORKERS" "$BATCH_CELLS" "$MAX_UPDATES" "$RUN_NAME" <<'PY'
import sys
from pathlib import Path
import yaml

src = Path(sys.argv[1]).resolve()
dst = Path(sys.argv[2]).resolve()
run_device = str(sys.argv[3]).strip().lower()
run_rollout_mode = str(sys.argv[4]).strip().lower()
n_rollout_workers = int(sys.argv[5])
batch_cells = int(sys.argv[6])
max_updates = int(sys.argv[7])
run_name = str(sys.argv[8]).strip()

with src.open("r", encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle)

run = cfg.setdefault("run", {})
run["device"] = run_device
run["rollout_mode"] = run_rollout_mode
run["n_rollout_workers"] = n_rollout_workers
run["batch_cells"] = batch_cells
run["max_updates"] = max_updates
if run_name:
    run["name"] = run_name

with dst.open("w", encoding="utf-8") as handle:
    yaml.safe_dump(cfg, handle, sort_keys=False)
PY

python scripts/run_ppo_training.py --config "$TMP_PPO_CONFIG"

conda deactivate
