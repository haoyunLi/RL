#!/bin/bash
#SBATCH --job-name=PPODebugApp
#SBATCH --account=kwang222-ic
#SBATCH --partition=bioshare
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=200G
#SBATCH --output=ppo_debug_app.out
#SBATCH --error=ppo_debug_app.err

set -euo pipefail

module load anaconda3

WORK_DIR="/taiga/illinois/vetmed/cb/kwang222/Haoyun_Li/RL"
cd "$WORK_DIR"

echo "Activating conda environment..."
source /sw/apps/anaconda3/2024.10/etc/profile.d/conda.sh
conda activate "$WORK_DIR/Bin2Cell_Validation"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/${USER}/mpl-cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/${USER}/xdg-cache}"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

PPO_EVAL_RUN_DIR="${PPO_EVAL_RUN_DIR:-runs/human_colorectal_ppo_eval_20260423T164043Z}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
RUN_DEVICE="${RUN_DEVICE:-cpu}"
CELL_ID="${CELL_ID:-}"
APP_PORT="${APP_PORT:-$(shuf -i 8501-8999 -n 1)}"
APP_ADDRESS="${APP_ADDRESS:-0.0.0.0}"

if [[ ! -d "$PPO_EVAL_RUN_DIR" ]]; then
  echo "PPO eval run directory not found: $PPO_EVAL_RUN_DIR" >&2
  exit 1
fi

echo "PPO_EVAL_RUN_DIR: $PPO_EVAL_RUN_DIR"
echo "CHECKPOINT_PATH: ${CHECKPOINT_PATH:-<from summary.json>}"
echo "RUN_DEVICE: $RUN_DEVICE"
echo "CELL_ID: ${CELL_ID:-<app default>}"
echo "APP_ADDRESS: $APP_ADDRESS"
echo "APP_PORT: $APP_PORT"

python - <<'PY'
import importlib.util
import sys

required = ("torch", "streamlit", "plotly")
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    print(
        "Missing required Python packages in current env: " + ", ".join(missing),
        file=sys.stderr,
    )
    sys.exit(1)
PY

NODE_HOST="$(hostname -s)"
NODE_IP="$(
python - <<'PY'
import socket

def resolve_ip() -> str:
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.connect(("8.8.8.8", 80))
            ip = sock.getsockname()[0]
            if ip and not ip.startswith("127."):
                return ip
        finally:
            sock.close()
    except Exception:
        pass

    try:
        for family, _, _, _, sockaddr in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = sockaddr[0]
            if ip and not ip.startswith("127."):
                return ip
    except Exception:
        pass

    return "127.0.0.1"

print(resolve_ip())
PY
)"
PUBLIC_HOST="${PUBLIC_HOST:-$NODE_IP}"

echo "======================================"
echo "PPO debug app starting"
echo "Node: $NODE_HOST"
echo "Node IP: $NODE_IP"
echo "Port: $APP_PORT"
echo "======================================"
echo "LAN URL (if your cluster network allows direct access):"
echo "http://${PUBLIC_HOST}:${APP_PORT}"
echo ""
echo "From your local machine, run:"
echo "ssh -N -L ${APP_PORT}:${NODE_HOST}:${APP_PORT} ${USER}@cc-login.campuscluster.illinois.edu"
echo ""
echo "Then open in your browser:"
echo "http://localhost:${APP_PORT}"
echo "======================================"

CMD=(python -m streamlit run scripts/ppo_debug_app.py
  --server.headless true
  --server.address "$APP_ADDRESS"
  --server.port "$APP_PORT"
  --browser.serverAddress "$PUBLIC_HOST"
  --browser.serverPort "$APP_PORT"
  --
  --ppo-eval-run-dir "$PPO_EVAL_RUN_DIR"
  --device "$RUN_DEVICE"
)

if [[ -n "$CHECKPOINT_PATH" ]]; then
  CMD+=(--checkpoint "$CHECKPOINT_PATH")
fi
if [[ -n "$CELL_ID" ]]; then
  CMD+=(--cell-id "$CELL_ID")
fi

"${CMD[@]}"

conda deactivate
