#!/bin/bash
#SBATCH --job-name=setup_stcs
#SBATCH --account=kwang222-ic
#SBATCH --partition=bioshare
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=setup_stcs_env.out
#SBATCH --error=setup_stcs_env.err

set -euo pipefail

WORK_DIR="/taiga/illinois/vetmed/cb/kwang222/Haoyun_Li/RL"
cd "$WORK_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
STCS_ENV_DIR="${STCS_ENV_DIR:-$WORK_DIR/stcs_env}"
STCS_REPO_DIR="${STCS_REPO_DIR:-$WORK_DIR/external/STCS}"

echo "Working directory: $WORK_DIR"
echo "Python: $($PYTHON_BIN --version)"
echo "STCS_ENV_DIR: $STCS_ENV_DIR"
echo "STCS_REPO_DIR: $STCS_REPO_DIR"

if [[ ! -d "$STCS_ENV_DIR" ]]; then
  "$PYTHON_BIN" -m venv "$STCS_ENV_DIR"
fi

source "$STCS_ENV_DIR/bin/activate"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib_${SLURM_JOB_ID:-$$}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/xdg_cache_${SLURM_JOB_ID:-$$}}"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

python -m pip install --upgrade pip wheel "setuptools<81"
python -m pip install \
  "numpy==1.26.4" "pandas<2.3" scipy scikit-learn h5py "anndata<0.12" "scanpy<1.11" squidpy \
  matplotlib tqdm tifffile imageio opencv-python-headless \
  stardist tensorflow decoupler celltypist shapely pyyaml datashader

mkdir -p "$(dirname "$STCS_REPO_DIR")"
if [[ ! -d "$STCS_REPO_DIR/.git" ]]; then
  git clone --depth 1 https://github.com/YangLabRutgers/STCS.git "$STCS_REPO_DIR"
else
  git -C "$STCS_REPO_DIR" pull --ff-only
fi

STCS_REPO_DIR="$STCS_REPO_DIR" python - <<'PY'
import importlib
import os
import sys

mods = ["scanpy", "squidpy", "stardist", "tensorflow", "decoupler", "celltypist", "cv2", "tifffile"]
for mod in mods:
    importlib.import_module(mod)
    print(f"{mod}: OK")

sys.path.insert(0, os.path.join(os.environ["STCS_REPO_DIR"], "STCS"))
importlib.import_module("STCS_main")
print("STCS_main: OK")
PY

echo "STCS environment setup complete at $(date)"
