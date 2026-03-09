#!/bin/bash
#SBATCH --job-name=EpisodeBuild
#SBATCH --account=kwang222-ic
#SBATCH --partition=bioshare
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --mem=900G
#SBATCH --output=episode_build.out
#SBATCH --error=episode_build.err

set -euo pipefail

module load anaconda3

WORK_DIR="/taiga/illinois/vetmed/cb/kwang222/Haoyun_Li/RL"
cd "$WORK_DIR"

echo "Activating conda environment..."
source /sw/apps/anaconda3/2024.10/etc/profile.d/conda.sh
conda activate "$WORK_DIR/Bin2Cell_Validation"

# Avoid thread oversubscription; episode build uses process-based workers.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

EPISODE_CONFIG="${EPISODE_CONFIG:-configs/episode_build.template.yaml}"
EPISODE_LIMIT_NUCLEI="${EPISODE_LIMIT_NUCLEI:-}"

echo "Running episode build with config: $EPISODE_CONFIG"
if [ -n "$EPISODE_LIMIT_NUCLEI" ]; then
  python scripts/run_episode_build.py --config "$EPISODE_CONFIG" --limit-nuclei "$EPISODE_LIMIT_NUCLEI"
else
  python scripts/run_episode_build.py --config "$EPISODE_CONFIG"
fi

conda deactivate
