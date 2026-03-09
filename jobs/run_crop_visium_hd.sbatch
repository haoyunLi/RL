#!/bin/bash
#SBATCH --job-name=Crop-VisiumHD
#SBATCH --account=kwang222-ic
#SBATCH --partition=bioshare
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=80G
#SBATCH --output=crop_visium_hd.out
#SBATCH --error=crop_visium_hd.err

module load anaconda3

# Set up environment
WORK_DIR="/taiga/illinois/vetmed/cb/kwang222/Haoyun_Li/RL"
cd $WORK_DIR

# Activate conda environment
echo "Activating Cellpose conda environment..."
source /sw/apps/anaconda3/2024.10/etc/profile.d/conda.sh
conda activate $WORK_DIR/Bin2Cell_Validation

# Run the Python script
mkdir -p workspace_outputs/human_colorectal/intermediate
python3 preprocessing/crop_visium_hd_script.py
