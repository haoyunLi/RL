#!/bin/bash
#SBATCH --job-name=MergeSquare002um
#SBATCH --account=kwang222-ic
#SBATCH --partition=bioshare
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem=160G
#SBATCH --output=merge_square_002um_expression_with_nuclear.out
#SBATCH --error=merge_square_002um_expression_with_nuclear.err

module load anaconda3

# Set up environment
WORK_DIR="/taiga/illinois/vetmed/cb/kwang222/Haoyun_Li/RL"
cd $WORK_DIR

# Activate conda environment
echo "Activating Bin2Cell_Validation conda environment..."
source /sw/apps/anaconda3/2024.10/etc/profile.d/conda.sh
conda activate $WORK_DIR/Bin2Cell_Validation

# Build all-bin RL metadata aligned to official square_002um barcodes
mkdir -p workspace_outputs/human_colorectal/intermediate/square_002um_nuclear
python3 preprocessing/merge_square_002um_expression_with_nuclear.py \
  --nuclear-annotation-path workspace_outputs/human_colorectal/intermediate/cropped_visium_hd_human_Colorectal_square_002um_nuclear_bins.csv.gz \
  --matrix-h5-path Human_Colorectal/output/binned_outputs/square_002um/raw_feature_bc_matrix.h5 \
  --tissue-positions-parquet Human_Colorectal/output/binned_outputs/square_002um/spatial/tissue_positions.parquet \
  --barcodes-tsv-path Human_Colorectal/output/binned_outputs/square_002um/raw_feature_bc_matrix/barcodes.tsv.gz \
  --output-dir workspace_outputs/human_colorectal/intermediate/square_002um_nuclear \
  --prefix human_colorectal_square_002um_rl

# Deactivate conda environment
conda deactivate
