#!/bin/bash
#SBATCH --job-name=Square002umNuclear
#SBATCH --account=kwang222-ic
#SBATCH --partition=bioshare
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --mem=120G
#SBATCH --output=pixel_to_square_002um_bins.out
#SBATCH --error=pixel_to_square_002um_bins.err

module load anaconda3

# Set up environment
WORK_DIR="/taiga/illinois/vetmed/cb/kwang222/Haoyun_Li/RL"
cd $WORK_DIR

# Activate conda environment
echo "Activating Bin2Cell_Validation conda environment..."
source /sw/apps/anaconda3/2024.10/etc/profile.d/conda.sh
conda activate $WORK_DIR/Bin2Cell_Validation

# Run the exact official-bin nuclear annotation step
mkdir -p workspace_outputs/human_colorectal/intermediate
python3 preprocessing/pixel_to_square_002um_bins.py \
  --pixel-mapping-csv workspace_outputs/human_colorectal/intermediate/cellpose_sam_human_colorectal_output/cropped_visium_hd_human_Colorectal_pixel_to_cell_mapping_full.csv.gz \
  --tissue-positions-parquet Human_Colorectal/output/binned_outputs/square_002um/spatial/tissue_positions.parquet \
  --output-path workspace_outputs/human_colorectal/intermediate/cropped_visium_hd_human_Colorectal_square_002um_nuclear_bins.csv.gz \
  --summary-path workspace_outputs/human_colorectal/intermediate/cropped_visium_hd_human_Colorectal_square_002um_nuclear_bins.summary.json \
  --overlay-path workspace_outputs/human_colorectal/intermediate/cropped_visium_hd_human_Colorectal_square_002um_nuclear_bins_overlay.png \
  --cropped-image-path workspace_outputs/human_colorectal/intermediate/cropped_visium_hd_human_Colorectal.png \
  --overlay-downsample-factor 2 \
  --microns-per-pixel 0.2737012522439323 \
  --chunk-size 10000000 \
  --nuclear-only

# Deactivate conda environment
conda deactivate
