#!/bin/bash
#SBATCH --job-name=PixelToBin
#SBATCH --account=kwang222-ic
#SBATCH --partition=bioshare
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem=90G
#SBATCH --output=pixel_to_bin_mapping.out
#SBATCH --error=pixel_to_bin_mapping.err

module load anaconda3

# Set up environment
WORK_DIR="/taiga/illinois/vetmed/cb/kwang222/Haoyun_Li/RL"
cd $WORK_DIR

# Activate conda environment
echo "Activating Bin2Cell_Validation conda environment..."
source /sw/apps/anaconda3/2024.10/etc/profile.d/conda.sh
conda activate $WORK_DIR/Bin2Cell_Validation

# Run the pixel -> 2 um bin aggregation script
mkdir -p workspace_outputs/human_colorectal/intermediate
python3 preprocessing/pixel_to_bin_mapping.py \
  --csv-path workspace_outputs/human_colorectal/intermediate/cellpose_sam_human_colorectal_output/cropped_visium_hd_human_Colorectal_pixel_to_cell_mapping_full.csv.gz \
  --output-path workspace_outputs/human_colorectal/intermediate/cropped_visium_hd_human_Colorectal_nuclear_bin_mapping.csv.gz \
  --summary-path workspace_outputs/human_colorectal/intermediate/cropped_visium_hd_human_Colorectal_nuclear_bin_mapping.summary.json \
  --microns-per-pixel 0.2737012522439323 \
  --bin-size-um 2.0 \
  --nuclear-only

# Deactivate conda environment
conda deactivate
