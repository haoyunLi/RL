#!/bin/bash
#SBATCH --job-name=BuildNucleiTable
#SBATCH --account=kwang222-ic
#SBATCH --partition=bioshare
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=80G
#SBATCH --output=build_nuclei_table.out
#SBATCH --error=build_nuclei_table.err

module load anaconda3

WORK_DIR="/taiga/illinois/vetmed/cb/kwang222/Haoyun_Li/RL"
cd "$WORK_DIR"

echo "Activating conda environment..."
source /sw/apps/anaconda3/2024.10/etc/profile.d/conda.sh
conda activate "$WORK_DIR/Bin2Cell_Validation"

mkdir -p workspace_outputs/human_colorectal/intermediate

python preprocessing/build_nuclei_table_from_pixel_segmentation.py \
  --pixel-mapping-csv workspace_outputs/human_colorectal/intermediate/cellpose_sam_human_colorectal_output/cropped_visium_hd_human_Colorectal_pixel_to_cell_mapping_full.csv.gz \
  --tissue-positions-parquet Human_Colorectal/output/binned_outputs/square_002um/spatial/tissue_positions.parquet \
  --output-path workspace_outputs/human_colorectal/intermediate/human_colorectal_nuclei.parquet \
  --summary-path workspace_outputs/human_colorectal/intermediate/human_colorectal_nuclei.summary.json \
  --microns-per-pixel 0.2737012522439323 \
  --chunk-size 1000000
