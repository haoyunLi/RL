#!/bin/bash
#SBATCH --job-name=PseudoSquare002um
#SBATCH --account=kwang222-ic
#SBATCH --partition=bioshare
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --mem=120G
#SBATCH --output=pseudo_pixel_to_square_002um_bins.out
#SBATCH --error=pseudo_pixel_to_square_002um_bins.err

set -euo pipefail

module load anaconda3

WORK_DIR="/taiga/illinois/vetmed/cb/kwang222/Haoyun_Li/RL"
cd "$WORK_DIR"

echo "Activating Bin2Cell_Validation conda environment..."
source /sw/apps/anaconda3/2024.10/etc/profile.d/conda.sh
conda activate "$WORK_DIR/Bin2Cell_Validation"

PSEUDO_ROOT="${PSEUDO_ROOT:-colorectal_pseudo_visium_hd_output_full_0.25}"
OUTPUT_DIR="${OUTPUT_DIR:-workspace_outputs/pseudo_human_colorectal/intermediate}"
PIXEL_MAPPING_CSV="${PIXEL_MAPPING_CSV:-$PSEUDO_ROOT/cropped_visium_hd_human_colorectal_pixel_to_cell_mapping_expanded.csv.gz}"
TISSUE_POSITIONS_PARQUET="${TISSUE_POSITIONS_PARQUET:-$PSEUDO_ROOT/spatial/tissue_positions.parquet}"
CROPPED_IMAGE_PATH="${CROPPED_IMAGE_PATH:-workspace_outputs/pseudo_human_colorectal/intermediate/cropped_visium_hd_human_Colorectal.png}"
MICRONS_PER_PIXEL="${MICRONS_PER_PIXEL:-0.2737012522439323}"
CHUNK_SIZE="${CHUNK_SIZE:-10000000}"
OVERLAY_DOWNSAMPLE_FACTOR="${OVERLAY_DOWNSAMPLE_FACTOR:-2}"
NUCLEAR_ONLY="${NUCLEAR_ONLY:-1}"
IN_TISSUE_ONLY="${IN_TISSUE_ONLY:-0}"

mkdir -p "$OUTPUT_DIR"

if [[ "$NUCLEAR_ONLY" == "1" ]]; then
  MODE_SUFFIX="nuclear_bins"
else
  MODE_SUFFIX="cell_bins"
fi

OUTPUT_PATH="$OUTPUT_DIR/cropped_visium_hd_human_colorectal_pseudo_square_002um_${MODE_SUFFIX}.csv.gz"
SUMMARY_PATH="$OUTPUT_DIR/cropped_visium_hd_human_colorectal_pseudo_square_002um_${MODE_SUFFIX}.summary.json"
OVERLAY_PATH="$OUTPUT_DIR/cropped_visium_hd_human_colorectal_pseudo_square_002um_${MODE_SUFFIX}_overlay.png"

echo "PIXEL_MAPPING_CSV: $PIXEL_MAPPING_CSV"
echo "TISSUE_POSITIONS_PARQUET: $TISSUE_POSITIONS_PARQUET"
echo "OUTPUT_PATH: $OUTPUT_PATH"
echo "SUMMARY_PATH: $SUMMARY_PATH"
echo "OVERLAY_PATH: $OVERLAY_PATH"
echo "NUCLEAR_ONLY: $NUCLEAR_ONLY"
echo "IN_TISSUE_ONLY: $IN_TISSUE_ONLY"

CMD=(
  python3 preprocessing/pixel_to_square_002um_bins.py
  --pixel-mapping-csv "$PIXEL_MAPPING_CSV"
  --tissue-positions-parquet "$TISSUE_POSITIONS_PARQUET"
  --output-path "$OUTPUT_PATH"
  --summary-path "$SUMMARY_PATH"
  --overlay-path "$OVERLAY_PATH"
  --cropped-image-path "$CROPPED_IMAGE_PATH"
  --overlay-downsample-factor "$OVERLAY_DOWNSAMPLE_FACTOR"
  --microns-per-pixel "$MICRONS_PER_PIXEL"
  --chunk-size "$CHUNK_SIZE"
)

if [[ "$NUCLEAR_ONLY" == "1" ]]; then
  CMD+=(--nuclear-only)
fi

if [[ "$IN_TISSUE_ONLY" == "1" ]]; then
  CMD+=(--in-tissue-only)
fi

"${CMD[@]}"

conda deactivate
