#!/bin/bash
#SBATCH --job-name=BuildRefCountsSCT
#SBATCH --account=kwang222-ic
#SBATCH --partition=bioshare
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=80G
#SBATCH --output=build_reference_counts_sct.out
#SBATCH --error=build_reference_counts_sct.err

module load anaconda3

WORK_DIR="/taiga/illinois/vetmed/cb/kwang222/Haoyun_Li/RL"
cd "$WORK_DIR"

echo "Activating conda environment..."
source /sw/apps/anaconda3/2024.10/etc/profile.d/conda.sh
conda activate "$WORK_DIR/Bin2Cell_Validation"

mkdir -p workspace_outputs/human_colorectal/intermediate/reference_sct

python preprocessing/build_hd_gene_allowlist.py \
  --selected-features-tsv workspace_outputs/human_colorectal/intermediate/square_002um_nuclear/human_colorectal_square_002um_rl.selected_features.tsv.gz \
  --output-txt workspace_outputs/human_colorectal/intermediate/reference_sct/hd_selected_feature_names_unique.txt

python preprocessing/build_reference_counts_from_scrna.py \
  --matrix-h5-path colorectal_sc_data/CRC_GSE166555_raw_tumor.h5 \
  --metadata-tsv-path colorectal_sc_data/GSE166555_meta_data_tumor.tsv \
  --cell-id-column cell_id \
  --cell-type-column sct_cell_type \
  --min-cells-per-type 50 \
  --gene-allowlist-path workspace_outputs/human_colorectal/intermediate/reference_sct/hd_selected_feature_names_unique.txt \
  --output-path workspace_outputs/human_colorectal/intermediate/reference_sct/reference_counts_sct_tumor_aligned_hd_overlap_unique.npz \
  --summary-path workspace_outputs/human_colorectal/intermediate/reference_sct/reference_counts_sct_tumor_aligned_hd_overlap_unique.summary.json

conda deactivate
