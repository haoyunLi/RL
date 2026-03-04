from __future__ import annotations

import gzip
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
import pandas as pd

from preprocessing.merge_square_002um_expression_with_nuclear import (
    build_square_002um_rl_metadata,
    write_rl_metadata_outputs,
)


class MergeSquare002umExpressionWithNuclearTests(unittest.TestCase):
    def test_builds_all_bin_metadata_with_left_joined_nuclear_info(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            nuclear_path = root / "nuclear.csv.gz"
            matrix_h5_path = root / "raw_feature_bc_matrix.h5"
            barcodes_tsv_path = root / "barcodes.tsv.gz"
            tissue_path = root / "tissue_positions.parquet"
            output_dir = root / "merged"

            nuclear_df = pd.DataFrame(
                {
                    "bin_id": ["BC0", "BC2", "BC2"],
                    "barcode": ["BC0", "BC2", "BC2"],
                    "array_row": [0, 1, 1],
                    "array_col": [0, 2, 2],
                    "in_tissue": [1, 1, 1],
                    "x_um": [1.0, 3.0, 3.0],
                    "y_um": [1.0, 1.0, 1.0],
                    "pxl_row_in_fullres": [1.0, 1.0, 1.0],
                    "pxl_col_in_fullres": [1.0, 3.0, 3.0],
                    "cell_id": [10, 20, 21],
                    "cell_type": [None, None, None],
                    "weight": [1.0, 0.6, 0.4],
                    "nuclear_pixel_count": [5, 6, 4],
                }
            )
            nuclear_df.to_csv(nuclear_path, index=False, compression="gzip")

            tissue_df = pd.DataFrame(
                {
                    "barcode": ["BC0", "BC1", "BC2"],
                    "in_tissue": [1, 1, 1],
                    "array_row": [0, 0, 1],
                    "array_col": [0, 1, 2],
                    "pxl_row_in_fullres": [1.0, 1.0, 3.0],
                    "pxl_col_in_fullres": [1.0, 3.0, 5.0],
                }
            )
            tissue_df.to_parquet(tissue_path, index=False)

            with h5py.File(matrix_h5_path, "w") as h5:
                g = h5.create_group("matrix")
                g.create_dataset("shape", data=np.array([3, 3], dtype=np.int32))
                g.create_dataset("barcodes", data=np.array([b"BC0", b"BC1", b"BC2"]))
                fg = g.create_group("features")
                fg.create_dataset("id", data=np.array([b"g0", b"g1", b"g2"]))
                fg.create_dataset("name", data=np.array([b"GENE0", b"GENE1", b"CTRL"]))
                fg.create_dataset("feature_type", data=np.array([b"Gene Expression", b"Gene Expression", b"Antibody Capture"]))
                fg.create_dataset("genome", data=np.array([b"GRCh38", b"GRCh38", b"GRCh38"]))

            with gzip.open(barcodes_tsv_path, "wt") as handle:
                handle.write("BC0\nBC1\nBC2\n")

            metadata_df, claims_df, selected_features_df, selected_feature_indices, manifest = build_square_002um_rl_metadata(
                nuclear_annotation_path=nuclear_path,
                matrix_h5_path=matrix_h5_path,
                tissue_positions_parquet=tissue_path,
                barcodes_tsv_path=barcodes_tsv_path,
                microns_per_pixel=1.0,
                feature_type_filter="Gene Expression",
            )

            self.assertEqual(metadata_df["barcode"].tolist(), ["BC0", "BC1", "BC2"])
            self.assertEqual(metadata_df["matrix_col_index"].tolist(), [0, 1, 2])
            self.assertEqual(metadata_df["has_nuclear_annotation"].tolist(), [True, False, True])
            self.assertEqual(metadata_df["total_nuclear_pixel_count"].tolist(), [5, 0, 10])
            self.assertEqual(metadata_df["n_cell_claims"].tolist(), [1, 0, 2])
            self.assertTrue(bool(metadata_df.loc[metadata_df["barcode"] == "BC2", "ambiguous_nuclear_assignment"].iloc[0]))
            self.assertEqual(claims_df["matrix_col_index"].tolist(), [0, 2, 2])
            self.assertEqual(selected_features_df["feature_name"].tolist(), ["GENE0", "GENE1"])
            self.assertEqual(selected_feature_indices.tolist(), [0, 1])
            self.assertEqual(manifest["n_barcodes"], 3)
            self.assertEqual(manifest["n_selected_features"], 2)

            paths = write_rl_metadata_outputs(
                metadata_df,
                claims_df,
                selected_features_df,
                selected_feature_indices,
                manifest,
                output_dir=output_dir,
                prefix="test_rl",
            )
            self.assertTrue(paths["metadata"].exists())
            self.assertTrue(paths["claims"].exists())
            self.assertTrue(paths["selected_feature_indices"].exists())
            self.assertTrue(paths["selected_features"].exists())
            self.assertTrue(paths["manifest"].exists())


if __name__ == "__main__":
    unittest.main()
