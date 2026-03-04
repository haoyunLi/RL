from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from preprocessing.build_nuclei_table_from_pixel_segmentation import build_nuclei_table_from_pixel_segmentation


class BuildNucleiTableFromPixelSegmentationTests(unittest.TestCase):
    def test_builds_centroid_and_equivalent_radius(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            pixel_path = root / 'pixel_mapping.csv.gz'
            tissue_path = root / 'tissue_positions.parquet'

            # 2x2 official grid with crop offsets row_left=0, col_up=0.
            tissue_df = pd.DataFrame(
                {
                    'barcode': ['b0', 'b1', 'b2', 'b3'],
                    'in_tissue': [1, 1, 1, 1],
                    'array_row': [0, 0, 1, 1],
                    'array_col': [0, 1, 0, 1],
                    'pxl_row_in_fullres': [1.0, 1.0, 3.0, 3.0],
                    'pxl_col_in_fullres': [1.0, 3.0, 1.0, 3.0],
                }
            )
            tissue_df.to_parquet(tissue_path, index=False)

            pixel_df = pd.DataFrame(
                {
                    'x': [0, 1, 4, 4, 5],
                    'y': [0, 0, 3, 4, 4],
                    'cell_id': [1, 1, 2, 2, 2],
                    'is_nuclear': [1, 1, 1, 1, 1],
                }
            )
            pixel_df.to_csv(pixel_path, index=False, compression='gzip')

            nuclei_df, crop_meta = build_nuclei_table_from_pixel_segmentation(
                pixel_mapping_csv=pixel_path,
                tissue_positions_parquet=tissue_path,
                microns_per_pixel=1.0,
                chunk_size=10,
            )

            self.assertEqual(crop_meta['col_up'], 0)
            self.assertEqual(crop_meta['row_left'], 0)
            self.assertEqual(nuclei_df['cell_id'].tolist(), [1, 2])
            self.assertAlmostEqual(float(nuclei_df.loc[0, 'center_x_um']), 0.5)
            self.assertAlmostEqual(float(nuclei_df.loc[0, 'center_y_um']), 0.0)
            self.assertEqual(int(nuclei_df.loc[0, 'nuclear_pixel_count']), 2)
            self.assertGreater(float(nuclei_df.loc[0, 'radius_um']), 0.0)
            self.assertAlmostEqual(float(nuclei_df.loc[1, 'center_x_um']), (4 + 4 + 5) / 3.0)
            self.assertAlmostEqual(float(nuclei_df.loc[1, 'center_y_um']), (3 + 4 + 4) / 3.0)


if __name__ == '__main__':
    unittest.main()
