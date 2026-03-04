from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import pandas as pd
from PIL import Image

from preprocessing.pixel_to_square_002um_bins import (
    _build_crop_region_bins,
    annotate_square_002um_bins_from_pixels,
    write_overlay_png,
)


class PixelToSquare002umBinsTests(unittest.TestCase):
    def test_maps_pixels_to_official_barcodes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            tissue_path = root / 'tissue_positions.parquet'
            pixel_path = root / 'pixel_mapping.csv.gz'

            tissue_df = pd.DataFrame(
                {
                    'barcode': ['s_002um_00000_00000-1', 's_002um_00000_00001-1', 's_002um_00001_00000-1', 's_002um_00001_00001-1'],
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
                    'x': [0, 1, 2, 3],
                    'y': [0, 0, 0, 0],
                    'cell_id': [7, 7, 7, 7],
                    'is_boundary': [0, 0, 0, 0],
                    'is_interior': [1, 1, 1, 1],
                    'is_nuclear': [1, 1, 1, 1],
                    'is_cytoplasm': [0, 0, 0, 0],
                }
            )
            pixel_df.to_csv(pixel_path, index=False, compression='gzip')

            out = annotate_square_002um_bins_from_pixels(
                pixel_mapping_csv=pixel_path,
                tissue_positions_parquet=tissue_path,
                microns_per_pixel=1.0,
                chunk_size=100,
                nuclear_only=True,
            )

            self.assertEqual(len(out), 2)
            self.assertEqual(sorted(out['barcode'].tolist()), ['s_002um_00000_00000-1', 's_002um_00000_00001-1'])
            self.assertTrue((out['bin_id'] == out['barcode']).all())
            self.assertTrue((out['is_nuclear'] == 1).all())
            self.assertTrue((out['weight'] == 1.0).all())
            self.assertEqual(int(out['nuclear_pixel_count'].sum()), 4)

    def test_overlay_png_is_written(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            tissue_path = root / 'tissue_positions.parquet'
            pixel_path = root / 'pixel_mapping.csv.gz'
            overlay_path = root / 'overlay.png'
            cropped_image_path = root / 'cropped.png'

            tissue_df = pd.DataFrame(
                {
                    'barcode': ['s_002um_00000_00000-1', 's_002um_00000_00001-1', 's_002um_00001_00000-1', 's_002um_00001_00001-1'],
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
                    'x': [0, 1, 2, 3],
                    'y': [0, 0, 0, 0],
                    'cell_id': [7, 7, 7, 7],
                    'is_boundary': [0, 0, 0, 0],
                    'is_interior': [1, 1, 1, 1],
                    'is_nuclear': [1, 1, 1, 1],
                    'is_cytoplasm': [0, 0, 0, 0],
                }
            )
            pixel_df.to_csv(pixel_path, index=False, compression='gzip')
            Image.new('RGB', (4, 4), color=(255, 255, 255)).save(cropped_image_path)

            out = annotate_square_002um_bins_from_pixels(
                pixel_mapping_csv=pixel_path,
                tissue_positions_parquet=tissue_path,
                microns_per_pixel=1.0,
                chunk_size=100,
                nuclear_only=True,
            )
            crop_bins, crop_meta = _build_crop_region_bins(pd.read_parquet(tissue_path), row_number=None, col_number=None)
            write_overlay_png(
                out,
                crop_bins,
                crop_meta,
                overlay_path,
                cropped_image_path=cropped_image_path,
                downsample_factor=1,
            )

            self.assertTrue(overlay_path.exists())
            self.assertGreater(overlay_path.stat().st_size, 0)


if __name__ == '__main__':
    unittest.main()
