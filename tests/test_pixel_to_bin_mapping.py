from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from preprocessing.pixel_to_bin_mapping import load_pixel_to_cell_mapping


class PixelToBinMappingTests(unittest.TestCase):
    def test_aggregates_pixels_into_2um_bins_and_splits_mixed_compartments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "pixel_mapping.csv.gz"
            df = pd.DataFrame(
                {
                    "x": [0, 1, 6, 6],
                    "y": [0, 0, 0, 0],
                    "cell_id": [11, 11, 11, 11],
                    "is_boundary": [0, 0, 0, 0],
                    "is_interior": [1, 1, 1, 1],
                    "is_nuclear": [1, 0, 1, 0],
                    "is_cytoplasm": [0, 1, 0, 1],
                }
            )
            df.to_csv(csv_path, index=False, compression="gzip")

            out = load_pixel_to_cell_mapping(
                csv_path=csv_path,
                aggregate_to_bins=True,
                microns_per_pixel=1.0,
                bin_size_um=2.0,
            )

            # Pixels x=0,1 fall into bin 0; x=6 falls into bin 3.
            self.assertEqual(len(out), 4)
            self.assertEqual(sorted(out["bin_x_index"].unique().tolist()), [0, 3])
            self.assertEqual(sorted(out["cell_id"].unique().tolist()), [11])
            self.assertEqual(int((out["is_nuclear"] == 1).sum()), 2)
            self.assertEqual(int((out["is_cytoplasm"] == 1).sum()), 2)

            first_bin = out[(out["bin_x_index"] == 0) & (out["is_nuclear"] == 1)].iloc[0]
            self.assertAlmostEqual(float(first_bin["weight"]), 0.5)
            self.assertEqual(int(first_bin["nuclear_pixel_count"]), 1)
            self.assertEqual(int(first_bin["cytoplasm_pixel_count"]), 1)

    def test_nuclear_only_filters_non_nuclear_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "pixel_mapping.csv.gz"
            df = pd.DataFrame(
                {
                    "x": [0, 1],
                    "y": [0, 0],
                    "cell_id": [5, 5],
                    "is_boundary": [0, 0],
                    "is_interior": [1, 1],
                    "is_nuclear": [1, 0],
                    "is_cytoplasm": [0, 1],
                }
            )
            df.to_csv(csv_path, index=False, compression="gzip")

            out = load_pixel_to_cell_mapping(
                csv_path=csv_path,
                aggregate_to_bins=True,
                microns_per_pixel=1.0,
                bin_size_um=2.0,
                nuclear_only=True,
            )

            self.assertEqual(len(out), 1)
            self.assertEqual(int(out.iloc[0]["is_nuclear"]), 1)
            self.assertEqual(int(out.iloc[0]["is_cytoplasm"]), 0)


if __name__ == "__main__":
    unittest.main()
