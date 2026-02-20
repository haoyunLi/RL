"""Tests for reproducible episode-build runner."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd
import yaml

from hd_cell_rl.episode_build import run_episode_build_from_config


class EpisodeBuildRunnerTest(unittest.TestCase):
    """Validate config-driven episode build and artifact outputs."""

    def _write_base_inputs(self, work_dir: Path) -> tuple[Path, Path]:
        nuclei = pd.DataFrame(
            {
                "cell_id": ["cell_a", "cell_b"],
                "center_x_um": [0.0, 50.0],
                "center_y_um": [0.0, 50.0],
                "radius_um": [10.0, 12.0],
            }
        )
        bins = pd.DataFrame(
            {
                "bin_id": ["b1", "b2", "b3", "b4"],
                "x_um": [1.0, 10.0, 45.0, 200.0],
                "y_um": [1.0, 4.0, 55.0, 200.0],
                "expr_g1": [1.0, 0.0, 2.5, 0.1],
                "expr_g2": [0.1, 2.0, 0.0, 0.5],
            }
        )

        nuclei_path = work_dir / "nuclei.csv"
        bins_path = work_dir / "bins.csv"
        nuclei.to_csv(nuclei_path, index=False)
        bins.to_csv(bins_path, index=False)
        return nuclei_path, bins_path

    def _write_config(self, work_dir: Path, nuclei_path: Path, bins_path: Path) -> Path:
        cfg = {
            "run": {
                "name": "test_build",
                "output_root": str(work_dir / "runs"),
                "seed": 123,
            },
            "inputs": {
                "nuclei_path": str(nuclei_path),
                "bins_path": str(bins_path),
                "nuclei_format": "csv",
                "bins_format": "csv",
                "nuclei_columns": {
                    "cell_id": "cell_id",
                    "center_x_um": "center_x_um",
                    "center_y_um": "center_y_um",
                    "radius_um": "radius_um",
                    "cell_type": None,
                },
                "bin_columns": {
                    "bin_id": "bin_id",
                    "x_um": "x_um",
                    "y_um": "y_um",
                },
                "expression": {
                    "mode": "prefix",
                    "prefix": "expr_",
                    "columns": [],
                },
            },
            "environment": {
                "max_center_distance_um": 80.0,
                "radius_band_um": 80.0,
                "strict_action_validation": True,
                "max_steps": None,
                "default_steps_multiplier": 3,
            },
        }

        config_path = work_dir / "config.yaml"
        with config_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(cfg, handle, sort_keys=False)

        return config_path

    def test_runner_writes_expected_artifacts(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            work_dir = Path(tmp_dir)
            nuclei_path, bins_path = self._write_base_inputs(work_dir)
            config_path = self._write_config(work_dir, nuclei_path, bins_path)

            run_dir = run_episode_build_from_config(config_path)

            self.assertTrue((run_dir / "config" / "config_resolved.yaml").exists())
            self.assertTrue((run_dir / "config" / "metadata.json").exists())
            self.assertTrue((run_dir / "logs" / "steps.jsonl").exists())
            self.assertTrue((run_dir / "summary.json").exists())
            self.assertTrue((run_dir / "episodes_index.csv").exists())

            states_dir = run_dir / "states"
            self.assertTrue(states_dir.exists())
            npz_files = list(states_dir.glob("*.npz"))
            self.assertEqual(len(npz_files), 2)

            summary = pd.read_json(run_dir / "summary.json", typ="series")
            self.assertEqual(int(summary["n_episodes"]), 2)
            self.assertEqual(int(summary["n_input_nuclei"]), 2)
            self.assertEqual(int(summary["n_input_bins"]), 4)

            with (run_dir / "logs" / "steps.jsonl").open("r", encoding="utf-8") as handle:
                events = [json.loads(line)["event"] for line in handle if line.strip()]

            self.assertIn("run_start", events)
            self.assertIn("state_snapshot_saved", events)
            self.assertIn("run_complete", events)

    def test_missing_required_column_fails_fast(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            work_dir = Path(tmp_dir)
            nuclei_path, bins_path = self._write_base_inputs(work_dir)
            config_path = self._write_config(work_dir, nuclei_path, bins_path)

            bins_df = pd.read_csv(bins_path)
            bins_df = bins_df.drop(columns=["x_um"])
            bins_df.to_csv(bins_path, index=False)

            with self.assertRaises(ValueError):
                run_episode_build_from_config(config_path)


if __name__ == "__main__":
    unittest.main()
