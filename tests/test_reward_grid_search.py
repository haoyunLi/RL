from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import yaml

from hd_cell_rl.reward_grid_search import (
    GridAxis,
    load_reward_grid_search_config,
    run_reward_grid_search_from_config,
)


class RewardGridSearchTests(unittest.TestCase):
    def test_grid_axis_is_inclusive(self) -> None:
        axis = GridAxis(start=0.1, stop=0.3, step=0.1)
        self.assertEqual(axis.values(), (0.1, 0.2, 0.3))

    def test_run_reward_grid_search_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            episodes_dir = root / "episodes"
            episodes_dir.mkdir(parents=True, exist_ok=True)

            reference_path = root / "reference_counts.csv"
            pd.DataFrame(
                {
                    "cell_type": ["A", "B"],
                    "gene_0": [10.0, 1.0],
                    "gene_1": [1.0, 10.0],
                }
            ).to_csv(reference_path, index=False)

            nuclei_path = root / "nuclei.csv"
            pd.DataFrame(
                {
                    "cell_id": ["cell_1", "cell_2"],
                    "center_x_um": [0.0, 20.0],
                    "center_y_um": [0.0, 0.0],
                }
            ).to_csv(nuclei_path, index=False)

            artifact_path = episodes_dir / "state_000000_cell_1.npz"
            np.savez_compressed(
                artifact_path,
                cell_id=np.asarray(["cell_1"], dtype=object),
                cell_type=np.asarray(["A"], dtype=object),
                nucleus_center_xy_um=np.asarray([0.0, 0.0], dtype=np.float32),
                nucleus_radius_um=np.asarray([5.0], dtype=np.float32),
                candidate_bin_ids=np.asarray(["bin_0", "bin_1"], dtype=object),
                candidate_bin_xy_um=np.asarray([[1.0, 0.0], [12.0, 0.0]], dtype=np.float32),
                candidate_expression=np.asarray([[5.0, 0.0], [0.0, 4.0]], dtype=np.float32),
            )

            episodes_index_path = root / "episodes_index.csv"
            pd.DataFrame(
                {
                    "episode_index": [0],
                    "cell_id": ["cell_1"],
                    "artifact_path": [str(artifact_path)],
                }
            ).to_csv(episodes_index_path, index=False)

            config_path = root / "reward_grid_search.yaml"
            config = {
                "run": {
                    "name": "reward_grid_search_test",
                    "output_root": str(root / "runs"),
                    "seed": 3,
                    "max_episodes": 1,
                },
                "inputs": {
                    "episodes_index_path": str(episodes_index_path),
                    "reference": {
                        "path": str(reference_path),
                        "format": "csv",
                        "cell_type_column": "cell_type",
                        "gene_mode": "prefix",
                        "gene_prefix": "gene_",
                    },
                    "nuclei": {
                        "path": str(nuclei_path),
                        "format": "csv",
                        "columns": {
                            "cell_id": "cell_id",
                            "center_x_um": "center_x_um",
                            "center_y_um": "center_y_um",
                        },
                    },
                },
                "reward": {
                    "objective": "mean_return",
                    "epsilon": 1.0e-8,
                    "r_max_um": 80.0,
                    "normalize_expression_zscore": False,
                    "zscore_delta": 1.0e-8,
                },
                "search": {
                    "w1": {"start": 0.5, "stop": 1.0, "step": 0.5},
                    "w2": {"start": 0.5, "stop": 0.5, "step": 0.5},
                    "w3": {"start": 0.5, "stop": 0.5, "step": 0.5},
                    "stop_lambda": {"start": 0.5, "stop": 1.0, "step": 0.5},
                },
            }
            with config_path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(config, handle, sort_keys=False)

            resolved = load_reward_grid_search_config(config_path)
            self.assertEqual(resolved.w1_axis.values(), (0.5, 1.0))

            run_dir = run_reward_grid_search_from_config(config_path)
            self.assertTrue((run_dir / "results.csv").exists())
            self.assertTrue((run_dir / "summary.json").exists())
            self.assertTrue((run_dir / "config" / "best_weights.yaml").exists())
            self.assertTrue((run_dir / "logs" / "steps.jsonl").exists())

            results = pd.read_csv(run_dir / "results.csv")
            self.assertEqual(len(results), 4)
            self.assertIn("objective_value", results.columns)

            with (run_dir / "summary.json").open("r", encoding="utf-8") as handle:
                summary = json.load(handle)
            self.assertEqual(summary["n_episodes"], 1)
            self.assertEqual(summary["n_combinations"], 4)
            self.assertIn("best", summary)


if __name__ == "__main__":
    unittest.main()
