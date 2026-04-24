from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import yaml

from hd_cell_rl.ppo_training import (
    ConfigError,
    EpisodeStep,
    EpisodeTrajectory,
    _build_rollout_buffer,
    _compute_full_grpo_episode_scores,
    _compute_group_relative_episode_advantages,
    _full_grpo_size_score,
    compute_discounted_returns,
    compute_gae_returns_and_advantages,
    load_ppo_training_config,
    run_ppo_training_from_config,
)


class PPOTrainingTests(unittest.TestCase):
    def test_gae_lambda_one_matches_discounted_returns(self) -> None:
        rewards = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
        values = np.asarray([0.5, 0.25, -0.5], dtype=np.float64)
        dones = np.asarray([False, False, True], dtype=bool)

        returns, advantages = compute_gae_returns_and_advantages(
            rewards,
            values,
            dones,
            gamma=0.99,
            gae_lambda=1.0,
        )

        expected_returns = compute_discounted_returns(rewards, gamma=0.99)
        np.testing.assert_allclose(returns, expected_returns, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(advantages, expected_returns - values, rtol=1e-6, atol=1e-6)

    def test_group_relative_advantages_standardize_within_group(self) -> None:
        trajectories = [
            EpisodeTrajectory(episode_slot=0, steps=tuple(), total_reward=1.0),
            EpisodeTrajectory(episode_slot=1, steps=tuple(), total_reward=3.0),
            EpisodeTrajectory(episode_slot=2, steps=tuple(), total_reward=2.0),
            EpisodeTrajectory(episode_slot=3, steps=tuple(), total_reward=6.0),
        ]

        bonuses = _compute_group_relative_episode_advantages(
            trajectories,
            group_size=2,
            norm_epsilon=1.0e-6,
            score="episode_total_reward",
        )

        np.testing.assert_allclose(
            bonuses,
            np.asarray([-0.999999, 0.999999, -0.9999995, 0.9999995], dtype=np.float64),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_rollout_buffer_without_group_relative_matches_gae(self) -> None:
        steps = (
            EpisodeStep(
                packed_membership_mask=np.zeros((1,), dtype=np.uint8),
                step_index=0,
                action=1,
                reward=1.0,
                done=False,
                old_log_prob=-0.5,
                old_value=0.2,
            ),
            EpisodeStep(
                packed_membership_mask=np.zeros((1,), dtype=np.uint8),
                step_index=1,
                action=0,
                reward=2.0,
                done=True,
                old_log_prob=-0.2,
                old_value=0.4,
            ),
        )
        traj = EpisodeTrajectory(episode_slot=0, steps=steps, total_reward=3.0)
        expected_returns, expected_advantages = compute_gae_returns_and_advantages(
            np.asarray([1.0, 2.0], dtype=np.float64),
            np.asarray([0.2, 0.4], dtype=np.float64),
            np.asarray([False, True], dtype=bool),
            gamma=0.99,
            gae_lambda=0.95,
        )

        transitions = _build_rollout_buffer(
            trajectories=[traj],
            gamma=0.99,
            gae_lambda=0.95,
            normalize_returns_per_episode=False,
            normalize_advantages=False,
            group_relative_enabled=False,
            group_relative_group_size=4,
            group_relative_mix_alpha=0.3,
            group_relative_norm_epsilon=1.0e-6,
            group_relative_score="episode_total_reward",
        )

        np.testing.assert_allclose(
            np.asarray([t.return_t for t in transitions], dtype=np.float64),
            expected_returns,
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            np.asarray([t.advantage for t in transitions], dtype=np.float64),
            expected_advantages,
            rtol=1e-6,
            atol=1e-6,
        )

    def test_load_config_rejects_invalid_group_relative_batch_divisibility(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = self._write_minimal_config(Path(tmp_dir), batch_cells=3, group_relative_enabled=True)
            with self.assertRaises(ConfigError):
                load_ppo_training_config(config_path)

    def test_full_grpo_size_score_penalizes_targeted_size_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = self._write_minimal_config(
                Path(tmp_dir),
                batch_cells=2,
                group_relative_enabled=True,
                training_mode="full_grpo",
            )
            config = load_ppo_training_config(config_path)

        small_over = _full_grpo_size_score(
            grow_ratio=4.0,
            lower=1.4,
            upper=2.4,
            bucket="small",
            config=config,
        )
        medium_over = _full_grpo_size_score(
            grow_ratio=4.0,
            lower=1.6,
            upper=2.8,
            bucket="medium",
            config=config,
        )
        large_under = _full_grpo_size_score(
            grow_ratio=0.5,
            lower=1.8,
            upper=3.4,
            bucket="large",
            config=config,
        )
        medium_under = _full_grpo_size_score(
            grow_ratio=0.5,
            lower=1.6,
            upper=2.8,
            bucket="medium",
            config=config,
        )

        self.assertLess(small_over, medium_over)
        self.assertLess(large_under, medium_under)

    def test_full_grpo_scores_use_size_aware_terminal_objective(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_path = self._write_minimal_config(
                root,
                batch_cells=2,
                group_relative_enabled=True,
                training_mode="full_grpo",
            )
            config = load_ppo_training_config(config_path)
            ctx = self._minimal_episode_context(config)
            stop_step = EpisodeStep(
                packed_membership_mask=np.zeros((1,), dtype=np.uint8),
                step_index=0,
                action=0,
                reward=0.0,
                done=True,
                old_log_prob=-0.5,
                old_value=0.0,
            )
            add_step = EpisodeStep(
                packed_membership_mask=np.zeros((1,), dtype=np.uint8),
                step_index=0,
                action=1,
                reward=0.0,
                done=True,
                old_log_prob=-0.5,
                old_value=0.0,
            )
            scores = _compute_full_grpo_episode_scores(
                [
                    EpisodeTrajectory(episode_slot=0, steps=(stop_step,), total_reward=0.0),
                    EpisodeTrajectory(episode_slot=1, steps=(add_step,), total_reward=0.0),
                ],
                [ctx, ctx],
                config,
            )
            self.assertEqual(scores.shape, (2,))
            self.assertTrue(np.isfinite(scores).all())

    def test_run_ppo_training_smoke_with_full_grpo(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_path = self._write_minimal_config(
                root,
                batch_cells=2,
                group_relative_enabled=True,
                training_mode="full_grpo",
            )

            result = run_ppo_training_from_config(config_path)

            self.assertTrue((result.run_dir / "summary.json").exists())
            self.assertTrue((result.run_dir / "checkpoints" / "final_model.pt").exists())

    def test_run_ppo_training_smoke_with_group_relative(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_path = self._write_minimal_config(root, batch_cells=2, group_relative_enabled=True)

            result = run_ppo_training_from_config(config_path)

            self.assertTrue(result.run_dir.exists())
            self.assertTrue((result.run_dir / "summary.json").exists())
            self.assertTrue((result.run_dir / "logs" / "steps.jsonl").exists())
            self.assertTrue((result.run_dir / "config" / "config_resolved.yaml").exists())
            self.assertTrue((result.run_dir / "checkpoints" / "final_model.pt").exists())

            with (result.run_dir / "summary.json").open("r", encoding="utf-8") as handle:
                summary = json.load(handle)
            self.assertEqual(summary["updates_completed"], 1)

    def _write_minimal_config(
        self,
        root: Path,
        *,
        batch_cells: int,
        group_relative_enabled: bool,
        training_mode: str = "ppo",
    ) -> Path:
        episodes_dir = root / "episodes"
        episodes_dir.mkdir(parents=True, exist_ok=True)

        reference_path = root / "reference_counts.csv"
        pd.DataFrame(
            {
                "gene_0": [8.0, 1.0],
                "gene_1": [1.0, 8.0],
            }
        ).to_csv(reference_path, index=False)

        nuclei_path = root / "nuclei.csv"
        pd.DataFrame(
            {
                "cell_id": ["cell_1"],
                "center_x_um": [0.0],
                "center_y_um": [0.0],
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
            candidate_bin_xy_um=np.asarray([[0.0, 0.0], [2.0, 0.0]], dtype=np.float32),
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

        config = {
            "run": {
                "name": "ppo_training_test",
                "output_root": str(root / "runs"),
                "seed": 7,
                "device": "cpu",
                "batch_cells": batch_cells,
                "rollout_mode": "vectorized",
                "n_rollout_workers": 1,
                "max_updates": 1,
                "max_steps_per_episode": 2,
                "training_mode": training_mode,
            },
            "inputs": {
                "episodes_index_path": str(episodes_index_path),
                "reference": {
                    "path": str(reference_path),
                    "format": "csv",
                    "array_key": "reference_counts",
                    "genes_key": "genes",
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
            "ppo": {
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "normalize_returns_per_episode": False,
                "normalize_advantages": False,
                "eps_clip": 0.2,
                "ppo_epochs": 1,
                "minibatch_size": 2,
                "learning_rate": 1.0e-4,
                "weight_decay": 0.0,
                "vf_coef": 0.5,
                "ent_coef": 0.0,
                "max_grad_norm": 1.0,
                "hidden_dim": 32,
                "target_kl": 0.01,
            },
            "group_relative": {
                "enabled": group_relative_enabled,
                "group_size": 2,
                "mix_alpha": 0.3,
                "norm_epsilon": 1.0e-6,
                "score": "episode_total_reward",
            },
            "full_grpo": {
                "reward_weight": 1.0,
                "size_weight": 0.8,
                "stop_weight": 0.4,
                "compact_weight": 0.2,
                "small_over_weight": 1.2,
                "large_under_weight": 1.2,
                "small_seed_max": 8,
                "large_seed_min": 17,
                "targets": {
                    "small": [1.4, 2.4],
                    "medium": [1.6, 2.8],
                    "large": [1.8, 3.4],
                },
            },
            "reward": {
                "epsilon": 1.0e-8,
                "r_max_um": 20.0,
                "w1": 0.45,
                "w2": 1.0,
                "w3": 1.0,
                "w4": 0.05,
                "stop_lambda": 0.7,
                "stop_stat": "topk_mean",
                "stop_top_k": 2,
                "expression_confidence_pseudocount": 5.0,
                "normalize_expression_zscore": True,
                "zscore_delta": 1.0e-8,
            },
            "stopping": {
                "moving_avg_window": 1,
                "min_improvement": 0.0,
                "patience": 1,
            },
        }
        config_path = root / "ppo_training.yaml"
        with config_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
        return config_path

    def _minimal_episode_context(self, config):
        from hd_cell_rl.ppo_training import EpisodeContext

        return EpisodeContext(
            cell_id="cell_1",
            candidate_bin_ids=("bin_0", "bin_1"),
            initial_membership_mask=np.asarray([1, 0], dtype=np.uint8),
            candidate_bin_xy_um=np.asarray([[0.0, 0.0], [2.0, 0.0]], dtype=np.float32),
            nucleus_center_xy_um=np.asarray([0.0, 0.0], dtype=np.float32),
            ll=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            p_dis=np.asarray([0.0, 0.1], dtype=np.float32),
            p_overlap=np.asarray([0.0, 0.0], dtype=np.float32),
            ll_mean_z=np.asarray([0.5, -0.5], dtype=np.float32),
            ll_max_z=np.asarray([0.5, -0.5], dtype=np.float32),
            base_penalty=np.asarray([0.0, 0.1], dtype=np.float32),
            expression_confidence=np.asarray([1.0, 1.0], dtype=np.float32),
            neighbor_index=np.asarray([[-1, -1, -1, -1, 1, -1, -1, -1], [-1, -1, -1, 0, -1, -1, -1, -1]], dtype=np.int32),
            max_steps=2,
            log_prior=-np.log(2.0),
            r_max_um=float(config.r_max_um),
            w1=float(config.w1),
            w2=float(config.w2),
            w3=float(config.w3),
            w4=float(config.w4),
            stop_lambda=float(config.stop_lambda),
            stop_stat=str(config.stop_stat),
            stop_top_k=int(config.stop_top_k),
            expression_confidence_pseudocount=float(config.expression_confidence_pseudocount),
            normalize_expression_zscore=bool(config.normalize_expression_zscore),
            zscore_delta=float(config.zscore_delta),
        )


if __name__ == "__main__":
    unittest.main()
