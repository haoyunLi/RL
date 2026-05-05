from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch
import yaml

from hd_cell_rl.ppo_feature_schema import (
    ACTION_FEATURE_DIM,
    A_CANDIDATE_COMPACTNESS_GAIN,
    A_CANDIDATE_NEIGHBOR_SUPPORT,
    A_IS_STOP_ACTION,
    GLOBAL_FEATURE_DIM,
)
from hd_cell_rl.ppo_training import (
    ActorCritic,
    ConfigError,
    AddStopCellEnv,
    EpisodeStep,
    EpisodeTrajectory,
    PlannerStep,
    PLANNER_MODE_COMPACT,
    _build_policy_observation_from_state,
    _build_planner_rollout_buffer,
    _build_rollout_buffer,
    _compute_full_grpo_episode_scores,
    _compute_group_relative_episode_advantages,
    _compute_state_feature_bundle,
    _full_grpo_frontier_quality,
    _full_grpo_key_node_score,
    _full_grpo_stop_score,
    _planner_logit_bias_from_action_features,
    _posterior_confidence_delta_per_bin,
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

    def test_full_grpo_frontier_quality_is_evidence_gated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = self._write_minimal_config(
                Path(tmp_dir),
                batch_cells=2,
                group_relative_enabled=True,
                training_mode="full_grpo",
            )
            config = load_ppo_training_config(config_path)

        low = _full_grpo_frontier_quality(topk_mean=-1.0, has_frontier=True, config=config)
        high = _full_grpo_frontier_quality(topk_mean=1.0, has_frontier=True, config=config)
        none = _full_grpo_frontier_quality(topk_mean=1.0, has_frontier=False, config=config)

        self.assertLess(low, 0.5)
        self.assertGreater(high, 0.5)
        self.assertEqual(none, 0.0)

    def test_full_grpo_stop_score_prefers_stop_when_frontier_quality_is_low(self) -> None:
        stop_low = _full_grpo_stop_score(stopped=True, frontier_quality=0.1, overgrowth_risk=0.4)
        keep_low = _full_grpo_stop_score(stopped=False, frontier_quality=0.1, overgrowth_risk=0.4)
        stop_high = _full_grpo_stop_score(stopped=True, frontier_quality=0.9, overgrowth_risk=0.0)
        keep_high = _full_grpo_stop_score(stopped=False, frontier_quality=0.9, overgrowth_risk=0.0)

        self.assertGreater(stop_low, keep_low)
        self.assertLess(stop_high, keep_high)

    def test_full_grpo_key_node_score_penalizes_repeated_compact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_path = self._write_minimal_config(
                root,
                batch_cells=2,
                group_relative_enabled=True,
                training_mode="full_grpo",
                planner_enabled=True,
            )
            config = load_ppo_training_config(config_path)
            ctx = self._minimal_episode_context(config)
            packed = np.packbits(np.asarray([1, 0], dtype=np.uint8))
            first = PlannerStep(
                packed_membership_mask=packed,
                step_index=0,
                mode=PLANNER_MODE_COMPACT,
                old_log_prob=-0.7,
                compact_streak=0,
            )
            repeated = PlannerStep(
                packed_membership_mask=packed,
                step_index=0,
                mode=PLANNER_MODE_COMPACT,
                old_log_prob=-0.7,
                compact_streak=8,
            )

        self.assertGreater(
            _full_grpo_key_node_score(ctx=ctx, planner_step=first, config=config),
            _full_grpo_key_node_score(ctx=ctx, planner_step=repeated, config=config),
        )

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

    def test_policy_observation_state_upgrade_shapes_are_finite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = load_ppo_training_config(
                self._write_minimal_config(Path(tmp_dir), batch_cells=2, group_relative_enabled=True)
            )
            ctx = self._minimal_episode_context(config)
            obs = _build_policy_observation_from_state(
                ctx=ctx,
                membership_mask=np.asarray([1, 0], dtype=np.uint8),
                step_index=0,
            )

        self.assertEqual(obs["global_features"].shape, (AddStopCellEnv.GLOBAL_FEATURE_DIM,))
        self.assertEqual(obs["action_features"].shape, (3, AddStopCellEnv.ACTION_FEATURE_DIM))
        self.assertEqual(AddStopCellEnv.GLOBAL_FEATURE_DIM, GLOBAL_FEATURE_DIM)
        self.assertEqual(AddStopCellEnv.ACTION_FEATURE_DIM, ACTION_FEATURE_DIM)
        self.assertTrue(np.isfinite(obs["global_features"]).all())
        self.assertTrue(np.isfinite(obs["action_features"]).all())

    def test_planner_config_defaults_disabled_and_preserves_policy_dims(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = load_ppo_training_config(
                self._write_minimal_config(Path(tmp_dir), batch_cells=2, group_relative_enabled=True)
            )

        self.assertFalse(config.planner_enabled)
        self.assertEqual(config.planner_modes, ("stop", "compact", "balanced", "explore"))
        model = ActorCritic(
            global_dim=AddStopCellEnv.GLOBAL_FEATURE_DIM,
            action_dim=AddStopCellEnv.ACTION_FEATURE_DIM,
            hidden_dim=8,
            planner_enabled=False,
        )
        self.assertFalse(hasattr(model, "planner_head"))
        self.assertEqual(AddStopCellEnv.GLOBAL_FEATURE_DIM, GLOBAL_FEATURE_DIM)
        self.assertEqual(AddStopCellEnv.ACTION_FEATURE_DIM, ACTION_FEATURE_DIM)

    def test_planner_logit_bias_is_soft_add_bias(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = load_ppo_training_config(
                self._write_minimal_config(
                    Path(tmp_dir),
                    batch_cells=2,
                    group_relative_enabled=True,
                    training_mode="full_grpo",
                    planner_enabled=True,
                )
            )

        action_features = torch.zeros((1, 3, AddStopCellEnv.ACTION_FEATURE_DIM), dtype=torch.float32)
        action_features[0, 0, A_IS_STOP_ACTION] = 1.0
        action_features[0, 1, A_CANDIDATE_COMPACTNESS_GAIN] = 0.1
        action_features[0, 2, A_CANDIDATE_COMPACTNESS_GAIN] = 0.1
        action_features[0, 1, A_CANDIDATE_NEIGHBOR_SUPPORT] = 0.0
        action_features[0, 2, A_CANDIDATE_NEIGHBOR_SUPPORT] = 1.0
        bias = _planner_logit_bias_from_action_features(
            action_features,
            torch.as_tensor([PLANNER_MODE_COMPACT], dtype=torch.long),
            config,
        )

        self.assertIsNotNone(bias)
        self.assertEqual(tuple(bias.shape), (1, 3))
        self.assertEqual(float(bias[0, 0]), 0.0)
        self.assertGreater(float(bias[0, 2]), float(bias[0, 1]))

    def test_planner_rollout_buffer_broadcasts_episode_bonus(self) -> None:
        planner_step = PlannerStep(
            packed_membership_mask=np.zeros((1,), dtype=np.uint8),
            step_index=0,
            mode=PLANNER_MODE_COMPACT,
            old_log_prob=-0.7,
        )
        trajectories = [
            EpisodeTrajectory(episode_slot=0, steps=tuple(), total_reward=1.0, planner_steps=(planner_step,)),
            EpisodeTrajectory(episode_slot=1, steps=tuple(), total_reward=2.0, planner_steps=(planner_step,)),
        ]

        transitions = _build_planner_rollout_buffer(
            trajectories=trajectories,
            episode_advantage_bonus=np.asarray([-1.0, 1.0], dtype=np.float64),
        )

        self.assertEqual(len(transitions), 2)
        self.assertEqual(transitions[0].mode, PLANNER_MODE_COMPACT)
        self.assertEqual(transitions[0].advantage, -1.0)
        self.assertEqual(transitions[1].advantage, 1.0)

    def test_radial_alignment_is_zero_without_centroid_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = load_ppo_training_config(
                self._write_minimal_config(Path(tmp_dir), batch_cells=2, group_relative_enabled=True)
            )
            ctx = self._minimal_episode_context(config)
            bundle = _compute_state_feature_bundle(
                ctx=ctx,
                membership_mask=np.asarray([1, 0], dtype=np.uint8),
                step_index=0,
            )

        np.testing.assert_allclose(bundle.radial_alignment_with_centroid_drift, np.zeros((2,), dtype=np.float32))

    def test_posterior_confidence_delta_rewards_consistent_bins(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = load_ppo_training_config(
                self._write_minimal_config(Path(tmp_dir), batch_cells=2, group_relative_enabled=True)
            )
            ctx = self._minimal_episode_context(config)
            ctx = ctx.__class__(
                **{
                    **ctx.__dict__,
                    "candidate_bin_ids": ("bin_0", "bin_1", "bin_2"),
                    "initial_membership_mask": np.asarray([1, 0, 0], dtype=np.uint8),
                    "candidate_bin_xy_um": np.asarray([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]], dtype=np.float32),
                    "ll": np.asarray([[2.0, 0.0], [2.0, 0.0], [0.0, 2.0]], dtype=np.float32),
                    "p_dis": np.asarray([0.0, 0.1, 0.2], dtype=np.float32),
                    "p_overlap": np.zeros((3,), dtype=np.float32),
                    "ll_mean_z": np.asarray([0.5, 0.5, -0.5], dtype=np.float32),
                    "ll_max_z": np.asarray([0.5, 0.5, -0.5], dtype=np.float32),
                    "base_penalty": np.asarray([0.0, 0.1, 0.2], dtype=np.float32),
                    "expression_confidence": np.ones((3,), dtype=np.float32),
                    "bin_count_totals": np.ones((3,), dtype=np.float32),
                    "neighbor_index": np.asarray(
                        [
                            [-1, -1, -1, -1, 1, -1, -1, -1],
                            [-1, -1, -1, 0, 2, -1, -1, -1],
                            [-1, -1, -1, 1, -1, -1, -1, -1],
                        ],
                        dtype=np.int32,
                    ),
                }
            )

            delta = _posterior_confidence_delta_per_bin(
                ctx=ctx,
                membership_mask=np.asarray([1, 0, 0], dtype=np.uint8),
            )

        self.assertGreater(float(delta[1]), 0.0)
        self.assertLess(float(delta[2]), 0.0)

    def test_w5_old_bin_compatibility_contributes_separately(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = load_ppo_training_config(
                self._write_minimal_config(Path(tmp_dir), batch_cells=2, group_relative_enabled=True)
            )
            ctx = self._minimal_episode_context(config)
            ctx = ctx.__class__(
                **{
                    **ctx.__dict__,
                    "candidate_bin_ids": ("bin_0", "bin_1", "bin_2"),
                    "initial_membership_mask": np.asarray([1, 0, 0], dtype=np.uint8),
                    "candidate_bin_xy_um": np.asarray([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]], dtype=np.float32),
                    "ll": np.asarray([[2.0, 0.0], [2.0, 0.0], [0.0, 2.0]], dtype=np.float32),
                    "p_dis": np.zeros((3,), dtype=np.float32),
                    "p_overlap": np.zeros((3,), dtype=np.float32),
                    "ll_mean_z": np.zeros((3,), dtype=np.float32),
                    "ll_max_z": np.zeros((3,), dtype=np.float32),
                    "base_penalty": np.zeros((3,), dtype=np.float32),
                    "expression_confidence": np.ones((3,), dtype=np.float32),
                    "bin_count_totals": np.ones((3,), dtype=np.float32),
                    "neighbor_index": np.asarray(
                        [
                            [-1, -1, -1, -1, 1, -1, -1, -1],
                            [-1, -1, -1, 0, 2, -1, -1, -1],
                            [-1, -1, -1, 1, -1, -1, -1, -1],
                        ],
                        dtype=np.int32,
                    ),
                    "w1": 0.0,
                    "w4": 0.0,
                    "w5": 1.0,
                    "normalize_expression_zscore": False,
                }
            )

            bundle = _compute_state_feature_bundle(
                ctx=ctx,
                membership_mask=np.asarray([1, 0, 0], dtype=np.uint8),
                step_index=0,
            )

        self.assertGreater(float(bundle.add_rewards[1]), float(bundle.add_rewards[2]))

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

    def test_run_ppo_training_smoke_with_full_grpo_planner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_path = self._write_minimal_config(
                root,
                batch_cells=2,
                group_relative_enabled=True,
                training_mode="full_grpo",
                planner_enabled=True,
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
        planner_enabled: bool = False,
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
            "planner": {
                "enabled": planner_enabled,
                "interval": 1,
                "modes": ["stop", "compact", "balanced", "explore"],
                "cot_weight": 0.4,
                "entropy_coef": 0.02,
                "logit_bias": {},
            },
            "full_grpo": {
                "reward_weight": 1.0,
                "evidence_growth_weight": 0.7,
                "stop_weight": 0.5,
                "overgrowth_weight": 0.8,
                "compact_weight": 0.25,
                "compact_streak_weight": 0.25,
                "explore_weight": 0.4,
                "tau_frontier": 0.0,
                "frontier_temp": 0.2,
            },
            "reward": {
                "epsilon": 1.0e-8,
                "r_max_um": 20.0,
                "w1": 0.45,
                "w2": 1.0,
                "w3": 1.0,
                "w4": 0.05,
                "w5": 0.10,
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
            bin_count_totals=np.asarray([5.0, 4.0], dtype=np.float32),
            neighbor_index=np.asarray([[-1, -1, -1, -1, 1, -1, -1, -1], [-1, -1, -1, 0, -1, -1, -1, -1]], dtype=np.int32),
            max_steps=2,
            log_prior=-np.log(2.0),
            r_max_um=float(config.r_max_um),
            w1=float(config.w1),
            w2=float(config.w2),
            w3=float(config.w3),
            w4=float(config.w4),
            w5=float(config.w5),
            stop_lambda=float(config.stop_lambda),
            stop_stat=str(config.stop_stat),
            stop_top_k=int(config.stop_top_k),
            expression_confidence_pseudocount=float(config.expression_confidence_pseudocount),
            normalize_expression_zscore=bool(config.normalize_expression_zscore),
            zscore_delta=float(config.zscore_delta),
        )


if __name__ == "__main__":
    unittest.main()
