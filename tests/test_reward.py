"""Tests for posterior-based add/stop reward implementation."""

from __future__ import annotations

import unittest

import numpy as np

from hd_cell_rl import Action
from hd_cell_rl.reward import (
    PosteriorAddBinReward,
    compute_bin_log_likelihood_by_type,
    compute_reference_distribution,
)


class PosteriorAddBinRewardTest(unittest.TestCase):
    """Validate reward formulas against expected numeric behavior."""

    def setUp(self) -> None:
        # Two cell types, two genes.
        self.reference_counts = np.array(
            [
                [9.0, 1.0],
                [1.0, 9.0],
            ],
            dtype=np.float64,
        )

        self.bin_ids = ["b0", "b1", "b2"]
        self.bin_expr = np.array(
            [
                [9.0, 1.0],  # type-0-like
                [1.0, 9.0],  # type-1-like
                [0.0, 0.0],  # Nb=0 edge case
            ],
            dtype=np.float64,
        )
        self.bin_xy = np.array(
            [
                [10.0, 0.0],
                [20.0, 0.0],
                [5.0, 0.0],
            ],
            dtype=np.float64,
        )
        self.nucleus_xy = np.array([0.0, 0.0], dtype=np.float64)
        self.other_nuclei = np.array([[12.0, 0.0]], dtype=np.float64)

    def test_reference_distribution_formula(self) -> None:
        theta = compute_reference_distribution(self.reference_counts, epsilon=0.0)
        expected = np.array(
            [
                [0.9, 0.1],
                [0.1, 0.9],
            ],
            dtype=np.float64,
        )
        self.assertTrue(np.allclose(theta, expected, atol=1e-12))

    def test_bin_log_likelihood_handles_zero_total_bin(self) -> None:
        theta = compute_reference_distribution(self.reference_counts, epsilon=0.0)
        ll = compute_bin_log_likelihood_by_type(self.bin_expr, theta)

        # For zero-count bin, LL[b,k] must be 0 for all k by definition.
        self.assertTrue(np.allclose(ll[2], np.zeros(2), atol=1e-12))

    def test_posterior_prefers_matching_type(self) -> None:
        reward = PosteriorAddBinReward(
            reference_counts=self.reference_counts,
            candidate_bin_ids=self.bin_ids,
            candidate_expression=self.bin_expr,
            candidate_bin_xy_um=self.bin_xy,
            nucleus_center_xy_um=self.nucleus_xy,
            other_nuclei_center_xy_um=self.other_nuclei,
            epsilon=1e-8,
            r_max_um=80.0,
            w1=1.0,
            w2=0.0,
            w3=0.0,
            stop_lambda=1.0,
        )

        # ST includes only b0 (type-0-like), so posterior should favor type 0.
        membership = np.array([1, 0, 0], dtype=np.int8)
        p = reward.posterior_given_state(membership)
        self.assertGreater(p[0], p[1])
        self.assertAlmostEqual(float(np.sum(p)), 1.0, places=10)

    def test_add_reward_formula_without_penalties(self) -> None:
        reward = PosteriorAddBinReward(
            reference_counts=self.reference_counts,
            candidate_bin_ids=self.bin_ids,
            candidate_expression=self.bin_expr,
            candidate_bin_xy_um=self.bin_xy,
            nucleus_center_xy_um=self.nucleus_xy,
            other_nuclei_center_xy_um=self.other_nuclei,
            epsilon=1e-8,
            r_max_um=80.0,
            w1=1.0,
            w2=0.0,
            w3=0.0,
            stop_lambda=1.0,
        )

        # Empty ST -> posterior is uniform over K=2.
        membership = np.array([0, 0, 0], dtype=np.int8)
        p = reward.posterior_given_state(membership)
        self.assertTrue(np.allclose(p, np.array([0.5, 0.5]), atol=1e-9))

        r_add = reward.add_reward_per_bin(membership)

        theta = compute_reference_distribution(self.reference_counts, epsilon=1e-8)
        ll = compute_bin_log_likelihood_by_type(self.bin_expr, theta)
        expected_expr = ll @ p

        self.assertTrue(np.allclose(r_add, expected_expr, atol=1e-10))

    def test_distance_and_overlap_penalties(self) -> None:
        reward = PosteriorAddBinReward(
            reference_counts=self.reference_counts,
            candidate_bin_ids=self.bin_ids,
            candidate_expression=self.bin_expr,
            candidate_bin_xy_um=self.bin_xy,
            nucleus_center_xy_um=self.nucleus_xy,
            other_nuclei_center_xy_um=self.other_nuclei,
            epsilon=1e-8,
            r_max_um=80.0,
            w1=0.0,
            w2=1.0,
            w3=1.0,
            stop_lambda=1.0,
        )

        membership = np.array([0, 0, 0], dtype=np.int8)
        r_add = reward.add_reward_per_bin(membership)

        # b0: d_n=10, d_other=2 => P_dis=10/80, P_overlap=(10-2)/80
        self.assertAlmostEqual(r_add[0], -(10.0 / 80.0) - (8.0 / 80.0), places=10)
        # b1: d_n=20, d_other=8 => P_dis=20/80, P_overlap=(20-8)/80
        self.assertAlmostEqual(r_add[1], -(20.0 / 80.0) - (12.0 / 80.0), places=10)
        # b2: d_n=5, d_other=7 => overlap term clipped at 0
        self.assertAlmostEqual(r_add[2], -(5.0 / 80.0), places=10)

    def test_stop_reward_uses_best_available_add(self) -> None:
        reward = PosteriorAddBinReward(
            reference_counts=self.reference_counts,
            candidate_bin_ids=self.bin_ids,
            candidate_expression=self.bin_expr,
            candidate_bin_xy_um=self.bin_xy,
            nucleus_center_xy_um=self.nucleus_xy,
            other_nuclei_center_xy_um=self.other_nuclei,
            epsilon=1e-8,
            r_max_um=80.0,
            w1=1.0,
            w2=0.2,
            w3=0.1,
            stop_lambda=2.0,
        )

        membership = np.array([0, 1, 0], dtype=np.int8)  # b1 already assigned, not add-eligible
        r_add = reward.add_reward_per_bin(membership)
        expected_delta_t = max(float(r_add[0]), float(r_add[2]))
        expected_r_stop = -2.0 * expected_delta_t

        self.assertAlmostEqual(reward.stop_reward(membership), expected_r_stop, places=10)

    def test_single_bin_helper_matches_vector_result(self) -> None:
        reward = PosteriorAddBinReward(
            reference_counts=self.reference_counts,
            candidate_bin_ids=self.bin_ids,
            candidate_expression=self.bin_expr,
            candidate_bin_xy_um=self.bin_xy,
            nucleus_center_xy_um=self.nucleus_xy,
            other_nuclei_center_xy_um=self.other_nuclei,
            epsilon=1e-8,
            r_max_um=80.0,
            w1=1.0,
            w2=0.25,
            w3=0.5,
            stop_lambda=1.0,
            normalize_expression_zscore=False,
        )

        membership = np.array([1, 0, 0], dtype=np.int8)
        vector_reward = reward.add_reward_per_bin(membership)

        for i, bin_id in enumerate(self.bin_ids):
            single = reward.add_reward_for_bin(membership, bin_id)
            self.assertAlmostEqual(single, float(vector_reward[i]), places=10)

    def test_compute_uses_action_semantics(self) -> None:
        reward = PosteriorAddBinReward(
            reference_counts=self.reference_counts,
            candidate_bin_ids=self.bin_ids,
            candidate_expression=self.bin_expr,
            candidate_bin_xy_um=self.bin_xy,
            nucleus_center_xy_um=self.nucleus_xy,
            other_nuclei_center_xy_um=self.other_nuclei,
            epsilon=1e-8,
            r_max_um=80.0,
            w1=1.0,
            w2=0.3,
            w3=0.2,
            stop_lambda=1.5,
            remove_reward=-0.25,
        )

        membership = np.array([0, 0, 0], dtype=np.int8)
        add_reward_vector = reward.add_reward_per_bin(membership)

        add_r = reward.compute(
            previous_membership_mask=membership,
            action=Action.add("b0"),
            new_membership_mask=np.array([1, 0, 0], dtype=np.int8),
            done=False,
            info={},
        )
        self.assertAlmostEqual(add_r, float(add_reward_vector[0]), places=10)

        stop_r = reward.compute(
            previous_membership_mask=membership,
            action=Action.stop(),
            new_membership_mask=membership,
            done=True,
            info={},
        )
        self.assertAlmostEqual(stop_r, reward.stop_reward(membership), places=10)

        rem_r = reward.compute(
            previous_membership_mask=membership,
            action=Action.remove("b0"),
            new_membership_mask=membership,
            done=False,
            info={},
        )
        self.assertAlmostEqual(rem_r, -0.25, places=12)


if __name__ == "__main__":
    unittest.main()
