"""Unit tests for HD spatial-cell RL environment scaffold."""

from __future__ import annotations

import unittest
from tempfile import TemporaryDirectory

import numpy as np

from hd_cell_rl import (
    Action,
    CellAssignmentState,
    CellAssignmentEnv,
    EnvironmentConfig,
    NucleusRecord,
    BinRecord,
    build_episodes,
)


class CellAssignmentEnvTest(unittest.TestCase):
    """Test environment transitions and candidate selection logic."""

    def _make_data(self) -> tuple[list[NucleusRecord], list[BinRecord]]:
        nuclei = [
            NucleusRecord(
                cell_id="cell_1",
                center_x_um=0.0,
                center_y_um=0.0,
                radius_um=10.0,
                cell_type=None,
            )
        ]

        bins = [
            BinRecord(bin_id="b1", x_um=5.0, y_um=0.0, expression=np.array([1.0, 0.0])),
            BinRecord(bin_id="b2", x_um=20.0, y_um=0.0, expression=np.array([0.0, 1.0])),
            BinRecord(bin_id="b3", x_um=85.0, y_um=0.0, expression=np.array([2.0, 1.0])),
        ]
        return nuclei, bins

    def test_candidate_selection_uses_80um_cap(self) -> None:
        nuclei, bins = self._make_data()
        config = EnvironmentConfig(max_center_distance_um=80.0, radius_band_um=None)
        episodes = build_episodes(nuclei=nuclei, bins=bins, config=config)

        candidate_ids = [b.bin_id for b in episodes[0].candidate_bins]
        self.assertEqual(candidate_ids, ["b1", "b2"])

    def test_add_remove_stop_updates_state(self) -> None:
        nuclei, bins = self._make_data()
        config = EnvironmentConfig(max_center_distance_um=80.0, radius_band_um=None, max_steps=10)
        episode = build_episodes(nuclei=nuclei, bins=bins, config=config)[0]
        env = CellAssignmentEnv(episode, config=config)

        obs, _ = env.reset()
        self.assertEqual(obs["belonged_bin_ids"], [])

        obs, _, terminated, truncated, _ = env.step(Action.add("b1"))
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(obs["belonged_bin_ids"], ["b1"])
        self.assertEqual(obs["membership_mask"].tolist(), [1, 0])
        self.assertEqual(obs["processed_mask"].tolist(), [1, 0])

        obs, _, terminated, truncated, _ = env.step(Action.remove("b1"))
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(obs["belonged_bin_ids"], [])
        self.assertEqual(obs["membership_mask"].tolist(), [0, 0])

        obs, _, terminated, truncated, _ = env.step(Action.stop())
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(obs["step_index"], 3)

    def test_strict_validation_rejects_invalid_remove(self) -> None:
        nuclei, bins = self._make_data()
        config = EnvironmentConfig(
            max_center_distance_um=80.0,
            radius_band_um=None,
            strict_action_validation=True,
        )
        episode = build_episodes(nuclei=nuclei, bins=bins, config=config)[0]
        env = CellAssignmentEnv(episode, config=config)
        env.reset()

        with self.assertRaises(ValueError):
            env.step(Action.remove("b1"))

    def test_non_strict_validation_makes_invalid_remove_noop(self) -> None:
        nuclei, bins = self._make_data()
        config = EnvironmentConfig(
            max_center_distance_um=80.0,
            radius_band_um=None,
            strict_action_validation=False,
        )
        episode = build_episodes(nuclei=nuclei, bins=bins, config=config)[0]
        env = CellAssignmentEnv(episode, config=config)
        env.reset()

        obs, _, terminated, truncated, _ = env.step(Action.remove("b1"))
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(obs["membership_mask"].tolist(), [0, 0])

    def test_current_state_matches_observation(self) -> None:
        nuclei, bins = self._make_data()
        config = EnvironmentConfig(max_center_distance_um=80.0, radius_band_um=None, max_steps=10)
        episode = build_episodes(nuclei=nuclei, bins=bins, config=config)[0]
        env = CellAssignmentEnv(episode, config=config)
        env.reset()
        obs, _, _, _, _ = env.step(Action.add("b1"))

        state = env.current_state()
        self.assertIsInstance(state, CellAssignmentState)
        self.assertEqual(state.cell_id, obs["cell_id"])
        self.assertEqual(list(state.candidate_bin_ids), obs["candidate_bin_ids"])
        self.assertEqual(list(state.belonged_bin_ids), obs["belonged_bin_ids"])
        self.assertEqual(state.membership_mask.tolist(), obs["membership_mask"].tolist())
        self.assertEqual(state.processed_mask.tolist(), obs["processed_mask"].tolist())

    def test_state_npz_round_trip(self) -> None:
        nuclei, bins = self._make_data()
        config = EnvironmentConfig(max_center_distance_um=80.0, radius_band_um=None, max_steps=10)
        episode = build_episodes(nuclei=nuclei, bins=bins, config=config)[0]
        env = CellAssignmentEnv(episode, config=config)
        env.reset()
        env.step(Action.add("b1"))
        env.step(Action.add("b2"))

        state_before = env.current_state()

        with TemporaryDirectory() as tmp_dir:
            path = f"{tmp_dir}/state_snapshot.npz"
            env.save_state_npz(path)
            state_after = CellAssignmentState.load_npz(path)

        self.assertEqual(state_before.cell_id, state_after.cell_id)
        self.assertEqual(state_before.cell_type, state_after.cell_type)
        self.assertEqual(state_before.candidate_bin_ids, state_after.candidate_bin_ids)
        self.assertEqual(state_before.belonged_bin_ids, state_after.belonged_bin_ids)
        self.assertTrue(np.array_equal(state_before.candidate_bin_xy_um, state_after.candidate_bin_xy_um))
        self.assertTrue(np.array_equal(state_before.candidate_expression, state_after.candidate_expression))
        self.assertTrue(np.array_equal(state_before.membership_mask, state_after.membership_mask))
        self.assertTrue(np.array_equal(state_before.processed_mask, state_after.processed_mask))


if __name__ == "__main__":
    unittest.main()
