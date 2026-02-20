"""Core RL environment for assigning HD bins to one nucleus-centered cell."""

from __future__ import annotations

from typing import Any

import numpy as np

from .actions import Action, ActionType
from .config import EnvironmentConfig
from .models import CellEpisodeData
from .reward import RewardFunction, ZeroReward
from .state import CellAssignmentState


class CellAssignmentEnv:
    """One-episode environment for one nucleus (one candidate cell).

    State tracks:
    - membership mask (0/1): whether each candidate bin currently belongs to cell
    - belonged bin IDs: list derived from membership mask
    - processed mask (f): whether a bin has been touched by ADD/REMOVE
    - bin expression vectors
    - bin coordinates
    - nucleus center/radius
    - optional cell type placeholder
    """

    def __init__(
        self,
        episode_data: CellEpisodeData,
        config: EnvironmentConfig | None = None,
        reward_fn: RewardFunction | None = None,
    ) -> None:
        """Create environment with immutable episode payload and runtime options."""
        self.episode_data = episode_data
        self.config = config or EnvironmentConfig()
        self.reward_fn = reward_fn or ZeroReward()

        self._candidate_bin_ids = [b.bin_id for b in self.episode_data.candidate_bins]
        self._bin_id_to_index = {bin_id: i for i, bin_id in enumerate(self._candidate_bin_ids)}

        if len(self._bin_id_to_index) != len(self._candidate_bin_ids):
            raise ValueError("candidate bin IDs must be unique per episode")

        if self.episode_data.n_candidate_bins > 0:
            self._candidate_bin_xy = np.asarray(
                [(b.x_um, b.y_um) for b in self.episode_data.candidate_bins],
                dtype=np.float32,
            )
            self._candidate_expression = np.vstack(
                [b.expression for b in self.episode_data.candidate_bins]
            ).astype(np.float32)
        else:
            # Keep empty arrays with stable rank so downstream code can rely on shape.
            self._candidate_bin_xy = np.zeros((0, 2), dtype=np.float32)
            self._candidate_expression = np.zeros((0, 0), dtype=np.float32)

        self._max_steps = self.config.resolve_max_steps(self.episode_data.n_candidate_bins)

        # Initialize mutable state buffers.
        self._membership_mask = np.zeros(self.episode_data.n_candidate_bins, dtype=np.int8)
        self._processed_mask = np.zeros(self.episode_data.n_candidate_bins, dtype=np.int8)
        self._step_index = 0
        self._terminated = False
        self._truncated = False

    def reset(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset dynamic state and return initial observation + info."""
        self._membership_mask.fill(0)
        self._processed_mask.fill(0)
        self._step_index = 0
        self._terminated = False
        self._truncated = False
        return self._build_observation(), self._build_info()

    def step(self, action: Action) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Apply one action and return (obs, reward, terminated, truncated, info)."""
        if self._terminated or self._truncated:
            raise RuntimeError("cannot call step() on a finished episode; call reset() first")

        if not isinstance(action, Action):
            raise TypeError("action must be an Action instance")

        previous_mask = self._membership_mask.copy()

        if action.kind == ActionType.STOP:
            self._terminated = True
        elif action.kind == ActionType.ADD_BIN:
            bin_index = self._resolve_bin_index(action)
            self._apply_add(bin_index)
            self._processed_mask[bin_index] = 1
        elif action.kind == ActionType.REMOVE_BIN:
            bin_index = self._resolve_bin_index(action)
            self._apply_remove(bin_index)
            self._processed_mask[bin_index] = 1
        else:
            raise ValueError(f"unsupported action kind: {action.kind!r}")

        self._step_index += 1

        if self._step_index >= self._max_steps and not self._terminated:
            self._truncated = True

        done = self._terminated or self._truncated
        info = self._build_info()

        reward = float(
            self.reward_fn.compute(
                previous_membership_mask=previous_mask,
                action=action,
                new_membership_mask=self._membership_mask.copy(),
                done=done,
                info=info,
            )
        )

        return self._build_observation(), reward, self._terminated, self._truncated, info

    def current_state(self) -> CellAssignmentState:
        """Return an immutable, typed snapshot of current environment state."""
        return self._build_state()

    def save_state_npz(self, path: str) -> None:
        """Persist current state snapshot to compressed NPZ on disk."""
        self.current_state().save_npz(path)

    def valid_actions(self) -> list[Action]:
        """Return current valid action set in deterministic order."""
        actions: list[Action] = [Action.stop()]

        # ADD actions are valid for bins that are not currently assigned.
        for i, assigned in enumerate(self._membership_mask):
            if assigned == 0:
                actions.append(Action.add(self._candidate_bin_ids[i]))

        # REMOVE actions are valid for bins that are currently assigned.
        for i, assigned in enumerate(self._membership_mask):
            if assigned == 1:
                actions.append(Action.remove(self._candidate_bin_ids[i]))

        return actions

    def belonged_bin_ids(self) -> list[str]:
        """Return IDs for bins currently assigned to this cell."""
        belonged: list[str] = []
        for i, assigned in enumerate(self._membership_mask):
            if assigned == 1:
                belonged.append(self._candidate_bin_ids[i])
        return belonged

    def _resolve_bin_index(self, action: Action) -> int:
        """Resolve action.bin_id into candidate-bin index with validation."""
        if action.bin_id is None:
            raise ValueError(f"action {action.kind.value} requires a bin_id")

        if action.bin_id not in self._bin_id_to_index:
            raise KeyError(f"bin_id {action.bin_id!r} is not in candidate bins")

        return self._bin_id_to_index[action.bin_id]

    def _apply_add(self, bin_index: int) -> None:
        """Apply ADD_BIN update to membership mask with strict/no-op behavior."""
        if self._membership_mask[bin_index] == 1:
            if self.config.strict_action_validation:
                raise ValueError("ADD_BIN is invalid because bin is already assigned")
            return

        self._membership_mask[bin_index] = 1

    def _apply_remove(self, bin_index: int) -> None:
        """Apply REMOVE_BIN update to membership mask with strict/no-op behavior."""
        if self._membership_mask[bin_index] == 0:
            if self.config.strict_action_validation:
                raise ValueError("REMOVE_BIN is invalid because bin is not assigned")
            return

        self._membership_mask[bin_index] = 0

    def _build_observation(self) -> dict[str, Any]:
        """Build state dictionary consumed by policy/value models."""
        return self._build_state().to_observation_dict()

    def _build_state(self) -> CellAssignmentState:
        """Build typed state snapshot used for debugging and persistence."""
        nucleus = self.episode_data.nucleus
        return CellAssignmentState(
            cell_id=nucleus.cell_id,
            cell_type=nucleus.cell_type,
            nucleus_center_xy_um=np.asarray([nucleus.center_x_um, nucleus.center_y_um], dtype=np.float32),
            nucleus_radius_um=float(nucleus.radius_um),
            candidate_bin_ids=tuple(self._candidate_bin_ids),
            candidate_bin_xy_um=self._candidate_bin_xy.copy(),
            candidate_expression=self._candidate_expression.copy(),
            membership_mask=self._membership_mask.copy(),
            belonged_bin_ids=tuple(self.belonged_bin_ids()),
            processed_mask=self._processed_mask.copy(),
            step_index=self._step_index,
            max_steps=self._max_steps,
        )

    def _build_info(self) -> dict[str, Any]:
        """Build side-channel info dictionary for debugging and logging."""
        return {
            "n_candidate_bins": self.episode_data.n_candidate_bins,
            "n_assigned_bins": int(self._membership_mask.sum()),
            "step_index": self._step_index,
            "max_steps": self._max_steps,
            "valid_actions": self.valid_actions(),
            "terminated": self._terminated,
            "truncated": self._truncated,
        }
