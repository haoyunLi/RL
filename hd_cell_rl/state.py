"""Explicit state snapshot model for one cell-assignment RL step."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CellAssignmentState:
    """Immutable snapshot of environment state for one timestep.

    This object centralizes the state fields used by the policy/value model and
    makes state persistence/debugging straightforward.
    """

    cell_id: str
    cell_type: str | None
    nucleus_center_xy_um: np.ndarray
    nucleus_radius_um: float
    candidate_bin_ids: tuple[str, ...]
    candidate_bin_xy_um: np.ndarray
    candidate_expression: np.ndarray
    membership_mask: np.ndarray
    belonged_bin_ids: tuple[str, ...]
    processed_mask: np.ndarray
    step_index: int
    max_steps: int

    def __post_init__(self) -> None:
        """Normalize arrays and validate shape consistency."""
        center = np.asarray(self.nucleus_center_xy_um, dtype=np.float32)
        bin_xy = np.asarray(self.candidate_bin_xy_um, dtype=np.float32)
        expr = np.asarray(self.candidate_expression, dtype=np.float32)
        membership = np.asarray(self.membership_mask, dtype=np.int8)
        processed = np.asarray(self.processed_mask, dtype=np.int8)

        if center.shape != (2,):
            raise ValueError("nucleus_center_xy_um must have shape (2,)")

        n_bins = len(self.candidate_bin_ids)

        if bin_xy.shape != (n_bins, 2):
            raise ValueError("candidate_bin_xy_um must have shape (n_bins, 2)")

        if expr.ndim != 2:
            raise ValueError("candidate_expression must be a 2D array")

        if expr.shape[0] != n_bins:
            raise ValueError("candidate_expression first dimension must equal n_bins")

        if membership.shape != (n_bins,):
            raise ValueError("membership_mask must have shape (n_bins,)")

        if processed.shape != (n_bins,):
            raise ValueError("processed_mask must have shape (n_bins,)")

        if self.nucleus_radius_um <= 0:
            raise ValueError("nucleus_radius_um must be > 0")

        if self.step_index < 0:
            raise ValueError("step_index must be >= 0")

        if self.max_steps <= 0:
            raise ValueError("max_steps must be > 0")

        if not set(np.unique(membership)).issubset({0, 1}):
            raise ValueError("membership_mask must contain only 0/1 values")

        if not set(np.unique(processed)).issubset({0, 1}):
            raise ValueError("processed_mask must contain only 0/1 values")

        candidate_set = set(self.candidate_bin_ids)
        belonged_set = set(self.belonged_bin_ids)

        if not belonged_set.issubset(candidate_set):
            raise ValueError("belonged_bin_ids must be a subset of candidate_bin_ids")

        # Build belonged IDs from membership for consistency checks.
        mask_belonged = {
            self.candidate_bin_ids[i]
            for i, assigned in enumerate(membership)
            if int(assigned) == 1
        }
        if belonged_set != mask_belonged:
            raise ValueError("belonged_bin_ids must match membership_mask")

        # Freeze arrays to make snapshots read-only and safer for debugging.
        center.setflags(write=False)
        bin_xy.setflags(write=False)
        expr.setflags(write=False)
        membership.setflags(write=False)
        processed.setflags(write=False)

        object.__setattr__(self, "nucleus_center_xy_um", center)
        object.__setattr__(self, "candidate_bin_xy_um", bin_xy)
        object.__setattr__(self, "candidate_expression", expr)
        object.__setattr__(self, "membership_mask", membership)
        object.__setattr__(self, "processed_mask", processed)

        if not isinstance(self.candidate_bin_ids, tuple):
            object.__setattr__(self, "candidate_bin_ids", tuple(self.candidate_bin_ids))

        if not isinstance(self.belonged_bin_ids, tuple):
            object.__setattr__(self, "belonged_bin_ids", tuple(self.belonged_bin_ids))

    def to_observation_dict(self) -> dict[str, Any]:
        """Return backward-compatible dict observation used by existing code."""
        return {
            "cell_id": self.cell_id,
            "cell_type": self.cell_type,
            "nucleus_center_xy_um": self.nucleus_center_xy_um.copy(),
            "nucleus_radius_um": float(self.nucleus_radius_um),
            "candidate_bin_ids": list(self.candidate_bin_ids),
            "candidate_bin_xy_um": self.candidate_bin_xy_um.copy(),
            "candidate_expression": self.candidate_expression.copy(),
            "membership_mask": self.membership_mask.copy(),
            "belonged_bin_ids": list(self.belonged_bin_ids),
            "processed_mask": self.processed_mask.copy(),
            "step_index": int(self.step_index),
            "max_steps": int(self.max_steps),
        }

    def save_npz(self, path: str | Path) -> None:
        """Save state snapshot to compressed NPZ for reproducible debugging."""
        out_path = Path(path)
        np.savez_compressed(
            out_path,
            cell_id=np.asarray([self.cell_id], dtype=object),
            cell_type=np.asarray([self.cell_type], dtype=object),
            nucleus_center_xy_um=self.nucleus_center_xy_um,
            nucleus_radius_um=np.asarray([self.nucleus_radius_um], dtype=np.float32),
            candidate_bin_ids=np.asarray(self.candidate_bin_ids, dtype=object),
            candidate_bin_xy_um=self.candidate_bin_xy_um,
            candidate_expression=self.candidate_expression,
            membership_mask=self.membership_mask,
            belonged_bin_ids=np.asarray(self.belonged_bin_ids, dtype=object),
            processed_mask=self.processed_mask,
            step_index=np.asarray([self.step_index], dtype=np.int32),
            max_steps=np.asarray([self.max_steps], dtype=np.int32),
        )

    @classmethod
    def load_npz(cls, path: str | Path) -> "CellAssignmentState":
        """Load a state snapshot from `save_npz` output."""
        with np.load(Path(path), allow_pickle=True) as data:
            return cls(
                cell_id=str(data["cell_id"][0]),
                cell_type=None if data["cell_type"][0] is None else str(data["cell_type"][0]),
                nucleus_center_xy_um=np.asarray(data["nucleus_center_xy_um"], dtype=np.float32),
                nucleus_radius_um=float(data["nucleus_radius_um"][0]),
                candidate_bin_ids=tuple(str(x) for x in data["candidate_bin_ids"]),
                candidate_bin_xy_um=np.asarray(data["candidate_bin_xy_um"], dtype=np.float32),
                candidate_expression=np.asarray(data["candidate_expression"], dtype=np.float32),
                membership_mask=np.asarray(data["membership_mask"], dtype=np.int8),
                belonged_bin_ids=tuple(str(x) for x in data["belonged_bin_ids"]),
                processed_mask=np.asarray(data["processed_mask"], dtype=np.int8),
                step_index=int(data["step_index"][0]),
                max_steps=int(data["max_steps"][0]),
            )
