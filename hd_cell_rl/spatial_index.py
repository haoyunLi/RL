"""Spatial index implementations for fast candidate-bin lookup."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .models import BinRecord


class BruteForceSpatialIndex:
    """Simple vectorized radius-query index over all bins.

    This class is intentionally minimal so it can be replaced later by a
    KD-tree or GPU-backed index without changing the environment API.
    """

    def __init__(self, bins: Iterable[BinRecord]) -> None:
        """Store bin coordinates and references for repeated queries."""
        self._bins = tuple(bins)

        if len(self._bins) == 0:
            self._xy = np.zeros((0, 2), dtype=np.float32)
            return

        coords = [(b.x_um, b.y_um) for b in self._bins]
        self._xy = np.asarray(coords, dtype=np.float32)

    def query_radius(self, center_x_um: float, center_y_um: float, radius_um: float) -> tuple[BinRecord, ...]:
        """Return all bins whose center is within `radius_um` of the query point."""
        if radius_um < 0:
            raise ValueError("radius_um must be >= 0")

        if len(self._bins) == 0:
            return ()

        # Compute squared distances in vectorized form for speed.
        dx = self._xy[:, 0] - center_x_um
        dy = self._xy[:, 1] - center_y_um
        dist2 = dx * dx + dy * dy
        threshold2 = float(radius_um * radius_um)

        # Build a compact tuple of matching bins in original order.
        keep = dist2 <= threshold2
        kept_indices = np.flatnonzero(keep)
        return tuple(self._bins[i] for i in kept_indices)
