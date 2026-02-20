"""Geometry helpers for spatial measurements in micrometre space."""

from __future__ import annotations

import math


def euclidean_distance_um(x1: float, y1: float, x2: float, y2: float) -> float:
    """Return Euclidean distance between two points in micrometres."""
    return math.hypot(x1 - x2, y1 - y2)
