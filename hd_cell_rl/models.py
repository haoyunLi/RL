"""Typed data models used by the HD spatial-cell RL environment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BinRecord:
    """One spatial bin with coordinates and expression vector."""

    bin_id: str
    x_um: float
    y_um: float
    expression: np.ndarray

    def __post_init__(self) -> None:
        """Normalize expression vectors to float32 and validate shape."""
        expr = np.asarray(self.expression, dtype=np.float32)

        if expr.ndim != 1:
            raise ValueError(f"expression for bin {self.bin_id!r} must be a 1D vector")

        if expr.size == 0:
            raise ValueError(f"expression for bin {self.bin_id!r} must not be empty")

        if not np.all(np.isfinite(expr)):
            raise ValueError(f"expression for bin {self.bin_id!r} contains non-finite values")

        object.__setattr__(self, "expression", expr)


@dataclass(frozen=True)
class NucleusRecord:
    """Nucleus attributes for one candidate cell episode."""

    cell_id: str
    center_x_um: float
    center_y_um: float
    radius_um: float
    cell_type: str | None = None

    def __post_init__(self) -> None:
        """Validate geometric attributes from nuclear segmentation."""
        if self.radius_um <= 0:
            raise ValueError(f"radius_um for cell {self.cell_id!r} must be > 0")


@dataclass(frozen=True)
class CellEpisodeData:
    """Precomputed episode payload for one nucleus-centered environment."""

    nucleus: NucleusRecord
    candidate_bins: tuple[BinRecord, ...]

    def __post_init__(self) -> None:
        """Validate candidate bins and enforce consistent expression dimension."""
        if not isinstance(self.candidate_bins, tuple):
            object.__setattr__(self, "candidate_bins", tuple(self.candidate_bins))

        if len(self.candidate_bins) == 0:
            # Empty candidate lists are allowed for debugging edge cases.
            return

        expected_dim = self.candidate_bins[0].expression.shape[0]

        for bin_record in self.candidate_bins:
            if bin_record.expression.shape[0] != expected_dim:
                raise ValueError(
                    "all candidate bins for one episode must have the same expression dimension"
                )

    @property
    def n_candidate_bins(self) -> int:
        """Return number of candidate bins in this episode."""
        return len(self.candidate_bins)

    @property
    def n_genes(self) -> int:
        """Return expression dimension, or 0 if no candidate bins exist."""
        if len(self.candidate_bins) == 0:
            return 0
        return self.candidate_bins[0].expression.shape[0]
