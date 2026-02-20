"""Configuration objects for HD spatial-cell reinforcement learning."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EnvironmentConfig:
    """Runtime configuration for a single-cell RL environment.

    The environment is designed around one nucleus (one potential cell) per episode.
    Candidate bins are selected around that nucleus before the RL loop begins.
    """

    # Maximum distance (in micrometres) from nucleus center to keep a bin.
    max_center_distance_um: float = 80.0

    # Optional radial-band rule: keep bin only if |rc - r| <= radius_band_um,
    # where rc is nucleus radius and r is bin distance from nucleus center.
    radius_band_um: float | None = 80.0

    # If True, invalid actions raise errors. If False, they become no-op updates.
    strict_action_validation: bool = True

    # Max number of steps per episode. If None, computed from number of bins.
    max_steps: int | None = None

    # Used only when max_steps is None. Total steps = max(1, n_bins * multiplier).
    default_steps_multiplier: int = 3

    # Optional random seed reserved for policies/samplers that use this config.
    seed: int | None = None

    def __post_init__(self) -> None:
        """Validate config values once at construction time."""
        if self.max_center_distance_um <= 0:
            raise ValueError("max_center_distance_um must be > 0")

        if self.radius_band_um is not None and self.radius_band_um < 0:
            raise ValueError("radius_band_um must be >= 0 when provided")

        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError("max_steps must be > 0 when provided")

        if self.default_steps_multiplier <= 0:
            raise ValueError("default_steps_multiplier must be > 0")

    def resolve_max_steps(self, n_candidate_bins: int) -> int:
        """Return max steps for an episode with `n_candidate_bins` bins."""
        if self.max_steps is not None:
            return self.max_steps

        safe_bins = max(1, n_candidate_bins)
        return safe_bins * self.default_steps_multiplier
