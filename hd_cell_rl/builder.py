"""Episode builders that transform segmentation + bin tables into RL episodes."""

from __future__ import annotations

from collections.abc import Iterable

from .config import EnvironmentConfig
from .geometry import euclidean_distance_um
from .models import BinRecord, CellEpisodeData, NucleusRecord
from .spatial_index import BruteForceSpatialIndex


def _is_candidate_bin(bin_record: BinRecord, nucleus: NucleusRecord, config: EnvironmentConfig) -> bool:
    """Check whether a bin satisfies geometric candidate rules for one nucleus."""
    distance_um = euclidean_distance_um(
        nucleus.center_x_um,
        nucleus.center_y_um,
        bin_record.x_um,
        bin_record.y_um,
    )

    # Hard cap: bins must be within max_center_distance_um from nucleus center.
    if distance_um > config.max_center_distance_um:
        return False

    # Optional band rule: |rc - r| <= radius_band_um.
    if config.radius_band_um is not None:
        radial_delta = abs(nucleus.radius_um - distance_um)
        if radial_delta > config.radius_band_um:
            return False

    return True


def build_episode_for_cell(
    nucleus: NucleusRecord,
    spatial_index: BruteForceSpatialIndex,
    config: EnvironmentConfig,
) -> CellEpisodeData:
    """Build one `CellEpisodeData` object for one nucleus-centered environment."""
    nearby = spatial_index.query_radius(
        center_x_um=nucleus.center_x_um,
        center_y_um=nucleus.center_y_um,
        radius_um=config.max_center_distance_um,
    )

    # Apply all geometric filters in one explicit pass for readability.
    candidates = []
    for bin_record in nearby:
        if _is_candidate_bin(bin_record, nucleus, config):
            candidates.append(bin_record)

    return CellEpisodeData(nucleus=nucleus, candidate_bins=tuple(candidates))


def build_episodes(
    nuclei: Iterable[NucleusRecord],
    bins: Iterable[BinRecord],
    config: EnvironmentConfig | None = None,
) -> list[CellEpisodeData]:
    """Build one episode per nucleus from shared bin data.

    Notes:
    - This function currently uses a brute-force index implementation.
    - The index can be replaced later without changing function signatures.
    """
    runtime_cfg = config or EnvironmentConfig()
    spatial_index = BruteForceSpatialIndex(bins)

    episodes: list[CellEpisodeData] = []
    for nucleus in nuclei:
        episodes.append(build_episode_for_cell(nucleus, spatial_index, runtime_cfg))

    return episodes
