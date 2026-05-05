"""Episode dataset and artifact loading helpers for PPO/GRPO training."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import re

import numpy as np
import pandas as pd

from .ppo_config import ConfigError, PPOTrainingConfig
from .ppo_state import EpisodeContext, _zscore_1d
from .reward import (
    build_eight_neighbor_index,
    compute_expression_confidence,
    compute_reference_distribution,
)
from .reward_grid_search import (
    _MatrixOnDemandExpressionLoader,
    _build_nuclei_centers,
    _build_nuclei_spatial_index,
    _load_episode_build_expression_context,
    _load_one_episode_artifact,
)


class EpisodeDataset:
    """On-demand loader for episode artifacts, optimized for many-cell datasets."""

    def __init__(self, config: PPOTrainingConfig, rng: np.random.Generator) -> None:
        self._config = config
        self._rng = rng

        if not config.episodes_index_path.exists():
            raise FileNotFoundError(f"episodes index file not found: {config.episodes_index_path}")
        index_df = pd.read_csv(config.episodes_index_path)
        required = ["cell_id", "artifact_path"]
        missing = [col for col in required if col not in index_df.columns]
        if missing:
            raise ValueError(f"episodes index missing required columns: {missing}")
        self._index_df = index_df.reset_index(drop=True)

        self._reference_counts = _load_reference_counts(
            path=config.reference_path,
            reference_format=config.reference_format,
            array_key=config.reference_array_key,
        )
        self._theta = compute_reference_distribution(self._reference_counts, epsilon=config.epsilon)
        self._log_theta = np.log(self._theta)

        nuclei_df = _load_table(config.nuclei_path, config.nuclei_format)
        centers = _build_nuclei_centers(df=nuclei_df, columns=config.nuclei_columns)
        self._nuclei_spatial_index = _build_nuclei_spatial_index(centers)

        expression_ctx = _load_episode_build_expression_context(config.episodes_index_path)
        self._expression_loader: _MatrixOnDemandExpressionLoader | None = None
        self._nuclear_barcode_to_cell: dict[str, str] = {}
        if expression_ctx is not None and config.reference_format == "npz":
            cache_size = (
                int(config.expression_cache_size)
                if config.expression_cache_size is not None
                else int(expression_ctx["cache_size"])
            )
            self._expression_loader = _MatrixOnDemandExpressionLoader(
                matrix_path=Path(expression_ctx["matrix_path"]),
                reference_npz_path=config.reference_path,
                reference_genes_key=config.reference_genes_key,
                cache_size=cache_size,
            )
            bins_path = expression_ctx.get("bins_path")
            if bins_path is not None:
                self._nuclear_barcode_to_cell = _load_nuclear_barcode_assignment_lookup(Path(bins_path))

    @property
    def n_cells(self) -> int:
        return int(len(self._index_df))

    def close(self) -> None:
        if self._expression_loader is not None:
            self._expression_loader.close()

    def sample_rows(self, n_rows: int) -> pd.DataFrame:
        """Sample rows from training set; with replacement if needed."""
        if n_rows <= 0:
            raise ValueError("n_rows must be > 0")
        if len(self._index_df) == 0:
            raise ValueError("episodes index is empty")

        replace = n_rows > len(self._index_df)
        sampled = self._rng.choice(len(self._index_df), size=n_rows, replace=replace)
        return self._index_df.iloc[np.asarray(sampled, dtype=np.int64)].reset_index(drop=True)

    def load_episode_context(
        self,
        cell_id: str,
        artifact_path: Path,
        max_steps_per_episode: int | None,
        *,
        include_candidate_bin_ids: bool = False,
    ) -> EpisodeContext | None:
        """Load one episode artifact and convert it into training-ready static context."""
        prepared = _load_one_episode_artifact(
            artifact_path=artifact_path,
            cell_id=cell_id,
            expression_loader=self._expression_loader,
            theta=self._theta,
            log_theta=self._log_theta,
            nuclei_spatial_index=self._nuclei_spatial_index,
            include_candidate_bin_ids=True,
        )
        if prepared is None:
            return None

        ll = np.asarray(prepared.precomputed_ll, dtype=np.float32)
        if ll.ndim != 2 or ll.shape[0] == 0 or ll.shape[1] == 0:
            return None

        bin_xy = np.asarray(prepared.candidate_bin_xy_um, dtype=np.float32)
        nucleus_center = np.asarray(prepared.nucleus_center_xy_um, dtype=np.float32)
        delta = bin_xy - nucleus_center[None, :]
        d_n = np.sqrt(np.sum(delta * delta, axis=1, dtype=np.float32))
        p_dis = (d_n / float(self._config.r_max_um)).astype(np.float32, copy=False)
        d_other = np.asarray(prepared.precomputed_d_other_um, dtype=np.float32)
        p_overlap = np.maximum(0.0, (d_n - d_other) / float(self._config.r_max_um)).astype(np.float32, copy=False)

        ll_mean = np.mean(ll, axis=1)
        ll_max = np.max(ll, axis=1)
        ll_mean_z = _zscore_1d(ll_mean).astype(np.float32, copy=False)
        ll_max_z = _zscore_1d(ll_max).astype(np.float32, copy=False)
        w2 = float(self._config.w2)
        w3 = float(self._config.w3)
        base_penalty = (w2 * p_dis + w3 * p_overlap).astype(np.float32, copy=False)
        expression_confidence = compute_expression_confidence(
            bin_count_totals=np.asarray(prepared.bin_count_totals, dtype=np.float64),
            pseudocount=float(self._config.expression_confidence_pseudocount),
        ).astype(np.float32, copy=False)
        neighbor_index = build_eight_neighbor_index(prepared.candidate_bin_ids, bin_xy).astype(np.int32, copy=False)
        initial_membership_mask = _build_initial_membership_mask(
            candidate_bin_ids=prepared.candidate_bin_ids,
            cell_id=str(prepared.cell_id),
            nuclear_barcode_to_cell=self._nuclear_barcode_to_cell,
        )

        max_steps = int(max_steps_per_episode) if max_steps_per_episode is not None else max(1, ll.shape[0] * 3)
        return EpisodeContext(
            cell_id=str(prepared.cell_id),
            candidate_bin_ids=tuple(prepared.candidate_bin_ids),
            initial_membership_mask=initial_membership_mask,
            candidate_bin_xy_um=bin_xy.astype(np.float32, copy=False),
            nucleus_center_xy_um=nucleus_center.astype(np.float32, copy=False),
            ll=ll,
            p_dis=p_dis,
            p_overlap=p_overlap,
            ll_mean_z=ll_mean_z,
            ll_max_z=ll_max_z,
            base_penalty=base_penalty,
            expression_confidence=expression_confidence,
            bin_count_totals=np.asarray(prepared.bin_count_totals, dtype=np.float32),
            neighbor_index=neighbor_index,
            max_steps=max_steps,
            log_prior=-np.log(float(ll.shape[1])),
            r_max_um=float(self._config.r_max_um),
            w1=float(self._config.w1),
            w2=w2,
            w3=w3,
            w4=float(self._config.w4),
            w5=float(self._config.w5),
            stop_lambda=float(self._config.stop_lambda),
            stop_stat=str(self._config.stop_stat),
            stop_top_k=int(self._config.stop_top_k),
            expression_confidence_pseudocount=float(self._config.expression_confidence_pseudocount),
            normalize_expression_zscore=bool(self._config.normalize_expression_zscore),
            zscore_delta=float(self._config.zscore_delta),
        )


def _normalize_cell_id(value: Any) -> str | None:
    """Normalize mixed int/float/string cell IDs into one stable string form."""
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        if not np.isfinite(value):
            return None
        if float(value).is_integer():
            return str(int(value))
        return str(value)
    text = str(value).strip()
    if not text:
        return None
    if re.fullmatch(r"[+-]?\\d+\\.0+", text):
        return text.split(".", 1)[0]
    return text


def _load_nuclear_barcode_assignment_lookup(bins_path: Path) -> dict[str, str]:
    """Load confident nuclear barcode -> cell_id assignments from merged bins metadata."""
    if not bins_path.exists():
        raise FileNotFoundError(f"episode-build bins metadata not found: {bins_path}")

    df = pd.read_parquet(
        bins_path,
        columns=[
            "barcode",
            "has_nuclear_annotation",
            "dominant_cell_id",
            "ambiguous_nuclear_assignment",
        ],
    )
    if len(df) == 0:
        return {}

    has_nuclear = df["has_nuclear_annotation"].fillna(False).astype(bool)
    ambiguous = df["ambiguous_nuclear_assignment"].fillna(False).astype(bool)
    out = df.loc[has_nuclear & ~ambiguous, ["barcode", "dominant_cell_id"]].copy()
    if out.empty:
        return {}

    out["cell_id"] = out["dominant_cell_id"].map(_normalize_cell_id)
    out = out.loc[out["cell_id"].notna()].copy()
    if out.empty:
        return {}

    return {
        str(barcode): str(cell_id)
        for barcode, cell_id in zip(out["barcode"].astype(str), out["cell_id"].astype(str), strict=False)
    }


def _build_initial_membership_mask(
    *,
    candidate_bin_ids: tuple[str, ...],
    cell_id: str,
    nuclear_barcode_to_cell: dict[str, str],
) -> np.ndarray:
    """Seed episode state with all confident nuclear bins for this cell."""
    if not candidate_bin_ids:
        return np.zeros((0,), dtype=np.uint8)
    normalized_cell_id = _normalize_cell_id(cell_id)
    return np.asarray(
        [
            1 if normalized_cell_id is not None and nuclear_barcode_to_cell.get(str(bin_id)) == normalized_cell_id else 0
            for bin_id in candidate_bin_ids
        ],
        dtype=np.uint8,
    )


def _load_reference_counts(path: Path, reference_format: str, array_key: str) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"reference file not found: {path}")
    if reference_format == "npy":
        arr = np.load(path)
        return _validate_reference_matrix(arr)
    if reference_format == "npz":
        with np.load(path) as data:
            if array_key not in data:
                raise ConfigError(f"reference array key {array_key!r} is not present in {path}")
            arr = data[array_key]
        return _validate_reference_matrix(arr)

    table = _load_table(path, reference_format)
    numeric_cols = [c for c in table.columns if pd.api.types.is_numeric_dtype(table[c])]
    if not numeric_cols:
        raise ValueError(f"reference table {path} has no numeric columns to form C[K,G]")
    arr = table.loc[:, numeric_cols].to_numpy(dtype=np.float64, copy=True)
    return _validate_reference_matrix(arr)


def _validate_reference_matrix(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("reference matrix must have shape (K, G)")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError("reference matrix must have positive shape in both dimensions")
    if not np.isfinite(arr).all():
        raise ValueError("reference matrix contains non-finite values")
    if (arr < 0).any():
        raise ValueError("reference matrix must be non-negative")
    return arr


def _load_table(path: Path, table_format: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"table file not found: {path}")
    if table_format == "parquet":
        return pd.read_parquet(path)
    if table_format == "csv":
        return pd.read_csv(path)
    if table_format == "tsv":
        return pd.read_csv(path, sep="\t")
    raise ConfigError(f"unsupported table format for loader: {table_format!r}")
