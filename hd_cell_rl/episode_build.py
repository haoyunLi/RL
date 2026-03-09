"""Reproducible episode-build pipeline for HD cell RL experiments."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import datetime as dt
import json
import multiprocessing as mp
from pathlib import Path
import platform
import re
import subprocess
import sys
from typing import Any

import h5py
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import yaml

from .builder import build_episodes
from .config import EnvironmentConfig
from .models import BinRecord, NucleusRecord


class ConfigError(ValueError):
    """Raised when episode-build config is invalid."""


_MATRIX_CHUNK_WORKER_CONTEXT: dict[str, Any] | None = None


@dataclass(frozen=True)
class EpisodeBuildConfig:
    """Resolved config for one reproducible episode-build run."""

    run_name: str
    output_root: Path
    seed: int | None
    nuclei_path: Path
    bins_path: Path
    nuclei_format: str
    bins_format: str
    nuclei_columns: dict[str, str | None]
    bin_columns: dict[str, str]
    expression_mode: str
    expression_prefix: str | None
    expression_columns: tuple[str, ...]
    expression_matrix_h5_path: Path | None
    expression_matrix_col_index_column: str
    expression_reference_npz_path: Path | None
    expression_reference_genes_key: str
    expression_nuclei_chunk_size: int
    expression_cache_size: int
    expression_n_workers: int
    expression_save_compressed: bool
    env_config: EnvironmentConfig

    def to_serializable_dict(self) -> dict[str, Any]:
        """Return config as plain Python types for YAML/JSON output."""
        return {
            "run": {
                "name": self.run_name,
                "output_root": str(self.output_root),
                "seed": self.seed,
            },
            "inputs": {
                "nuclei_path": str(self.nuclei_path),
                "bins_path": str(self.bins_path),
                "nuclei_format": self.nuclei_format,
                "bins_format": self.bins_format,
                "nuclei_columns": self.nuclei_columns,
                "bin_columns": self.bin_columns,
                "expression": {
                    "mode": self.expression_mode,
                    "prefix": self.expression_prefix,
                    "columns": list(self.expression_columns),
                    "matrix_h5_path": None if self.expression_matrix_h5_path is None else str(self.expression_matrix_h5_path),
                    "matrix_col_index_column": self.expression_matrix_col_index_column,
                    "reference_npz_path": None if self.expression_reference_npz_path is None else str(self.expression_reference_npz_path),
                    "reference_genes_key": self.expression_reference_genes_key,
                    "nuclei_chunk_size": self.expression_nuclei_chunk_size,
                    "cache_size": self.expression_cache_size,
                    "n_workers": self.expression_n_workers,
                    "save_compressed": self.expression_save_compressed,
                },
            },
            "environment": {
                "max_center_distance_um": self.env_config.max_center_distance_um,
                "radius_band_um": self.env_config.radius_band_um,
                "strict_action_validation": self.env_config.strict_action_validation,
                "max_steps": self.env_config.max_steps,
                "default_steps_multiplier": self.env_config.default_steps_multiplier,
            },
        }


def load_episode_build_config(config_path: str | Path) -> EpisodeBuildConfig:
    """Load and validate YAML config for episode building."""
    path = Path(config_path)
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if not isinstance(raw, dict):
        raise ConfigError("config root must be a mapping")

    run = _as_dict(raw.get("run"), "run")
    inputs = _as_dict(raw.get("inputs"), "inputs")
    environment = _as_dict(raw.get("environment"), "environment")

    run_name = str(run.get("name", "episode_build"))
    if not run_name.strip():
        raise ConfigError("run.name must be a non-empty string")

    output_root = Path(str(run.get("output_root", "runs"))).expanduser().resolve()
    seed = run.get("seed")
    if seed is not None:
        seed = int(seed)

    nuclei_path = Path(str(_require(inputs, "nuclei_path", "inputs"))).expanduser().resolve()
    bins_path = Path(str(_require(inputs, "bins_path", "inputs"))).expanduser().resolve()

    nuclei_format = _normalize_format(str(inputs.get("nuclei_format", "auto")), nuclei_path)
    bins_format = _normalize_format(str(inputs.get("bins_format", "auto")), bins_path)

    nuclei_columns = _default_nuclei_columns(_as_dict(inputs.get("nuclei_columns", {}), "inputs.nuclei_columns"))
    bin_columns = _default_bin_columns(_as_dict(inputs.get("bin_columns", {}), "inputs.bin_columns"))

    expression_cfg = _as_dict(inputs.get("expression", {}), "inputs.expression")
    expression_mode = str(expression_cfg.get("mode", "prefix"))
    expression_prefix = expression_cfg.get("prefix")
    expression_columns_raw = expression_cfg.get("columns", [])
    expression_matrix_h5_path_raw = expression_cfg.get("matrix_h5_path")
    expression_matrix_col_index_column = str(expression_cfg.get("matrix_col_index_column", "matrix_col_index"))
    expression_reference_npz_path_raw = expression_cfg.get("reference_npz_path")
    expression_reference_genes_key = str(expression_cfg.get("reference_genes_key", "genes"))
    expression_nuclei_chunk_size_raw = expression_cfg.get("nuclei_chunk_size", 512)
    expression_cache_size_raw = expression_cfg.get("cache_size", 20000)
    expression_n_workers_raw = expression_cfg.get("n_workers", 1)
    expression_save_compressed_raw = expression_cfg.get("save_compressed", True)

    if expression_columns_raw is None:
        expression_columns_raw = []
    if not isinstance(expression_columns_raw, list):
        raise ConfigError("inputs.expression.columns must be a list")

    expression_columns = tuple(str(col) for col in expression_columns_raw)
    expression_matrix_h5_path = (
        None
        if expression_matrix_h5_path_raw is None
        else Path(str(expression_matrix_h5_path_raw)).expanduser().resolve()
    )
    expression_reference_npz_path = (
        None
        if expression_reference_npz_path_raw is None
        else Path(str(expression_reference_npz_path_raw)).expanduser().resolve()
    )
    expression_nuclei_chunk_size = int(expression_nuclei_chunk_size_raw)
    expression_cache_size = int(expression_cache_size_raw)
    expression_n_workers = int(expression_n_workers_raw)
    expression_save_compressed = bool(expression_save_compressed_raw)

    if expression_mode not in {"prefix", "list", "matrix_h5"}:
        raise ConfigError("inputs.expression.mode must be 'prefix', 'list', or 'matrix_h5'")

    if expression_mode == "prefix":
        if expression_prefix is None or not str(expression_prefix):
            raise ConfigError("inputs.expression.prefix must be set when mode='prefix'")
        expression_prefix = str(expression_prefix)
    elif expression_mode == "matrix_h5":
        expression_prefix = None
        if expression_matrix_h5_path is None:
            raise ConfigError("inputs.expression.matrix_h5_path must be set when mode='matrix_h5'")
        if not expression_matrix_col_index_column:
            raise ConfigError("inputs.expression.matrix_col_index_column must be non-empty when mode='matrix_h5'")
        if expression_nuclei_chunk_size <= 0:
            raise ConfigError("inputs.expression.nuclei_chunk_size must be > 0")
        if expression_cache_size < 0:
            raise ConfigError("inputs.expression.cache_size must be >= 0")
        if expression_n_workers <= 0:
            raise ConfigError("inputs.expression.n_workers must be > 0")

    max_center_distance_um = float(environment.get("max_center_distance_um", 80.0))
    radius_band_um_raw = environment.get("radius_band_um", 80.0)
    radius_band_um = None if radius_band_um_raw is None else float(radius_band_um_raw)
    strict_action_validation = bool(environment.get("strict_action_validation", True))

    max_steps_raw = environment.get("max_steps", None)
    max_steps = None if max_steps_raw is None else int(max_steps_raw)

    default_steps_multiplier = int(environment.get("default_steps_multiplier", 3))

    env_config = EnvironmentConfig(
        max_center_distance_um=max_center_distance_um,
        radius_band_um=radius_band_um,
        strict_action_validation=strict_action_validation,
        max_steps=max_steps,
        default_steps_multiplier=default_steps_multiplier,
        seed=seed,
    )

    return EpisodeBuildConfig(
        run_name=run_name,
        output_root=output_root,
        seed=seed,
        nuclei_path=nuclei_path,
        bins_path=bins_path,
        nuclei_format=nuclei_format,
        bins_format=bins_format,
        nuclei_columns=nuclei_columns,
        bin_columns=bin_columns,
        expression_mode=expression_mode,
        expression_prefix=expression_prefix,
        expression_columns=expression_columns,
        expression_matrix_h5_path=expression_matrix_h5_path,
        expression_matrix_col_index_column=expression_matrix_col_index_column,
        expression_reference_npz_path=expression_reference_npz_path,
        expression_reference_genes_key=expression_reference_genes_key,
        expression_nuclei_chunk_size=expression_nuclei_chunk_size,
        expression_cache_size=expression_cache_size,
        expression_n_workers=expression_n_workers,
        expression_save_compressed=expression_save_compressed,
        env_config=env_config,
    )


def run_episode_build(
    config: EpisodeBuildConfig,
    limit_nuclei: int | None = None,
    limit_bins: int | None = None,
) -> Path:
    """Run full episode-build pipeline and return output run directory."""
    config.output_root.mkdir(parents=True, exist_ok=True)

    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = config.output_root / f"{_slugify(config.run_name)}_{timestamp}"
    run_dir.mkdir(parents=False, exist_ok=False)

    config_dir = run_dir / "config"
    logs_dir = run_dir / "logs"
    states_dir = run_dir / "states"
    index_chunks_dir = run_dir / "index_chunks"
    config_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    states_dir.mkdir(parents=True, exist_ok=True)
    index_chunks_dir.mkdir(parents=True, exist_ok=True)

    steps_log_path = logs_dir / "steps.jsonl"
    _append_step_log(
        steps_log_path,
        event="run_start",
        payload={"run_name": config.run_name, "seed": config.seed},
    )

    rng = np.random.default_rng(config.seed)

    nuclei_load_columns = [config.nuclei_columns["cell_id"], config.nuclei_columns["center_x_um"], config.nuclei_columns["center_y_um"], config.nuclei_columns["radius_um"]]
    if config.nuclei_columns["cell_type"] is not None:
        nuclei_load_columns.append(config.nuclei_columns["cell_type"])
    nuclei_df = _load_table(config.nuclei_path, config.nuclei_format, columns=_dedupe_keep_order(nuclei_load_columns))

    if config.expression_mode == "matrix_h5":
        bins_load_columns = _dedupe_keep_order(
            [
                config.bin_columns["bin_id"],
                config.bin_columns["x_um"],
                config.bin_columns["y_um"],
                config.expression_matrix_col_index_column,
            ]
        )
    elif config.expression_mode == "list":
        bins_load_columns = _dedupe_keep_order(
            [
                config.bin_columns["bin_id"],
                config.bin_columns["x_um"],
                config.bin_columns["y_um"],
                *config.expression_columns,
            ]
        )
    else:
        bins_load_columns = None

    bins_df = _load_table(config.bins_path, config.bins_format, columns=bins_load_columns)
    _append_step_log(
        steps_log_path,
        event="tables_loaded",
        payload={"n_input_nuclei": int(len(nuclei_df)), "n_input_bins": int(len(bins_df))},
    )

    if limit_nuclei is not None:
        nuclei_df = _sample_rows(nuclei_df, int(limit_nuclei), rng)
        _append_step_log(
            steps_log_path,
            event="limit_nuclei_applied",
            payload={"limit_nuclei": int(limit_nuclei), "n_nuclei_after_limit": int(len(nuclei_df))},
        )

    if limit_bins is not None:
        bins_df = _sample_rows(bins_df, int(limit_bins), rng)
        _append_step_log(
            steps_log_path,
            event="limit_bins_applied",
            payload={"limit_bins": int(limit_bins), "n_bins_after_limit": int(len(bins_df))},
        )

    _validate_nuclei_schema(nuclei_df, config.nuclei_columns)
    _validate_bins_schema_common(bins_df, config.bin_columns)
    _append_step_log(steps_log_path, event="schema_validated", payload={})

    nuclei_records = _build_nucleus_records(nuclei_df, config.nuclei_columns)
    if config.expression_mode == "matrix_h5":
        episode_index, expression_dim = _build_and_save_episodes_matrix_backed(
            nuclei_records=nuclei_records,
            bins_df=bins_df,
            bin_columns=config.bin_columns,
            env_config=config.env_config,
            matrix_h5_path=config.expression_matrix_h5_path,
            matrix_col_index_column=config.expression_matrix_col_index_column,
            reference_npz_path=config.expression_reference_npz_path,
            reference_genes_key=config.expression_reference_genes_key,
            nuclei_chunk_size=config.expression_nuclei_chunk_size,
            expression_cache_size=config.expression_cache_size,
            n_workers=config.expression_n_workers,
            save_compressed=config.expression_save_compressed,
            states_dir=states_dir,
            index_chunks_dir=index_chunks_dir,
            steps_log_path=steps_log_path,
        )
        resolved_expression_columns = ("__matrix_h5__",)
    else:
        expression_columns = _resolve_expression_columns(
            bins_df=bins_df,
            mode=config.expression_mode,
            prefix=config.expression_prefix,
            configured_columns=config.expression_columns,
        )
        _append_step_log(
            steps_log_path,
            event="expression_columns_resolved",
            payload={"expression_dim": int(len(expression_columns))},
        )
        _validate_bins_expression_columns(bins_df, expression_columns)
        bin_records = _build_bin_records(bins_df, config.bin_columns, expression_columns)
        _append_step_log(
            steps_log_path,
            event="records_built",
            payload={"n_nucleus_records": int(len(nuclei_records)), "n_bin_records": int(len(bin_records))},
        )

        episodes = build_episodes(nuclei=nuclei_records, bins=bin_records, config=config.env_config)
        _append_step_log(
            steps_log_path,
            event="episodes_built",
            payload={"n_episodes": int(len(episodes))},
        )
        expression_dim = len(expression_columns)
        resolved_expression_columns = expression_columns
        episode_index = _save_episode_artifacts(
            episodes=episodes,
            states_dir=states_dir,
            steps_log_path=steps_log_path,
        )
    summary = _compute_summary(
        n_input_nuclei=len(nuclei_df),
        n_input_bins=len(bins_df),
        expression_dim=expression_dim,
        episode_index=episode_index,
    )

    _write_yaml(config_dir / "config_resolved.yaml", _merge_config_with_resolved_columns(config, resolved_expression_columns))
    _write_json(config_dir / "metadata.json", _build_metadata(config=config, run_dir=run_dir))
    _write_json(run_dir / "summary.json", summary)
    pd.DataFrame(episode_index).to_csv(run_dir / "episodes_index.csv", index=False)
    _append_step_log(
        steps_log_path,
        event="run_complete",
        payload={
            "n_episodes": int(summary["n_episodes"]),
            "empty_episode_count": int(summary["empty_episode_count"]),
        },
    )

    return run_dir


def run_episode_build_from_config(
    config_path: str | Path,
    limit_nuclei: int | None = None,
    limit_bins: int | None = None,
) -> Path:
    """Convenience wrapper: load config, execute build, and return run dir."""
    config = load_episode_build_config(config_path)
    return run_episode_build(config=config, limit_nuclei=limit_nuclei, limit_bins=limit_bins)


def _as_dict(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"{name} must be a mapping")
    return value


def _require(mapping: dict[str, Any], key: str, section: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"missing required key '{key}' in section '{section}'")
    return mapping[key]


def _default_nuclei_columns(overrides: dict[str, Any]) -> dict[str, str | None]:
    cols = {
        "cell_id": "cell_id",
        "center_x_um": "center_x_um",
        "center_y_um": "center_y_um",
        "radius_um": "radius_um",
        "cell_type": None,
    }
    cols.update(overrides)

    if cols["cell_type"] is not None:
        cols["cell_type"] = str(cols["cell_type"])

    for required_key in ("cell_id", "center_x_um", "center_y_um", "radius_um"):
        if cols.get(required_key) is None:
            raise ConfigError(f"inputs.nuclei_columns.{required_key} must not be null")
        cols[required_key] = str(cols[required_key])

    return cols


def _default_bin_columns(overrides: dict[str, Any]) -> dict[str, str]:
    cols = {
        "bin_id": "bin_id",
        "x_um": "x_um",
        "y_um": "y_um",
    }
    cols.update(overrides)

    for key in ("bin_id", "x_um", "y_um"):
        if cols.get(key) is None:
            raise ConfigError(f"inputs.bin_columns.{key} must not be null")
        cols[key] = str(cols[key])

    return cols


def _normalize_format(raw_format: str, path: Path) -> str:
    value = raw_format.strip().lower()
    if value == "auto":
        suffix = path.suffix.lower()
        if suffix in {".csv"}:
            return "csv"
        if suffix in {".tsv", ".txt"}:
            return "tsv"
        if suffix in {".parquet", ".pq"}:
            return "parquet"
        raise ConfigError(f"cannot infer format from extension for file: {path}")

    if value not in {"csv", "tsv", "parquet"}:
        raise ConfigError(f"unsupported table format: {raw_format!r}")
    return value


def _load_table(path: Path, table_format: str, columns: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"table file not found: {path}")

    if table_format == "csv":
        return pd.read_csv(path, usecols=columns)
    if table_format == "tsv":
        return pd.read_csv(path, sep="\t", usecols=columns)
    if table_format == "parquet":
        return pd.read_parquet(path, columns=columns)

    raise ConfigError(f"unsupported format in loader: {table_format!r}")


def _sample_rows(df: pd.DataFrame, limit: int, rng: np.random.Generator) -> pd.DataFrame:
    if limit <= 0:
        raise ValueError("limit values must be > 0")

    if len(df) <= limit:
        return df.reset_index(drop=True)

    keep = rng.choice(len(df), size=limit, replace=False)
    keep_sorted = np.sort(keep)
    return df.iloc[keep_sorted].reset_index(drop=True)


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _resolve_expression_columns(
    bins_df: pd.DataFrame,
    mode: str,
    prefix: str | None,
    configured_columns: tuple[str, ...],
) -> tuple[str, ...]:
    if mode == "prefix":
        assert prefix is not None
        columns = tuple(col for col in bins_df.columns if str(col).startswith(prefix))
        if not columns:
            raise ConfigError(
                f"no expression columns found with prefix {prefix!r}; "
                "check inputs.expression.prefix and input bins table"
            )
        return columns

    if mode == "list":
        if not configured_columns:
            raise ConfigError("inputs.expression.columns must be non-empty when mode='list'")
        missing = [col for col in configured_columns if col not in bins_df.columns]
        if missing:
            raise ConfigError(f"expression columns missing in bins table: {missing}")
        return configured_columns

    raise ConfigError(f"unsupported expression mode: {mode!r}")


def _validate_nuclei_schema(df: pd.DataFrame, columns: dict[str, str | None]) -> None:
    required = [columns["cell_id"], columns["center_x_um"], columns["center_y_um"], columns["radius_um"]]
    _require_columns(df, required, table_name="nuclei")

    cell_id_col = columns["cell_id"]
    assert cell_id_col is not None
    if df[cell_id_col].isna().any():
        raise ValueError("nuclei cell_id column contains missing values")

    if df[cell_id_col].duplicated().any():
        dup_value = str(df.loc[df[cell_id_col].duplicated(), cell_id_col].iloc[0])
        raise ValueError(f"duplicate cell_id found in nuclei table: {dup_value!r}")

    for key in ("center_x_um", "center_y_um", "radius_um"):
        col = columns[key]
        assert col is not None
        _validate_numeric_series(df[col], f"nuclei.{col}")

    radius_col = columns["radius_um"]
    assert radius_col is not None
    if (pd.to_numeric(df[radius_col], errors="coerce") <= 0).any():
        raise ValueError("nuclei radius_um must be > 0 for all rows")

    cell_type_col = columns["cell_type"]
    if cell_type_col is not None:
        if cell_type_col not in df.columns:
            raise ValueError(f"nuclei cell_type column {cell_type_col!r} is not present")


def _validate_bins_schema_common(df: pd.DataFrame, columns: dict[str, str]) -> None:
    required = [columns["bin_id"], columns["x_um"], columns["y_um"]]
    _require_columns(df, required, table_name="bins")

    bin_id_col = columns["bin_id"]
    if df[bin_id_col].isna().any():
        raise ValueError("bins bin_id column contains missing values")

    if df[bin_id_col].duplicated().any():
        dup_value = str(df.loc[df[bin_id_col].duplicated(), bin_id_col].iloc[0])
        raise ValueError(f"duplicate bin_id found in bins table: {dup_value!r}")

    for key in ("x_um", "y_um"):
        _validate_numeric_series(df[columns[key]], f"bins.{columns[key]}")

def _validate_bins_expression_columns(df: pd.DataFrame, expression_columns: tuple[str, ...]) -> None:
    for expr_col in expression_columns:
        if expr_col not in df.columns:
            raise ValueError(f"expression column missing in bins table: {expr_col!r}")
        _validate_numeric_series(df[expr_col], f"bins.{expr_col}")


def _require_columns(df: pd.DataFrame, columns: list[str], table_name: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"missing columns in {table_name} table: {missing}")


def _validate_numeric_series(series: pd.Series, name: str) -> None:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any():
        raise ValueError(f"column {name} contains non-numeric or missing values")

    values = numeric.to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(values).all():
        raise ValueError(f"column {name} contains non-finite values")


def _build_nucleus_records(df: pd.DataFrame, columns: dict[str, str | None]) -> list[NucleusRecord]:
    cell_id_col = columns["cell_id"]
    cx_col = columns["center_x_um"]
    cy_col = columns["center_y_um"]
    radius_col = columns["radius_um"]

    assert cell_id_col is not None
    assert cx_col is not None
    assert cy_col is not None
    assert radius_col is not None

    cell_type_col = columns["cell_type"]

    cell_ids = df[cell_id_col].astype(str).to_numpy()
    center_x = pd.to_numeric(df[cx_col], errors="raise").to_numpy(dtype=np.float32)
    center_y = pd.to_numeric(df[cy_col], errors="raise").to_numpy(dtype=np.float32)
    radius = pd.to_numeric(df[radius_col], errors="raise").to_numpy(dtype=np.float32)

    if cell_type_col is not None:
        raw_cell_types = df[cell_type_col].to_numpy()
    else:
        raw_cell_types = np.full(len(df), None, dtype=object)

    records: list[NucleusRecord] = []
    for i in range(len(df)):
        raw_ct = raw_cell_types[i]
        cell_type = None if pd.isna(raw_ct) else str(raw_ct)
        records.append(
            NucleusRecord(
                cell_id=str(cell_ids[i]),
                center_x_um=float(center_x[i]),
                center_y_um=float(center_y[i]),
                radius_um=float(radius[i]),
                cell_type=cell_type,
            )
        )

    return records


def _build_bin_records(
    df: pd.DataFrame,
    columns: dict[str, str],
    expression_columns: tuple[str, ...],
) -> list[BinRecord]:
    bin_ids = df[columns["bin_id"]].astype(str).to_numpy()
    x_um = pd.to_numeric(df[columns["x_um"]], errors="raise").to_numpy(dtype=np.float32)
    y_um = pd.to_numeric(df[columns["y_um"]], errors="raise").to_numpy(dtype=np.float32)

    expression_matrix = (
        df.loc[:, list(expression_columns)]
        .apply(pd.to_numeric, errors="raise")
        .to_numpy(dtype=np.float32, copy=True)
    )

    records: list[BinRecord] = []
    for i in range(len(df)):
        records.append(
            BinRecord(
                bin_id=str(bin_ids[i]),
                x_um=float(x_um[i]),
                y_um=float(y_um[i]),
                expression=expression_matrix[i],
            )
        )

    return records


def _build_and_save_episodes_matrix_backed(
    nuclei_records: list[NucleusRecord],
    bins_df: pd.DataFrame,
    bin_columns: dict[str, str],
    env_config: EnvironmentConfig,
    matrix_h5_path: Path | None,
    matrix_col_index_column: str,
    reference_npz_path: Path | None,
    reference_genes_key: str,
    nuclei_chunk_size: int,
    expression_cache_size: int,
    n_workers: int,
    save_compressed: bool,
    states_dir: Path,
    index_chunks_dir: Path,
    steps_log_path: Path,
) -> tuple[list[dict[str, Any]], int]:
    """Build and save episodes by loading expression vectors from 10x H5 via matrix_col_index."""
    if matrix_h5_path is None:
        raise ConfigError("matrix_h5_path must be set for matrix_h5 expression mode")
    if not matrix_h5_path.exists():
        raise FileNotFoundError(f"matrix H5 file not found: {matrix_h5_path}")
    if matrix_col_index_column not in bins_df.columns:
        raise ValueError(
            f"bins table missing matrix col index column {matrix_col_index_column!r} "
            f"required for matrix_h5 mode"
        )

    bin_ids = bins_df[bin_columns["bin_id"]].astype(str).to_numpy()
    x_um = pd.to_numeric(bins_df[bin_columns["x_um"]], errors="raise").to_numpy(dtype=np.float32)
    y_um = pd.to_numeric(bins_df[bin_columns["y_um"]], errors="raise").to_numpy(dtype=np.float32)
    matrix_col_index = pd.to_numeric(bins_df[matrix_col_index_column], errors="raise").to_numpy(dtype=np.int64)

    if (matrix_col_index < 0).any():
        raise ValueError(f"column bins.{matrix_col_index_column} must be >= 0")

    if n_workers <= 0:
        raise ValueError("n_workers must be > 0")

    selected_feature_indices: np.ndarray
    n_features: int
    n_cols: int
    with h5py.File(matrix_h5_path, "r") as h5:
        if "matrix" not in h5:
            raise ValueError(f"H5 file does not contain 'matrix' group: {matrix_h5_path}")
        mg = h5["matrix"]
        for key in ("data", "indices", "indptr", "shape", "features"):
            if key not in mg:
                raise ValueError(f"H5 matrix group missing key: matrix/{key}")
        fg = mg["features"]
        if "name" not in fg:
            raise ValueError("H5 matrix/features missing 'name' dataset")

        shape = tuple(int(v) for v in mg["shape"][:].tolist())
        if len(shape) != 2:
            raise ValueError(f"matrix/shape in {matrix_h5_path} is invalid: {shape}")
        n_features, n_cols = int(shape[0]), int(shape[1])
        if (matrix_col_index >= n_cols).any():
            bad = int(matrix_col_index[matrix_col_index >= n_cols][0])
            raise ValueError(
                f"bins.{matrix_col_index_column} contains index {bad} outside matrix column range [0, {n_cols})"
            )

        feature_names = np.asarray([x.decode("utf-8") for x in fg["name"][:]], dtype="U")
        selected_feature_indices = _resolve_matrix_feature_indices(
            feature_names=feature_names,
            reference_npz_path=reference_npz_path,
            reference_genes_key=reference_genes_key,
        )
        if selected_feature_indices.size == 0:
            raise ValueError("matrix_h5 mode selected zero genes")
        if selected_feature_indices.max() >= n_features:
            raise ValueError("selected feature index exceeds matrix row count")

    expression_dim = int(selected_feature_indices.size)
    _append_step_log(
        steps_log_path,
        event="expression_columns_resolved",
        payload={
            "expression_dim": expression_dim,
            "matrix_h5_path": str(matrix_h5_path),
            "n_matrix_features": n_features,
            "n_matrix_barcodes": n_cols,
            "nuclei_chunk_size": int(nuclei_chunk_size),
            "expression_cache_size": int(expression_cache_size),
            "n_workers": int(n_workers),
            "save_compressed": bool(save_compressed),
        },
    )

    chunk_specs: list[tuple[int, int, int, list[NucleusRecord]]] = []
    n_nuclei = len(nuclei_records)
    for chunk_id, chunk_start in enumerate(range(0, n_nuclei, nuclei_chunk_size)):
        chunk_end = min(chunk_start + nuclei_chunk_size, n_nuclei)
        chunk_specs.append((chunk_id, chunk_start, chunk_end, nuclei_records[chunk_start:chunk_end]))

    worker_context: dict[str, Any] = {
        "bin_ids": bin_ids,
        "x_um": x_um,
        "y_um": y_um,
        "matrix_col_index": matrix_col_index,
        "max_center_distance_um": float(env_config.max_center_distance_um),
        "radius_band_um": env_config.radius_band_um,
        "states_dir": str(states_dir),
        "index_chunks_dir": str(index_chunks_dir),
        "save_compressed": bool(save_compressed),
    }

    global _MATRIX_CHUNK_WORKER_CONTEXT
    _MATRIX_CHUNK_WORKER_CONTEXT = worker_context
    chunk_results: list[tuple[int, int, int, str, int]] = []
    try:
        use_parallel = n_workers > 1 and len(chunk_specs) > 1
        if use_parallel:
            try:
                mp_ctx = mp.get_context("fork")
            except ValueError:
                use_parallel = False
                _append_step_log(
                    steps_log_path,
                    event="parallel_fallback",
                    payload={"reason": "fork_start_method_unavailable"},
                )

        if use_parallel:
            try:
                with mp_ctx.Pool(processes=int(n_workers)) as pool:
                    for result in pool.imap_unordered(_process_matrix_chunk_worker, chunk_specs):
                        chunk_results.append(result)
                        chunk_id, chunk_start, chunk_end, _, n_rows = result
                        _append_step_log(
                            steps_log_path,
                            event="matrix_chunk_completed",
                            payload={
                                "chunk_id": int(chunk_id),
                                "chunk_start": int(chunk_start),
                                "chunk_end_exclusive": int(chunk_end),
                                "n_rows": int(n_rows),
                            },
                        )
            except (PermissionError, OSError) as exc:
                use_parallel = False
                _append_step_log(
                    steps_log_path,
                    event="parallel_fallback",
                    payload={"reason": f"pool_creation_failed: {type(exc).__name__}", "detail": str(exc)},
                )

        if not use_parallel:
            for spec in chunk_specs:
                result = _process_matrix_chunk_worker(spec)
                chunk_results.append(result)
                chunk_id, chunk_start, chunk_end, _, n_rows = result
                _append_step_log(
                    steps_log_path,
                    event="matrix_chunk_completed",
                    payload={
                        "chunk_id": int(chunk_id),
                        "chunk_start": int(chunk_start),
                        "chunk_end_exclusive": int(chunk_end),
                        "n_rows": int(n_rows),
                    },
                )
    finally:
        _MATRIX_CHUNK_WORKER_CONTEXT = None

    chunk_results.sort(key=lambda x: x[0])
    index_frames: list[pd.DataFrame] = []
    for _, _, _, chunk_index_path, _ in chunk_results:
        path = Path(chunk_index_path)
        if not path.exists():
            raise FileNotFoundError(f"chunk index file not found: {path}")
        index_frames.append(pd.read_csv(path))

    if index_frames:
        index_df = pd.concat(index_frames, ignore_index=True)
    else:
        index_df = pd.DataFrame(columns=list(_episode_index_columns()))
    index_rows = index_df.to_dict(orient="records")

    _append_step_log(
        steps_log_path,
        event="records_built",
        payload={"n_nucleus_records": int(len(nuclei_records)), "n_bin_records": int(len(bin_ids))},
    )
    _append_step_log(
        steps_log_path,
        event="episodes_built",
        payload={"n_episodes": int(len(index_rows))},
    )
    return index_rows, expression_dim


def _process_matrix_chunk_worker(
    chunk_spec: tuple[int, int, int, list[NucleusRecord]],
) -> tuple[int, int, int, str, int]:
    """Worker entrypoint: build one nuclei chunk and write one chunk index CSV."""
    if _MATRIX_CHUNK_WORKER_CONTEXT is None:
        raise RuntimeError("matrix chunk worker context is not initialized")

    chunk_id, chunk_start, chunk_end, chunk_records = chunk_spec
    ctx = _MATRIX_CHUNK_WORKER_CONTEXT

    tree = ctx.get("_tree")
    if tree is None:
        x_um = np.asarray(ctx["x_um"], dtype=np.float32)
        y_um = np.asarray(ctx["y_um"], dtype=np.float32)
        xy = np.column_stack((x_um, y_um)).astype(np.float64, copy=False)
        tree = cKDTree(xy) if len(xy) else None
        ctx["_tree"] = tree

    bin_ids = np.asarray(ctx["bin_ids"])
    x_um = np.asarray(ctx["x_um"], dtype=np.float32)
    y_um = np.asarray(ctx["y_um"], dtype=np.float32)
    matrix_col_index = np.asarray(ctx["matrix_col_index"], dtype=np.int64)

    max_center_distance_um = float(ctx["max_center_distance_um"])
    radius_band_um = ctx["radius_band_um"]
    states_dir = Path(str(ctx["states_dir"]))
    index_chunks_dir = Path(str(ctx["index_chunks_dir"]))
    save_compressed = bool(ctx["save_compressed"])

    rows: list[dict[str, Any]] = []
    for offset, nucleus in enumerate(chunk_records):
        episode_index = chunk_start + offset

        if tree is None:
            candidate_idx = np.zeros(0, dtype=np.int64)
        else:
            candidate_idx = np.asarray(
                tree.query_ball_point(
                    [float(nucleus.center_x_um), float(nucleus.center_y_um)],
                    r=max_center_distance_um,
                ),
                dtype=np.int64,
            )

        if candidate_idx.size > 0:
            dx = x_um[candidate_idx].astype(np.float64, copy=False) - float(nucleus.center_x_um)
            dy = y_um[candidate_idx].astype(np.float64, copy=False) - float(nucleus.center_y_um)
            dist = np.sqrt(dx * dx + dy * dy)
            keep = dist <= max_center_distance_um
            if radius_band_um is not None:
                keep &= np.abs(float(nucleus.radius_um) - dist) <= float(radius_band_um)
            candidate_idx = candidate_idx[keep]

        if candidate_idx.size == 0:
            candidate_ids = np.asarray([], dtype=object)
            candidate_xy = np.zeros((0, 2), dtype=np.float32)
            candidate_col_index = np.zeros((0,), dtype=np.int64)
        else:
            candidate_idx.sort()
            candidate_ids = bin_ids[candidate_idx].astype(object)
            candidate_xy = np.column_stack((x_um[candidate_idx], y_um[candidate_idx])).astype(np.float32, copy=False)
            candidate_col_index = matrix_col_index[candidate_idx].astype(np.int64, copy=False)

        rows.append(
            _save_one_episode_artifact(
                episode_index=episode_index,
                nucleus=nucleus,
                candidate_bin_ids=candidate_ids,
                candidate_bin_xy_um=candidate_xy,
                candidate_expression=None,
                candidate_matrix_col_index=candidate_col_index,
                states_dir=states_dir,
                steps_log_path=None,
                save_compressed=save_compressed,
            )
        )

    chunk_index_path = index_chunks_dir / f"chunk_{chunk_id:06d}.csv"
    pd.DataFrame(rows, columns=list(_episode_index_columns())).to_csv(chunk_index_path, index=False)
    return (int(chunk_id), int(chunk_start), int(chunk_end), str(chunk_index_path), int(len(rows)))


def _get_matrix_column_selected_expression(
    col_index: int,
    data_ds: h5py.Dataset,
    indices_ds: h5py.Dataset,
    indptr: np.ndarray,
    feature_lookup: np.ndarray,
    expression_dim: int,
    cache: OrderedDict[int, np.ndarray],
    cache_size: int,
) -> np.ndarray:
    """Load one matrix column and project onto selected gene coordinates."""
    cached = cache.get(col_index)
    if cached is not None:
        cache.move_to_end(col_index)
        return cached

    start = int(indptr[col_index])
    end = int(indptr[col_index + 1])
    expr = np.zeros(expression_dim, dtype=np.float32)
    if end > start:
        col_feature_idx = np.asarray(indices_ds[start:end], dtype=np.int64)
        col_values = np.asarray(data_ds[start:end], dtype=np.float32)
        selected_pos = feature_lookup[col_feature_idx]
        keep = selected_pos >= 0
        if keep.any():
            expr[selected_pos[keep]] = col_values[keep]

    if cache_size > 0:
        cache[col_index] = expr
        cache.move_to_end(col_index)
        while len(cache) > cache_size:
            cache.popitem(last=False)
    return expr


def _resolve_matrix_feature_indices(
    feature_names: np.ndarray,
    reference_npz_path: Path | None,
    reference_genes_key: str,
) -> np.ndarray:
    """Resolve matrix row indices (genes) for expression extraction."""
    if reference_npz_path is None:
        indices = np.arange(feature_names.size, dtype=np.int64)
    else:
        if not reference_npz_path.exists():
            raise FileNotFoundError(f"reference NPZ file not found: {reference_npz_path}")
        with np.load(reference_npz_path) as data:
            if reference_genes_key not in data:
                raise ConfigError(
                    f"inputs.expression.reference_genes_key {reference_genes_key!r} "
                    f"is not present in {reference_npz_path}"
                )
            ordered_genes = data[reference_genes_key].astype(str)

        first_index_by_name: dict[str, int] = {}
        for i, name in enumerate(feature_names.astype(str)):
            if name not in first_index_by_name:
                first_index_by_name[name] = i

        missing = [g for g in ordered_genes if g not in first_index_by_name]
        if missing:
            preview = missing[:5]
            raise ValueError(
                f"{len(missing)} reference genes are missing in matrix features; "
                f"first missing: {preview}"
            )

        indices = np.asarray([first_index_by_name[g] for g in ordered_genes], dtype=np.int64)

    return indices


def _episode_index_columns() -> tuple[str, ...]:
    return (
        "episode_index",
        "cell_id",
        "cell_type",
        "nucleus_center_x_um",
        "nucleus_center_y_um",
        "nucleus_radius_um",
        "n_candidate_bins",
        "artifact_path",
    )


def _save_one_episode_artifact(
    episode_index: int,
    nucleus: NucleusRecord,
    candidate_bin_ids: np.ndarray,
    candidate_bin_xy_um: np.ndarray,
    candidate_expression: np.ndarray | None,
    candidate_matrix_col_index: np.ndarray | None,
    states_dir: Path,
    steps_log_path: Path | None,
    save_compressed: bool = True,
) -> dict[str, Any]:
    """Save one episode snapshot and return one row for episodes_index.csv."""
    cell_id = nucleus.cell_id
    safe_cell_id = _slugify(cell_id)
    file_name = f"state_{episode_index:06d}_{safe_cell_id}.npz"
    out_path = states_dir / file_name

    payload = {
        "cell_id": np.asarray([nucleus.cell_id], dtype=object),
        "cell_type": np.asarray([nucleus.cell_type], dtype=object),
        "nucleus_center_xy_um": np.asarray([nucleus.center_x_um, nucleus.center_y_um], dtype=np.float32),
        "nucleus_radius_um": np.asarray([nucleus.radius_um], dtype=np.float32),
        "candidate_bin_ids": np.asarray(candidate_bin_ids, dtype=object),
        "candidate_bin_xy_um": np.asarray(candidate_bin_xy_um, dtype=np.float32),
    }
    if candidate_expression is not None:
        payload["candidate_expression"] = np.asarray(candidate_expression, dtype=np.float32)
    if candidate_matrix_col_index is not None:
        payload["candidate_matrix_col_index"] = np.asarray(candidate_matrix_col_index, dtype=np.int64)
    if save_compressed:
        np.savez_compressed(out_path, **payload)
    else:
        np.savez(out_path, **payload)

    n_candidate_bins = int(len(candidate_bin_ids))
    row = {
        "episode_index": int(episode_index),
        "cell_id": nucleus.cell_id,
        "cell_type": nucleus.cell_type,
        "nucleus_center_x_um": nucleus.center_x_um,
        "nucleus_center_y_um": nucleus.center_y_um,
        "nucleus_radius_um": nucleus.radius_um,
        "n_candidate_bins": n_candidate_bins,
        "artifact_path": str(out_path),
    }
    if steps_log_path is not None:
        _append_step_log(
            steps_log_path,
            event="state_snapshot_saved",
            payload={
                "episode_index": int(episode_index),
                "cell_id": cell_id,
                "n_candidate_bins": n_candidate_bins,
                "state_path": str(out_path),
            },
        )
    return row


def _save_episode_artifacts(
    episodes: list[Any],
    states_dir: Path,
    steps_log_path: Path,
) -> list[dict[str, Any]]:
    index_rows: list[dict[str, Any]] = []

    for idx, episode in enumerate(episodes):
        cell_id = episode.nucleus.cell_id
        safe_cell_id = _slugify(cell_id)
        file_name = f"state_{idx:06d}_{safe_cell_id}.npz"
        out_path = states_dir / file_name

        candidate_bin_ids = [b.bin_id for b in episode.candidate_bins]
        if episode.n_candidate_bins > 0:
            candidate_bin_xy = np.asarray([(b.x_um, b.y_um) for b in episode.candidate_bins], dtype=np.float32)
            candidate_expression = np.vstack([b.expression for b in episode.candidate_bins]).astype(np.float32)
        else:
            candidate_bin_xy = np.zeros((0, 2), dtype=np.float32)
            candidate_expression = np.zeros((0, 0), dtype=np.float32)

        np.savez_compressed(
            out_path,
            cell_id=np.asarray([episode.nucleus.cell_id], dtype=object),
            cell_type=np.asarray([episode.nucleus.cell_type], dtype=object),
            nucleus_center_xy_um=np.asarray(
                [episode.nucleus.center_x_um, episode.nucleus.center_y_um], dtype=np.float32
            ),
            nucleus_radius_um=np.asarray([episode.nucleus.radius_um], dtype=np.float32),
            candidate_bin_ids=np.asarray(candidate_bin_ids, dtype=object),
            candidate_bin_xy_um=candidate_bin_xy,
            candidate_expression=candidate_expression,
        )

        index_rows.append(
            {
                "episode_index": idx,
                "cell_id": episode.nucleus.cell_id,
                "cell_type": episode.nucleus.cell_type,
                "nucleus_center_x_um": episode.nucleus.center_x_um,
                "nucleus_center_y_um": episode.nucleus.center_y_um,
                "nucleus_radius_um": episode.nucleus.radius_um,
                "n_candidate_bins": episode.n_candidate_bins,
                "artifact_path": str(out_path),
            }
        )
        _append_step_log(
            steps_log_path,
            event="state_snapshot_saved",
            payload={
                "episode_index": int(idx),
                "cell_id": cell_id,
                "n_candidate_bins": int(episode.n_candidate_bins),
                "state_path": str(out_path),
            },
        )

    return index_rows


def _compute_summary(
    n_input_nuclei: int,
    n_input_bins: int,
    expression_dim: int,
    episode_index: list[dict[str, Any]],
) -> dict[str, Any]:
    if episode_index:
        candidate_counts = np.asarray([int(row["n_candidate_bins"]) for row in episode_index], dtype=np.int64)
    else:
        candidate_counts = np.zeros(0, dtype=np.int64)

    n_empty = int((candidate_counts == 0).sum()) if len(candidate_counts) else 0

    summary = {
        "n_input_nuclei": int(n_input_nuclei),
        "n_input_bins": int(n_input_bins),
        "expression_dim": int(expression_dim),
        "n_episodes": int(len(episode_index)),
        "total_candidate_bins": int(candidate_counts.sum()) if len(candidate_counts) else 0,
        "empty_episode_count": n_empty,
        "nonempty_episode_count": int(len(episode_index) - n_empty),
        "nonempty_episode_rate": float((len(episode_index) - n_empty) / len(episode_index)) if episode_index else 0.0,
        "candidate_bins_mean": float(candidate_counts.mean()) if len(candidate_counts) else 0.0,
        "candidate_bins_median": float(np.median(candidate_counts)) if len(candidate_counts) else 0.0,
        "candidate_bins_p90": float(np.percentile(candidate_counts, 90)) if len(candidate_counts) else 0.0,
        "candidate_bins_p99": float(np.percentile(candidate_counts, 99)) if len(candidate_counts) else 0.0,
    }
    return summary


def _merge_config_with_resolved_columns(
    config: EpisodeBuildConfig,
    resolved_expression_columns: tuple[str, ...],
) -> dict[str, Any]:
    data = config.to_serializable_dict()
    data["inputs"]["expression"]["resolved_columns"] = list(resolved_expression_columns)
    return data


def _build_metadata(config: EpisodeBuildConfig, run_dir: Path) -> dict[str, Any]:
    now = dt.datetime.now(dt.timezone.utc).isoformat()

    metadata = {
        "timestamp_utc": now,
        "run_dir": str(run_dir),
        "seed": config.seed,
        "python_version": sys.version,
        "platform": platform.platform(),
        "git_commit": _try_git_commit(),
    }
    return metadata


def _try_git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return completed.stdout.strip() or None
    except Exception:
        return None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _append_step_log(path: Path, event: str, payload: dict[str, Any]) -> None:
    entry = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "event": event,
        "payload": payload,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=False))
        handle.write("\n")


def _slugify(text: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    normalized = normalized.strip("_")
    return normalized or "item"
