"""Reproducible episode-build pipeline for HD cell RL experiments."""

from __future__ import annotations

from dataclasses import dataclass
import datetime as dt
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .builder import build_episodes
from .config import EnvironmentConfig
from .models import BinRecord, NucleusRecord


class ConfigError(ValueError):
    """Raised when episode-build config is invalid."""


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

    if expression_columns_raw is None:
        expression_columns_raw = []
    if not isinstance(expression_columns_raw, list):
        raise ConfigError("inputs.expression.columns must be a list")

    expression_columns = tuple(str(col) for col in expression_columns_raw)

    if expression_mode not in {"prefix", "list"}:
        raise ConfigError("inputs.expression.mode must be 'prefix' or 'list'")

    if expression_mode == "prefix":
        if expression_prefix is None or not str(expression_prefix):
            raise ConfigError("inputs.expression.prefix must be set when mode='prefix'")
        expression_prefix = str(expression_prefix)

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
    config_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    states_dir.mkdir(parents=True, exist_ok=True)

    steps_log_path = logs_dir / "steps.jsonl"
    _append_step_log(
        steps_log_path,
        event="run_start",
        payload={"run_name": config.run_name, "seed": config.seed},
    )

    rng = np.random.default_rng(config.seed)

    nuclei_df = _load_table(config.nuclei_path, config.nuclei_format)
    bins_df = _load_table(config.bins_path, config.bins_format)
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

    _validate_nuclei_schema(nuclei_df, config.nuclei_columns)
    _validate_bins_schema(bins_df, config.bin_columns, expression_columns)
    _append_step_log(steps_log_path, event="schema_validated", payload={})

    nuclei_records = _build_nucleus_records(nuclei_df, config.nuclei_columns)
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

    episode_index = _save_episode_artifacts(
        episodes=episodes,
        states_dir=states_dir,
        steps_log_path=steps_log_path,
    )
    summary = _compute_summary(
        n_input_nuclei=len(nuclei_df),
        n_input_bins=len(bins_df),
        expression_dim=len(expression_columns),
        episode_index=episode_index,
    )

    _write_yaml(config_dir / "config_resolved.yaml", _merge_config_with_resolved_columns(config, expression_columns))
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


def _load_table(path: Path, table_format: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"table file not found: {path}")

    if table_format == "csv":
        return pd.read_csv(path)
    if table_format == "tsv":
        return pd.read_csv(path, sep="\t")
    if table_format == "parquet":
        return pd.read_parquet(path)

    raise ConfigError(f"unsupported format in loader: {table_format!r}")


def _sample_rows(df: pd.DataFrame, limit: int, rng: np.random.Generator) -> pd.DataFrame:
    if limit <= 0:
        raise ValueError("limit values must be > 0")

    if len(df) <= limit:
        return df.reset_index(drop=True)

    keep = rng.choice(len(df), size=limit, replace=False)
    keep_sorted = np.sort(keep)
    return df.iloc[keep_sorted].reset_index(drop=True)


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


def _validate_bins_schema(df: pd.DataFrame, columns: dict[str, str], expression_columns: tuple[str, ...]) -> None:
    required = [columns["bin_id"], columns["x_um"], columns["y_um"], *expression_columns]
    _require_columns(df, required, table_name="bins")

    bin_id_col = columns["bin_id"]
    if df[bin_id_col].isna().any():
        raise ValueError("bins bin_id column contains missing values")

    if df[bin_id_col].duplicated().any():
        dup_value = str(df.loc[df[bin_id_col].duplicated(), bin_id_col].iloc[0])
        raise ValueError(f"duplicate bin_id found in bins table: {dup_value!r}")

    for key in ("x_um", "y_um"):
        _validate_numeric_series(df[columns[key]], f"bins.{columns[key]}")

    for expr_col in expression_columns:
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
