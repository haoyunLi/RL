"""Helpers for loading 10x matrix sources from either H5 files or matrix directories."""

from __future__ import annotations

import gc
import logging
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.io import mmread
from scipy.sparse import csc_matrix


LOGGER = logging.getLogger(__name__)
_CACHE_DIRNAME = ".rl_cache"
_CACHE_H5_NAME = "matrix_csc_cache.h5"


def resolve_matrix_csc_h5_path(matrix_path: str | Path) -> Path:
    """Return an H5 path for a 10x matrix source.

    Accepted inputs:
    - existing `.h5` file
    - 10x matrix directory containing `matrix.mtx[.gz]`, `barcodes.tsv.gz`, `features.tsv.gz`

    If a directory is passed and a sibling `<dir>.h5` exists, that sibling is used.
    Otherwise a cached CSC H5 is created once under `<dir>/.rl_cache/matrix_csc_cache.h5`.
    """
    source = Path(matrix_path).expanduser().resolve()
    if source.is_file():
        if source.suffix.lower() != ".h5":
            raise ValueError(f"matrix source file must end with .h5: {source}")
        return source
    if not source.exists():
        raise FileNotFoundError(f"matrix source not found: {source}")
    if not source.is_dir():
        raise ValueError(f"matrix source must be a .h5 file or 10x matrix directory: {source}")

    sibling_h5 = source.with_suffix(".h5")
    if sibling_h5.exists():
        return sibling_h5.resolve()

    matrix_mtx_path = _resolve_matrix_mtx_path(source)
    barcodes_path = source / "barcodes.tsv.gz"
    features_path = source / "features.tsv.gz"
    missing = [str(p) for p in (matrix_mtx_path, barcodes_path, features_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"10x matrix directory is missing required files: {missing}")

    cache_dir = source / _CACHE_DIRNAME
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_h5 = cache_dir / _CACHE_H5_NAME
    if _is_cache_fresh(cache_h5, [matrix_mtx_path, barcodes_path, features_path]):
        return cache_h5.resolve()

    _build_cached_h5_from_10x_dir(
        matrix_dir=source,
        matrix_mtx_path=matrix_mtx_path,
        barcodes_path=barcodes_path,
        features_path=features_path,
        cache_h5_path=cache_h5,
    )
    return cache_h5.resolve()


def load_matrix_barcodes_in_order(
    matrix_path: str | Path,
    *,
    barcodes_tsv_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load matrix barcodes in column order from a 10x H5 or matrix directory."""
    source = Path(matrix_path).expanduser().resolve()
    explicit_barcodes = None if barcodes_tsv_path is None else Path(barcodes_tsv_path).expanduser().resolve()
    if explicit_barcodes is not None and explicit_barcodes.exists():
        barcodes_df = _read_barcodes_tsv(explicit_barcodes)
    elif source.is_dir():
        inferred = source / "barcodes.tsv.gz"
        if not inferred.exists():
            raise FileNotFoundError(f"barcodes.tsv.gz not found under matrix directory: {source}")
        barcodes_df = _read_barcodes_tsv(inferred)
    else:
        resolved_h5 = resolve_matrix_csc_h5_path(source)
        with h5py.File(resolved_h5, "r") as h5:
            raw = h5["matrix/barcodes"][:]
        barcodes_df = pd.DataFrame({"barcode": [x.decode("utf-8") for x in raw]})

    barcodes_df["matrix_col_index"] = np.arange(len(barcodes_df), dtype=np.int64)
    return barcodes_df


def load_selected_feature_metadata(
    matrix_path: str | Path,
    *,
    feature_type_filter: str | None,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, int]]:
    """Load selected feature metadata from a 10x H5 or matrix directory."""
    source = Path(matrix_path).expanduser().resolve()
    if source.is_dir():
        features_df = _read_features_tsv(source / "features.tsv.gz")
        feature_ids = features_df["feature_id"].to_numpy(dtype=object)
        feature_names = features_df["feature_name"].to_numpy(dtype=object)
        feature_types = features_df["feature_type"].to_numpy(dtype=object)
        genomes = features_df["genome"].to_numpy(dtype=object)
    else:
        resolved_h5 = resolve_matrix_csc_h5_path(source)
        with h5py.File(resolved_h5, "r") as h5:
            feature_ids = np.asarray([x.decode("utf-8") for x in h5["matrix/features/id"][:]], dtype=object)
            feature_names = np.asarray([x.decode("utf-8") for x in h5["matrix/features/name"][:]], dtype=object)
            feature_types = np.asarray(
                [x.decode("utf-8") for x in h5["matrix/features/feature_type"][:]],
                dtype=object,
            )
            genomes = np.asarray([x.decode("utf-8") for x in h5["matrix/features/genome"][:]], dtype=object)

    if feature_type_filter is None:
        selected = np.arange(len(feature_ids), dtype=np.int64)
    else:
        selected = np.flatnonzero(feature_types == str(feature_type_filter)).astype(np.int64)
        if len(selected) == 0:
            raise ValueError(f"feature_type_filter {feature_type_filter!r} matched zero features")

    selected_features_df = pd.DataFrame(
        {
            "matrix_feature_index": selected,
            "feature_id": feature_ids[selected],
            "feature_name": feature_names[selected],
            "feature_type": feature_types[selected],
            "genome": genomes[selected],
        }
    )
    manifest = {"n_total_features": int(len(feature_ids))}
    return selected_features_df, selected, manifest


def _resolve_matrix_mtx_path(matrix_dir: Path) -> Path:
    gzip_path = matrix_dir / "matrix.mtx.gz"
    if gzip_path.exists():
        return gzip_path
    plain_path = matrix_dir / "matrix.mtx"
    if plain_path.exists():
        return plain_path
    return gzip_path


def _read_barcodes_tsv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["barcode"],
        dtype={"barcode": "string"},
    )
    df["barcode"] = df["barcode"].astype(str)
    return df


def _read_features_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"features.tsv.gz not found: {path}")
    df = pd.read_csv(path, sep="\t", header=None, dtype="string")
    if df.shape[1] < 2:
        raise ValueError(f"features file must have at least 2 columns: {path}")
    while df.shape[1] < 4:
        df[df.shape[1]] = ""
    df = df.iloc[:, :4].copy()
    df.columns = ["feature_id", "feature_name", "feature_type", "genome"]
    return df


def _is_cache_fresh(cache_h5_path: Path, source_paths: list[Path]) -> bool:
    if not cache_h5_path.exists():
        return False
    cache_mtime = cache_h5_path.stat().st_mtime
    return all(cache_mtime >= p.stat().st_mtime for p in source_paths if p.exists())


def _build_cached_h5_from_10x_dir(
    *,
    matrix_dir: Path,
    matrix_mtx_path: Path,
    barcodes_path: Path,
    features_path: Path,
    cache_h5_path: Path,
) -> None:
    LOGGER.info("Building cached CSC H5 from 10x matrix directory: %s", matrix_dir)
    feature_df = _read_features_tsv(features_path)
    barcodes = _read_barcodes_tsv(barcodes_path)["barcode"].tolist()

    matrix = mmread(str(matrix_mtx_path))
    if not isinstance(matrix, csc_matrix):
        matrix = matrix.tocsc()
    matrix.sort_indices()

    data = np.asarray(matrix.data)
    if np.issubdtype(data.dtype, np.integer):
        if data.size and data.min() >= np.iinfo(np.int32).min and data.max() <= np.iinfo(np.int32).max:
            data = data.astype(np.int32, copy=False)
    else:
        data = data.astype(np.float32, copy=False)

    indices = np.asarray(matrix.indices, dtype=np.int32)
    indptr = np.asarray(matrix.indptr, dtype=np.int64)
    shape = np.asarray(matrix.shape, dtype=np.int64)

    tmp_path = cache_h5_path.with_name(f"{cache_h5_path.name}.tmp.{os.getpid()}")
    if tmp_path.exists():
        tmp_path.unlink()

    try:
        with h5py.File(tmp_path, "w") as h5:
            mg = h5.create_group("matrix")
            mg.create_dataset("data", data=data)
            mg.create_dataset("indices", data=indices)
            mg.create_dataset("indptr", data=indptr)
            mg.create_dataset("shape", data=shape)
            mg.create_dataset("barcodes", data=np.asarray(barcodes, dtype="S"))

            fg = mg.create_group("features")
            fg.create_dataset("id", data=np.asarray(feature_df["feature_id"].astype(str).tolist(), dtype="S"))
            fg.create_dataset("name", data=np.asarray(feature_df["feature_name"].astype(str).tolist(), dtype="S"))
            fg.create_dataset(
                "feature_type",
                data=np.asarray(feature_df["feature_type"].astype(str).tolist(), dtype="S"),
            )
            fg.create_dataset("genome", data=np.asarray(feature_df["genome"].astype(str).tolist(), dtype="S"))

            h5.attrs["source_matrix_dir"] = str(matrix_dir)
            h5.attrs["source_format"] = "10x_mtx_dir"

        os.replace(tmp_path, cache_h5_path)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise
    finally:
        del matrix
        gc.collect()

    LOGGER.info("Cached CSC H5 written to %s", cache_h5_path)
