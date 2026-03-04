#!/usr/bin/env python
"""Aggregate pixel-level Cellpose segmentation outputs into Visium-style bin-level tables.

This script is the bridge between pixel-resolution segmentation and the bin-level RL
state/action space. It keeps segmentation itself at pixel level, then converts pixels
into fixed-size spatial bins (2 um by default) using the microscope resolution.

Key design choices:
- Input remains the pixel-level mapping produced by `cellpose_segmentation.py`.
- Output is a bin-level table with one or more rows per `(bin_x, bin_y, cell_id)`.
- Mixed compartments are split into separate rows so nuclear and cytoplasm evidence do
  not get collapsed into a single ambiguous label.
- `cell_type` is treated as optional because current segmentation output does not carry
  that column yet.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)

# Current colorectal image resolution supplied by the user.
DEFAULT_MICRONS_PER_PIXEL = 0.2737012522439323
DEFAULT_BIN_SIZE_UM = 2.0
DEFAULT_CHUNK_SIZE = 1_000_000

REQUIRED_PIXEL_COLUMNS = (
    "x",
    "y",
    "cell_id",
    "is_boundary",
    "is_interior",
    "is_nuclear",
    "is_cytoplasm",
)


def configure_logging(verbose: bool = False) -> None:
    """Configure a simple process-wide logger for CLI usage."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")


def load_pixel_to_cell_mapping(
    csv_path: str | Path,
    *,
    aggregate_to_bins: bool = True,
    microns_per_pixel: float = DEFAULT_MICRONS_PER_PIXEL,
    bin_size_um: float = DEFAULT_BIN_SIZE_UM,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    x_offset_px: int = 0,
    y_offset_px: int = 0,
    nuclear_only: bool = False,
) -> pd.DataFrame:
    """Load a pixel-to-cell mapping file and optionally aggregate it to bins.

    Parameters
    ----------
    csv_path:
        Path to the pixel-level CSV or CSV.GZ produced by Cellpose segmentation.
    aggregate_to_bins:
        If True, return a bin-level table. If False, return pixel-level rows after
        schema validation and optional `cell_type` normalization.
    microns_per_pixel:
        Physical microscope resolution used to convert pixels into micrometres.
    bin_size_um:
        Spatial bin size in micrometres. Default is 2 um for Visium HD 2 um bins.
    chunk_size:
        Number of pixel rows to read per chunk when streaming the input file.
    x_offset_px, y_offset_px:
        Optional crop offsets to map crop-relative pixels back to full-image coordinates.
        Leave them at 0 if you want the aggregated bins to stay crop-relative.
    nuclear_only:
        If True, keep only nuclear-compartment rows in the final bin table.

    Returns
    -------
    pd.DataFrame
        If `aggregate_to_bins=False`, returns the validated pixel-level mapping.
        If `aggregate_to_bins=True`, returns a bin-level table with columns such as:
        `bin_x_index`, `bin_y_index`, `bin_center_x_um`, `bin_center_y_um`, `cell_id`,
        optional `cell_type`, compartment flags, `weight`, and pixel-count summaries.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"pixel mapping file not found: {path}")

    if microns_per_pixel <= 0:
        raise ValueError("microns_per_pixel must be > 0")
    if bin_size_um <= 0:
        raise ValueError("bin_size_um must be > 0")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")

    LOGGER.info("Loading pixel-to-cell mapping from %s", path)

    if not aggregate_to_bins:
        df = pd.read_csv(path, compression="infer")
        df = _normalize_pixel_chunk(df)
        LOGGER.info("Loaded %d pixel rows without aggregation", len(df))
        return df

    aggregated_chunks: list[pd.DataFrame] = []
    total_rows = 0
    for chunk in _read_pixel_chunks(path=path, chunk_size=chunk_size):
        total_rows += len(chunk)
        chunk_grouped = _aggregate_chunk_to_bin_counts(
            chunk=chunk,
            microns_per_pixel=microns_per_pixel,
            bin_size_um=bin_size_um,
            x_offset_px=x_offset_px,
            y_offset_px=y_offset_px,
        )
        aggregated_chunks.append(chunk_grouped)
        LOGGER.info(
            "Processed chunk with %d pixels into %d bin/cell groups",
            len(chunk),
            len(chunk_grouped),
        )

    if not aggregated_chunks:
        return _empty_bin_table(include_cell_type=True)

    combined = pd.concat(aggregated_chunks, ignore_index=True)
    merged = _merge_chunk_aggregates(combined)
    bin_df = _split_compartments(merged, bin_size_um=bin_size_um)

    if nuclear_only:
        bin_df = bin_df[bin_df["is_nuclear"] == 1].reset_index(drop=True)

    LOGGER.info("Aggregated %d pixels into %d bin-level rows", total_rows, len(bin_df))
    return bin_df


def write_bin_mapping(df: pd.DataFrame, output_path: str | Path) -> Path:
    """Write aggregated bin-level table to disk using suffix-driven format selection."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    suffixes = path.suffixes
    if suffixes[-2:] == [".csv", ".gz"]:
        df.to_csv(path, index=False, compression="gzip")
    elif suffixes[-1:] == [".csv"]:
        df.to_csv(path, index=False)
    elif suffixes[-1:] == [".parquet"]:
        df.to_parquet(path, index=False)
    else:
        raise ValueError("output_path must end with .csv, .csv.gz, or .parquet")

    LOGGER.info("Wrote bin-level mapping to %s", path)
    return path


def write_summary_json(df: pd.DataFrame, output_path: str | Path) -> Path:
    """Write a compact summary JSON for quick sanity checks."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "n_rows": int(len(df)),
        "n_unique_bins": int(df[["bin_x_index", "bin_y_index"]].drop_duplicates().shape[0]) if len(df) else 0,
        "n_unique_cells": int(df["cell_id"].nunique()) if len(df) else 0,
        "n_nuclear_rows": int((df["is_nuclear"] == 1).sum()) if len(df) else 0,
        "n_cytoplasm_rows": int((df["is_cytoplasm"] == 1).sum()) if len(df) else 0,
        "n_boundary_rows": int((df["is_boundary"] == 1).sum()) if len(df) else 0,
    }

    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")

    LOGGER.info("Wrote summary JSON to %s", path)
    return path


def _read_pixel_chunks(path: Path, chunk_size: int) -> Iterator[pd.DataFrame]:
    """Stream the pixel mapping file in chunks to keep memory bounded."""
    for chunk in pd.read_csv(path, compression="infer", chunksize=chunk_size):
        yield _normalize_pixel_chunk(chunk)


def _normalize_pixel_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """Validate required columns and normalize dtypes on one pixel chunk."""
    missing = [col for col in REQUIRED_PIXEL_COLUMNS if col not in chunk.columns]
    if missing:
        raise ValueError(f"pixel mapping is missing required columns: {missing}")

    df = chunk.copy()

    # `cell_type` is optional at this stage. Keep the column so downstream code can rely on it.
    if "cell_type" not in df.columns:
        df["cell_type"] = pd.Series([None] * len(df), dtype="object")

    numeric_columns = [
        "x",
        "y",
        "cell_id",
        "is_boundary",
        "is_interior",
        "is_nuclear",
        "is_cytoplasm",
    ]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="raise")

    if not np.isfinite(df[numeric_columns].to_numpy(dtype=np.float64)).all():
        raise ValueError("pixel mapping contains non-finite numeric values")

    # Use compact integer dtypes because these columns are large and repeatedly grouped.
    df["x"] = df["x"].astype(np.int32)
    df["y"] = df["y"].astype(np.int32)
    df["cell_id"] = df["cell_id"].astype(np.int64)
    for col in ("is_boundary", "is_interior", "is_nuclear", "is_cytoplasm"):
        df[col] = df[col].astype(np.int8)

    return df


def _aggregate_chunk_to_bin_counts(
    chunk: pd.DataFrame,
    *,
    microns_per_pixel: float,
    bin_size_um: float,
    x_offset_px: int,
    y_offset_px: int,
) -> pd.DataFrame:
    """Convert one pixel chunk to aggregated counts keyed by `(bin, cell)`."""
    df = chunk.copy()

    # Convert crop-relative pixels to the requested coordinate frame first.
    x_px = df["x"].to_numpy(dtype=np.int64, copy=False) + int(x_offset_px)
    y_px = df["y"].to_numpy(dtype=np.int64, copy=False) + int(y_offset_px)

    # Use floor division in micrometre space to map each pixel center to a fixed-size bin.
    x_um = x_px.astype(np.float64) * float(microns_per_pixel)
    y_um = y_px.astype(np.float64) * float(microns_per_pixel)
    df["bin_x_index"] = np.floor(x_um / float(bin_size_um)).astype(np.int32)
    df["bin_y_index"] = np.floor(y_um / float(bin_size_um)).astype(np.int32)

    group_columns = ["bin_x_index", "bin_y_index", "cell_id", "cell_type"]
    grouped = (
        df.groupby(group_columns, dropna=False, sort=False)
        .agg(
            nuclear_pixel_count=("is_nuclear", "sum"),
            cytoplasm_pixel_count=("is_cytoplasm", "sum"),
            boundary_pixel_count=("is_boundary", "sum"),
            interior_pixel_count=("is_interior", "sum"),
        )
        .reset_index()
    )
    return grouped


def _merge_chunk_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """Merge per-chunk aggregates into one final count table."""
    merged = (
        df.groupby(["bin_x_index", "bin_y_index", "cell_id", "cell_type"], dropna=False, sort=False)
        .agg(
            nuclear_pixel_count=("nuclear_pixel_count", "sum"),
            cytoplasm_pixel_count=("cytoplasm_pixel_count", "sum"),
            boundary_pixel_count=("boundary_pixel_count", "sum"),
            interior_pixel_count=("interior_pixel_count", "sum"),
        )
        .reset_index()
    )

    merged["total_compartment_pixels"] = (
        merged["nuclear_pixel_count"]
        + merged["cytoplasm_pixel_count"]
        + merged["boundary_pixel_count"]
    )
    merged = merged[merged["total_compartment_pixels"] > 0].reset_index(drop=True)
    return merged


def _split_compartments(counts: pd.DataFrame, *, bin_size_um: float) -> pd.DataFrame:
    """Split mixed bins into separate compartment rows with weights."""
    if len(counts) == 0:
        return _empty_bin_table(include_cell_type=True)

    shared = [
        "bin_x_index",
        "bin_y_index",
        "cell_id",
        "cell_type",
        "nuclear_pixel_count",
        "cytoplasm_pixel_count",
        "boundary_pixel_count",
        "interior_pixel_count",
        "total_compartment_pixels",
    ]

    nuclear_bins = counts[counts["nuclear_pixel_count"] > 0][shared].copy()
    nuclear_bins["weight"] = nuclear_bins["nuclear_pixel_count"] / nuclear_bins["total_compartment_pixels"]
    nuclear_bins["is_nuclear"] = np.int8(1)
    nuclear_bins["is_cytoplasm"] = np.int8(0)
    nuclear_bins["is_boundary"] = np.int8(0)
    nuclear_bins["is_interior"] = np.int8(1)

    cytoplasm_bins = counts[counts["cytoplasm_pixel_count"] > 0][shared].copy()
    cytoplasm_bins["weight"] = cytoplasm_bins["cytoplasm_pixel_count"] / cytoplasm_bins["total_compartment_pixels"]
    cytoplasm_bins["is_nuclear"] = np.int8(0)
    cytoplasm_bins["is_cytoplasm"] = np.int8(1)
    cytoplasm_bins["is_boundary"] = np.int8(0)
    cytoplasm_bins["is_interior"] = np.int8(1)

    boundary_bins = counts[
        (counts["boundary_pixel_count"] > 0)
        & (counts["nuclear_pixel_count"] == 0)
        & (counts["cytoplasm_pixel_count"] == 0)
    ][shared].copy()
    boundary_bins["weight"] = 1.0
    boundary_bins["is_nuclear"] = np.int8(0)
    boundary_bins["is_cytoplasm"] = np.int8(0)
    boundary_bins["is_boundary"] = np.int8(1)
    boundary_bins["is_interior"] = np.int8(0)

    out = pd.concat([nuclear_bins, cytoplasm_bins, boundary_bins], ignore_index=True)

    out["bin_center_x_um"] = (out["bin_x_index"].astype(np.float64) + 0.5) * float(bin_size_um)
    out["bin_center_y_um"] = (out["bin_y_index"].astype(np.float64) + 0.5) * float(bin_size_um)
    out["bin_id"] = out["bin_x_index"].astype(str) + "_" + out["bin_y_index"].astype(str)

    ordered_columns = [
        "bin_id",
        "bin_x_index",
        "bin_y_index",
        "bin_center_x_um",
        "bin_center_y_um",
        "cell_id",
        "cell_type",
        "is_nuclear",
        "is_cytoplasm",
        "is_boundary",
        "is_interior",
        "weight",
        "nuclear_pixel_count",
        "cytoplasm_pixel_count",
        "boundary_pixel_count",
        "interior_pixel_count",
        "total_compartment_pixels",
    ]
    return out.loc[:, ordered_columns].reset_index(drop=True)


def _empty_bin_table(include_cell_type: bool) -> pd.DataFrame:
    """Return an empty output table with the final schema."""
    columns = [
        "bin_id",
        "bin_x_index",
        "bin_y_index",
        "bin_center_x_um",
        "bin_center_y_um",
        "cell_id",
        "is_nuclear",
        "is_cytoplasm",
        "is_boundary",
        "is_interior",
        "weight",
        "nuclear_pixel_count",
        "cytoplasm_pixel_count",
        "boundary_pixel_count",
        "interior_pixel_count",
        "total_compartment_pixels",
    ]
    if include_cell_type:
        columns.insert(6, "cell_type")
    return pd.DataFrame(columns=columns)


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for the converter."""
    parser = argparse.ArgumentParser(description="Aggregate pixel-level Cellpose output into 2 um bins")
    parser.add_argument("--csv-path", required=True, help="Path to pixel-level mapping CSV/CSV.GZ")
    parser.add_argument("--output-path", required=True, help="Output path (.csv, .csv.gz, or .parquet)")
    parser.add_argument(
        "--summary-path",
        default=None,
        help="Optional JSON summary output path",
    )
    parser.add_argument(
        "--microns-per-pixel",
        type=float,
        default=DEFAULT_MICRONS_PER_PIXEL,
        help=f"Microscope resolution in um/pixel (default: {DEFAULT_MICRONS_PER_PIXEL})",
    )
    parser.add_argument(
        "--bin-size-um",
        type=float,
        default=DEFAULT_BIN_SIZE_UM,
        help=f"Bin size in micrometres (default: {DEFAULT_BIN_SIZE_UM})",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"CSV chunk size for streaming input (default: {DEFAULT_CHUNK_SIZE})",
    )
    parser.add_argument(
        "--x-offset-px",
        type=int,
        default=0,
        help="Optional x-offset in pixels to map crop-relative coordinates back to full image",
    )
    parser.add_argument(
        "--y-offset-px",
        type=int,
        default=0,
        help="Optional y-offset in pixels to map crop-relative coordinates back to full image",
    )
    parser.add_argument(
        "--nuclear-only",
        action="store_true",
        help="Keep only nuclear rows in the final bin table",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser


def main() -> None:
    """CLI entrypoint."""
    args = _build_arg_parser().parse_args()
    configure_logging(verbose=args.verbose)

    df = load_pixel_to_cell_mapping(
        csv_path=args.csv_path,
        aggregate_to_bins=True,
        microns_per_pixel=args.microns_per_pixel,
        bin_size_um=args.bin_size_um,
        chunk_size=args.chunk_size,
        x_offset_px=args.x_offset_px,
        y_offset_px=args.y_offset_px,
        nuclear_only=args.nuclear_only,
    )
    write_bin_mapping(df, args.output_path)

    if args.summary_path is not None:
        write_summary_json(df, args.summary_path)


if __name__ == "__main__":
    main()
