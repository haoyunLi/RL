#!/usr/bin/env python
"""Map pixel-level segmentation output onto the official Space Ranger square_002um bins.

This script is the exact bin-alignment version of pixel aggregation. It does not create a
new synthetic 2 um grid. Instead, it reconstructs the official crop-region bin map from
`spatial/tissue_positions.parquet` and assigns each segmentation pixel to the matching
Space Ranger barcode.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Iterator

import matplotlib
matplotlib.use("Agg")
from matplotlib import colormaps
import numpy as np
import pandas as pd
from numba import jit
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


LOGGER = logging.getLogger(__name__)
DEFAULT_MICRONS_PER_PIXEL = 0.2737012522439323
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


@jit(nopython=True)
def fill_pixels(pixels, row_starts, row_ends, col_starts, col_ends, indices):
    """Fill a crop-shaped pixel array with official bin indices."""
    for i in range(len(indices)):
        pixels[row_starts[i] : row_ends[i], col_starts[i] : col_ends[i]] = indices[i]
    return pixels


@jit(nopython=True)
def fill_value_pixels(pixels, row_starts, row_ends, col_starts, col_ends, values):
    """Fill a downsampled crop image with scalar values for overlay rendering."""
    for i in range(len(values)):
        pixels[row_starts[i] : row_ends[i], col_starts[i] : col_ends[i]] = values[i]
    return pixels


def configure_logging(verbose: bool = False) -> None:
    """Configure process-wide logging for CLI usage."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")


def annotate_square_002um_bins_from_pixels(
    pixel_mapping_csv: str | Path,
    tissue_positions_parquet: str | Path,
    *,
    microns_per_pixel: float = DEFAULT_MICRONS_PER_PIXEL,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    nuclear_only: bool = False,
    in_tissue_only: bool = False,
    row_number: int | None = None,
    col_number: int | None = None,
) -> pd.DataFrame:
    """Aggregate pixel-level segmentation onto official square_002um barcodes.

    Parameters
    ----------
    pixel_mapping_csv:
        Pixel-level segmentation CSV/CSV.GZ from `cellpose_segmentation.py`.
        Coordinates are expected to be crop-relative.
    tissue_positions_parquet:
        Official Space Ranger `spatial/tissue_positions.parquet` file for `square_002um`.
    microns_per_pixel:
        Microscope resolution used to convert official pixel coordinates into micrometres.
    chunk_size:
        Number of pixel rows to stream per chunk.
    nuclear_only:
        If True, keep only nuclear rows in the final output.
    in_tissue_only:
        If True, keep only official bins with `in_tissue == 1` in the final output.
    row_number, col_number:
        Optional grid dimensions. If omitted, inferred from max `array_row/array_col`.
    """
    pixel_path = Path(pixel_mapping_csv)
    tissue_path = Path(tissue_positions_parquet)
    if not pixel_path.exists():
        raise FileNotFoundError(f"pixel mapping file not found: {pixel_path}")
    if not tissue_path.exists():
        raise FileNotFoundError(f"tissue positions file not found: {tissue_path}")
    if microns_per_pixel <= 0:
        raise ValueError("microns_per_pixel must be > 0")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")

    LOGGER.info("Loading official tissue positions from %s", tissue_path)
    tissue_df = pd.read_parquet(tissue_path)
    crop_bins, crop_meta = _build_crop_region_bins(
        tissue_df=tissue_df,
        row_number=row_number,
        col_number=col_number,
    )
    LOGGER.info(
        "Crop-region official bins: %d, crop shape: (%d, %d)",
        len(crop_bins),
        crop_meta["crop_height_px"], crop_meta["crop_width_px"],
    )

    pixel_to_bin = _build_crop_pixel_to_bin_map(crop_bins, crop_meta)

    aggregated_chunks: list[pd.DataFrame] = []
    total_rows = 0
    total_assigned_rows = 0

    for chunk in _read_pixel_chunks(pixel_path, chunk_size=chunk_size):
        total_rows += len(chunk)
        grouped, n_assigned = _aggregate_pixel_chunk_to_official_bins(
            chunk=chunk,
            pixel_to_bin=pixel_to_bin,
            crop_bins=crop_bins,
        )
        total_assigned_rows += n_assigned
        aggregated_chunks.append(grouped)
        LOGGER.info(
            "Processed chunk with %d pixels; %d landed inside official bins; %d grouped rows",
            len(chunk), n_assigned, len(grouped),
        )

    if not aggregated_chunks:
        return _empty_official_output(include_cell_type=True)

    combined = pd.concat(aggregated_chunks, ignore_index=True)
    merged = _merge_aggregates(combined)
    annotated = _attach_official_bin_metadata(merged, crop_bins, microns_per_pixel=microns_per_pixel)
    output = _split_compartments(annotated)

    if nuclear_only:
        output = output[output["is_nuclear"] == 1].reset_index(drop=True)
    if in_tissue_only:
        output = output[output["in_tissue"] == 1].reset_index(drop=True)

    LOGGER.info(
        "Aggregated %d pixels into %d official bin/cell rows; %d source pixels landed in crop bins",
        total_rows, len(output), total_assigned_rows,
    )
    return output


def write_official_bin_table(df: pd.DataFrame, output_path: str | Path) -> Path:
    """Write the official-bin annotation table to disk."""
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
    LOGGER.info("Wrote official bin annotation table to %s", path)
    return path


def write_summary_json(df: pd.DataFrame, crop_meta: dict[str, int | float], output_path: str | Path) -> Path:
    """Write summary JSON with crop metadata and output shape."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **crop_meta,
        "n_rows": int(len(df)),
        "n_unique_barcodes": int(df["barcode"].nunique()) if len(df) else 0,
        "n_unique_cells": int(df["cell_id"].nunique()) if len(df) else 0,
        "n_nuclear_rows": int((df["is_nuclear"] == 1).sum()) if len(df) else 0,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    LOGGER.info("Wrote summary JSON to %s", path)
    return path


def write_overlay_png(
    df: pd.DataFrame,
    crop_bins: pd.DataFrame,
    crop_meta: dict[str, int | float],
    output_path: str | Path,
    *,
    cropped_image_path: str | Path | None = None,
    downsample_factor: int = 4,
) -> Path:
    """Render a downsampled overlay of nuclear bin signal on the cropped image.

    The overlay is based on official square_002um bins, collapsed to one scalar per
    barcode using summed `nuclear_pixel_count`. Values are log-scaled and clipped at
    the 99th percentile for visibility.
    """
    if downsample_factor <= 0:
        raise ValueError("downsample_factor must be > 0")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    crop_height = int(crop_meta["crop_height_px"])
    crop_width = int(crop_meta["crop_width_px"])
    ds = int(downsample_factor)
    ds_height = int(math.ceil(crop_height / ds))
    ds_width = int(math.ceil(crop_width / ds))

    # Collapse multi-cell rows to one scalar per official barcode for visualization.
    per_barcode = (
        df.groupby("barcode", sort=False)
        .agg(nuclear_signal=("nuclear_pixel_count", "sum"))
        .reset_index()
    )
    overlay_bins = crop_bins.merge(per_barcode, on="barcode", how="left")
    overlay_bins["nuclear_signal"] = overlay_bins["nuclear_signal"].fillna(0.0).astype(np.float32)

    positive = overlay_bins["nuclear_signal"].to_numpy(dtype=np.float32)
    if np.any(positive > 0):
        log_signal = np.log1p(positive)
        vmax = float(np.quantile(log_signal[log_signal > 0], 0.99))
        if vmax <= 0:
            vmax = float(np.max(log_signal))
        scaled = np.clip(log_signal / max(vmax, 1e-8), 0.0, 1.0)
        values_uint8 = np.rint(scaled * 255.0).astype(np.uint8)
    else:
        values_uint8 = np.zeros(len(overlay_bins), dtype=np.uint8)

    row_starts = (overlay_bins["pxl_row_left_in_crop"].to_numpy(dtype=np.int32, copy=False) // ds).astype(np.int32)
    row_ends = ((overlay_bins["pxl_row_right_in_crop"].to_numpy(dtype=np.int32, copy=False) + ds - 1) // ds).astype(np.int32)
    col_starts = (overlay_bins["pxl_col_up_in_crop"].to_numpy(dtype=np.int32, copy=False) // ds).astype(np.int32)
    col_ends = ((overlay_bins["pxl_col_down_in_crop"].to_numpy(dtype=np.int32, copy=False) + ds - 1) // ds).astype(np.int32)

    row_starts = np.clip(row_starts, 0, ds_height)
    row_ends = np.clip(row_ends, 0, ds_height)
    col_starts = np.clip(col_starts, 0, ds_width)
    col_ends = np.clip(col_ends, 0, ds_width)

    heatmap = np.zeros((ds_height, ds_width), dtype=np.uint8)
    heatmap = fill_value_pixels(heatmap, row_starts, row_ends, col_starts, col_ends, values_uint8)

    if cropped_image_path is not None:
        image = Image.open(cropped_image_path)
        image = image.resize((ds_width, ds_height), resample=Image.BILINEAR)
        base = np.asarray(image, dtype=np.float32)
        if base.ndim == 2:
            base = np.stack([base, base, base], axis=-1)
        elif base.shape[2] == 4:
            base = base[:, :, :3]
    else:
        base = np.zeros((ds_height, ds_width, 3), dtype=np.float32)

    cmap_rgb = colormaps["autumn"](heatmap.astype(np.float32) / 255.0)[..., :3]
    overlay_rgb = (cmap_rgb * 255.0).astype(np.float32)
    alpha = (heatmap.astype(np.float32) / 255.0) * 0.75
    alpha = alpha[..., None]
    composite = np.clip(base * (1.0 - alpha) + overlay_rgb * alpha, 0.0, 255.0).astype(np.uint8)

    Image.fromarray(composite).save(path)
    LOGGER.info("Wrote official-bin overlay PNG to %s", path)
    return path


def _build_crop_region_bins(
    tissue_df: pd.DataFrame,
    *,
    row_number: int | None,
    col_number: int | None,
) -> tuple[pd.DataFrame, dict[str, int | float]]:
    """Recreate the crop-region bin table used by the crop script."""
    required = [
        "barcode",
        "in_tissue",
        "array_row",
        "array_col",
        "pxl_row_in_fullres",
        "pxl_col_in_fullres",
    ]
    missing = [col for col in required if col not in tissue_df.columns]
    if missing:
        raise ValueError(f"tissue_positions is missing required columns: {missing}")

    df = tissue_df.loc[:, required].copy()

    if row_number is None:
        row_number = int(df["array_row"].max()) + 1
    if col_number is None:
        col_number = int(df["array_col"].max()) + 1

    avg_row = (df["pxl_row_in_fullres"].max() - df["pxl_row_in_fullres"].min()) / (2.0 * float(row_number))
    avg_col = (df["pxl_col_in_fullres"].max() - df["pxl_col_in_fullres"].min()) / (2.0 * float(col_number))

    df["pxl_row_left_in_fullres"] = np.rint(df["pxl_row_in_fullres"] - avg_row).astype(np.int32)
    df["pxl_row_right_in_fullres"] = np.rint(df["pxl_row_in_fullres"] + avg_row).astype(np.int32)
    df["pxl_col_up_in_fullres"] = np.rint(df["pxl_col_in_fullres"] - avg_col).astype(np.int32)
    df["pxl_col_down_in_fullres"] = np.rint(df["pxl_col_in_fullres"] + avg_col).astype(np.int32)

    in_tissue = df[df["in_tissue"] == 1]
    start_row_spot = int(in_tissue["array_row"].min())
    end_row_spot = int(in_tissue["array_row"].max()) + 1
    start_col_spot = int(in_tissue["array_col"].min())
    end_col_spot = int(in_tissue["array_col"].max()) + 1

    crop_bins = df[
        (df["array_row"] >= start_row_spot)
        & (df["array_row"] < end_row_spot)
        & (df["array_col"] >= start_col_spot)
        & (df["array_col"] < end_col_spot)
    ].copy().reset_index(drop=True)

    row_left = int(max(crop_bins["pxl_row_left_in_fullres"].min(), 0))
    row_right = int(crop_bins["pxl_row_right_in_fullres"].max())
    col_up = int(max(crop_bins["pxl_col_up_in_fullres"].min(), 0))
    col_down = int(crop_bins["pxl_col_down_in_fullres"].max())

    crop_bins["pxl_row_left_in_crop"] = crop_bins["pxl_row_left_in_fullres"] - row_left
    crop_bins["pxl_row_right_in_crop"] = crop_bins["pxl_row_right_in_fullres"] - row_left
    crop_bins["pxl_col_up_in_crop"] = crop_bins["pxl_col_up_in_fullres"] - col_up
    crop_bins["pxl_col_down_in_crop"] = crop_bins["pxl_col_down_in_fullres"] - col_up

    crop_meta = {
        "row_left": row_left,
        "row_right": row_right,
        "col_up": col_up,
        "col_down": col_down,
        "crop_height_px": int(row_right - row_left),
        "crop_width_px": int(col_down - col_up),
        "start_row_spot": start_row_spot,
        "end_row_spot": end_row_spot,
        "start_col_spot": start_col_spot,
        "end_col_spot": end_col_spot,
        "row_number": int(row_number),
        "col_number": int(col_number),
    }
    return crop_bins, crop_meta


def _build_crop_pixel_to_bin_map(crop_bins: pd.DataFrame, crop_meta: dict[str, int | float]) -> np.ndarray:
    """Build crop-relative pixel -> official bin row index lookup array."""
    crop_height = int(crop_meta["crop_height_px"])
    crop_width = int(crop_meta["crop_width_px"])
    pixels = -1 * np.ones((crop_height, crop_width), dtype=np.int32)

    row_starts = crop_bins["pxl_row_left_in_crop"].to_numpy(dtype=np.int32, copy=False)
    row_ends = crop_bins["pxl_row_right_in_crop"].to_numpy(dtype=np.int32, copy=False)
    col_starts = crop_bins["pxl_col_up_in_crop"].to_numpy(dtype=np.int32, copy=False)
    col_ends = crop_bins["pxl_col_down_in_crop"].to_numpy(dtype=np.int32, copy=False)
    indices = np.arange(len(crop_bins), dtype=np.int32)

    LOGGER.info("Building crop pixel -> official bin map array of shape %s", pixels.shape)
    return fill_pixels(pixels, row_starts, row_ends, col_starts, col_ends, indices)


def _read_pixel_chunks(path: Path, *, chunk_size: int) -> Iterator[pd.DataFrame]:
    """Stream the pixel mapping file in chunks to keep memory bounded."""
    for chunk in pd.read_csv(path, compression="infer", chunksize=chunk_size):
        yield _normalize_pixel_chunk(chunk)


def _normalize_pixel_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """Validate required columns and normalize dtypes on one chunk."""
    missing = [col for col in REQUIRED_PIXEL_COLUMNS if col not in chunk.columns]
    if missing:
        raise ValueError(f"pixel mapping is missing required columns: {missing}")

    df = chunk.copy()
    if "cell_type" not in df.columns:
        df["cell_type"] = pd.Series([None] * len(df), dtype="object")

    numeric_columns = list(REQUIRED_PIXEL_COLUMNS)
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="raise")

    if not np.isfinite(df[numeric_columns].to_numpy(dtype=np.float64)).all():
        raise ValueError("pixel mapping contains non-finite numeric values")

    df["x"] = df["x"].astype(np.int32)
    df["y"] = df["y"].astype(np.int32)
    df["cell_id"] = df["cell_id"].astype(np.int64)
    for col in ("is_boundary", "is_interior", "is_nuclear", "is_cytoplasm"):
        df[col] = df[col].astype(np.int8)
    return df


def _aggregate_pixel_chunk_to_official_bins(
    chunk: pd.DataFrame,
    *,
    pixel_to_bin: np.ndarray,
    crop_bins: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    """Assign one pixel chunk to official bins and aggregate counts."""
    df = chunk.copy()

    # Background pixels contribute zero to every compartment count. Drop them early
    # to avoid unnecessary pixel-map lookups and groupby load on the 500M+ crop grid.
    has_signal = (
        (df["is_nuclear"] > 0)
        | (df["is_cytoplasm"] > 0)
        | (df["is_boundary"] > 0)
    )
    df = df.loc[has_signal].copy()
    if len(df) == 0:
        return _empty_grouped_chunk(), 0

    crop_height, crop_width = pixel_to_bin.shape
    inside_crop = (
        (df["x"] >= 0)
        & (df["x"] < crop_width)
        & (df["y"] >= 0)
        & (df["y"] < crop_height)
    )
    df = df.loc[inside_crop].copy()
    if len(df) == 0:
        return _empty_grouped_chunk(), 0

    lookup = pixel_to_bin[
        df["y"].to_numpy(dtype=np.int32, copy=False),
        df["x"].to_numpy(dtype=np.int32, copy=False),
    ]
    df["crop_bin_index"] = lookup
    df = df[df["crop_bin_index"] >= 0].copy()
    if len(df) == 0:
        return _empty_grouped_chunk(), 0

    # Attach the exact official barcode and grid coordinates before grouping.
    metadata = crop_bins.loc[:, ["barcode", "array_row", "array_col", "in_tissue"]].copy()
    metadata["crop_bin_index"] = np.arange(len(metadata), dtype=np.int32)
    df = df.merge(metadata, on="crop_bin_index", how="left", validate="many_to_one")

    grouped = (
        df.groupby(
            ["barcode", "array_row", "array_col", "in_tissue", "cell_id", "cell_type"],
            dropna=False,
            sort=False,
        )
        .agg(
            nuclear_pixel_count=("is_nuclear", "sum"),
            cytoplasm_pixel_count=("is_cytoplasm", "sum"),
            boundary_pixel_count=("is_boundary", "sum"),
            interior_pixel_count=("is_interior", "sum"),
        )
        .reset_index()
    )
    return grouped, int(len(df))


def _empty_grouped_chunk() -> pd.DataFrame:
    """Return an empty grouped-chunk table with stable columns."""
    return pd.DataFrame(
        columns=[
            "barcode",
            "array_row",
            "array_col",
            "in_tissue",
            "cell_id",
            "cell_type",
            "nuclear_pixel_count",
            "cytoplasm_pixel_count",
            "boundary_pixel_count",
            "interior_pixel_count",
        ]
    )


def _merge_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """Merge grouped chunk results into one aggregated table."""
    merged = (
        df.groupby(
            ["barcode", "array_row", "array_col", "in_tissue", "cell_id", "cell_type"],
            dropna=False,
            sort=False,
        )
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
    return merged[merged["total_compartment_pixels"] > 0].reset_index(drop=True)


def _attach_official_bin_metadata(
    aggregated: pd.DataFrame,
    crop_bins: pd.DataFrame,
    *,
    microns_per_pixel: float,
) -> pd.DataFrame:
    """Attach official full-resolution bin-center coordinates to grouped counts."""
    metadata = crop_bins.loc[
        :,
        ["barcode", "array_row", "array_col", "in_tissue", "pxl_row_in_fullres", "pxl_col_in_fullres"],
    ].drop_duplicates(subset=["barcode"]).copy()

    out = aggregated.merge(
        metadata,
        on=["barcode", "array_row", "array_col", "in_tissue"],
        how="left",
        validate="many_to_one",
    )
    out["x_um"] = out["pxl_col_in_fullres"].astype(np.float64) * float(microns_per_pixel)
    out["y_um"] = out["pxl_row_in_fullres"].astype(np.float64) * float(microns_per_pixel)
    out["bin_id"] = out["barcode"]
    return out


def _split_compartments(counts: pd.DataFrame) -> pd.DataFrame:
    """Split mixed official bins into separate compartment rows with weights."""
    if len(counts) == 0:
        return _empty_official_output(include_cell_type=True)

    shared = [
        "bin_id",
        "barcode",
        "array_row",
        "array_col",
        "in_tissue",
        "x_um",
        "y_um",
        "pxl_row_in_fullres",
        "pxl_col_in_fullres",
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
    ordered_columns = [
        "bin_id",
        "barcode",
        "array_row",
        "array_col",
        "in_tissue",
        "x_um",
        "y_um",
        "pxl_row_in_fullres",
        "pxl_col_in_fullres",
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


def _empty_official_output(include_cell_type: bool) -> pd.DataFrame:
    """Return an empty output table with the final official-bin schema."""
    columns = [
        "bin_id",
        "barcode",
        "array_row",
        "array_col",
        "in_tissue",
        "x_um",
        "y_um",
        "pxl_row_in_fullres",
        "pxl_col_in_fullres",
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
        columns.insert(10, "cell_type")
    return pd.DataFrame(columns=columns)


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(description="Annotate official square_002um bins from pixel-level segmentation")
    parser.add_argument("--pixel-mapping-csv", required=True, help="Path to pixel-level mapping CSV/CSV.GZ")
    parser.add_argument("--tissue-positions-parquet", required=True, help="Path to square_002um spatial/tissue_positions.parquet")
    parser.add_argument("--output-path", required=True, help="Output path (.csv, .csv.gz, or .parquet)")
    parser.add_argument("--summary-path", default=None, help="Optional summary JSON path")
    parser.add_argument("--overlay-path", default=None, help="Optional overlay PNG output path")
    parser.add_argument("--cropped-image-path", default=None, help="Optional cropped image path for overlay background")
    parser.add_argument("--overlay-downsample-factor", type=int, default=4, help="Overlay downsample factor (default: 4)")
    parser.add_argument("--microns-per-pixel", type=float, default=DEFAULT_MICRONS_PER_PIXEL)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--nuclear-only", action="store_true", help="Keep only nuclear rows in final output")
    parser.add_argument("--in-tissue-only", action="store_true", help="Keep only in-tissue bins in final output")
    parser.add_argument("--row-number", type=int, default=None, help="Optional grid row count override")
    parser.add_argument("--col-number", type=int, default=None, help="Optional grid col count override")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    """CLI entrypoint."""
    args = _build_arg_parser().parse_args()
    configure_logging(verbose=args.verbose)

    tissue_df = pd.read_parquet(args.tissue_positions_parquet)
    _, crop_meta = _build_crop_region_bins(
        tissue_df=tissue_df,
        row_number=args.row_number,
        col_number=args.col_number,
    )
    df = annotate_square_002um_bins_from_pixels(
        pixel_mapping_csv=args.pixel_mapping_csv,
        tissue_positions_parquet=args.tissue_positions_parquet,
        microns_per_pixel=args.microns_per_pixel,
        chunk_size=args.chunk_size,
        nuclear_only=args.nuclear_only,
        in_tissue_only=args.in_tissue_only,
        row_number=args.row_number,
        col_number=args.col_number,
    )
    write_official_bin_table(df, args.output_path)
    if args.summary_path is not None:
        write_summary_json(df, crop_meta, args.summary_path)
    if args.overlay_path is not None:
        crop_bins, _ = _build_crop_region_bins(
            tissue_df=tissue_df,
            row_number=args.row_number,
            col_number=args.col_number,
        )
        write_overlay_png(
            df,
            crop_bins,
            crop_meta,
            args.overlay_path,
            cropped_image_path=args.cropped_image_path,
            downsample_factor=args.overlay_downsample_factor,
        )


if __name__ == "__main__":
    main()
