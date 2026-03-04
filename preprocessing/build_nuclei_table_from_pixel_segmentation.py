#!/usr/bin/env python
"""Build a one-row-per-nucleus table from pixel-level nuclear segmentation output.

This script computes nucleus centroids from the pixel-level `is_nuclear` mask produced by
`cellpose_segmentation.py` and converts them into the same full-resolution micron
coordinate system used by the official `square_002um` bins.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)
DEFAULT_MICRONS_PER_PIXEL = 0.2737012522439323
DEFAULT_CHUNK_SIZE = 1_000_000
REQUIRED_PIXEL_COLUMNS = ("x", "y", "cell_id", "is_nuclear")


def configure_logging(verbose: bool = False) -> None:
    """Configure process-wide logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")


def build_nuclei_table_from_pixel_segmentation(
    pixel_mapping_csv: str | Path,
    tissue_positions_parquet: str | Path,
    *,
    microns_per_pixel: float = DEFAULT_MICRONS_PER_PIXEL,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    row_number: int | None = None,
    col_number: int | None = None,
) -> tuple[pd.DataFrame, dict[str, int | float]]:
    """Compute one nucleus row per cell from crop-relative pixel segmentation output.

    Returns
    -------
    nuclei_df:
        One row per `cell_id` with centroid and equivalent-radius information.
    crop_meta:
        Crop offsets and dimensions used to convert crop-relative pixels into full-image
        coordinates.
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

    crop_meta = _compute_crop_metadata(
        tissue_positions_parquet=tissue_path,
        row_number=row_number,
        col_number=col_number,
    )
    LOGGER.info("Using crop offsets row_left=%d col_up=%d", crop_meta["row_left"], crop_meta["col_up"])

    chunk_aggregates: list[pd.DataFrame] = []
    total_rows = 0
    total_nuclear_rows = 0
    for chunk in _read_pixel_chunks(pixel_path, chunk_size=chunk_size):
        total_rows += len(chunk)
        grouped = _aggregate_nuclear_pixels_in_chunk(chunk)
        total_nuclear_rows += int(grouped["nuclear_pixel_count"].sum()) if len(grouped) else 0
        chunk_aggregates.append(grouped)
        LOGGER.info(
            "Processed chunk with %d pixels into %d nucleus groups",
            len(chunk),
            len(grouped),
        )

    if not chunk_aggregates:
        return _empty_nuclei_table(), crop_meta

    combined = pd.concat(chunk_aggregates, ignore_index=True)
    grouped = (
        combined.groupby("cell_id", sort=False)
        .agg(
            nuclear_pixel_count=("nuclear_pixel_count", "sum"),
            sum_x_crop_px=("sum_x_crop_px", "sum"),
            sum_y_crop_px=("sum_y_crop_px", "sum"),
            min_x_crop_px=("min_x_crop_px", "min"),
            max_x_crop_px=("max_x_crop_px", "max"),
            min_y_crop_px=("min_y_crop_px", "min"),
            max_y_crop_px=("max_y_crop_px", "max"),
        )
        .reset_index()
    )

    nuclei_df = _finalize_nuclei_table(grouped, crop_meta=crop_meta, microns_per_pixel=microns_per_pixel)
    LOGGER.info(
        "Built nuclei table with %d rows from %d nuclear pixels",
        len(nuclei_df),
        total_nuclear_rows,
    )
    return nuclei_df, crop_meta


def write_nuclei_table(df: pd.DataFrame, output_path: str | Path) -> Path:
    """Write nuclei table to `.csv`, `.csv.gz`, or `.parquet`."""
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
    LOGGER.info("Wrote nuclei table to %s", path)
    return path


def write_summary_json(
    nuclei_df: pd.DataFrame,
    crop_meta: dict[str, int | float],
    output_path: str | Path,
) -> Path:
    """Write a compact summary JSON for sanity checking."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **crop_meta,
        "n_nuclei": int(len(nuclei_df)),
        "mean_radius_um": float(nuclei_df["radius_um"].mean()) if len(nuclei_df) else 0.0,
        "median_radius_um": float(nuclei_df["radius_um"].median()) if len(nuclei_df) else 0.0,
        "max_radius_um": float(nuclei_df["radius_um"].max()) if len(nuclei_df) else 0.0,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    LOGGER.info("Wrote nuclei summary to %s", path)
    return path


def _read_pixel_chunks(path: Path, *, chunk_size: int) -> Iterator[pd.DataFrame]:
    """Stream pixel CSV chunks with schema validation."""
    for chunk in pd.read_csv(path, compression="infer", chunksize=chunk_size):
        missing = [col for col in REQUIRED_PIXEL_COLUMNS if col not in chunk.columns]
        if missing:
            raise ValueError(f"pixel mapping is missing required columns: {missing}")
        df = chunk.loc[:, list(REQUIRED_PIXEL_COLUMNS)].copy()
        for col in REQUIRED_PIXEL_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="raise")
        df["x"] = df["x"].astype(np.int32)
        df["y"] = df["y"].astype(np.int32)
        df["cell_id"] = df["cell_id"].astype(np.int64)
        df["is_nuclear"] = df["is_nuclear"].astype(np.int8)
        yield df


def _aggregate_nuclear_pixels_in_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """Aggregate nuclear pixels in one chunk by `cell_id`."""
    df = chunk[(chunk["is_nuclear"] > 0) & (chunk["cell_id"] > 0)].copy()
    if len(df) == 0:
        return pd.DataFrame(
            columns=[
                "cell_id",
                "nuclear_pixel_count",
                "sum_x_crop_px",
                "sum_y_crop_px",
                "min_x_crop_px",
                "max_x_crop_px",
                "min_y_crop_px",
                "max_y_crop_px",
            ]
        )

    grouped = (
        df.groupby("cell_id", sort=False)
        .agg(
            nuclear_pixel_count=("is_nuclear", "sum"),
            sum_x_crop_px=("x", "sum"),
            sum_y_crop_px=("y", "sum"),
            min_x_crop_px=("x", "min"),
            max_x_crop_px=("x", "max"),
            min_y_crop_px=("y", "min"),
            max_y_crop_px=("y", "max"),
        )
        .reset_index()
    )
    return grouped


def _compute_crop_metadata(
    *,
    tissue_positions_parquet: Path,
    row_number: int | None,
    col_number: int | None,
) -> dict[str, int | float]:
    """Recompute the crop offsets used when the cropped image was generated."""
    df = pd.read_parquet(tissue_positions_parquet)
    required = [
        "array_row",
        "array_col",
        "in_tissue",
        "pxl_row_in_fullres",
        "pxl_col_in_fullres",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"tissue_positions is missing required columns: {missing}")

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

    crop_df = df[
        (df["array_row"] >= start_row_spot)
        & (df["array_row"] < end_row_spot)
        & (df["array_col"] >= start_col_spot)
        & (df["array_col"] < end_col_spot)
    ].copy()

    row_left = int(max(crop_df["pxl_row_left_in_fullres"].min(), 0))
    row_right = int(crop_df["pxl_row_right_in_fullres"].max())
    col_up = int(max(crop_df["pxl_col_up_in_fullres"].min(), 0))
    col_down = int(crop_df["pxl_col_down_in_fullres"].max())

    return {
        "row_left": row_left,
        "row_right": row_right,
        "col_up": col_up,
        "col_down": col_down,
        "crop_height_px": int(row_right - row_left),
        "crop_width_px": int(col_down - col_up),
        "row_number": int(row_number),
        "col_number": int(col_number),
    }


def _finalize_nuclei_table(
    grouped: pd.DataFrame,
    *,
    crop_meta: dict[str, int | float],
    microns_per_pixel: float,
) -> pd.DataFrame:
    """Convert grouped crop-relative aggregates into full-resolution nuclei metadata."""
    out = grouped.copy()

    out["center_x_crop_px"] = out["sum_x_crop_px"] / out["nuclear_pixel_count"]
    out["center_y_crop_px"] = out["sum_y_crop_px"] / out["nuclear_pixel_count"]

    out["center_x_fullres_px"] = out["center_x_crop_px"] + float(crop_meta["col_up"])
    out["center_y_fullres_px"] = out["center_y_crop_px"] + float(crop_meta["row_left"])

    out["center_x_um"] = out["center_x_fullres_px"] * float(microns_per_pixel)
    out["center_y_um"] = out["center_y_fullres_px"] * float(microns_per_pixel)

    nuclear_area_um2 = out["nuclear_pixel_count"].astype(np.float64) * float(microns_per_pixel) * float(microns_per_pixel)
    out["radius_um"] = np.sqrt(nuclear_area_um2 / math.pi)

    out["bbox_min_x_fullres_px"] = out["min_x_crop_px"] + int(crop_meta["col_up"])
    out["bbox_max_x_fullres_px"] = out["max_x_crop_px"] + int(crop_meta["col_up"])
    out["bbox_min_y_fullres_px"] = out["min_y_crop_px"] + int(crop_meta["row_left"])
    out["bbox_max_y_fullres_px"] = out["max_y_crop_px"] + int(crop_meta["row_left"])

    out = out.rename(columns={"cell_id": "cell_id"})
    ordered = [
        "cell_id",
        "center_x_um",
        "center_y_um",
        "radius_um",
        "nuclear_pixel_count",
        "center_x_fullres_px",
        "center_y_fullres_px",
        "bbox_min_x_fullres_px",
        "bbox_max_x_fullres_px",
        "bbox_min_y_fullres_px",
        "bbox_max_y_fullres_px",
    ]
    out = out.loc[:, ordered].sort_values("cell_id").reset_index(drop=True)
    return out


def _empty_nuclei_table() -> pd.DataFrame:
    """Return an empty nuclei table with the final schema."""
    return pd.DataFrame(
        columns=[
            "cell_id",
            "center_x_um",
            "center_y_um",
            "radius_um",
            "nuclear_pixel_count",
            "center_x_fullres_px",
            "center_y_fullres_px",
            "bbox_min_x_fullres_px",
            "bbox_max_x_fullres_px",
            "bbox_min_y_fullres_px",
            "bbox_max_y_fullres_px",
        ]
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(description="Build one-row-per-nucleus table from pixel-level segmentation")
    parser.add_argument("--pixel-mapping-csv", required=True, help="Path to pixel-level segmentation CSV/CSV.GZ")
    parser.add_argument("--tissue-positions-parquet", required=True, help="Path to official square_002um tissue_positions.parquet")
    parser.add_argument("--output-path", required=True, help="Output path (.csv, .csv.gz, or .parquet)")
    parser.add_argument("--summary-path", default=None, help="Optional summary JSON path")
    parser.add_argument("--microns-per-pixel", type=float, default=DEFAULT_MICRONS_PER_PIXEL)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--row-number", type=int, default=None)
    parser.add_argument("--col-number", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    """CLI entrypoint."""
    args = _build_arg_parser().parse_args()
    configure_logging(verbose=args.verbose)

    nuclei_df, crop_meta = build_nuclei_table_from_pixel_segmentation(
        pixel_mapping_csv=args.pixel_mapping_csv,
        tissue_positions_parquet=args.tissue_positions_parquet,
        microns_per_pixel=args.microns_per_pixel,
        chunk_size=args.chunk_size,
        row_number=args.row_number,
        col_number=args.col_number,
    )
    write_nuclei_table(nuclei_df, args.output_path)
    if args.summary_path is not None:
        write_summary_json(nuclei_df, crop_meta, args.summary_path)


if __name__ == "__main__":
    main()
