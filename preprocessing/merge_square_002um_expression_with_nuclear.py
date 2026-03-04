#!/usr/bin/env python
"""Build all-bin RL metadata by left-joining nuclear annotations onto official square_002um bins.

This script uses the official 10x / Space Ranger barcode order as the source of truth.
It does not duplicate the full expression matrix. Instead it writes metadata aligned to
matrix columns, plus a manifest that points back to the original matrix H5.

Outputs:
- metadata.parquet: one row per official barcode/bin, in matrix column order
- barcode_cell_claims.parquet: original nuclear claim rows with matrix column index added
- selected_feature_indices.npy: feature row indices to use from the matrix H5
- selected_features.tsv.gz: metadata for the selected feature rows
- manifest.json: paths, counts, and loading metadata
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)
DEFAULT_MICRONS_PER_PIXEL = 0.2737012522439323
DEFAULT_FEATURE_TYPE_FILTER = "Gene Expression"


def configure_logging(verbose: bool = False) -> None:
    """Configure process-wide logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")


def build_square_002um_rl_metadata(
    nuclear_annotation_path: str | Path,
    matrix_h5_path: str | Path,
    tissue_positions_parquet: str | Path,
    *,
    barcodes_tsv_path: str | Path | None = None,
    microns_per_pixel: float = DEFAULT_MICRONS_PER_PIXEL,
    feature_type_filter: str | None = DEFAULT_FEATURE_TYPE_FILTER,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, dict[str, Any]]:
    """Build all-bin metadata aligned to the official matrix column order.

    Parameters
    ----------
    nuclear_annotation_path:
        Official-barcode nuclear annotation CSV/CSV.GZ.
    matrix_h5_path:
        10x matrix H5. For all bins, use `raw_feature_bc_matrix.h5`.
    tissue_positions_parquet:
        Official `spatial/tissue_positions.parquet` file with per-bin coordinates.
    barcodes_tsv_path:
        Optional barcode TSV path. If omitted, inferred from the H5 parent folder.
    microns_per_pixel:
        Used to derive `x_um` and `y_um` from official full-resolution pixel positions.
    feature_type_filter:
        Optional filter for matrix feature rows. Default keeps `Gene Expression` rows.

    Returns
    -------
    metadata_df:
        One row per official matrix barcode, in matrix column order.
    claims_df:
        Original nuclear annotation rows with `matrix_col_index` attached.
    selected_features_df:
        Feature metadata for the selected feature subset.
    selected_feature_indices:
        Matrix row indices corresponding to `selected_features_df`.
    manifest:
        Plain-Python metadata about the aligned dataset.
    """
    nuclear_path = Path(nuclear_annotation_path)
    matrix_path = Path(matrix_h5_path)
    tissue_path = Path(tissue_positions_parquet)
    if not nuclear_path.exists():
        raise FileNotFoundError(f"nuclear annotation file not found: {nuclear_path}")
    if not matrix_path.exists():
        raise FileNotFoundError(f"matrix H5 file not found: {matrix_path}")
    if not tissue_path.exists():
        raise FileNotFoundError(f"tissue positions file not found: {tissue_path}")
    if microns_per_pixel <= 0:
        raise ValueError("microns_per_pixel must be > 0")

    claims_df = _load_nuclear_claims(nuclear_path)
    LOGGER.info("Loaded %d nuclear claim rows", len(claims_df))

    nuclear_barcode_df = _aggregate_nuclear_claims_to_barcodes(claims_df)
    LOGGER.info("Collapsed nuclear claims to %d unique barcodes", len(nuclear_barcode_df))

    tissue_df = _load_official_tissue_positions(tissue_path, microns_per_pixel=microns_per_pixel)
    LOGGER.info("Loaded %d official tissue-position rows", len(tissue_df))

    matrix_barcodes_df = _load_matrix_barcodes_in_order(matrix_path, barcodes_tsv_path=barcodes_tsv_path)
    LOGGER.info("Loaded %d matrix barcodes in official order", len(matrix_barcodes_df))

    metadata_df = matrix_barcodes_df.merge(tissue_df, on="barcode", how="left", validate="one_to_one")
    if metadata_df["barcode"].isna().any():
        raise ValueError("matrix barcode table contains missing barcodes after load")
    if metadata_df[["array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"]].isna().any().any():
        raise ValueError("some official matrix barcodes are missing tissue-position metadata")

    metadata_df = metadata_df.merge(nuclear_barcode_df, on="barcode", how="left", validate="one_to_one")
    metadata_df = _fill_missing_nuclear_metadata(metadata_df)
    metadata_df["bin_id"] = metadata_df["barcode"]

    claims_df = claims_df.merge(
        metadata_df.loc[:, ["barcode", "matrix_col_index"]],
        on="barcode",
        how="inner",
        validate="many_to_one",
    )

    selected_features_df, selected_feature_indices, feature_manifest = _load_selected_feature_metadata(
        matrix_path,
        feature_type_filter=feature_type_filter,
    )

    manifest = {
        "matrix_h5_path": str(matrix_path.resolve()),
        "barcodes_tsv_path": None if barcodes_tsv_path is None else str(Path(barcodes_tsv_path).resolve()),
        "tissue_positions_parquet": str(tissue_path.resolve()),
        "feature_type_filter": feature_type_filter,
        "n_barcodes": int(len(metadata_df)),
        "n_selected_features": int(len(selected_feature_indices)),
        "n_total_features": int(feature_manifest["n_total_features"]),
        "n_nuclear_barcodes": int(metadata_df["has_nuclear_annotation"].sum()),
        "n_ambiguous_nuclear_barcodes": int(metadata_df["ambiguous_nuclear_assignment"].sum()),
    }
    return metadata_df, claims_df, selected_features_df, selected_feature_indices, manifest


def write_rl_metadata_outputs(
    metadata_df: pd.DataFrame,
    claims_df: pd.DataFrame,
    selected_features_df: pd.DataFrame,
    selected_feature_indices: np.ndarray,
    manifest: dict[str, Any],
    output_dir: str | Path,
    *,
    prefix: str = "square_002um_rl",
) -> dict[str, Path]:
    """Write aligned metadata outputs to disk and return their paths."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = out_dir / f"{prefix}.metadata.parquet"
    claims_path = out_dir / f"{prefix}.barcode_cell_claims.parquet"
    feature_indices_path = out_dir / f"{prefix}.selected_feature_indices.npy"
    features_path = out_dir / f"{prefix}.selected_features.tsv.gz"
    manifest_path = out_dir / f"{prefix}.manifest.json"

    metadata_df.to_parquet(metadata_path, index=False)
    claims_df.to_parquet(claims_path, index=False)
    np.save(feature_indices_path, selected_feature_indices)
    with gzip.open(features_path, "wt") as handle:
        selected_features_df.to_csv(handle, sep="\t", index=False)

    manifest_payload = dict(manifest)
    manifest_payload.update(
        {
            "metadata_path": str(metadata_path.resolve()),
            "claims_path": str(claims_path.resolve()),
            "selected_feature_indices_path": str(feature_indices_path.resolve()),
            "selected_features_path": str(features_path.resolve()),
        }
    )
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest_payload, handle, indent=2)
        handle.write("\n")

    LOGGER.info("Wrote aligned metadata to %s", metadata_path)
    LOGGER.info("Wrote barcode-cell claims to %s", claims_path)
    LOGGER.info("Wrote selected feature indices to %s", feature_indices_path)
    LOGGER.info("Wrote selected feature metadata to %s", features_path)
    LOGGER.info("Wrote manifest to %s", manifest_path)
    return {
        "metadata": metadata_path,
        "claims": claims_path,
        "selected_feature_indices": feature_indices_path,
        "selected_features": features_path,
        "manifest": manifest_path,
    }


def _load_nuclear_claims(path: Path) -> pd.DataFrame:
    """Load nuclear claims and validate required columns."""
    df = pd.read_csv(path, compression="infer")
    required = [
        "barcode",
        "bin_id",
        "array_row",
        "array_col",
        "in_tissue",
        "x_um",
        "y_um",
        "pxl_row_in_fullres",
        "pxl_col_in_fullres",
        "cell_id",
        "cell_type",
        "weight",
        "nuclear_pixel_count",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"nuclear annotation file is missing required columns: {missing}")
    df["barcode"] = df["barcode"].astype(str)
    return df


def _aggregate_nuclear_claims_to_barcodes(claims_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-barcode-cell nuclear claims into one row per barcode."""
    sort_df = claims_df.sort_values(
        by=["barcode", "nuclear_pixel_count", "weight"],
        ascending=[True, False, False],
        kind="mergesort",
    )
    dominant = sort_df.drop_duplicates(subset=["barcode"], keep="first").copy()

    agg = claims_df.groupby("barcode", sort=False).agg(
        total_nuclear_pixel_count=("nuclear_pixel_count", "sum"),
        max_nuclear_pixel_count=("nuclear_pixel_count", "max"),
        n_cell_claims=("cell_id", pd.Series.nunique),
        max_weight=("weight", "max"),
    ).reset_index()

    dominant_cols = dominant.loc[:, ["barcode", "cell_id", "cell_type", "weight"]].rename(
        columns={
            "cell_id": "dominant_cell_id",
            "cell_type": "dominant_cell_type",
            "weight": "dominant_weight",
        }
    )

    out = agg.merge(dominant_cols, on="barcode", how="left", validate="one_to_one")
    out["has_nuclear_annotation"] = True
    out["ambiguous_nuclear_assignment"] = out["n_cell_claims"] > 1
    return out


def _load_official_tissue_positions(path: Path, *, microns_per_pixel: float) -> pd.DataFrame:
    """Load official tissue positions and derive coordinate metadata."""
    df = pd.read_parquet(path)
    required = [
        "barcode",
        "in_tissue",
        "array_row",
        "array_col",
        "pxl_row_in_fullres",
        "pxl_col_in_fullres",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"tissue_positions is missing required columns: {missing}")

    out = df.loc[:, required].copy()
    out["barcode"] = out["barcode"].astype(str)
    out["x_um"] = out["pxl_col_in_fullres"].astype(np.float64) * float(microns_per_pixel)
    out["y_um"] = out["pxl_row_in_fullres"].astype(np.float64) * float(microns_per_pixel)
    return out


def _load_matrix_barcodes_in_order(matrix_h5_path: Path, *, barcodes_tsv_path: str | Path | None) -> pd.DataFrame:
    """Load official matrix barcodes in matrix-column order."""
    if barcodes_tsv_path is None:
        inferred = matrix_h5_path.parent / matrix_h5_path.stem.replace('.h5', '') / 'barcodes.tsv.gz'
        # fallback for 10x layout where H5 sits beside a same-named folder is not reliable
        # so just use the sibling folder named after the matrix stem if it exists.
        if inferred.exists():
            barcodes_tsv_path = inferred
        else:
            sibling = matrix_h5_path.parent / matrix_h5_path.name.replace('.h5', '') / 'barcodes.tsv.gz'
            if sibling.exists():
                barcodes_tsv_path = sibling

    if barcodes_tsv_path is not None and Path(barcodes_tsv_path).exists():
        barcodes_df = pd.read_csv(
            barcodes_tsv_path,
            sep="\t",
            header=None,
            names=["barcode"],
            dtype={"barcode": "string"},
        )
        barcodes_df["barcode"] = barcodes_df["barcode"].astype(str)
    else:
        with h5py.File(matrix_h5_path, "r") as h5:
            raw = h5["matrix/barcodes"][:]
        barcodes_df = pd.DataFrame({"barcode": [x.decode("utf-8") for x in raw]})

    barcodes_df["matrix_col_index"] = np.arange(len(barcodes_df), dtype=np.int64)
    return barcodes_df


def _load_selected_feature_metadata(
    matrix_h5_path: Path,
    *,
    feature_type_filter: str | None,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    """Load feature metadata and compute selected row indices."""
    with h5py.File(matrix_h5_path, "r") as h5:
        feature_ids = np.asarray([x.decode("utf-8") for x in h5["matrix/features/id"][:]], dtype=object)
        feature_names = np.asarray([x.decode("utf-8") for x in h5["matrix/features/name"][:]], dtype=object)
        feature_types = np.asarray([x.decode("utf-8") for x in h5["matrix/features/feature_type"][:]], dtype=object)
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


def _fill_missing_nuclear_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing nuclear annotation columns for non-nuclear bins."""
    out = df.copy()
    if "bin_id" not in out.columns:
        out["bin_id"] = out["barcode"]
    if "has_nuclear_annotation" not in out.columns:
        out["has_nuclear_annotation"] = np.zeros(len(out), dtype=bool)
    else:
        has_nuclear = out["has_nuclear_annotation"].to_numpy(dtype=object, copy=False)
        out["has_nuclear_annotation"] = pd.Series(
            [False if value is None or pd.isna(value) else bool(value) for value in has_nuclear],
            index=out.index,
            dtype=bool,
        )

    for col in ["total_nuclear_pixel_count", "max_nuclear_pixel_count", "n_cell_claims", "max_weight", "dominant_weight"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)
    if "n_cell_claims" in out.columns:
        out["n_cell_claims"] = out["n_cell_claims"].astype(np.int32)
    for col in ["total_nuclear_pixel_count", "max_nuclear_pixel_count"]:
        if col in out.columns:
            out[col] = out[col].astype(np.int32)

    if "ambiguous_nuclear_assignment" not in out.columns:
        out["ambiguous_nuclear_assignment"] = np.zeros(len(out), dtype=bool)
    else:
        ambiguous = out["ambiguous_nuclear_assignment"].to_numpy(dtype=object, copy=False)
        out["ambiguous_nuclear_assignment"] = pd.Series(
            [False if value is None or pd.isna(value) else bool(value) for value in ambiguous],
            index=out.index,
            dtype=bool,
        )

    for col in ["dominant_cell_id", "dominant_cell_type"]:
        if col not in out.columns:
            out[col] = pd.Series([None] * len(out), dtype="object")

    ordered = [
        "matrix_col_index",
        "bin_id",
        "barcode",
        "array_row",
        "array_col",
        "in_tissue",
        "x_um",
        "y_um",
        "pxl_row_in_fullres",
        "pxl_col_in_fullres",
        "has_nuclear_annotation",
        "dominant_cell_id",
        "dominant_cell_type",
        "dominant_weight",
        "total_nuclear_pixel_count",
        "max_nuclear_pixel_count",
        "n_cell_claims",
        "ambiguous_nuclear_assignment",
        "max_weight",
    ]
    return out.loc[:, ordered].reset_index(drop=True)


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(description="Build all-bin square_002um RL metadata from official expression + nuclear annotations")
    parser.add_argument("--nuclear-annotation-path", required=True, help="Path to barcode-aligned nuclear annotation CSV/CSV.GZ")
    parser.add_argument("--matrix-h5-path", required=True, help="Path to raw_feature_bc_matrix.h5 or filtered_feature_bc_matrix.h5")
    parser.add_argument("--tissue-positions-parquet", required=True, help="Path to spatial/tissue_positions.parquet")
    parser.add_argument("--output-dir", required=True, help="Directory for merged outputs")
    parser.add_argument("--prefix", default="square_002um_rl", help="Output file prefix")
    parser.add_argument("--barcodes-tsv-path", default=None, help="Optional path to barcodes.tsv.gz")
    parser.add_argument("--microns-per-pixel", type=float, default=DEFAULT_MICRONS_PER_PIXEL)
    parser.add_argument("--feature-type-filter", default=DEFAULT_FEATURE_TYPE_FILTER, help="Feature type to keep from matrix H5 (default: Gene Expression)")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    """CLI entrypoint."""
    args = _build_arg_parser().parse_args()
    configure_logging(verbose=args.verbose)

    metadata_df, claims_df, selected_features_df, selected_feature_indices, manifest = build_square_002um_rl_metadata(
        nuclear_annotation_path=args.nuclear_annotation_path,
        matrix_h5_path=args.matrix_h5_path,
        tissue_positions_parquet=args.tissue_positions_parquet,
        barcodes_tsv_path=args.barcodes_tsv_path,
        microns_per_pixel=args.microns_per_pixel,
        feature_type_filter=args.feature_type_filter,
    )
    write_rl_metadata_outputs(
        metadata_df,
        claims_df,
        selected_features_df,
        selected_feature_indices,
        manifest,
        output_dir=args.output_dir,
        prefix=args.prefix,
    )


if __name__ == "__main__":
    main()
