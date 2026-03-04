#!/usr/bin/env python
"""Build per-cell-type reference count matrix C[K,G] from scRNA 10x-style H5 + metadata."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from scipy import sparse


LOGGER = logging.getLogger(__name__)


def configure_logging(verbose: bool = False) -> None:
    """Configure process logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")


def build_reference_counts(
    matrix_h5_path: str | Path,
    metadata_tsv_path: str | Path,
    *,
    cell_id_column: str = "cell_id",
    cell_type_column: str = "sct_cell_type",
    min_cells_per_type: int = 50,
    feature_type_filter: str | None = "Gene Expression",
    gene_allowlist_path: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Aggregate raw scRNA counts into a per-cell-type reference matrix.

    Returns
    -------
    reference_counts:
        Dense matrix with shape (K, G), where K is kept cell types and G is selected genes.
    cell_types:
        Length-K unicode array of kept cell-type labels.
    genes:
        Length-G unicode array of gene names, aligned to reference_counts columns.
    n_cells_per_type:
        Length-K int64 array of cell counts in each kept type.
    summary:
        JSON-serializable metadata.
    """
    h5_path = Path(matrix_h5_path)
    meta_path = Path(metadata_tsv_path)
    if not h5_path.exists():
        raise FileNotFoundError(f"matrix H5 not found: {h5_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata TSV not found: {meta_path}")
    if min_cells_per_type <= 0:
        raise ValueError("min_cells_per_type must be > 0")

    x_csc, barcodes, genes, feature_types = _load_10x_h5_matrix(h5_path)
    n_genes, n_cells = x_csc.shape
    LOGGER.info("Loaded matrix with %d genes x %d cells", n_genes, n_cells)

    metadata = pd.read_csv(
        meta_path,
        sep="\t",
        low_memory=False,
        usecols=[cell_id_column, cell_type_column],
    )
    if cell_id_column not in metadata.columns:
        raise ValueError(f"metadata is missing cell ID column: {cell_id_column!r}")
    if cell_type_column not in metadata.columns:
        raise ValueError(f"metadata is missing cell type column: {cell_type_column!r}")

    meta_min = metadata.loc[:, [cell_id_column, cell_type_column]].copy()
    meta_min[cell_id_column] = meta_min[cell_id_column].astype(str)
    meta_min = meta_min.drop_duplicates(subset=[cell_id_column], keep="first")

    matrix_cells = pd.DataFrame(
        {
            "matrix_col_index": np.arange(n_cells, dtype=np.int64),
            "cell_id": barcodes.astype(str),
        }
    )
    aligned = matrix_cells.merge(
        meta_min,
        left_on="cell_id",
        right_on=cell_id_column,
        how="left",
        validate="one_to_one",
    )

    raw_cell_types = aligned[cell_type_column].astype("string")
    missing_type_mask = raw_cell_types.isna().to_numpy()
    n_missing_type = int(missing_type_mask.sum())
    if n_missing_type > 0:
        LOGGER.warning("%d cells have missing %s and will be dropped", n_missing_type, cell_type_column)

    value_counts = raw_cell_types.value_counts(dropna=True)
    keep_types = value_counts[value_counts >= int(min_cells_per_type)].index.astype(str).tolist()
    if not keep_types:
        raise ValueError(
            f"no cell types meet min_cells_per_type={min_cells_per_type}; "
            f"max observed type size={int(value_counts.max()) if len(value_counts) else 0}"
        )
    LOGGER.info("Keeping %d cell types with >= %d cells", len(keep_types), min_cells_per_type)

    keep_type_set = set(keep_types)
    selected_mask = raw_cell_types.notna().to_numpy() & np.asarray(
        [str(v) in keep_type_set for v in raw_cell_types.to_numpy(dtype=object)],
        dtype=bool,
    )
    selected_indices = np.flatnonzero(selected_mask)
    if len(selected_indices) == 0:
        raise ValueError("no cells selected after applying cell-type filtering")

    kept_type_counts = (
        pd.Series(raw_cell_types[selected_mask].astype(str))
        .value_counts()
        .sort_values(ascending=False)
    )
    cell_type_labels = kept_type_counts.index.to_list()
    type_to_code = {ct: i for i, ct in enumerate(cell_type_labels)}

    type_codes = np.full(n_cells, -1, dtype=np.int32)
    selected_types = raw_cell_types[selected_mask].astype(str).to_numpy()
    type_codes[selected_indices] = np.asarray([type_to_code[t] for t in selected_types], dtype=np.int32)
    k = len(cell_type_labels)
    n_cells_per_type = np.bincount(type_codes[selected_indices], minlength=k).astype(np.int64)

    gene_mask = np.ones(n_genes, dtype=bool)
    if feature_type_filter is not None:
        if feature_types is None:
            LOGGER.warning("feature_type_filter was set but feature_type is absent in H5; skipping this filter")
        else:
            gene_mask &= feature_types == str(feature_type_filter)
            LOGGER.info("Feature-type filter kept %d/%d genes", int(gene_mask.sum()), n_genes)

    if gene_allowlist_path is not None:
        allow_genes = _load_gene_allowlist(Path(gene_allowlist_path))
        if not allow_genes:
            raise ValueError(f"gene allowlist is empty: {gene_allowlist_path}")
        allow_mask = np.isin(genes, np.asarray(sorted(allow_genes), dtype=genes.dtype))
        gene_mask &= allow_mask
        LOGGER.info("Gene allowlist filter kept %d genes", int(gene_mask.sum()))

    selected_gene_idx = np.flatnonzero(gene_mask)
    if len(selected_gene_idx) == 0:
        raise ValueError("no genes remain after filtering")

    x_sel = x_csc[selected_gene_idx, :]
    indicator = sparse.csc_matrix(
        (
            np.ones(len(selected_indices), dtype=np.int8),
            (selected_indices, type_codes[selected_indices]),
        ),
        shape=(n_cells, k),
    )
    c_sparse = x_sel.dot(indicator)
    reference_counts = np.asarray(c_sparse.toarray(), dtype=np.float64).T
    selected_genes = genes[selected_gene_idx].astype("U")

    summary: dict[str, Any] = {
        "matrix_h5_path": str(h5_path.resolve()),
        "metadata_tsv_path": str(meta_path.resolve()),
        "cell_id_column": cell_id_column,
        "cell_type_column": cell_type_column,
        "min_cells_per_type": int(min_cells_per_type),
        "feature_type_filter": feature_type_filter,
        "gene_allowlist_path": None if gene_allowlist_path is None else str(Path(gene_allowlist_path).resolve()),
        "n_input_cells": int(n_cells),
        "n_cells_missing_type": int(n_missing_type),
        "n_selected_cells": int(len(selected_indices)),
        "n_kept_cell_types": int(k),
        "n_selected_genes": int(len(selected_gene_idx)),
        "n_total_genes": int(n_genes),
    }
    return (
        reference_counts,
        np.asarray(cell_type_labels, dtype="U"),
        selected_genes,
        n_cells_per_type,
        summary,
    )


def write_reference_counts_npz(
    output_path: str | Path,
    *,
    reference_counts: np.ndarray,
    cell_types: np.ndarray,
    genes: np.ndarray,
    n_cells_per_type: np.ndarray,
) -> Path:
    """Write reference-count bundle to compressed NPZ."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        reference_counts=np.asarray(reference_counts, dtype=np.float64),
        cell_types=np.asarray(cell_types, dtype="U"),
        genes=np.asarray(genes, dtype="U"),
        n_cells_per_type=np.asarray(n_cells_per_type, dtype=np.int64),
    )
    LOGGER.info("Wrote reference counts NPZ: %s", out)
    return out


def write_summary_json(output_path: str | Path, summary: dict[str, Any], *, cell_types: np.ndarray, n_cells_per_type: np.ndarray) -> Path:
    """Write compact JSON summary for audit/debug."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(summary)
    payload["cell_type_counts"] = {
        str(cell_types[i]): int(n_cells_per_type[i]) for i in range(len(cell_types))
    }
    with out.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    LOGGER.info("Wrote summary JSON: %s", out)
    return out


def _load_10x_h5_matrix(path: Path) -> tuple[sparse.csc_matrix, np.ndarray, np.ndarray, np.ndarray | None]:
    """Load 10x-format sparse matrix and aligned labels from H5."""
    with h5py.File(path, "r") as h5:
        if "matrix" not in h5:
            raise ValueError(f"H5 file does not contain 'matrix' group: {path}")
        mg = h5["matrix"]
        for key in ("data", "indices", "indptr", "shape", "barcodes"):
            if key not in mg:
                raise ValueError(f"H5 matrix group missing key: matrix/{key}")
        if "features" not in mg:
            raise ValueError("H5 matrix group missing features subgroup")
        fg = mg["features"]
        if "name" not in fg:
            raise ValueError("H5 features subgroup missing matrix/features/name")

        data = mg["data"][:]
        indices = mg["indices"][:]
        indptr = mg["indptr"][:]
        shape = tuple(int(v) for v in mg["shape"][:].tolist())
        barcodes = np.asarray([x.decode("utf-8") for x in mg["barcodes"][:]], dtype="U")
        genes = np.asarray([x.decode("utf-8") for x in fg["name"][:]], dtype="U")
        feature_types = None
        if "feature_type" in fg:
            feature_types = np.asarray([x.decode("utf-8") for x in fg["feature_type"][:]], dtype="U")

    x_csc = sparse.csc_matrix((data, indices, indptr), shape=shape, dtype=np.int64)
    if x_csc.shape[0] != len(genes):
        raise ValueError("gene count does not match matrix rows")
    if x_csc.shape[1] != len(barcodes):
        raise ValueError("barcode count does not match matrix columns")
    return x_csc, barcodes, genes, feature_types


def _load_gene_allowlist(path: Path) -> set[str]:
    """Load gene allowlist from text/csv/tsv file."""
    if not path.exists():
        raise FileNotFoundError(f"gene allowlist not found: {path}")

    suffixes = path.suffixes
    if suffixes[-1:] == [".csv"]:
        df = pd.read_csv(path)
        if df.shape[1] == 0:
            return set()
        genes = df.iloc[:, 0].astype(str).str.strip()
        return {g for g in genes.tolist() if g}
    if suffixes[-1:] == [".tsv"]:
        df = pd.read_csv(path, sep="\t")
        if df.shape[1] == 0:
            return set()
        genes = df.iloc[:, 0].astype(str).str.strip()
        return {g for g in genes.tolist() if g}

    genes: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            gene = line.strip()
            if not gene or gene.startswith("#"):
                continue
            genes.add(gene)
    return genes


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Build C[K,G] reference counts from scRNA 10x H5 + metadata")
    parser.add_argument("--matrix-h5-path", required=True, help="Path to 10x-style raw scRNA H5")
    parser.add_argument("--metadata-tsv-path", required=True, help="Path to metadata TSV with cell labels")
    parser.add_argument("--output-path", required=True, help="Output NPZ path (e.g., reference_counts.npz)")
    parser.add_argument("--summary-path", default=None, help="Optional summary JSON output path")
    parser.add_argument("--cell-id-column", default="cell_id", help="Metadata column matching matrix barcodes")
    parser.add_argument("--cell-type-column", default="sct_cell_type", help="Metadata column to group by")
    parser.add_argument("--min-cells-per-type", type=int, default=50, help="Drop cell types with fewer cells")
    parser.add_argument(
        "--feature-type-filter",
        default="Gene Expression",
        help="Feature type to keep; pass NONE to disable",
    )
    parser.add_argument("--gene-allowlist-path", default=None, help="Optional gene allowlist (txt/csv/tsv)")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    """CLI entrypoint."""
    args = _build_arg_parser().parse_args()
    configure_logging(verbose=args.verbose)

    feature_type_filter = None if str(args.feature_type_filter).upper() == "NONE" else str(args.feature_type_filter)
    reference_counts, cell_types, genes, n_cells_per_type, summary = build_reference_counts(
        matrix_h5_path=args.matrix_h5_path,
        metadata_tsv_path=args.metadata_tsv_path,
        cell_id_column=str(args.cell_id_column),
        cell_type_column=str(args.cell_type_column),
        min_cells_per_type=int(args.min_cells_per_type),
        feature_type_filter=feature_type_filter,
        gene_allowlist_path=args.gene_allowlist_path,
    )

    write_reference_counts_npz(
        args.output_path,
        reference_counts=reference_counts,
        cell_types=cell_types,
        genes=genes,
        n_cells_per_type=n_cells_per_type,
    )
    if args.summary_path is not None:
        write_summary_json(
            args.summary_path,
            summary,
            cell_types=cell_types,
            n_cells_per_type=n_cells_per_type,
        )

    LOGGER.info(
        "Done: C shape=%s, K=%d, G=%d",
        tuple(reference_counts.shape),
        int(reference_counts.shape[0]),
        int(reference_counts.shape[1]),
    )


if __name__ == "__main__":
    main()
