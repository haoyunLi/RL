#!/usr/bin/env python
"""Build a unique HD gene allowlist text file from selected-features TSV."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd


LOGGER = logging.getLogger(__name__)


def configure_logging(verbose: bool = False) -> None:
    """Configure process logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")


def build_hd_gene_allowlist(
    selected_features_tsv: str | Path,
    output_txt: str | Path,
    *,
    gene_column: str = "feature_name",
    sort_values: bool = True,
) -> Path:
    """Extract unique non-empty gene names and write one gene per line."""
    in_path = Path(selected_features_tsv)
    out_path = Path(output_txt)
    if not in_path.exists():
        raise FileNotFoundError(f"selected features file not found: {in_path}")

    df = pd.read_csv(in_path, sep="\t")
    if gene_column not in df.columns:
        raise ValueError(f"missing required gene column {gene_column!r} in {in_path}")

    genes = [g.strip() for g in df[gene_column].astype(str).tolist()]
    genes = [g for g in genes if g and g.lower() != "nan"]
    unique = sorted(set(genes)) if sort_values else list(dict.fromkeys(genes))
    if not unique:
        raise ValueError("no valid genes extracted from selected features table")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for gene in unique:
            handle.write(gene)
            handle.write("\n")

    LOGGER.info("Wrote %d unique genes to %s", len(unique), out_path)
    return out_path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build unique HD gene allowlist from selected-features TSV")
    parser.add_argument("--selected-features-tsv", required=True, help="Path to selected_features.tsv.gz")
    parser.add_argument("--output-txt", required=True, help="Output TXT path (one gene per line)")
    parser.add_argument("--gene-column", default="feature_name", help="Column name containing gene symbols")
    parser.add_argument("--keep-order", action="store_true", help="Keep first-seen order instead of sorting")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    configure_logging(verbose=args.verbose)
    build_hd_gene_allowlist(
        selected_features_tsv=args.selected_features_tsv,
        output_txt=args.output_txt,
        gene_column=args.gene_column,
        sort_values=not args.keep_order,
    )


if __name__ == "__main__":
    main()
