#!/usr/bin/env python
"""
Bin2Cell Analysis for Pseudo Visium HD Data

This script:
1. Loads pseudo Visium HD 2μm bin data with spatial coordinates
2. Performs destriping to correct for variable bin dimensions
3. Generates scaled H&E images at specified resolution (mpp)
4. Performs StarDist segmentation on H&E (nuclei) and gene expression (cells)
5. Expands nuclear labels to include cytoplasm
6. Combines H&E and GEX segmentation results
7. Groups 2μm bins into cells
8. Outputs cell-level AnnData object ready for downstream analysis

Usage:
    python run_bin2cell_analysis.py \
        --pseudo_hd_dir pseudo_visium_hd_outpu_full \
        --image_path cropped_visium_hd_human_colorectal.png \
        --output_dir bin2cell_results_colorectal \
        --dataset_name human_colorectal \
        --mpp 0.5
"""

from __future__ import annotations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scanpy as sc
import numpy as np
import os
import argparse
from pathlib import Path
import logging
import sys
import json

# Disable OpenCV pixel limit to allow loading very large images (BTF files)
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2,40))  # Allow up to 1 trillion pixels

# Disable PIL decompression bomb protection for large microscopy images
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

import cv2

import bin2cell as b2c
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from preprocessing.ppo_format_assignment_eval import (
    coerce_bool_series,
    normalize_cell_id,
    run_ppo_format_assignment_evaluation,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bin2cell_analysis.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run bin2cell analysis on pseudo Visium HD data'
    )

    # Required arguments
    parser.add_argument(
        '--pseudo_hd_dir',
        type=str,
        required=True,
        help='Path to pseudo Visium HD directory (containing filtered_feature_bc_matrix and spatial folders)'
    )

    parser.add_argument(
        '--image_path',
        type=str,
        required=True,
        help='Path to cropped H&E tissue image (e.g., cropped_visium_hd_human_colorectal.png)'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for results'
    )

    # Optional arguments
    parser.add_argument(
        '--dataset_name',
        type=str,
        default='sample',
        help='Dataset name for output files (default: sample)'
    )

    parser.add_argument(
        '--mpp',
        type=float,
        default=0.5,
        help='Microns per pixel for image scaling (default: 0.5)'
    )

    parser.add_argument(
        '--prob_thresh_he',
        type=float,
        default=0.4,
        help='Probability threshold for H&E StarDist segmentation (default: 0.01, lower=more nuclei)'
    )

    parser.add_argument(
        '--prob_thresh_gex',
        type=float,
        default=0.05,
        help='Probability threshold for GEX StarDist segmentation (default: 0.05, lower=more cells)'
    )

    parser.add_argument(
        '--nms_thresh_gex',
        type=float,
        default=0.5,
        help='NMS threshold for GEX StarDist segmentation (default: 0.5, higher=less merging)'
    )

    parser.add_argument(
        '--max_bin_distance',
        type=int,
        default=5,
        help='Maximum bin distance for label expansion (default: 2, ~4μm cytoplasm radius)'
    )

    parser.add_argument(
        '--gaussian_sigma',
        type=float,
        default=5.0,
        help='Gaussian filter sigma for GEX image smoothing (default: 5.0, higher=smoother)'
    )

    parser.add_argument(
        '--ppo_eval_run_dir',
        type=str,
        default=None,
        help='Optional PPO evaluation run directory containing per_episode.csv and config_used.yaml. '
             'If provided, evaluate bin2cell on the exact same episode cell set.'
    )

    parser.add_argument(
        '--eval_run_name',
        type=str,
        default='human_colorectal_bin2cell_eval',
        help='Run-name prefix for PPO-format bin2cell evaluation outputs'
    )

    parser.add_argument(
        '--eval_output_root',
        type=str,
        default='runs',
        help='Root directory for PPO-format bin2cell evaluation outputs'
    )

    parser.add_argument(
        '--overlay_max_cells',
        type=int,
        default=300,
        help='Number of overlay PNGs to write for PPO-format bin2cell evaluation'
    )

    parser.add_argument(
        '--overlay_selection',
        type=str,
        choices=('first', 'random', 'best_iou', 'worst_iou'),
        default='first',
        help='How to choose per-cell overlays for PPO-format bin2cell evaluation'
    )

    parser.add_argument(
        '--eval_seed',
        type=int,
        default=7,
        help='Seed for PPO-format bin2cell evaluation overlay selection'
    )

    parser.add_argument(
        '--pred_min_nuclear_overlap_frac',
        type=float,
        default=0.3,
        help='Minimum fraction of PPO episode nuclear bins needed to accept a bin2cell match'
    )

    parser.add_argument(
        '--pred_min_nuclear_overlap_bins',
        type=int,
        default=2,
        help='Minimum number of overlapping nuclear bins needed to accept a bin2cell match'
    )

    parser.add_argument(
        '--gt_cell_bins_path',
        type=str,
        default=None,
        help='Optional GT full-cell bin table used for PPO-format IoU/Dice evaluation'
    )

    parser.add_argument(
        '--gt_nuclear_bins_path',
        type=str,
        default=None,
        help='Optional GT nuclear-bin table used for PPO-format GT matching'
    )

    parser.add_argument(
        '--gt_min_nuclear_overlap_frac',
        type=float,
        default=0.3,
        help='Minimum fraction of PPO episode nuclear bins needed to accept a GT match'
    )

    parser.add_argument(
        '--gt_min_nuclear_overlap_bins',
        type=int,
        default=2,
        help='Minimum number of overlapping nuclear bins needed to accept a GT match'
    )

    parser.add_argument(
        '--external_nuclear_bins_path',
        type=str,
        default=None,
        help='Optional external nuclear-bin CSV/CSV.GZ used to directly populate adata.obs["labels_he"]. '
             'When provided, bin2cell skips its own H&E nuclear segmentation and uses these labels instead.'
    )

    return parser.parse_args()


def get_paths(args):
    """Get all file paths from arguments."""
    # Convert to Path objects
    pseudo_hd_dir = Path(args.pseudo_hd_dir)
    image_path = Path(args.image_path)
    output_dir = Path(args.output_dir)
    dataset_name = args.dataset_name

    # Validate input paths exist
    if not pseudo_hd_dir.exists():
        raise FileNotFoundError(f"Pseudo HD directory not found: {pseudo_hd_dir}")
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Create output directory
    output_dir.mkdir(exist_ok=True, parents=True)

    # Create subdirectory for StarDist outputs
    stardist_dir = output_dir / 'stardist'
    stardist_dir.mkdir(exist_ok=True)

    # Create subdirectory for figures
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)

    paths = {
        'pseudo_hd_dir': pseudo_hd_dir,
        'image_path': image_path,
        'output_dir': output_dir,
        'stardist_dir': stardist_dir,
        'figures_dir': figures_dir,
        'dataset_name': dataset_name
    }

    logger.info(f"Paths configuration:")
    for key, value in paths.items():
        logger.info(f"  {key}: {value}")

    return paths
def run_ppo_format_bin2cell_evaluation(
    *,
    assignments_csv: Path,
    args: argparse.Namespace,
) -> Path:
    nuclear_source = (
        "external_ppo_aligned"
        if args.external_nuclear_bins_path is not None
        else "bin2cell_he_stardist"
    )
    pipeline_config = {
        "pseudo_hd_dir": str(Path(str(args.pseudo_hd_dir)).expanduser().resolve()),
        "image_path": str(Path(str(args.image_path)).expanduser().resolve()),
        "dataset_name": str(args.dataset_name),
        "nuclear_source": nuclear_source,
        "external_nuclear_bins_path": None if args.external_nuclear_bins_path is None else str(Path(str(args.external_nuclear_bins_path)).expanduser().resolve()),
        "mpp": float(args.mpp),
        "prob_thresh_he": float(args.prob_thresh_he),
        "prob_thresh_gex": float(args.prob_thresh_gex),
        "nms_thresh_gex": float(args.nms_thresh_gex),
        "max_bin_distance": int(args.max_bin_distance),
        "gaussian_sigma": float(args.gaussian_sigma),
    }
    return run_ppo_format_assignment_evaluation(
        assignments_csv=assignments_csv,
        method_name="bin2cell",
        method_label="Bin2Cell",
        nuclear_source=nuclear_source,
        external_nuclear_bins_path=None if args.external_nuclear_bins_path is None else Path(str(args.external_nuclear_bins_path)).expanduser().resolve(),
        args=args,
        pipeline_config=pipeline_config,
    )


def load_visium_data(pseudo_hd_dir, image_path):
    """
    Load Visium HD data using bin2cell's loader.
    Handles both H5 and MTX matrix formats.

    Args:
        pseudo_hd_dir: Path to pseudo Visium HD output directory
        image_path: Path to cropped H&E image

    Returns:
        adata: AnnData object with spatial coordinates
    """
    logger.info(f"Loading Visium HD data from {pseudo_hd_dir}")
    logger.info(f"Using H&E image: {image_path}")

    # Check if H5 file exists, otherwise use MTX format
    pseudo_hd_dir = Path(pseudo_hd_dir)
    h5_file = pseudo_hd_dir / "filtered_feature_bc_matrix.h5"
    mtx_dir = pseudo_hd_dir / "filtered_feature_bc_matrix"

    if h5_file.exists():
        logger.info("  Loading from H5 format...")
        adata = b2c.read_visium(
            str(pseudo_hd_dir),
            source_image_path=str(image_path)
        )
    elif mtx_dir.exists():
        logger.info("  Loading from MTX format...")
        # Load using scanpy's MTX reader
        adata = sc.read_10x_mtx(str(mtx_dir), var_names='gene_symbols', cache=True)

        # Load spatial data manually
        spatial_dir = pseudo_hd_dir / "spatial"
        tissue_positions_file = spatial_dir / "tissue_positions.parquet"

        if tissue_positions_file.exists():
            logger.info("  Loading spatial coordinates from Parquet...")
            import pandas as pd
            tissue_positions = pd.read_parquet(tissue_positions_file)

            # Match barcodes between expression and spatial data
            common_barcodes = adata.obs_names.intersection(tissue_positions['barcode'])
            adata = adata[common_barcodes].copy()
            tissue_positions = tissue_positions.set_index('barcode').loc[common_barcodes]

            # Add spatial information to adata
            adata.obs['in_tissue'] = tissue_positions['in_tissue'].values
            adata.obs['array_row'] = tissue_positions['array_row'].values
            adata.obs['array_col'] = tissue_positions['array_col'].values

            # Add spatial coordinates to obsm
            adata.obsm['spatial'] = tissue_positions[['pxl_row_in_fullres', 'pxl_col_in_fullres']].values

            # Load and add image to uns
            from PIL import Image
            img = np.array(Image.open(image_path))

            # Load scalefactors if available
            scalefactors_file = spatial_dir / "scalefactors_json.json"
            if scalefactors_file.exists():
                import json
                with open(scalefactors_file, 'r') as f:
                    scalefactors = json.load(f)
            else:
                scalefactors = {
                    'tissue_hires_scalef': 1.0,
                    'spot_diameter_fullres': 2.0
                }

            # Treat the provided pseudo Visium HD image as the full resolution
            # morphology image so bin2cell's dimension checks stay consistent.
            scalefactors['tissue_hires_scalef'] = 1.0

            # Create spatial structure matching bin2cell expectations
            adata.uns['spatial'] = {
                'VisiumHD': {
                    'images': {'hires': img},
                    'scalefactors': scalefactors,
                    'metadata': {
                        'source_image_path': str(image_path)
                    }
                }
            }

            logger.info("  Spatial data loaded successfully")
        else:
            raise FileNotFoundError(f"Spatial positions file not found: {tissue_positions_file}")
    else:
        raise FileNotFoundError(f"Neither H5 nor MTX format found in {pseudo_hd_dir}")

    # Make variable names unique
    adata.var_names_make_unique()

    logger.info(f"Loaded AnnData object:")
    logger.info(f"  n_obs (bins): {adata.n_obs}")
    logger.info(f"  n_vars (genes): {adata.n_vars}")
    logger.info(f"  obs columns: {list(adata.obs.columns)}")
    logger.info(f"  obsm keys: {list(adata.obsm.keys())}")

    return adata


def filter_data(adata):
    """
    Filter genes and cells with minimal QC.

    Args:
        adata: AnnData object

    Returns:
        adata: Filtered AnnData object
    """
    logger.info("Filtering data...")

    # Filter genes: require genes to show up in at least 3 spots
    logger.info(f"  Before filtering: {adata.n_vars} genes")
    sc.pp.filter_genes(adata, min_cells=3)
    logger.info(f"  After gene filtering: {adata.n_vars} genes")

    # Filter cells: require spots to have any information at all
    logger.info(f"  Before filtering: {adata.n_obs} bins")
    sc.pp.filter_cells(adata, min_counts=1)
    logger.info(f"  After bin filtering: {adata.n_obs} bins")

    return adata


def destripe_data(adata):
    """
    Correct for variable bin dimensions using destriping.

    Args:
        adata: AnnData object
    """
    logger.info("Performing destriping to correct for variable bin dimensions...")

    # Destripe and adjust counts
    b2c.destripe(adata, adjust_counts=True)

    logger.info("  Destriping complete")
    logger.info(f"  Added 'n_counts_adjusted' to adata.obs")


def generate_scaled_he_image(adata, mpp, stardist_dir):
    """
    Generate scaled H&E image at specified resolution.

    Args:
        adata: AnnData object
        mpp: Microns per pixel
        stardist_dir: Directory to save scaled image

    Returns:
        he_image_path: Path to saved H&E image
    """
    logger.info(f"Generating scaled H&E image at {mpp} mpp...")

    he_image_path = stardist_dir / "he.tiff"

    # Generate and save scaled H&E image
    b2c.scaled_he_image(
        adata,
        mpp=mpp,
        save_path=str(he_image_path)
    )

    logger.info(f"  Scaled H&E image saved to {he_image_path}")
    logger.info(f"  Added 'spatial_cropped' to adata.obsm")
    logger.info(f"  Added '{mpp}_mpp' to adata.uns['spatial']")

    return he_image_path


def segment_he(he_image_path, stardist_dir, prob_thresh):
    """
    Perform StarDist segmentation on H&E image (nuclei).

    Args:
        he_image_path: Path to H&E image
        stardist_dir: Directory to save segmentation results
        prob_thresh: Probability threshold for segmentation

    Returns:
        he_labels_path: Path to segmentation labels
    """
    logger.info("Performing H&E (nuclear) segmentation with StarDist...")
    logger.info(f"  Using model: 2D_versatile_he")
    logger.info(f"  Probability threshold: {prob_thresh}")

    he_labels_path = stardist_dir / "he.npz"

    # Run StarDist segmentation
    b2c.stardist(
        image_path=str(he_image_path),
        labels_npz_path=str(he_labels_path),
        stardist_model="2D_versatile_he",
        prob_thresh=prob_thresh
    )

    logger.info(f"  H&E segmentation complete, saved to {he_labels_path}")

    return he_labels_path


def insert_he_labels(adata, he_labels_path, mpp):
    """
    Insert H&E segmentation labels into AnnData object.

    Args:
        adata: AnnData object
        he_labels_path: Path to H&E segmentation labels
        mpp: Microns per pixel
    """
    logger.info("Inserting H&E labels into AnnData object...")

    # Check which spatial key to use - look for cropped versions first
    # b2c.scaled_he_image() creates spatial_cropped_XXX_buffer by default
    available_keys = list(adata.obsm.keys())
    logger.info(f"  Available spatial keys: {available_keys}")

    # Find the cropped spatial key (could be spatial_cropped_150_buffer or similar)
    spatial_key = None
    for key in available_keys:
        if 'spatial_cropped' in key:
            spatial_key = key
            logger.info(f"  Using cropped spatial key: '{spatial_key}'")
            break

    if spatial_key is None:
        spatial_key = 'spatial'
        logger.info(f"  No cropped spatial key found, using 'spatial'")

    b2c.insert_labels(
        adata,
        labels_npz_path=str(he_labels_path),
        basis="spatial",
        spatial_key=spatial_key,
        mpp=mpp,
        labels_key="labels_he"
    )

    n_labeled = (adata.obs['labels_he'] > 0).sum()
    logger.info(f"  Labeled {n_labeled} bins with H&E segmentation")
    logger.info(f"  Added 'labels_he' to adata.obs")

    if n_labeled == 0:
        logger.warning("  WARNING: No bins were labeled!")
        logger.warning(f"  This may indicate a coordinate mismatch between image and spatial data")


def insert_external_nuclear_labels(adata, external_nuclear_bins_path):
    """
    Populate adata.obs['labels_he'] directly from an external nuclear-bin table.

    Expected columns:
      - barcode
      - cell_id
      - is_nuclear
    """
    external_path = Path(external_nuclear_bins_path).expanduser().resolve()
    logger.info("Injecting external nuclear labels instead of running H&E nuclear segmentation...")
    logger.info(f"  External nuclear bins path: {external_path}")
    if not external_path.exists():
        raise FileNotFoundError(f"External nuclear bins file not found: {external_path}")

    usecols = ["barcode", "cell_id", "is_nuclear"]
    external = pd.read_csv(external_path, usecols=usecols, compression="infer")
    external = external.dropna(subset=["barcode", "cell_id"]).copy()
    external["barcode"] = external["barcode"].astype(str)
    external["cell_id"] = external["cell_id"].map(normalize_cell_id)
    external["is_nuclear"] = coerce_bool_series(external["is_nuclear"]).fillna(False)
    external = external.loc[external["cell_id"].notna() & external["is_nuclear"]].copy()
    if external.empty:
        raise ValueError(f"No nuclear rows found in external nuclear bins file: {external_path}")

    label_map = {}
    for row in external.itertuples(index=False):
        try:
            label_map[str(row.barcode)] = int(float(row.cell_id))
        except Exception as exc:
            raise ValueError(f"Failed to coerce external cell_id {row.cell_id!r} to int for barcode {row.barcode!r}") from exc

    labels_he = np.zeros((adata.n_obs,), dtype=np.int32)
    obs_barcodes = pd.Index(adata.obs_names.astype(str))
    obs_positions = pd.Series(np.arange(adata.n_obs, dtype=np.int64), index=obs_barcodes)
    overlap = external.loc[external["barcode"].isin(obs_positions.index), ["barcode", "cell_id"]].copy()
    if overlap.empty:
        raise ValueError(
            "External nuclear bins had zero barcode overlap with loaded pseudo HD matrix. "
            "Check that you are using the same square_002um barcode space as PPO."
        )
    overlap["cell_id_int"] = overlap["cell_id"].astype(float).astype(np.int64)
    for barcode, cell_id_int in zip(overlap["barcode"].tolist(), overlap["cell_id_int"].tolist()):
        labels_he[int(obs_positions[str(barcode)])] = int(cell_id_int)

    adata.obs["labels_he"] = pd.Series(labels_he, index=adata.obs_names, dtype=np.int32)
    n_labeled = int((labels_he > 0).sum())
    logger.info("  Injected external labels_he for %d bins", n_labeled)
    logger.info("  Unique external nuclear cells overlapping matrix: %d", int(pd.Series(labels_he).loc[lambda s: s > 0].nunique()))


def expand_he_labels(adata, max_bin_distance):
    """
    Expand nuclear labels to include cytoplasm.

    Args:
        adata: AnnData object
        max_bin_distance: Maximum bin distance for expansion
    """
    logger.info(f"Expanding H&E labels to include cytoplasm...")
    logger.info(f"  Max bin distance: {max_bin_distance}")

    b2c.expand_labels(
        adata,
        labels_key='labels_he',
        expanded_labels_key="labels_he_expanded",
        max_bin_distance=max_bin_distance
    )

    n_labeled = (adata.obs['labels_he_expanded'] > 0).sum()
    logger.info(f"  Labeled {n_labeled} bins after expansion")
    logger.info(f"  Added 'labels_he_expanded' to adata.obs")


def generate_gex_image(adata, mpp, gaussian_sigma, stardist_dir):
    """
    Generate gene expression image for segmentation.

    Args:
        adata: AnnData object
        mpp: Microns per pixel
        gaussian_sigma: Sigma for Gaussian smoothing
        stardist_dir: Directory to save image

    Returns:
        gex_image_path: Path to GEX image
    """
    logger.info(f"Generating gene expression image...")
    logger.info(f"  Using adjusted counts with Gaussian smoothing (sigma={gaussian_sigma})")

    gex_image_path = stardist_dir / "gex.tiff"

    # Create GEX image using bin2cell's built-in save_path
    # This ensures proper image format handling
    b2c.grid_image(adata, "n_counts_adjusted", mpp=mpp, sigma=gaussian_sigma, save_path=str(gex_image_path))

    logger.info(f"  GEX image saved to {gex_image_path}")

    return gex_image_path


def segment_gex(gex_image_path, stardist_dir, prob_thresh, nms_thresh):
    """
    Perform StarDist segmentation on gene expression image (cells).

    Args:
        gex_image_path: Path to GEX image
        stardist_dir: Directory to save segmentation results
        prob_thresh: Probability threshold for segmentation
        nms_thresh: NMS threshold for segmentation

    Returns:
        gex_labels_path: Path to segmentation labels
    """
    logger.info("Performing GEX (cell) segmentation with StarDist...")
    logger.info(f"  Using model: 2D_versatile_fluo")
    logger.info(f"  Probability threshold: {prob_thresh}")
    logger.info(f"  NMS threshold: {nms_thresh}")

    gex_labels_path = stardist_dir / "gex.npz"

    # Run StarDist segmentation
    b2c.stardist(
        image_path=str(gex_image_path),
        labels_npz_path=str(gex_labels_path),
        stardist_model="2D_versatile_fluo",
        prob_thresh=prob_thresh,
        nms_thresh=nms_thresh
    )

    logger.info(f"  GEX segmentation complete, saved to {gex_labels_path}")

    return gex_labels_path


def insert_gex_labels(adata, gex_labels_path, mpp):
    """
    Insert GEX segmentation labels into AnnData object.

    Args:
        adata: AnnData object
        gex_labels_path: Path to GEX segmentation labels
        mpp: Microns per pixel
    """
    logger.info("Inserting GEX labels into AnnData object...")

    b2c.insert_labels(
        adata,
        labels_npz_path=str(gex_labels_path),
        basis="array",
        mpp=mpp,
        labels_key="labels_gex"
    )

    n_labeled = (adata.obs['labels_gex'] > 0).sum()
    logger.info(f"  Labeled {n_labeled} bins with GEX segmentation")
    logger.info(f"  Added 'labels_gex' to adata.obs")


def combine_labels(adata):
    """
    Combine H&E and GEX labels, filling gaps with GEX calls.

    Args:
        adata: AnnData object
    """
    logger.info("Combining H&E and GEX labels...")
    logger.info("  Using H&E as primary, GEX to fill gaps")

    b2c.salvage_secondary_labels(
        adata,
        primary_label="labels_he_expanded",
        secondary_label="labels_gex",
        labels_key="labels_joint"
    )

    n_labeled = (adata.obs['labels_joint'] > 0).sum()
    n_he = (adata.obs['labels_joint_source'] == 'primary').sum()
    n_gex = (adata.obs['labels_joint_source'] == 'secondary').sum()

    logger.info(f"  Total labeled bins: {n_labeled}")
    logger.info(f"    From H&E: {n_he}")
    logger.info(f"    From GEX: {n_gex}")
    logger.info(f"  Added 'labels_joint' and 'labels_joint_source' to adata.obs")


def bin_to_cell_conversion(adata):
    """
    Convert bins to cells by grouping based on labels.

    Args:
        adata: AnnData object with bin-level data

    Returns:
        cdata: AnnData object with cell-level data
    """
    logger.info("Converting bins to cells...")

    # Determine which spatial keys are available
    spatial_keys = ["spatial"]
    available_keys = list(adata.obsm.keys())

    # Find any cropped spatial key
    for key in available_keys:
        if 'spatial_cropped' in key:
            spatial_keys.append(key)
            logger.info(f"  Using spatial keys: {spatial_keys}")
            break

    if len(spatial_keys) == 1:
        logger.info(f"  Using only 'spatial' coordinates")

    cdata = b2c.bin_to_cell(
        adata,
        labels_key="labels_joint",
        spatial_keys=spatial_keys
    )

    logger.info(f"Cell-level data created:")
    logger.info(f"  n_obs (cells): {cdata.n_obs}")
    logger.info(f"  n_vars (genes): {cdata.n_vars}")
    logger.info(f"  obs columns: {list(cdata.obs.columns)}")

    return cdata


def create_visualizations(adata, cdata, mpp, figures_dir):
    """
    Create visualizations of the results.

    Args:
        adata: Bin-level AnnData object
        cdata: Cell-level AnnData object
        mpp: Microns per pixel
        figures_dir: Directory to save figures
    """
    logger.info("Creating visualizations...")

    # Set up matplotlib parameters and configure scanpy to save to correct directory
    sc.set_figure_params(figsize=[10, 10], dpi=100)
    sc.settings.figdir = figures_dir  # Set scanpy figure directory to our output folder
    logger.info(f"  Scanpy figure directory set to: {figures_dir}")

    # Find the correct spatial and image keys (created by b2c.scaled_he_image)
    # Demo uses: basis="spatial_cropped_150_buffer", img_key="0.5_mpp_150_buffer"
    spatial_basis = None
    img_key = None

    # Find spatial_cropped key
    for key in adata.obsm.keys():
        if 'spatial_cropped' in key:
            spatial_basis = key
            break

    # Find img_key in uns['spatial']
    if 'spatial' in adata.uns:
        for lib_key in adata.uns['spatial']:
            if 'images' in adata.uns['spatial'][lib_key]:
                for key in adata.uns['spatial'][lib_key]['images']:
                    if 'mpp' in key and 'buffer' in key:
                        img_key = key
                        break

    # Fallback to simple keys if dynamic detection fails
    if spatial_basis is None:
        spatial_basis = 'spatial'
        logger.warning("  Could not find spatial_cropped key, using 'spatial'")
    if img_key is None:
        img_key = f"{mpp}_mpp_150_buffer"
        logger.warning(f"  Could not find img_key, trying '{img_key}'")

    logger.info(f"  Using spatial basis: {spatial_basis}")
    logger.info(f"  Using image key: {img_key}")

    # Select a region for detailed visualization
    # Use a central region
    mask = (
        (adata.obs['array_row'] >= adata.obs['array_row'].quantile(0.4)) &
        (adata.obs['array_row'] <= adata.obs['array_row'].quantile(0.6)) &
        (adata.obs['array_col'] >= adata.obs['array_col'].quantile(0.4)) &
        (adata.obs['array_col'] <= adata.obs['array_col'].quantile(0.6))
    )

    bdata = adata[mask].copy()

    try:
        # 1. Visualize destriping effect
        logger.info("  Creating destriping visualization...")
        bdata_counts = bdata[bdata.obs['n_counts'] > 0].copy()
        sc.pl.spatial(
            bdata_counts,
            color=["n_counts", "n_counts_adjusted"],
            img_key=img_key,
            basis=spatial_basis,
            save='_destriping.pdf',
            cmap='Reds',
            show=False
        )

        # 2. Visualize H&E labels
        logger.info("  Creating H&E labels visualization...")
        bdata_he = bdata[bdata.obs['labels_he_expanded'] > 0].copy()
        if len(bdata_he) > 0:
            bdata_he.obs['labels_he_expanded'] = bdata_he.obs['labels_he_expanded'].astype(str)
            sc.pl.spatial(
                bdata_he,
                color="labels_he_expanded",
                img_key=img_key,
                basis=spatial_basis,
                save='_he_labels.pdf',
                show=False,
                legend_loc=None
            )

        # 3. Visualize joint labels
        logger.info("  Creating joint labels visualization...")
        bdata_joint = bdata[bdata.obs['labels_joint'] > 0].copy()
        if len(bdata_joint) > 0:
            bdata_joint.obs['labels_joint'] = bdata_joint.obs['labels_joint'].astype(str)
            sc.pl.spatial(
                bdata_joint,
                color=["labels_joint_source", "labels_joint"],
                img_key=img_key,
                basis=spatial_basis,
                save='_joint_labels.pdf',
                show=False,
                legend_loc=None
            )

        # 4. Visualize cells
        logger.info("  Creating cell-level visualization...")
        cell_mask = (
            (cdata.obs['array_row'] >= cdata.obs['array_row'].quantile(0.4)) &
            (cdata.obs['array_row'] <= cdata.obs['array_row'].quantile(0.6)) &
            (cdata.obs['array_col'] >= cdata.obs['array_col'].quantile(0.4)) &
            (cdata.obs['array_col'] <= cdata.obs['array_col'].quantile(0.6))
        )

        ddata = cdata[cell_mask].copy()
        if len(ddata) > 0:
            sc.set_figure_params(fontsize=20, figsize=[7, 7])
            sc.pl.spatial(
                ddata,
                color=["bin_count", "labels_joint_source"],
                img_key=img_key,
                basis=spatial_basis,
                s=4,
                save='_cells.pdf',
                show=False
            )

        logger.info(f"  Figures saved to {figures_dir}")

    except Exception as e:
        logger.warning(f"  Error creating visualizations: {e}")
        logger.warning("  Continuing without visualizations...")


def save_nuclear_assignments(adata, output_dir, dataset_name):
    """
    Save nuclear bin assignments for each cell (similar to SMURF format).
    This allows for easy validation against ground truth.

    Args:
        adata: Bin-level AnnData object with labels
        output_dir: Output directory
        dataset_name: Name of dataset
    """
    logger.info("Saving nuclear bin assignments for validation...")

    import pandas as pd

    # Get bins with nuclear labels (before expansion)
    nuclear_bins = adata.obs[adata.obs['labels_he'] > 0].copy()

    # Create assignment dataframe
    assignments = pd.DataFrame({
        'barcode': nuclear_bins.index,
        'cell_id': nuclear_bins['labels_he'].astype(int),
        'array_row': nuclear_bins['array_row'].astype(int),
        'array_col': nuclear_bins['array_col'].astype(int),
        'is_nuclear': True,  # All these are nuclear bins
        'segmentation_source': 'H&E'  # From H&E segmentation
    })

    # Add expanded cytoplasm bins (non-nuclear)
    expanded_bins = adata.obs[
        (adata.obs['labels_he_expanded'] > 0) &
        (adata.obs['labels_he'] == 0)
    ].copy()

    if len(expanded_bins) > 0:
        expanded_assignments = pd.DataFrame({
            'barcode': expanded_bins.index,
            'cell_id': expanded_bins['labels_he_expanded'].astype(int),
            'array_row': expanded_bins['array_row'].astype(int),
            'array_col': expanded_bins['array_col'].astype(int),
            'is_nuclear': False,  # These are cytoplasm bins
            'segmentation_source': 'H&E_expanded'
        })

        assignments = pd.concat([assignments, expanded_assignments], ignore_index=True)

    # Add GEX-derived bins if any
    gex_bins = adata.obs[
        (adata.obs['labels_joint_source'] == 'secondary') &
        (adata.obs['labels_gex'] > 0)
    ].copy()

    if len(gex_bins) > 0:
        gex_assignments = pd.DataFrame({
            'barcode': gex_bins.index,
            'cell_id': gex_bins['labels_gex'].astype(int),
            'array_row': gex_bins['array_row'].astype(int),
            'array_col': gex_bins['array_col'].astype(int),
            'is_nuclear': False,  # GEX doesn't distinguish nuclear
            'segmentation_source': 'GEX'
        })

        assignments = pd.concat([assignments, gex_assignments], ignore_index=True)

    # Sort by cell_id and barcode
    assignments = assignments.sort_values(['cell_id', 'barcode'])

    # Save to CSV
    assignment_path = output_dir / f'{dataset_name}_bin2cell_assignments.csv'
    assignments.to_csv(assignment_path, index=False)
    logger.info(f"  Bin assignments saved to {assignment_path}")
    logger.info(f"    Total bins assigned: {len(assignments)}")
    logger.info(f"    Nuclear bins: {assignments['is_nuclear'].sum()}")
    logger.info(f"    Cytoplasm bins: {(~assignments['is_nuclear']).sum()}")
    logger.info(f"    Unique cells: {assignments['cell_id'].nunique()}")

    # Also save a summary per cell (like SMURF's cell-level output)
    cell_summary = assignments.groupby('cell_id').agg({
        'barcode': 'count',  # Total bins per cell
        'is_nuclear': 'sum',  # Nuclear bins per cell
        'array_row': 'mean',  # Centroid row
        'array_col': 'mean',  # Centroid col
        'segmentation_source': lambda x: x.mode()[0] if len(x) > 0 else 'unknown'  # Primary source
    }).rename(columns={'barcode': 'total_bins', 'is_nuclear': 'nuclear_bins'})

    cell_summary['cytoplasm_bins'] = cell_summary['total_bins'] - cell_summary['nuclear_bins']
    cell_summary['nuclear_ratio'] = cell_summary['nuclear_bins'] / cell_summary['total_bins']

    # Save cell summary
    summary_path = output_dir / f'{dataset_name}_bin2cell_cell_summary.csv'
    cell_summary.to_csv(summary_path)
    logger.info(f"  Cell summary saved to {summary_path}")

    return assignments, cell_summary, assignment_path, summary_path


def save_results(adata, cdata, output_dir, dataset_name):
    """
    Save results to disk.

    Args:
        adata: Bin-level AnnData object
        cdata: Cell-level AnnData object
        output_dir: Output directory
        dataset_name: Name of dataset
    """
    logger.info("Saving results...")

    # Save bin-level data
    bin_path = output_dir / f'{dataset_name}_bin2cell_2um.h5ad'
    adata.write_h5ad(bin_path)
    logger.info(f"  Bin-level data saved to {bin_path}")

    # Save cell-level data
    cell_path = output_dir / f'{dataset_name}_bin2cell_cells.h5ad'
    cdata.write_h5ad(cell_path)
    logger.info(f"  Cell-level data saved to {cell_path}")

    # Save nuclear bin assignments for validation
    _, _, assignment_path, cell_summary_path = save_nuclear_assignments(adata, output_dir, dataset_name)
    return {
        'bin_h5ad_path': bin_path,
        'cell_h5ad_path': cell_path,
        'assignments_csv_path': assignment_path,
        'cell_summary_csv_path': cell_summary_path,
    }


def main():
    """Main function."""
    args = parse_args()
    if args.overlay_max_cells < 0:
        raise ValueError('--overlay_max_cells must be >= 0')
    if args.pred_min_nuclear_overlap_frac < 0 or args.pred_min_nuclear_overlap_frac > 1:
        raise ValueError('--pred_min_nuclear_overlap_frac must be in [0, 1]')
    if args.gt_min_nuclear_overlap_frac < 0 or args.gt_min_nuclear_overlap_frac > 1:
        raise ValueError('--gt_min_nuclear_overlap_frac must be in [0, 1]')
    if args.pred_min_nuclear_overlap_bins < 0:
        raise ValueError('--pred_min_nuclear_overlap_bins must be >= 0')
    if args.gt_min_nuclear_overlap_bins < 0:
        raise ValueError('--gt_min_nuclear_overlap_bins must be >= 0')

    logger.info("="*80)
    logger.info("Starting bin2cell analysis")
    logger.info("="*80)
    logger.info(f"Dataset name: {args.dataset_name}")
    logger.info(f"Input paths:")
    logger.info(f"  Pseudo HD dir: {args.pseudo_hd_dir}")
    logger.info(f"  Image: {args.image_path}")
    logger.info(f"  Output: {args.output_dir}")
    logger.info(f"Parameters:")
    logger.info(f"  mpp: {args.mpp}")
    logger.info(f"  prob_thresh_he: {args.prob_thresh_he}")
    logger.info(f"  prob_thresh_gex: {args.prob_thresh_gex}")
    logger.info(f"  nms_thresh_gex: {args.nms_thresh_gex}")
    logger.info(f"  max_bin_distance: {args.max_bin_distance}")
    logger.info(f"  gaussian_sigma: {args.gaussian_sigma}")
    logger.info(f"  external_nuclear_bins_path: {args.external_nuclear_bins_path}")
    logger.info(f"  ppo_eval_run_dir: {args.ppo_eval_run_dir}")
    logger.info(f"  eval_run_name: {args.eval_run_name}")
    logger.info(f"  eval_output_root: {args.eval_output_root}")
    logger.info("="*80)

    # Get paths
    paths = get_paths(args)

    # Load data
    adata = load_visium_data(paths['pseudo_hd_dir'], paths['image_path'])

    # Filter data
    adata = filter_data(adata)

    # Destripe data
    destripe_data(adata)

    # Populate nuclear labels either from external PPO-aligned labels or from bin2cell's own H&E segmentation.
    if args.external_nuclear_bins_path is not None:
        logger.info("External nuclear labels provided; skipping scaled H&E generation and H&E nuclear segmentation.")
        insert_external_nuclear_labels(adata, args.external_nuclear_bins_path)
    else:
        # Generate scaled H&E image only when bin2cell is responsible for its own nuclear segmentation.
        he_image_path = generate_scaled_he_image(adata, args.mpp, paths['stardist_dir'])
        he_labels_path = segment_he(he_image_path, paths['stardist_dir'], args.prob_thresh_he)
        insert_he_labels(adata, he_labels_path, args.mpp)
    expand_he_labels(adata, args.max_bin_distance)

    # Generate GEX image and perform segmentation
    gex_image_path = generate_gex_image(adata, args.mpp, args.gaussian_sigma, paths['stardist_dir'])
    gex_labels_path = segment_gex(gex_image_path, paths['stardist_dir'], args.prob_thresh_gex, args.nms_thresh_gex)
    insert_gex_labels(adata, gex_labels_path, args.mpp)

    # Combine labels
    combine_labels(adata)

    # Convert bins to cells
    cdata = bin_to_cell_conversion(adata)

    # Create visualizations
    create_visualizations(adata, cdata, args.mpp, paths['figures_dir'])

    # Save results
    saved_paths = save_results(adata, cdata, paths['output_dir'], paths['dataset_name'])
    nuclear_source = (
        "external_ppo_aligned"
        if args.external_nuclear_bins_path is not None
        else "bin2cell_he_stardist"
    )
    pipeline_summary = {
        "method": "bin2cell_pipeline",
        "nuclear_source": nuclear_source,
        "external_nuclear_bins_path": None if args.external_nuclear_bins_path is None else str(Path(str(args.external_nuclear_bins_path)).expanduser().resolve()),
        "pseudo_hd_dir": str(Path(args.pseudo_hd_dir).expanduser().resolve()),
        "image_path": str(Path(args.image_path).expanduser().resolve()),
        "output_dir": str(paths['output_dir']),
        "dataset_name": str(args.dataset_name),
        "mpp": float(args.mpp),
        "prob_thresh_he": float(args.prob_thresh_he),
        "prob_thresh_gex": float(args.prob_thresh_gex),
        "nms_thresh_gex": float(args.nms_thresh_gex),
        "max_bin_distance": int(args.max_bin_distance),
        "gaussian_sigma": float(args.gaussian_sigma),
        "bin_h5ad_path": str(saved_paths['bin_h5ad_path']),
        "cell_h5ad_path": str(saved_paths['cell_h5ad_path']),
        "assignments_csv_path": str(saved_paths['assignments_csv_path']),
        "cell_summary_csv_path": str(saved_paths['cell_summary_csv_path']),
    }
    pipeline_summary_path = paths['output_dir'] / f"{paths['dataset_name']}_bin2cell_pipeline_summary.json"
    with pipeline_summary_path.open("w", encoding="utf-8") as handle:
        json.dump(pipeline_summary, handle, indent=2, sort_keys=False)
        handle.write("\n")
    logger.info("  Pipeline summary saved to %s", pipeline_summary_path)

    if args.ppo_eval_run_dir is not None:
        logger.info("Running PPO-format bin2cell evaluation on the exact PPO-eval cell set...")
        eval_run_dir = run_ppo_format_bin2cell_evaluation(
            assignments_csv=Path(saved_paths['assignments_csv_path']).expanduser().resolve(),
            args=args,
        )
        logger.info("PPO-format bin2cell evaluation outputs saved to: %s", eval_run_dir)

    logger.info("="*80)
    logger.info("Bin2cell analysis complete!")
    logger.info("="*80)
    logger.info(f"Results saved to: {paths['output_dir']}")
    logger.info(f"  Bin-level data: {saved_paths['bin_h5ad_path']}")
    logger.info(f"  Cell-level data: {saved_paths['cell_h5ad_path']}")
    logger.info(f"  Assignments CSV: {saved_paths['assignments_csv_path']}")
    logger.info(f"  Cell summary CSV: {saved_paths['cell_summary_csv_path']}")
    logger.info(f"  Nuclear source: {nuclear_source}")
    logger.info(f"  Pipeline summary JSON: {pipeline_summary_path}")
    logger.info(f"  Figures: {paths['figures_dir']}")
    logger.info(f"  StarDist outputs: {paths['stardist_dir']}")
    logger.info("="*80)


if __name__ == '__main__':
    main()
