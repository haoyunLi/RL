#!/usr/bin/env python
"""
Run Cellpose-SAM cell segmentation on Visium HD images.

This script uses Cellpose-SAM to segment cells in large tissue images and
saves the results including pixel-level cell assignments to CSV files.

Usage:
    conda activate ./cellpose
    python cellpose_segmentation.py --img_path <image> --output_dir <output>
"""

import numpy as np
from cellpose import models, core, io, plot
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for HPC
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import logging
import sys
import argparse

# Disable PIL decompression bomb protection for large images
Image.MAX_IMAGE_PIXELS = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cellpose_sam_segmentation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

io.logger_setup()  # Cellpose logging


def match_nuclei_to_cells(cell_masks, nuclear_masks):
    """
    Match nuclear masks to cell masks ensuring 1 cell = 1 nucleus.

    Uses sparse matrix operations for fast computation on large images.

    Args:
        cell_masks: Cell segmentation masks (2D array with cell IDs)
        nuclear_masks: Nuclear segmentation masks (2D array with nucleus IDs)

    Returns:
        nuclear_mask_matched: Binary mask of nuclear pixels matched to cells
    """
    logger.info("Matching nuclei to cells (1 cell = 1 nucleus) using fast sparse matrix method...")

    from scipy.sparse import coo_matrix

    # Flatten masks for vectorized operations
    cell_flat = cell_masks.flatten()
    nuclear_flat = nuclear_masks.flatten()

    # Only consider pixels that have both a cell and nucleus
    mask = (cell_flat > 0) & (nuclear_flat > 0)
    cell_ids = cell_flat[mask]
    nuclear_ids = nuclear_flat[mask]

    logger.info(f"  Processing {len(cell_ids):,} pixels with both cell and nucleus labels...")

    # Create sparse overlap matrix: rows=cells, cols=nuclei, values=overlap counts
    # Use coo_matrix which is efficient for construction
    num_cells = np.max(cell_masks)
    num_nuclei = np.max(nuclear_masks)

    # Count overlaps using sparse matrix
    # Each (cell_id, nuclear_id) pair contributes 1 to the overlap count
    overlap_matrix = coo_matrix(
        (np.ones(len(cell_ids), dtype=np.int32), (cell_ids, nuclear_ids)),
        shape=(num_cells + 1, num_nuclei + 1)
    ).tocsr()  # Convert to CSR for efficient row operations

    logger.info(f"  Built overlap matrix: {num_cells:,} cells × {num_nuclei:,} nuclei")

    # Find best nucleus for ALL cells at once 
    logger.info("  Finding best nucleus for each cell (fully vectorized)...")

    # Step 1: Get the best nucleus for each cell using sparse matrix argmax
    # argmax on axis=1 gives the nucleus with max overlap for each cell
    best_nuclei_indices = np.zeros(num_cells + 1, dtype=np.int32)

    # For CSR matrix, we can efficiently extract argmax per row
    for cell_id in range(1, num_cells + 1):
        row = overlap_matrix[cell_id]
        if row.nnz > 0:  # Check if row has any non-zero elements
            best_nuclei_indices[cell_id] = row.indices[np.argmax(row.data)]

    # Count cells with/without nuclei
    cells_with_nucleus = np.sum(best_nuclei_indices > 0)
    cells_without_nucleus = num_cells - cells_with_nucleus

    logger.info(f"  Cells with nucleus: {cells_with_nucleus:,}")
    logger.info(f"  Cells without nucleus: {cells_without_nucleus:,}")

    # Step 2: Create matched nuclear mask using FULLY VECTORIZED operations 
    logger.info("  Creating matched nuclear mask (single pass over all pixels)...")

    # For each pixel, get its cell's best nucleus ID
    best_nucleus_per_pixel = best_nuclei_indices[cell_masks]

    # CORRECT LOGIC: Mark pixels that are BOTH:
    # (1) Part of a nucleus (nuclear_masks > 0), AND
    # (2) That nucleus is the best match for the cell (nuclear_masks == best_nucleus_per_pixel)
    nuclear_mask_matched = ((nuclear_masks > 0) & (nuclear_masks == best_nucleus_per_pixel)).astype(np.uint8)

    logger.info(f"  Total nuclear pixels (matched): {nuclear_mask_matched.sum():,}")

    return nuclear_mask_matched


def save_pixel_to_cell_csv(masks, nuclear_masks, output_dir, image_name):
    """
    Save pixel-level cell assignments to CSV files.

    Args:
        masks: Cell segmentation masks from Cellpose (2D array with cell IDs)
        nuclear_masks: Nuclear segmentation masks from Cellpose (2D array with nucleus IDs)
        output_dir: Directory to save CSV files
        image_name: Name of the image (for naming output files)

    Returns:
        Paths to the saved CSV files and matched nuclear mask
    """
    logger.info("Creating pixel-to-cell mapping CSV files...")

    height, width = masks.shape
    logger.info(f"  Image dimensions: {width} x {height}")
    logger.info(f"  Total pixels: {width * height:,}")

    # Create coordinate grids
    y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')

    # Flatten arrays
    x_flat = x_coords.flatten()
    y_flat = y_coords.flatten()
    cell_id_flat = masks.flatten()

    # Detect cell boundaries for membrane identification
    logger.info("Detecting cell boundaries (vectorized)...")
    from scipy import ndimage

    # Method: A pixel is a boundary if it belongs to a cell AND has a neighbor with different cell_id
    # Use 4-connectivity for thinner, more precise boundaries
    structure = np.array([[0, 1, 0],
                          [1, 1, 1],
                          [0, 1, 0]])  # 4-connected neighborhood (cross shape)

    # Vectorized approach: compute max and min neighbors for ALL pixels at once
    neighbor_max = ndimage.maximum_filter(masks, footprint=structure)
    neighbor_min = ndimage.minimum_filter(masks, footprint=structure)

    # A pixel is a boundary if:
    # 1. It belongs to a cell (masks > 0)
    # 2. Its max neighbor OR min neighbor is different from itself
    cell_boundaries = ((masks > 0) & ((neighbor_max != masks) | (neighbor_min != masks))).astype(np.uint8)

    logger.info(f"  Boundary pixels detected: {cell_boundaries.sum():,}")

    # DEBUG: Check nuclear masks before matching
    logger.info(f"  Total nuclear mask pixels (from Cellpose): {(nuclear_masks > 0).sum():,}")
    logger.info(f"  Nuclear masks as % of image: {100.0 * (nuclear_masks > 0).sum() / nuclear_masks.size:.1f}%")

    # Match nuclei to cells (1 cell = 1 nucleus)
    nuclear_mask_binary = match_nuclei_to_cells(masks, nuclear_masks)

    # IMPORTANT: Nuclear pixels should ONLY be from interior regions, not boundaries
    # First compute interior (non-boundary pixels inside cells)
    is_interior = ((masks > 0) & (cell_boundaries == 0)).astype(np.uint8)

    logger.info(f"  Interior pixels: {is_interior.sum():,}")
    logger.info(f"  Matched nuclear pixels (before interior filter): {nuclear_mask_binary.sum():,}")

    # Nuclear pixels = matched nuclear regions AND interior (exclude boundaries)
    nuclear_mask_binary = (nuclear_mask_binary & is_interior).astype(np.uint8)

    logger.info(f"  Nuclear pixels (interior only): {nuclear_mask_binary.sum():,}")
    logger.info(f"  Nuclear pixels as % of interior: {100.0 * nuclear_mask_binary.sum() / is_interior.sum():.1f}%")

    # Cytoplasm = interior - nuclear
    is_cytoplasm = (is_interior & (nuclear_mask_binary == 0)).astype(np.uint8)

    logger.info(f"  Cytoplasm pixels: {is_cytoplasm.sum():,}")
    logger.info(f"  Cytoplasm as % of interior: {100.0 * is_cytoplasm.sum() / is_interior.sum():.1f}%")

    # Create full pixel mapping DataFrame
    df_full = pd.DataFrame({
        'x': x_flat,
        'y': y_flat,
        'cell_id': cell_id_flat,
        'is_boundary': cell_boundaries.flatten().astype(np.uint8),
        'is_interior': is_interior.flatten().astype(np.uint8),
        'is_nuclear': nuclear_mask_binary.flatten().astype(np.uint8),
        'is_cytoplasm': is_cytoplasm.flatten().astype(np.uint8),
    })

    # Save full resolution (compressed)
    full_csv_path = output_dir / f'{image_name}_pixel_to_cell_mapping_full.csv.gz'
    df_full.to_csv(full_csv_path, index=False, compression='gzip')
    logger.info(f"Full-resolution mapping saved (compressed): {full_csv_path}")
    logger.info(f"  Total pixels: {len(df_full):,}")
    logger.info(f"  Boundary (membrane) pixels: {df_full['is_boundary'].sum():,}")
    logger.info(f"  Nuclear pixels: {df_full['is_nuclear'].sum():,}")
    logger.info(f"  Cytoplasm pixels: {df_full['is_cytoplasm'].sum():,}")
    logger.info(f"  Interior (total) pixels: {df_full['is_interior'].sum():,}")
    logger.info(f"  Background pixels: {(df_full['cell_id'] == 0).sum():,}")

    # Cell-level summary is optional
    logger.info("Skipping cell-level summary (can be computed from full CSV if needed later)")
    logger.info(f"  Total cells detected: {len(np.unique(masks)) - 1:,}")
    summary_csv_path = None

    # Save masks as numpy array for easy loading
    numpy_path = output_dir / f'{image_name}_cell_masks.npy'
    np.save(numpy_path, masks)
    logger.info(f"Cell masks array saved: {numpy_path}")

    # Save boundary mask
    boundary_path = output_dir / f'{image_name}_boundary_mask.npy'
    np.save(boundary_path, cell_boundaries)
    logger.info(f"Boundary mask array saved: {boundary_path}")

    # Save nuclear binary mask
    nuclear_path = output_dir / f'{image_name}_nuclear_binary_mask.npy'
    np.save(nuclear_path, nuclear_mask_binary)
    logger.info(f"Nuclear binary mask array saved: {nuclear_path}")

    # Save original nuclear segmentation masks
    nuclear_masks_path = output_dir / f'{image_name}_nuclear_masks.npy'
    np.save(nuclear_masks_path, nuclear_masks)
    logger.info(f"Nuclear segmentation masks saved: {nuclear_masks_path}")

    return full_csv_path, summary_csv_path, nuclear_mask_binary


def create_visualization(img, masks, nuclear_masks, output_dir, image_name, downsample_factor=1, nuclear_mask_binary=None):
    """
    Create and save visualization of segmentation results.

    For very large images, this is downsampled to speed up visualization.

    Args:
        img: Original image
        masks: Cell masks from Cellpose
        nuclear_masks: Nuclear segmentation masks from Cellpose
        output_dir: Directory to save outputs
        image_name: Name of the image
        downsample_factor: Factor to downsample by (default 1 for full resolution)
        nuclear_mask_binary: Binary mask of nuclear pixels matched to cells (optional)
    """
    logger.info(f"Creating visualization (downsampled by {downsample_factor}x for speed)...")

    # Get image dimensions
    h, w = masks.shape
    logger.info(f"Original image size: {w}x{h}")

    # Downsample for visualization only
    new_h = h // downsample_factor
    new_w = w // downsample_factor
    logger.info(f"Downsampling to {new_w}x{new_h} for faster visualization...")

    from skimage.transform import resize

    # Downsample image
    if len(img.shape) == 3:
        img_small = resize(img, (new_h, new_w, img.shape[2]), order=1, preserve_range=True, anti_aliasing=True).astype(img.dtype)
    else:
        img_small = resize(img, (new_h, new_w), order=1, preserve_range=True, anti_aliasing=True).astype(img.dtype)

    # Downsample masks (use nearest neighbor to preserve labels)
    masks_small = resize(masks, (new_h, new_w), order=0, preserve_range=True, anti_aliasing=False).astype(masks.dtype)

    # Downsample nuclear masks
    nuclear_masks_small = resize(nuclear_masks, (new_h, new_w), order=0, preserve_range=True, anti_aliasing=False).astype(nuclear_masks.dtype)

    # Downsample binary nuclear mask if provided
    if nuclear_mask_binary is not None:
        nuclear_mask_binary_small = resize(nuclear_mask_binary, (new_h, new_w), order=0, preserve_range=True, anti_aliasing=False).astype(nuclear_mask_binary.dtype)
    else:
        nuclear_mask_binary_small = None

    logger.info("Creating cell and nuclear overlay visualization...")
    from skimage.color import label2rgb

    # Create cell segmentation as colored background
    cell_colored = label2rgb(masks_small, bg_label=0)

    # Create nuclear segmentation overlay
    # Use a different colormap or transparency to show nuclei on top of cells
    nuclear_colored = label2rgb(nuclear_masks_small, bg_label=0)

    # Create overlay: cells (colored) with nuclei (outlined or semi-transparent)
    # Method: Show cell colors, then overlay nuclear regions in a contrasting way

    # Option 1: Overlay nuclei as semi-transparent colored regions
    composite = cell_colored.copy()

    # Create nuclear overlay with partial transparency
    # Blend 70% nuclei color + 30% cell color where nuclei exist
    nuclear_mask_viz = nuclear_masks_small > 0
    composite[nuclear_mask_viz] = 0.3 * cell_colored[nuclear_mask_viz] + 0.7 * nuclear_colored[nuclear_mask_viz]

    overlay_path = output_dir / f'{image_name}_cell_nuclear_overlay_downsampled_{downsample_factor}x.png'
    plt.imsave(overlay_path, composite)
    logger.info(f"Cell+Nuclear overlay saved (downsampled {downsample_factor}x): {overlay_path}")
    logger.info(f"  Resolution: {new_w} x {new_h} pixels")
    logger.info(f"  Visualization shows cells (background colors) with nuclei (overlayed colors)")

    logger.info("Note: Full resolution masks are saved in the TIFF and numpy files")


def main():
    """Run Cellpose-SAM segmentation on the cropped Visium HD brain image."""

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run Cellpose-SAM cell segmentation on Visium HD images')
    parser.add_argument('--img_path', type=str,
                       default='cropped_visium_hd_human_colorectal.png',
                       help='Path to input image')
    parser.add_argument('--output_dir', type=str,
                       default='cellpose_sam_human_colorectal_output',
                       help='Output directory for segmentation results')
    args = parser.parse_args()

    # Set paths from arguments
    img_path = Path(args.img_path)
    output_dir = Path(args.output_dir)

    # Create output directory
    output_dir.mkdir(exist_ok=True)

    # Check if input image exists
    if not img_path.exists():
        logger.error(f"Input image not found: {img_path}")
        raise FileNotFoundError(f"Input image not found: {img_path}")

    logger.info("="*70)
    logger.info("Cellpose-SAM Cell Segmentation")
    logger.info("="*70)
    logger.info(f"Input image: {img_path}")
    logger.info(f"Output directory: {output_dir}")

    # Check GPU availability
    use_gpu = core.use_gpu()
    if use_gpu:
        logger.info("GPU available and will be used for acceleration")
    else:
        logger.warning("GPU not available, using CPU (will be slower)")

    # Load image
    logger.info("Loading image...")
    img = io.imread(img_path)
    logger.info(f"Image loaded: shape={img.shape}, dtype={img.dtype}")

    # Determine channels
    if len(img.shape) == 3:
        logger.info(f"Multi-channel image detected with {img.shape[2]} channels")
        # For RGB/multi-channel images, Cellpose can use:
        # - channels=[0,0] for grayscale (single channel)
        # - channels=[2,3] for cytoplasm in channel 2, nucleus in channel 3
        # - channels=[0,0] with RGB will use all channels
        channels = [0, 0]  # Use all channels (grayscale mode)
    else:
        logger.info("Grayscale image detected")
        channels = [0, 0]

    # Initialize Cellpose cyto3 model (best for brightfield with nuclear detection)
    logger.info("Initializing Cellpose cyto3 model...")
    model = models.CellposeModel(gpu=use_gpu, model_type='cyto3')
    logger.info("Model initialized successfully (Cellpose cyto3 - detects cells and nuclei)")

    # Set segmentation parameters
    flow_threshold = 0.4  # Default: 0.4. Standard quality control for cell shapes
    cellprob_threshold = 0.0  # Default: 0.0. DECREASE to include more pixels = BIGGER cells
    diameter = None 
    tile_norm_blocksize = 0  # Set to 100-200 for inhomogeneous brightness

    logger.info("Running segmentation...")
    logger.info(f"Parameters:")
    logger.info(f"  - flow_threshold: {flow_threshold}")
    logger.info(f"  - cellprob_threshold: {cellprob_threshold}")
    logger.info(f"  - diameter: {diameter} (auto-estimate)")
    logger.info(f"  - tile_norm_blocksize: {tile_norm_blocksize}")
    logger.info(f"  - channels: {channels}")

    # Run Cellpose-SAM segmentation
    masks, flows, styles = model.eval(
        img,
        channels=channels,
        diameter=diameter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        normalize={'tile_norm_blocksize': tile_norm_blocksize}
    )

    logger.info(f"Cell segmentation complete!")
    logger.info(f"  - Number of cells detected: {len(np.unique(masks)) - 1}")  # -1 to exclude background
    logger.info("")

    # Extract nuclear segmentation from cyto3 model
    logger.info("Extracting nuclear segmentation from cyto3 model...")
    logger.info("cyto3 provides nuclear detection built-in for brightfield images")

    # cyto3 returns nuclear predictions in flows[2] (nuclear probability)
    # We need to threshold this to get nuclear masks
    if len(flows) > 2 and flows[2] is not None:
        nuclear_prob = flows[2]
        logger.info(f"  Nuclear probability map shape: {nuclear_prob.shape}")

        # Threshold nuclear probability to create nuclear masks
        from skimage.measure import label
        nuclear_threshold = 0.6 # Higher threshold = smaller nuclei
        nuclear_binary = (nuclear_prob > nuclear_threshold).astype(np.uint8)

        # Label connected components as individual nuclei
        nuclear_masks = label(nuclear_binary, connectivity=2).astype(np.int32)

        logger.info(f"  Nuclear threshold: {nuclear_threshold}")
        logger.info(f"  Number of nuclei detected: {len(np.unique(nuclear_masks)) - 1}")
        logger.info(f"  Nuclear pixels: {(nuclear_masks > 0).sum():,}")
    else:
        logger.warning("  cyto3 model did not return nuclear predictions!")
        logger.warning("  Falling back to intensity-based nuclear detection...")

        # Fallback: use intensity thresholding
        from skimage.filters import threshold_otsu
        if len(img.shape) == 3:
            nuclear_channel = img[:, :, 0]
        else:
            nuclear_channel = img

        threshold = threshold_otsu(nuclear_channel[masks > 0])
        nuclear_binary = ((nuclear_channel > threshold) & (masks > 0)).astype(np.uint8)
        nuclear_masks = label(nuclear_binary, connectivity=2).astype(np.int32)

        logger.info(f"  Number of nuclei detected (fallback): {len(np.unique(nuclear_masks)) - 1}")

    logger.info("")

    # Save results
    logger.info("Saving results...")

    image_name = img_path.stem

    # Save cell masks in multiple formats
    masks_tif_path = output_dir / f'{image_name}_cell_masks.tif'
    io.imsave(masks_tif_path, masks)
    logger.info(f"Cell masks saved (TIFF): {masks_tif_path}")

    # Save nuclear masks
    nuclear_masks_tif_path = output_dir / f'{image_name}_nuclear_masks.tif'
    io.imsave(nuclear_masks_tif_path, nuclear_masks)
    logger.info(f"Nuclear masks saved (TIFF): {nuclear_masks_tif_path}")

    # Save pixel-to-cell mapping CSV (includes nuclear-to-cell matching)
    logger.info("Generating pixel-to-cell mapping CSV files...")
    full_csv_path, summary_csv_path, nuclear_mask_binary = save_pixel_to_cell_csv(masks, nuclear_masks, output_dir, image_name)

    # Create visualization with nuclear segmentation
    create_visualization(img, masks, nuclear_masks, output_dir, image_name, nuclear_mask_binary=nuclear_mask_binary)

    # Summary
    logger.info("Processing complete!")
    logger.info("Output files:")
    logger.info(f"\nSegmentation results:")
    logger.info(f"  - Cell masks (TIFF): {masks_tif_path}")
    logger.info(f"  - Nuclear masks (TIFF): {nuclear_masks_tif_path}")
    logger.info(f"\nVisualization (all at same resolution):")
    logger.info(f"  - Cell segmentation: {output_dir / f'{image_name}_cell_seg_downsampled_1x.png'}")
    logger.info(f"  - Nuclear segmentation: {output_dir / f'{image_name}_nuclear_seg_downsampled_1x.png'}")
    logger.info(f"  - Cell+Nuclear overlay: {output_dir / f'{image_name}_cell_nuclear_overlay_downsampled_1x.png'}")
    logger.info(f"\nPixel-level data:")
    logger.info(f"  - Full pixel mapping (compressed): {full_csv_path}")
    logger.info(f"  - Cell masks array: {output_dir / f'{image_name}_cell_masks.npy'}")
    logger.info(f"  - Nuclear masks array: {output_dir / f'{image_name}_nuclear_masks.npy'}")
    logger.info(f"  - Nuclear binary mask array (matched to cells): {output_dir / f'{image_name}_nuclear_binary_mask.npy'}")
    logger.info(f"  - Boundary mask array: {output_dir / f'{image_name}_boundary_mask.npy'}")
    logger.info(f"\nLog file: cellpose_sam_segmentation.log")
    logger.info("\nCSV file format:")
    logger.info("  {image_name}_pixel_to_cell_mapping_full.csv.gz columns:")
    logger.info("    - x, y: pixel coordinates")
    logger.info("    - cell_id: which cell this pixel belongs to (0 = background)")
    logger.info("    - is_boundary: 1 if pixel is at cell boundary (membrane), 0 otherwise")
    logger.info("    - is_nuclear: 1 if pixel is in nucleus (matched to cell), 0 otherwise")
    logger.info("    - is_interior: 1 if pixel is inside cell (non-boundary), 0 otherwise")
    logger.info("    - is_cytoplasm: 1 if pixel is in cytoplasm (interior - nuclear), 0 otherwise")

    return masks


if __name__ == "__main__":
    masks = main()
