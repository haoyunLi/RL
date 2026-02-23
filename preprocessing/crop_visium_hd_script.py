#!/usr/bin/env python3
"""
Crop Visium HD Image Script

This script crops the Visium HD microscope image to the tissue region using
the prepare_dataframe_image function. This is adapted from crop_visium_hd_notebook.ipynb
to run as a batch job with sufficient memory.

Usage:
    python crop_visium_hd_script.py
"""

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for batch processing
import matplotlib.pyplot as plt
import sys
import os
from datetime import datetime

# Import the cropping function
from crop_visium_hd_image import prepare_dataframe_image, visualize_crop_boundaries


def main():
    """Main function to crop Visium HD image"""

    print("Visium HD Image Cropping Script")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Path to tissue positions data (contains spot coordinates)
    df_path = 'Human_Colorectal/output/binned_outputs/square_002um/spatial/tissue_positions.parquet'

    # Path to the full-resolution microscope image
    img_path = 'Human_Colorectal/input/Visium_HD_Human_Colon_Cancer_tissue_image.btf'

    # Output path for the cropped image
    output_path = 'workspace_outputs/cropped_visium_hd_human_Colorectal.png'

    # Image format: 'HE' for H&E staining, 'DAPI' for fluorescence
    image_format = 'HE'

    # Visium HD array dimensions (default: 3350 x 3350 for 2um resolution)
    row_number = 3350
    col_number = 3350

    print("Checking input files...")
    if not os.path.exists(df_path):
        print(f"ERROR: Tissue positions file not found: {df_path}")
        sys.exit(1)
    print(f"✓ Found tissue positions: {df_path}")

    if not os.path.exists(img_path):
        print(f"ERROR: Image file not found: {img_path}")
        sys.exit(1)
    print(f"✓ Found image file: {img_path}")
    print()

    print("Starting image cropping process...")
    print()

    try:
        result = prepare_dataframe_image(
            df_path=df_path,
            img_path=img_path,
            image_format=image_format,
            row_number=row_number,
            col_number=col_number,
            save_cropped_path=output_path
        )
        print()
        print("✓ Image cropping completed successfully!")

    except Exception as e:
        print(f"ERROR: Image cropping failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print()
    print("Results Summary")
    print(f"Original image shape: {result['image_array'].shape}")
    print(f"Cropped image shape: {result['cropped_image'].shape}")
    print(f"Total spots: {len(result['df']):,}")
    print(f"Spots in cropped region: {len(result['df_temp']):,}")
    print(f"Spots marked as in_tissue: {result['df'][result['df']['in_tissue'] == 1].shape[0]:,}")
    print()

    print("Crop boundaries:")
    for key, value in result['boundaries'].items():
        print(f"  {key}: {value}")
    print()

    

if __name__ == "__main__":
    main()
