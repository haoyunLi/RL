"""
This module extracts and crops the Visium HD region from the original microscope image.
This is the first step in the SMURF pipeline for soft segmentation of VisiumHD data.

The main function `prepare_dataframe_image` performs the following:
1. Reads the full-resolution microscope image (HE or DAPI)
2. Reads the tissue position data from Visium HD
3. Calculates pixel boundaries for each spot
4. Identifies the tissue region (spots marked as in_tissue)
5. Crops the image to only the Visium HD covered area
6. Creates a spatial object with the cropped image and mapped spot data

"""

import numpy as np
import pandas as pd
from PIL import Image
from numba import jit


@jit(nopython=True)
def fill_pixels(pixels, row_starts, row_ends, col_starts, col_ends, indices):
    """
    Fill pixels array with spot indices using optimized numba JIT compilation.

    This function assigns each pixel to a spot ID based on the calculated boundaries.

    :param pixels: Array to fill with spot indices
    :param row_starts: Starting row positions for each spot
    :param row_ends: Ending row positions for each spot
    :param col_starts: Starting column positions for each spot
    :param col_ends: Ending column positions for each spot
    :param indices: Spot indices to assign
    :return: Filled pixels array
    """
    for i in range(len(indices)):
        pixels[row_starts[i] : row_ends[i], col_starts[i] : col_ends[i]] = indices[i]
    return pixels


def prepare_dataframe_image(
    df_path,
    img_path,
    image_format="HE",
    row_number=3350,
    col_number=3350,
    save_cropped_path=None,
):
    """
    Prepares and crops the microscope image to the Visium HD tissue region.

    This is the first step in SMURF processing. It reads the full-resolution microscope
    image and crops it to only include the region covered by Visium HD spots marked as
    "in_tissue". The function also maps each pixel to its corresponding spot.

    Parameters
    ----------
    df_path : str
        Path to the tissue_positions.parquet file containing spot position data.
        Example: 'square_002um/spatial/tissue_positions.parquet'

        Expected columns in the parquet file:
        - pxl_row_in_fullres: Row pixel position in full resolution image
        - pxl_col_in_fullres: Column pixel position in full resolution image
        - array_row: Row index in the spot array
        - array_col: Column index in the spot array
        - in_tissue: Boolean flag (1 if spot is in tissue, 0 otherwise)

    img_path : str
        Path to the full-resolution microscope image file.
        Example: 'Visium_HD_Mouse_Small_Intestine_tissue_image.btf'
        Supports formats: .tif, .tiff, .btf, .png, .jpg, etc.

    image_format : str, optional
        The staining format of the image. Must be either 'HE' or 'DAPI'.
        - 'HE': Hematoxylin and Eosin staining (RGB image)
        - 'DAPI': DAPI fluorescence staining (grayscale)
        Default: 'HE'

    row_number : int, optional
        Total number of rows in the spot array. Used for calculating average spot size.
        Default: 3350 (typical for Visium HD 2um resolution)

    col_number : int, optional
        Total number of columns in the spot array. Used for calculating average spot size.
        Default: 3350 (typical for Visium HD 2um resolution)

    save_cropped_path : str, optional
        Path where the cropped image will be saved. If None, the image is not saved.
        Example: 'cropped_visium_hd.tif'
        Default: None

    Returns
    -------
    dict
        A dictionary containing the cropped image and mapping information:
        - 'image_array': Full resolution original image (numpy array)
        - 'cropped_image': Cropped image containing only Visium HD region (numpy array)
        - 'df': Complete DataFrame with all spot information
        - 'df_temp': Filtered DataFrame with only spots in tissue region
        - 'pixels': Pixel-to-spot mapping array (-1 for background, spot_id otherwise)
        - 'boundaries': Dictionary containing cropping boundaries:
            - 'row_left', 'row_right': Row boundaries for cropping
            - 'col_up', 'col_down': Column boundaries for cropping
            - 'start_row_spot', 'end_row_spot': Row range of spots in tissue
            - 'start_col_spot', 'end_col_spot': Column range of spots in tissue
        - 'image_format': The image format ('HE' or 'DAPI')

    Raises
    ------
    ValueError
        If image_format is not 'HE' or 'DAPI'
    IOError
        If the image file cannot be opened or processed

    Examples
    --------
    >>> # Crop HE stained image
    >>> result = prepare_dataframe_image(
    ...     df_path='square_002um/spatial/tissue_positions.parquet',
    ...     img_path='Visium_HD_Mouse_Brain_tissue_image.tif',
    ...     image_format='HE'
    ... )
    >>> print(f"Original image shape: {result['image_array'].shape}")
    >>> print(f"Cropped image shape: {result['cropped_image'].shape}")
    >>> print(f"Number of spots in tissue: {len(result['df_temp'])}")

    >>> # Crop DAPI image and save the cropped result
    >>> result = prepare_dataframe_image(
    ...     df_path='square_002um/spatial/tissue_positions.parquet',
    ...     img_path='DAPI_image.tif',
    ...     image_format='DAPI',
    ...     save_cropped_path='cropped_dapi.tif'
    ... )

    Notes
    -----
    - The function automatically removes PIL's image size limit to handle large images
    - Pixel boundaries are calculated based on average spot spacing
    - The cropped region includes a small margin around the tissue area
    - Background pixels in the pixel mapping array are marked as -1
    - The pixel mapping allows quick lookup of which spot a pixel belongs to
    """

    # Validate image format
    if image_format not in ["HE", "DAPI"]:
        raise ValueError("image_format must be in ['HE','DAPI']")

    # Read the full-resolution image
    Image.MAX_IMAGE_PIXELS = None  # Remove PIL's image size limit

    try:
        image = Image.open(img_path)
        image_array = np.array(image)
        print(f'Original image shape: {image_array.shape}')
    except IOError as e:
        raise IOError(f"Error opening or processing image: {e}")

    # Read the spot position data
    # Use fastparquet to avoid PyArrow extension type conflicts
    df = pd.read_parquet(df_path, engine="fastparquet")
    print(f'Total number of spots: {len(df)}')

    # Calculate the average spot size (distance between spot centers)
    avg_row = (df["pxl_row_in_fullres"].max() - df["pxl_row_in_fullres"].min()) / (
        2 * row_number
    )
    avg_col = (df["pxl_col_in_fullres"].max() - df["pxl_col_in_fullres"].min()) / (
        2 * col_number
    )
    print(f'Average spot size: {avg_row:.2f} x {avg_col:.2f} pixels')

    # Calculate pixel boundaries for each spot (left, right, top, bottom)
    df["pxl_row_left_in_fullres"] = df["pxl_row_in_fullres"] - avg_row
    df["pxl_row_right_in_fullres"] = df["pxl_row_in_fullres"] + avg_row
    df["pxl_col_up_in_fullres"] = df["pxl_col_in_fullres"] - avg_col
    df["pxl_col_down_in_fullres"] = df["pxl_col_in_fullres"] + avg_col

    # Round boundaries to integer pixel values
    df["pxl_row_left_in_fullres"] = df["pxl_row_left_in_fullres"].round().astype(int)
    df["pxl_row_right_in_fullres"] = df["pxl_row_right_in_fullres"].round().astype(int)
    df["pxl_col_up_in_fullres"] = df["pxl_col_up_in_fullres"].round().astype(int)
    df["pxl_col_down_in_fullres"] = df["pxl_col_down_in_fullres"].round().astype(int)

    # Find the range of spots that are in tissue (in_tissue == 1)
    spots_in_tissue = df[df["in_tissue"] == 1]
    start_row_spot = spots_in_tissue["array_row"].min()
    end_row_spot = spots_in_tissue["array_row"].max() + 1
    start_col_spot = spots_in_tissue["array_col"].min()
    end_col_spot = spots_in_tissue["array_col"].max() + 1

    print(f'Spots in tissue: {len(spots_in_tissue)}')
    print(f'Spot array range: rows [{start_row_spot}, {end_row_spot}), cols [{start_col_spot}, {end_col_spot})')

    # Create a filtered DataFrame with only spots in the tissue region
    df_temp = df[
        (df["array_row"] >= start_row_spot)
        & (df["array_row"] < end_row_spot)
        & (df["array_col"] >= start_col_spot)
        & (df["array_col"] < end_col_spot)
    ].copy()

    # Calculate the cropping boundaries (with bounds checking)
    row_left = max(df_temp["pxl_row_left_in_fullres"].min(), 0)
    row_right = min(df_temp["pxl_row_right_in_fullres"].max(), image_array.shape[0])
    col_up = max(df_temp["pxl_col_up_in_fullres"].min(), 0)
    col_down = min(df_temp["pxl_col_down_in_fullres"].max(), image_array.shape[1])

    print(f'Cropping boundaries: rows [{row_left}, {row_right}), cols [{col_up}, {col_down})')

    # Adjust pixel boundaries relative to the cropped image
    df_temp.loc[:, "pxl_row_left_in_fullres_temp"] = (
        df_temp.loc[:, "pxl_row_left_in_fullres"] - row_left
    )
    df_temp.loc[:, "pxl_row_right_in_fullres_temp"] = (
        df_temp.loc[:, "pxl_row_right_in_fullres"] - row_left
    )
    df_temp.loc[:, "pxl_col_up_in_fullres_temp"] = (
        df_temp.loc[:, "pxl_col_up_in_fullres"] - col_up
    )
    df_temp.loc[:, "pxl_col_down_in_fullres_temp"] = (
        df_temp.loc[:, "pxl_col_down_in_fullres"] - col_up
    )

    # Crop the image to the Visium HD region
    if image_format == "HE":
        cropped_image = image_array[row_left:row_right, col_up:col_down, :]
    else:  # DAPI
        cropped_image = image_array[row_left:row_right, col_up:col_down]

    print(f'Cropped image shape: {cropped_image.shape}')

    # Save the cropped image if output path is provided
    if save_cropped_path is not None:
        cropped_img = Image.fromarray(cropped_image.astype(np.uint8))
        cropped_img.save(save_cropped_path)
        print(f'Cropped image saved to: {save_cropped_path}')

    # Create pixel-to-spot mapping
    # Extract spot boundaries for the JIT-compiled function
    row_starts = df_temp["pxl_row_left_in_fullres_temp"].values
    row_ends = df_temp["pxl_row_right_in_fullres_temp"].values
    col_starts = df_temp["pxl_col_up_in_fullres_temp"].values
    col_ends = df_temp["pxl_col_down_in_fullres_temp"].values
    indices = df_temp.index.to_numpy()

    # Initialize pixels array with -1 (background)
    pixels = -1 * np.ones(cropped_image.shape[:2], dtype=np.int32)

    # Fill the pixels array with spot indices
    print('Mapping pixels to spots...')
    pixels = fill_pixels(pixels, row_starts, row_ends, col_starts, col_ends, indices)
    print('Pixel mapping complete!')

    # Return all relevant information in a dictionary
    return {
        'image_array': image_array,
        'cropped_image': cropped_image,
        'df': df,
        'df_temp': df_temp,
        'pixels': pixels,
        'boundaries': {
            'row_left': row_left,
            'row_right': row_right,
            'col_up': col_up,
            'col_down': col_down,
            'start_row_spot': start_row_spot,
            'end_row_spot': end_row_spot,
            'start_col_spot': start_col_spot,
            'end_col_spot': end_col_spot,
        },
        'image_format': image_format,
    }


def save_cropped_image(result, output_path):
    """
    Save the cropped image to a file.

    :param result: Dictionary returned by prepare_dataframe_image
    :param output_path: Path where the cropped image will be saved
    """
    cropped_img = Image.fromarray(result['cropped_image'].astype(np.uint8))
    cropped_img.save(output_path)
    print(f'Cropped image saved to: {output_path}')


def visualize_crop_boundaries(result):
    """
    Visualize the original image with crop boundaries marked.

    :param result: Dictionary returned by prepare_dataframe_image
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

        # Show original image with crop box
        ax1.imshow(result['image_array'])
        boundaries = result['boundaries']
        rect = patches.Rectangle(
            (boundaries['col_up'], boundaries['row_left']),
            boundaries['col_down'] - boundaries['col_up'],
            boundaries['row_right'] - boundaries['row_left'],
            linewidth=3, edgecolor='r', facecolor='none'
        )
        ax1.add_patch(rect)
        ax1.set_title('Original Image with Crop Boundary (Red Box)')
        ax1.axis('off')

        # Show cropped image
        ax2.imshow(result['cropped_image'])
        ax2.set_title('Cropped Visium HD Region')
        ax2.axis('off')

        plt.tight_layout()
        plt.show()

    except ImportError:
        print("Matplotlib is required for visualization. Install it with: pip install matplotlib")


if __name__ == "__main__":
    pass
