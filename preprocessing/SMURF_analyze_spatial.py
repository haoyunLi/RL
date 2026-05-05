import os
# Suppress TensorFlow informational messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=filter INFO, 2=filter INFO+WARNING, 3=filter all
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom ops messages

import argparse
import copy
import gzip
import json
import pickle
import sys
from pathlib import Path

import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import smurf as su
import squidpy as sq
import yaml
from csbdeep.utils import normalize
from stardist.models import StarDist2D

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from preprocessing.ppo_format_assignment_eval import (
    add_ppo_format_assignment_eval_args,
    coerce_bool_series,
    run_ppo_format_assignment_evaluation,
    validate_ppo_format_assignment_eval_args,
)


def stardist_2D_versatile_he(img, nms_thresh=None, prob_thresh=None, return_details=False):
    """StarDist segmentation function for HE images"""
    axis_norm = (0, 1, 2)  # normalize channels jointly
    img = normalize(img, 1, 99.8, axis=axis_norm)
    model = StarDist2D.from_pretrained("2D_versatile_he")

    # Override model thresholds with our custom values if provided
    # This ensures our prob_thresh parameter is actually used
    if prob_thresh is not None:
        model.thresholds = {'prob': prob_thresh, 'nms': nms_thresh if nms_thresh is not None else model.thresholds.get('nms', 0.3)}
    elif nms_thresh is not None:
        model.thresholds = {'prob': model.thresholds.get('prob', 0.5), 'nms': nms_thresh}

    labels, details = model.predict_instances(
        img, nms_thresh=nms_thresh, prob_thresh=prob_thresh
    )
    if return_details:
        return labels, details
    return labels


def segment_image_tiles(image, loop=4700, gap=80, prob_thresh_default=0.01, nms_thresh_default=0.3):
    """Segment large image by processing it in tiles"""
    i_max = image.shape[0]
    j_max = image.shape[1]
    segmentation_results = {}
    confidence_scores = {}

    print(f"Processing image of size {i_max} x {j_max} with tiles of size {loop} x {loop}")
    print(f"Using prob_thresh={prob_thresh_default}, nms_thresh={nms_thresh_default}")

    tiles_i = (i_max + loop - 1) // loop
    tiles_j = (j_max + loop - 1) // loop
    total_tiles = tiles_i * tiles_j
    tile_count = 0

    for i in range(0, i_max, loop):
        for j in range(0, j_max, loop):
            tile_count += 1
            print(f"Processing tile {tile_count}/{total_tiles} at position ({i}, {j})")

            a_t = image[max(0, i-gap):min(i+loop, i_max), max(0, j-gap):min(j+loop, j_max)]

            # Get segmentation with confidence scores
            labels, details = stardist_2D_versatile_he(a_t, nms_thresh=nms_thresh_default,
                                                      prob_thresh=prob_thresh_default,
                                                      return_details=True)

            segmentation_results[(i, j)] = labels
            # Store confidence scores for this tile
            if 'prob' in details:
                confidence_scores[(i, j)] = details['prob']
            else:
                confidence_scores[(i, j)] = np.array([])

    return segmentation_results, confidence_scores


def combine_segmentation_tiles(segmentation_results, confidence_scores, i_max, j_max, loop, gap):
    """Combine segmentation results from different tiles and track confidence scores"""
    print("Combining segmentation tiles...")

    segmentation_results1 = copy.deepcopy(segmentation_results)
    confidence_mapping = {}  # Maps final cell ID to confidence score
    num = 0

    for i in range(0, i_max, loop):
        for j in range(0, j_max, loop):
            mask = segmentation_results1[(i, j)] != 0
            old_labels = np.unique(segmentation_results1[(i, j)][mask])
            old_labels = old_labels[old_labels > 0]

            # Update labels and map confidence scores
            segmentation_results1[(i, j)][mask] = segmentation_results1[(i, j)][mask] + num

            # Map confidence scores to new labels
            if (i, j) in confidence_scores and len(confidence_scores[(i, j)]) > 0:
                for idx, old_label in enumerate(old_labels):
                    new_label = old_label + num
                    if idx < len(confidence_scores[(i, j)]):
                        confidence_mapping[new_label] = confidence_scores[(i, j)][idx]

            num = max(num, segmentation_results1[(i, j)].max())

    segmentation_final = np.zeros((i_max, j_max))

    for i in range(0, i_max, loop):
        for j in range(0, j_max, loop):
            if i == 0 and j == 0:
                segmentation_final[i:min(i+loop, i_max), j:min(j+loop, j_max)] = segmentation_results1[(i, j)]
            elif i == 0 and j != 0:
                segmentation_final[i:min(i+loop, i_max), j:min(j+loop, j_max)] = segmentation_results1[(i, j)][:, gap:]
            elif i != 0 and j == 0:
                segmentation_final[i:min(i+loop, i_max), j:min(j+loop, j_max)] = segmentation_results1[(i, j)][gap:, :]
            else:
                segmentation_final[i:min(i+loop, i_max), j:min(j+loop, j_max)] = segmentation_results1[(i, j)][gap:, gap:]

    print("Resolving tile overlaps...")
    for i in range(0, i_max, loop):
        for j in range(0, j_max, loop):
            if j != 0:
                if i == 0:
                    a = copy.deepcopy(segmentation_results1[(i, (j-loop))][:, -gap:])
                    b = copy.deepcopy(segmentation_results1[(i, j)][:, :gap])
                else:
                    a = copy.deepcopy(segmentation_results1[(i, (j-loop))][gap:, -gap:])
                    b = copy.deepcopy(segmentation_results1[(i, j)][gap:, :gap])

                uni_a = np.unique(a[:, -1:])
                uni_a = uni_a[uni_a > 0]
                for ele in uni_a:
                    uni_b = np.unique(b[a == ele])
                    uni_b = uni_b[uni_b > 0]
                    if len(uni_b) > 0:
                        a[a == ele] = 0
                        for eleb in uni_b:
                            a[b == eleb] = eleb

                if i == 0:
                    segmentation_final[i:min(i+loop, i_max), min(j, j_max)-gap:min(j, j_max)] = a
                else:
                    segmentation_final[i:min(i+loop, i_max), min(j, j_max)-gap:min(j, j_max)] = a

            if i != 0:
                if j == 0:
                    a = copy.deepcopy(segmentation_results1[((i-loop), j)][-gap:, :])
                    b = copy.deepcopy(segmentation_results1[(i, j)][:gap, :])
                else:
                    a = copy.deepcopy(segmentation_results1[((i-loop), j)][-gap:, gap:])
                    b = copy.deepcopy(segmentation_results1[(i, j)][:gap, gap:])

                uni_a = np.unique(a[-1:, :])
                uni_a = uni_a[uni_a > 0]
                for ele in uni_a:
                    uni_b = np.unique(b[a == ele])
                    uni_b = uni_b[uni_b > 0]
                    if len(uni_b) > 0:
                        a[a == ele] = 0
                        for eleb in uni_b:
                            a[b == eleb] = eleb

                if j == 0:
                    segmentation_final[(min(i, i_max)-gap):min(i, i_max), j:min(j+loop, j_max)] = a
                else:
                    segmentation_final[(min(i, i_max)-gap):min(i, i_max), j:min(j+loop, j_max)] = a

    return segmentation_final, confidence_mapping


def filter_cells_by_confidence(segmentation_final, confidence_mapping, confidence_threshold=0.6):
    """Filter out cells with confidence scores below threshold"""
    print(f"Filtering cells with confidence < {confidence_threshold}")

    # Create filtered segmentation
    filtered_segmentation = segmentation_final.copy()
    removed_cells = []

    for cell_id, confidence in confidence_mapping.items():
        if confidence < confidence_threshold:
            filtered_segmentation[segmentation_final == cell_id] = 0
            removed_cells.append((cell_id, confidence))

    # Relabel remaining cells to be consecutive
    unique_labels = np.unique(filtered_segmentation)
    unique_labels = unique_labels[unique_labels > 0]

    final_segmentation = np.zeros_like(filtered_segmentation)
    new_confidence_mapping = {}

    for new_label, old_label in enumerate(unique_labels, 1):
        final_segmentation[filtered_segmentation == old_label] = new_label
        if old_label in confidence_mapping:
            new_confidence_mapping[new_label] = confidence_mapping[old_label]

    print(f"Removed {len(removed_cells)} cells with low confidence")
    print(f"Remaining cells: {int(final_segmentation.max())}")

    return final_segmentation, new_confidence_mapping, removed_cells


def save_confidence_scores(confidence_mapping, save_path):
    """Save confidence scores to a readable file"""
    confidence_df = pd.DataFrame([
        {'cell_id': cell_id, 'confidence_score': score}
        for cell_id, score in confidence_mapping.items()
    ])
    confidence_df = confidence_df.sort_values('confidence_score', ascending=False)

    csv_path = os.path.join(save_path, 'cell_confidence_scores.csv')
    confidence_df.to_csv(csv_path, index=False)
    print(f"Confidence scores saved to: {csv_path}")

    # Print summary statistics
    print(f"Confidence score statistics:")
    print(f"  Mean: {confidence_df['confidence_score'].mean():.3f}")
    print(f"  Median: {confidence_df['confidence_score'].median():.3f}")
    print(f"  Min: {confidence_df['confidence_score'].min():.3f}")
    print(f"  Max: {confidence_df['confidence_score'].max():.3f}")
    print(f"  Cells with confidence >= 0.8: {(confidence_df['confidence_score'] >= 0.8).sum()}")
    print(f"  Cells with confidence >= 0.9: {(confidence_df['confidence_score'] >= 0.9).sum()}")

    return confidence_df


def load_external_nuclear_bins(external_nuclear_bins_path):
    """Load PPO-aligned nuclear bins and keep only nuclear barcode -> cell_id assignments."""
    usecols = ["barcode", "cell_id", "is_nuclear"]
    external = pd.read_csv(external_nuclear_bins_path, usecols=usecols, compression="infer")
    external["is_nuclear"] = coerce_bool_series(external["is_nuclear"]).fillna(False)
    external = external.loc[external["is_nuclear"]].copy()
    external["cell_id"] = pd.to_numeric(external["cell_id"], errors="coerce")
    external = external.loc[external["cell_id"].notna()].copy()
    external["cell_id"] = external["cell_id"].astype(np.int64)
    external["barcode"] = external["barcode"].astype(str)
    external = external.drop_duplicates(subset=["barcode"], keep="last").copy()
    return external


def build_external_segmentation_from_nuclear_bins(so, external_nuclear_bins_path):
    """
    Populate so.segmentation_final directly from external PPO-aligned nuclear bins.

    Each nuclear barcode is expanded to the full 2um-bin pixel block in so.pixels, so SMURF starts from
    the same nuclear seeds as PPO instead of running its own H&E nuclei segmentation.
    """
    external = load_external_nuclear_bins(external_nuclear_bins_path)
    print(f"Loaded {len(external)} external nuclear barcodes from {external_nuclear_bins_path}")

    spot_df = so.df.loc[:, ["barcode"]].copy()
    spot_df["barcode"] = spot_df["barcode"].astype(str)
    merged = spot_df.merge(
        external.loc[:, ["barcode", "cell_id"]],
        on="barcode",
        how="left",
    )
    cell_ids_by_index = pd.to_numeric(merged["cell_id"], errors="coerce").fillna(0).astype(np.int32).to_numpy()

    max_index = max(int(so.df.index.max()), int(np.max(so.pixels))) if so.pixels.size > 0 else int(so.df.index.max())
    lookup = np.zeros(max_index + 1, dtype=np.int32)
    lookup[np.asarray(so.df.index, dtype=np.int64)] = cell_ids_by_index

    segmentation_final = np.zeros_like(so.pixels, dtype=np.int32)
    valid = so.pixels >= 0
    segmentation_final[valid] = lookup[np.asarray(so.pixels[valid], dtype=np.int64)]
    so.segmentation_final = segmentation_final

    matched_bins = int(np.count_nonzero(cell_ids_by_index > 0))
    matched_pixels = int(np.count_nonzero(segmentation_final > 0))
    print(f"External nuclei injected into SMURF seeds:")
    print(f"  matched nuclear bins: {matched_bins}")
    print(f"  matched nuclear pixels: {matched_pixels}")
    print(f"  unique nuclear cells: {int(np.unique(cell_ids_by_index[cell_ids_by_index > 0]).size)}")
    return external


def build_spot_composition_dict(so, adata, cells_before_ml, spot_cell_dic, spots_id_dic, spots_id_dic_prop, nonzero_indices_dic):
    """Rebuild per-spot cell composition from SMURF optimization outputs without per-pixel sampling."""
    spots_composition = {}
    for i in so.index_toset.keys():
        spots_composition[i] = {}

    for cell_id in cells_before_ml.keys():
        for spot_info in cells_before_ml[cell_id]:
            spots_composition[spot_info[0]][cell_id] = spot_info[1]

    for i in spot_cell_dic.keys():
        for j in range(len(spot_cell_dic[i])):
            for k in range(len(spot_cell_dic[i][j])):
                spots_composition[spots_id_dic[i][j][0]][nonzero_indices_dic[i][j][k]] = (
                    spot_cell_dic[i][j][k] * spots_id_dic_prop[i][j][0]
                )

    keys = list(spots_composition.keys())
    for spot_id in keys:
        if len(spots_composition[spot_id]) == 0:
            del spots_composition[spot_id]
            continue
        total = float(sum(spots_composition[spot_id].values()))
        if abs(total - 1.0) > 0.0001:
            raise ValueError("Proportion Sum for spot " + str(spot_id) + " is not 1.")
        for cell_id in list(spots_composition[spot_id].keys()):
            spots_composition[spot_id][cell_id] = float(spots_composition[spot_id][cell_id] / total)

    df1 = copy.deepcopy(adata.obs)
    df1["barcode"] = df1.index
    df2 = copy.deepcopy(so.df)
    df2["index"] = df2.index
    df = pd.merge(df1, df2, on=["barcode"], how="inner")
    df_index = np.asarray(df["index"], dtype=np.int64)

    spots_dict_new = {}
    for spot_id in spots_composition.keys():
        spots_dict_new[int(df_index[spot_id])] = dict(spots_composition[spot_id])
    return spots_dict_new


def save_smurf_assignments(
    *,
    so,
    adata,
    cells_before_ml,
    spot_cell_dic,
    spots_id_dic,
    spots_id_dic_prop,
    nonzero_indices_dic,
    output_dir,
    dataset_name,
    microns_per_pixel,
    external_nuclear_bins_path=None,
):
    """Save one row per 2um bin with final SMURF cell assignment and nuclear/cytoplasm flag."""
    print("\nSaving SMURF bin assignments...")
    spots_dict_new = build_spot_composition_dict(
        so,
        adata,
        cells_before_ml,
        spot_cell_dic,
        spots_id_dic,
        spots_id_dic_prop,
        nonzero_indices_dic,
    )

    external_nuclear_barcodes = None
    if external_nuclear_bins_path is not None:
        external_nuclear_barcodes = set(load_external_nuclear_bins(external_nuclear_bins_path)["barcode"].astype(str).tolist())
    else:
        nuclear_spot_ids = np.unique(np.asarray(so.pixels[so.segmentation_final > 0], dtype=np.int64))
        nuclear_spot_ids = nuclear_spot_ids[nuclear_spot_ids >= 0]
        external_nuclear_barcodes = set(so.df.loc[nuclear_spot_ids, "barcode"].astype(str).tolist())

    rows = []
    for spot_index, cell_probs in spots_dict_new.items():
        if not cell_probs:
            continue
        dominant_cell_id, dominant_prob = max(cell_probs.items(), key=lambda kv: (kv[1], kv[0]))
        spot = so.df.loc[int(spot_index)]
        barcode = str(spot["barcode"])
        rows.append(
            {
                "barcode": barcode,
                "cell_id": int(dominant_cell_id),
                "is_nuclear": bool(barcode in external_nuclear_barcodes),
                "dominant_probability": float(dominant_prob),
                "array_row": int(spot["array_row"]),
                "array_col": int(spot["array_col"]),
                "x_um": float(spot["pxl_col_in_fullres"]) * float(microns_per_pixel),
                "y_um": float(spot["pxl_row_in_fullres"]) * float(microns_per_pixel),
            }
        )

    assignments = pd.DataFrame(rows)
    assignments = assignments.sort_values(["cell_id", "barcode"], kind="stable").reset_index(drop=True)
    assignments_path = Path(output_dir) / f"{dataset_name}_smurf_assignments.csv"
    assignments.to_csv(assignments_path, index=False)

    print(f"  Saved {len(assignments)} assigned bins to {assignments_path}")
    print(f"  Unique SMURF cells: {assignments['cell_id'].nunique()}")
    print(f"  Nuclear bins: {int(assignments['is_nuclear'].sum())}")
    print(f"  Cytoplasm bins: {int((~assignments['is_nuclear']).sum())}")
    return assignments_path


def main():
    """Main SMURF analysis workflow."""

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run SMURF spatial transcriptomics analysis')
    parser.add_argument('--tissue_positions', type=str,
                       default='colorectal_pseudo_visium_hd_output_full_0.25/spatial/tissue_positions.parquet',
                       help='Path to tissue positions parquet file')
    parser.add_argument('--tissue_image', type=str,
                       default='Human_Colorectal/input/Visium_HD_Human_Colon_Cancer_tissue_image.btf',
                       help='Path to full-resolution tissue image aligned to tissue_positions')
    parser.add_argument('--pseudo_hd_dir', type=str,
                       default='colorectal_pseudo_visium_hd_output_full_0.25',
                       help='Path to pseudo Visium HD output directory')
    parser.add_argument('--output_dir', type=str,
                       default='workspace_outputs/pseudo_human_colorectal/smurf_results_colorectal_0.25',
                       help='Output directory for SMURF results')
    parser.add_argument('--dataset_name', type=str,
                       default='human_colorectal',
                       help='Dataset name prefix for saved outputs')
    parser.add_argument('--microns_per_pixel', type=float,
                       default=0.2737012522439323,
                       help='Microns per pixel conversion factor')
    parser.add_argument('--external_nuclear_bins_path', type=str, default=None,
                       help='Optional PPO-aligned nuclear-bin CSV/CSV.GZ. If provided, skip SMURF H&E nuclei segmentation.')
    add_ppo_format_assignment_eval_args(
        parser,
        default_eval_run_name='human_colorectal_smurf_eval',
        method_label='SMURF',
    )
    args = parser.parse_args()
    validate_ppo_format_assignment_eval_args(args)

    save_path = Path(args.output_dir).expanduser().resolve()
    pseudo_hd_dir = args.pseudo_hd_dir
    external_nuclear_bins_path = None if args.external_nuclear_bins_path is None else Path(args.external_nuclear_bins_path).expanduser().resolve()
    nuclear_source = "external_ppo_aligned" if external_nuclear_bins_path is not None else "smurf_he_stardist"

    save_path.mkdir(parents=True, exist_ok=True)

    so = su.prepare_dataframe_image(args.tissue_positions, args.tissue_image, 'HE')
    plt.imshow(so.image_temp())
    plt.savefig(save_path / 'tissue_image.png', dpi=300, bbox_inches='tight')
    plt.close()
    segmentation_path = save_path / 'segmentation_final.npy'
    if external_nuclear_bins_path is not None:
        print("External nuclear bins provided. Skipping SMURF H&E nuclei segmentation.")
        build_external_segmentation_from_nuclear_bins(so, external_nuclear_bins_path)
        np.save(segmentation_path, so.segmentation_final)
        print(f"External segmentation seed saved to: {segmentation_path}")
    else:
        with gzip.GzipFile(save_path / 'image_to_segmentation.npy.gz', 'w') as f:
            np.save(f, so.image_temp())

        print("Image saved for segmentation.")

        if segmentation_path.exists():
            print("Loading existing segmentation results...")
            so.segmentation_final = np.load(segmentation_path)
            print("Segmentation results loaded successfully.")
        else:
            print("Segmentation file not found. Performing nuclei segmentation...")

            with gzip.GzipFile(save_path / 'image_to_segmentation.npy.gz', 'r') as f:
                image = np.load(f)

            print(f"Performing segmentation on image of size: {image.shape}")

            segmentation_results, confidence_scores = segment_image_tiles(image, loop=4700, gap=80,
                                                                         prob_thresh_default=0.01,
                                                                         nms_thresh_default=0.3)

            i_max, j_max = image.shape[:2]
            segmentation_final_array, confidence_mapping = combine_segmentation_tiles(
                segmentation_results, confidence_scores, i_max, j_max, loop=4700, gap=80)

            print(f"Initial segmentation completed! Found {int(segmentation_final_array.max())} nuclei.")

            save_confidence_scores(confidence_mapping, save_path)

            with open(save_path / 'confidence_mapping.pkl', 'wb') as f:
                pickle.dump(confidence_mapping, f)

            np.save(segmentation_path, segmentation_final_array)
            so.segmentation_final = segmentation_final_array

            print(f"Segmentation saved to: {segmentation_path}")
            print(f"Confidence scores saved to: {save_path / 'cell_confidence_scores.csv'}")

    # Generate cell and spot information
    print("Generating cell and spot information...")
    so.generate_cell_spots_information()

    # Visualize segmentation results
    su.plot_results(so.image_temp(), so.segmentation_final, dpi=1500,
                    save=os.path.join(save_path, 'segmentation_overlay.png'))

    # Load gene expression data
    print("Loading gene expression data...")
    adata = sc.read_10x_mtx(os.path.join(pseudo_hd_dir, 'filtered_feature_bc_matrix'))
    adata = copy.deepcopy(adata[so.df[so.df.in_tissue == 1]['barcode']])

    # Filter genes
    sc.pp.filter_genes(adata, min_counts=100)

    # Generate nuclei*genes matrix
    print("Generating nuclei-genes matrix...")
    su.nuclei_rna(adata, so)
    adata_sc = copy.deepcopy(so.final_nuclei)

    # Filter cells
    sc.pp.filter_cells(adata_sc, min_counts=500)
    adata_raw = copy.deepcopy(adata_sc)

    # Initial single cell analysis
    print("Performing initial single cell analysis...")
    adata_sc = su.singlecellanalysis(adata_sc, resolution=0.5)

    # Iterative arrangement (this is time-consuming)
    print("Starting iterative arrangement (this may take a while)...")
    su.itering_arragement(adata_sc, adata_raw, adata, so, resolution=0.5,
                         save_folder=os.path.abspath(save_path) + '/', show=True, keep_previous=False)

    # Load iteration results
    adatas_path = save_path / 'adatas.h5ad'
    if not adatas_path.exists():
        # Check if files were saved with incorrect path concatenation
        adatas_path = Path(str(save_path) + 'adatas.h5ad')
        if not adatas_path.exists():
            raise FileNotFoundError(f"Could not find adatas.h5ad in either {save_path / 'adatas.h5ad'} or {str(save_path) + 'adatas.h5ad'}")
    adatas_final = sc.read_h5ad(adatas_path)

    # Load cells_final.pkl
    cells_path = save_path / 'cells_final.pkl'
    if not cells_path.exists():
        cells_path = Path(str(save_path) + 'cells_final.pkl')
    with open(cells_path, 'rb') as file:
        cells_final = pickle.load(file)

    # Load weights_record.pkl
    weights_path = save_path / 'weights_record.pkl'
    if not weights_path.exists():
        weights_path = Path(str(save_path) + 'weights_record.pkl')
    with open(weights_path, 'rb') as file:
        weights_record = pickle.load(file)

    # Prepare data for deep learning
    print("Preparing data for deep learning optimization...")
    pct_toml_dic, spots_X_dic, celltypes_dic, cells_X_plus_dic, nonzero_indices_dic, nonzero_indices_toml, \
    cells_before_ml, cells_before_ml_x, groups_combined, spots_id_dic, spots_id_dic_prop = \
        su.make_preparation(cells_final, so, adatas_final, adata, weights_record, maximum_cells=2000)

    # Setup CUDA for optimization
    import torch
    import gc

    # Clear any existing GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    # Enable memory fragmentation fix
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Start optimization
    print("Starting deep learning optimization...")
    spot_cell_dic = su.start_optimization(spots_X_dic, celltypes_dic, cells_X_plus_dic,
                                         nonzero_indices_toml, device,
                                         num_epochs=1000, learning_rate=0.1,
                                         print_each=100, epsilon=0.00001)

    # Save optimization results
    with open(save_path / 'spot_cell_dic.pkl', 'wb') as f:
        pickle.dump(spot_cell_dic, f)

    # Generate final pixel-to-cell mapping
    print("Generating final pixel-to-cell mapping...")
    su.make_pixels_cells(so, adata, cells_before_ml, spot_cell_dic, spots_id_dic,
                        spots_id_dic_prop, nonzero_indices_dic)

    # Plot final results
    su.plot_results(so.image_temp(), so.pixels_cells, dpi=1500,
                   save=save_path / 'final_results.pdf')

    # Generate final cell-level data
    print("Generating final cell-level data...")
    adata_sc_final = su.get_finaldata(adata, adatas_final, spot_cell_dic, weights_record,
                                    cells_before_ml, groups_combined, pct_toml_dic,
                                    nonzero_indices_dic, spots_X_dic,
                                    cells_before_ml_x=cells_before_ml_x, so=so)

    # Save nuclear bin information for validation
    print("\nSaving nuclear bin information for each final cell...")

    # Create nuclear bin mapping from segmentation_final and pixels_cells
    # so.segmentation_final = nuclear segmentation mask (which pixels are nuclear)
    # so.pixels_cells = final pixel-to-cell mapping (which pixels belong to which cell)

    print("  Building nuclear bin information from segmentation and cell assignments...")

    # Get unique cell IDs from pixels_cells
    unique_cells = np.unique(so.pixels_cells)
    unique_cells = unique_cells[unique_cells > 0]  # Exclude background (0)

    # Bin size parameters (matching validation scripts)
    microns_per_pixel = args.microns_per_pixel
    bin_size = 2.0  # 2 micron bins

    # Dictionaries to store bin information per cell
    cell_nuclear_bins = {}  # cell_id -> set of (bin_x, bin_y) that are nuclear
    cell_all_bins = {}      # cell_id -> set of all (bin_x, bin_y)

    print(f"  Processing {len(unique_cells)} cells...")

    for cell_id in unique_cells:
        # Get pixel coordinates for this cell from pixels_cells
        y_coords, x_coords = np.where(so.pixels_cells == cell_id)

        # Convert pixels to bins and track nuclear status
        all_bins_set = set()
        nuclear_bins_set = set()

        for y, x in zip(y_coords, x_coords):
            # Convert pixel to bin coordinates
            bin_x = int(x * microns_per_pixel / bin_size)
            bin_y = int(y * microns_per_pixel / bin_size)
            all_bins_set.add((bin_x, bin_y))

            # Check if this pixel is nuclear (from initial segmentation)
            if so.segmentation_final[y, x] > 0:
                nuclear_bins_set.add((bin_x, bin_y))

        cell_all_bins[int(cell_id)] = all_bins_set
        cell_nuclear_bins[int(cell_id)] = nuclear_bins_set

    # Save nuclear bin information
    print(f"  Saving nuclear bin information for {len(cell_nuclear_bins)} cells...")
    with open(save_path / 'cell_nuclear_bins.pkl', 'wb') as f:
        pickle.dump({
            'nuclear_bins': cell_nuclear_bins,
            'all_bins': cell_all_bins
        }, f)

    # Calculate and save summary statistics
    cell_stats = []
    for cell_id in unique_cells:
        n_total = len(cell_all_bins[int(cell_id)])
        n_nuclear = len(cell_nuclear_bins[int(cell_id)])
        nuclear_pct = (n_nuclear / n_total * 100) if n_total > 0 else 0.0

        cell_stats.append({
            'cell_id': int(cell_id),
            'n_total_bins': n_total,
            'n_nuclear_bins': n_nuclear,
            'nuclear_percentage': nuclear_pct
        })

    df_nuclear_stats = pd.DataFrame(cell_stats)
    df_nuclear_stats.to_csv(save_path / 'cell_nuclear_bins_summary.csv', index=False)

    print(f"  Mean nuclear bins per cell: {df_nuclear_stats['n_nuclear_bins'].mean():.1f}")
    print(f"  Mean nuclear percentage: {df_nuclear_stats['nuclear_percentage'].mean():.2f}%")

    assignments_path = save_smurf_assignments(
        so=so,
        adata=adata,
        cells_before_ml=cells_before_ml,
        spot_cell_dic=spot_cell_dic,
        spots_id_dic=spots_id_dic,
        spots_id_dic_prop=spots_id_dic_prop,
        nonzero_indices_dic=nonzero_indices_dic,
        output_dir=save_path,
        dataset_name=args.dataset_name,
        microns_per_pixel=args.microns_per_pixel,
        external_nuclear_bins_path=external_nuclear_bins_path,
    )

    pipeline_summary = {
        "method": "smurf",
        "dataset_name": args.dataset_name,
        "tissue_positions": str(Path(args.tissue_positions).expanduser().resolve()),
        "tissue_image": str(Path(args.tissue_image).expanduser().resolve()),
        "pseudo_hd_dir": str(Path(args.pseudo_hd_dir).expanduser().resolve()),
        "output_dir": str(save_path),
        "nuclear_source": nuclear_source,
        "external_nuclear_bins_path": None if external_nuclear_bins_path is None else str(external_nuclear_bins_path),
        "smurf_assignments_path": str(assignments_path),
        "microns_per_pixel": float(args.microns_per_pixel),
    }
    with (save_path / f"{args.dataset_name}_smurf_pipeline_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(pipeline_summary, handle, indent=2, sort_keys=False)
        handle.write("\n")

    print("\nSaving final results...")
    with open(save_path / "so.pkl", 'wb') as f:
        pickle.dump(so, f)

    adata_sc_final.write(save_path / "adata_sc_final.h5ad")

    # Print what nuclear information is available in 'so' object
    print("\nNuclear information available in 'so' object:")
    print(f"  so.segmentation_final: Nuclear segmentation mask (shape: {so.segmentation_final.shape})")
    if hasattr(so, 'cell_spot_dic'):
        print(f"  so.cell_spot_dic: Cells -> nuclear bins mapping ({len(so.cell_spot_dic)} cells)")
    if hasattr(so, 'final_nuclei'):
        print(f"  so.final_nuclei: Nuclear gene expression matrix ({so.final_nuclei.shape})")
    if hasattr(so, 'df'):
        print(f"  so.df: Bin information ({len(so.df)} bins)")

    if args.ppo_eval_run_dir is not None:
        print("\nRunning PPO-format SMURF evaluation...")
        eval_run_dir = run_ppo_format_assignment_evaluation(
            assignments_csv=assignments_path,
            method_name="smurf",
            method_label="SMURF",
            nuclear_source=nuclear_source,
            external_nuclear_bins_path=external_nuclear_bins_path,
            args=args,
            pipeline_config=pipeline_summary,
        )
        print(f"PPO-format SMURF evaluation complete: {eval_run_dir}")

    print("\nSMURF analysis completed successfully!")
    print(f"Final cell-level data saved as: {save_path / 'adata_sc_final.h5ad'}")
    print(f"Final results visualization saved as: {save_path / 'final_results.pdf'}")
    print(f"Nuclear bin information saved in: {save_path / 'so.pkl'}")


if __name__ == "__main__":
    main()
