import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from pathlib import Path
from matplotlib import colormaps
from typing import Dict, List, Union, Tuple
from mpl_toolkits.axes_grid1 import make_axes_locatable
from gbm_bench.utils.utils import compute_center_of_mass, load_mri_data, load_and_resample_mri_data, merge_pdfs
from gbm_bench.utils.constants import (
        BRAIN_MASK_SCHEMA,
        LONGITUDINAL_WARP_SCHEMA,
        MODALITY_CONVERTED_SCHEMA,
        MODALITY_STRIPPED_SCHEMA,
        MODEL_PLAN_SCHEMA,
        PREDICTION_OUTPUT_SCHEMA,
        RECURRENCE_SCHEMA,
        STANDARD_PLAN_SCHEMA,
        TISSUE_SEG_SCHEMA,
        TISSUE_PBMAP_SCHEMA,
        TUMORSEG_CORE_SCHEMA,
        TUMORSEG_SCHEMA
        )


def get_slices(center: Tuple[int, int, int], num_slices: int, step_size: int, patient_dim: Tuple[int, int, int]):
    axial_slices = [center[2] + ind * step_size - 2 * step_size for ind in range(0, num_slices)]
    axial_slices = [min(max(0, ax_slice), patient_dim[2]-1) for ax_slice in axial_slices]
    coronal_slices = [center[1] + ind * step_size - 2 * step_size for ind in range(0, num_slices)]
    coronal_slices = [min(max(0, cor_slice), patient_dim[1]-1) for cor_slice in coronal_slices]
    return axial_slices, coronal_slices


def get_cmap_norm_patches_tumorseg(classes_of_interest: List[int]):
    # Tumor segmentation legend (1: non enhancing, 2: edema, 3: enhancing)
    colors = [(0,0,0,0), (1, 127/255, 0, 1), (30/255, 144/255, 1, 1), (138/255, 43/255, 226/255, 1)]
    color_labels = ["Non-enhancing Tumor", "Peritumoral Edema", "Enhancing Tumor"]
    cmap = mcolors.ListedColormap(colors)
    bounds = [0, 0.5, 1.5, 2.5, 3.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    patches = [mpatches.Patch(color=c, label=l) for (c, l) in zip(colors[1:], color_labels)]
    return cmap, norm, patches


def get_segmentation_projection(segmentation: np.ndarray, label: int, axis: int) -> np.ndarray:
    seg_data = segmentation.copy()
    seg_data[seg_data!=label]=0
    projection = np.rint(np.sum(seg_data, axis=axis) > 0)
    return projection


def grid_plot(image_tensor: np.ndarray, imshow_args: List[Dict], header: str, col_titles: List[str], row_titles: List[str],
              outfile: str, legend_handles: List[mpatches.Patch] = None ) -> None:
    """
    A generic function to create a grid plot with multiple layers / overlays.

    Args:
        image_tensor: A numpy array with dimension 3 (n_layers, n_cols, n_rows) where each point is a 2D-image or None.
        imshow_args: A list of dictionaries containing arguments for imshow calls for each image layer (e.g. {"cmap": "gray"}).
        header: String to be displayd at the top of the image.
        col_titles: List of strings used as column titles.
        row_titles: List of strings used as row titles.
        outfile: File that the pdf is saved to.
        legend_handles: List of matplotlib.patches.Patch to be displayed in a legend.
    """

    if image_tensor.ndim != 3:
        raise ValueError("Dimension mismatch. image_tensor dimension should be 3: (n_layers, n_cols, n_rows)")

    if len(imshow_args) != image_tensor.shape[0]:
        raise ValueError(f"Dimension mismatch. imshow_args should be the same length as image_tensor.shape[0] = {image_tensor.shape[0]}.")

    if len(row_titles) != image_tensor.shape[1]:
        raise ValueError(f"Dimension mismatch. row_titles should be the same length as image_tensor.shape[1] = {image_tensor.shape[1]}.")

    if len(col_titles) != image_tensor.shape[2]:
        raise ValueError(f"Dimension mismatch. col_titles should be the same length as image_tensor.shape[2] = {image_tensor.shape[2]}.")

    n_row = image_tensor.shape[1]
    n_col = image_tensor.shape[2]
    non_gray_cmaps = [mpcmp for mpcmp in colormaps() if mpcmp not in ["grey", "gray"]]

    # Create figure and fill axes
    fig, axs = plt.subplots(n_row, n_col, figsize=(5 * n_col, 4 * n_row))
    for image_layer, imshow_args in zip(image_tensor, imshow_args):
        for row in range(n_row):
            for col in range(n_col):
                if image_layer[row, col] is not None:
                    img = axs[row, col].imshow(np.rot90(image_layer[row, col]), **imshow_args)
                    axs[row, col].axis("off")

                    # Add colorbar if non gray colormap is used
                    if "cmap" in imshow_args.keys() and imshow_args["cmap"] in non_gray_cmaps:
                        divider = make_axes_locatable(axs[row, col])
                        cax = divider.append_axes("right", size="5%", pad=0.05)
                        plt.colorbar(img, cax=cax)

    # Column titles
    for ind, col_title in enumerate(col_titles):
        axs[0, ind].set_title(col_title, fontsize=16, fontweight="bold", pad=20)

    # Row titles
    for ind, row_title in enumerate(row_titles):
        axs[ind, 0].axis("on")
        axs[ind, 0].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        axs[ind, 0].set_ylabel(row_title, fontweight="bold", labelpad=20, fontsize=16)

    # Header
    fig.subplots_adjust(top=0.85, wspace=0, hspace=0)
    fig.suptitle(
            header,
            horizontalalignment="left",
            fontsize=20,
            fontweight="bold",
            color="black",
            y=0.92,
            x=0.0665,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
            )

    # Color legends
    if legend_handles is not None:
        fig.legend(handles=legend_handles, loc="upper right", bbox_to_anchor=(0.96, 0.890), ncol=3)

    plt.tight_layout(rect=[0, 0, 1, 0.9])
    Path(outfile).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile, format="pdf")
    print(f"Plot saved as {outfile}")
    plt.close(fig)


def plot_model_multislice(patient_identifier: str, exam_identifier: str, algorithm_identifier: str, exam_dir: Path,
                          outfile: str, classes_of_interest: List[int] = [1, 2, 3]) -> None:

    c_threshold = 0.01    # tumor cell concentration threshold
    n_layers = 3    # one layer for each imshow config

    # Load data
    t1c_data = load_mri_data(MODALITY_STRIPPED_SCHEMA.format(base_dir=exam_dir, modality="t1c"))
    tumorseg_data = load_mri_data(TUMORSEG_SCHEMA.format(base_dir=exam_dir))
    tissueseg_data = load_mri_data(TISSUE_SEG_SCHEMA.format(base_dir=exam_dir))
    model_data = load_and_resample_mri_data(PREDICTION_OUTPUT_SCHEMA.format(base_dir=exam_dir, algo_id=algorithm_identifier.lower()), resample_params=t1c_data.shape, interp_type=1)

    # Mask data outside of the brain
    #NOTE: do we want this

    # Compute tumor center of mass
    center = compute_center_of_mass(tumorseg_data, t1c_data, classes_of_interest)

    # Create axial/coronal slices
    step_size = 10
    num_slices = 5
    patient_dim = t1c_data.shape
    axial_slices, coronal_slices = get_slices(center, num_slices, step_size, patient_dim)

    # Tumor segmentation args
    cmap, norm, patches = get_cmap_norm_patches_tumorseg(classes_of_interest)

    # Read recurrence coverage for title
    coverage_str = ""
    #coverage_dir = os.path.join(os.path.dirname(model_dir), "coverage.pkl") #TODO: update path
    #if os.path.isfile(coverage_dir):
    #    coverage = pickle.load(open(coverage_dir, "rb"))
    #    coverage_str = (
    #            f"Coverage (conventional / model): {100*coverage['recurrence_coverage_standard']:.1f}% / {100*coverage['recurrence_coverage_model']:.1f}%\n"
    #            f"CoverageAll (conventional / model): {100*coverage['recurrence_coverage_standard_all']:.1f}% / {100*coverage['recurrence_coverage_model_all']:.1f}%"
    #            )
    #else:
    #    coverage_str = ""

    # Titles
    col_titles = ["T1C", "TUMORSEG", f"{algorithm_identifier.upper()}", "TISSUESEG"]
    row_titles = axial_slices + coronal_slices
    header = (
            f"Patient: {patient_identifier}\n"
            f"Exam: {exam_identifier}\n"
            f"Algorithm: {algorithm_identifier}\n"
            f"Tumor cell concentration threshold: {c_threshold}\n" + coverage_str
            )

    # Build image tensor
    image_tensor = np.empty((n_layers, num_slices*2, 4), dtype=object)
    
    # Layer 1: T1c, T1c, T1c, Tissueseg
    layer_1_args = {"cmap": "gray", "interpolation": "none"}
    for ind, ax_slice, cor_slice in zip(range(num_slices), axial_slices, coronal_slices):
        
        image_tensor[0, ind, 0] = t1c_data[:, :, ax_slice]
        image_tensor[0, ind, 1] = t1c_data[:, :, ax_slice]
        image_tensor[0, ind, 2] = t1c_data[:, :, ax_slice]
        image_tensor[0, ind, 3] = tissueseg_data[:, :, ax_slice]

        image_tensor[0, ind+num_slices, 0] = t1c_data[:, cor_slice, :]
        image_tensor[0, ind+num_slices, 1] = t1c_data[:, cor_slice, :]
        image_tensor[0, ind+num_slices, 2] = t1c_data[:, cor_slice, :]
        image_tensor[0, ind+num_slices, 3] = tissueseg_data[:, cor_slice, :]

    # Layer 2: None, Tumorseg, None, None
    layer_2_args = {"cmap": cmap, "norm": norm, "alpha": 0.9, "interpolation": "none"}
    for ind, ax_slice, cor_slice in zip(range(num_slices), axial_slices, coronal_slices):
        
        image_tensor[1, ind, 1] = tumorseg_data[:, :, ax_slice]
        image_tensor[1, ind+num_slices, 1] = tumorseg_data[:, cor_slice, :]

    # Layer 3: None, None, Model, None
    layer_3_args = {"cmap": "inferno", "alpha": 0.90, "vmin": 0.0, "vmax": 1.0, "interpolation": "none"}
    for ind, ax_slice, cor_slice in zip(range(num_slices), axial_slices, coronal_slices):
        
        image_tensor[2, ind, 2] = model_data[:, :, ax_slice]
        image_tensor[2, ind+num_slices, 2] = model_data[:, cor_slice, :]

    # Imshow arguments
    imshow_args = [layer_1_args, layer_2_args, layer_3_args]

    grid_plot(
            image_tensor=image_tensor,
            imshow_args=imshow_args,
            header=header,
            col_titles=col_titles,
            row_titles=row_titles,
            outfile=outfile,
            legend_handles=patches
            )


def plot_recurrence_multislice(patient_identifier: str, exam_identifier_pre: str, exam_identifier_followup: str,
                               exam_dir_preop: Path, exam_dir_followup: Path, outfile: str,
                               classes_of_interest: List[int] = [1, 2, 3]) -> None:

    n_layers = 2    # one layer for each imshow config

    # Paths
    t1c_pre_dir = MODALITY_STRIPPED_SCHEMA.format(base_dir=exam_dir_preop, modality="t1c")
    t1c_post_dir = LONGITUDINAL_WARP_SCHEMA.format(base_dir=exam_dir_followup)
    #t1c_post_dir = MODALITY_STRIPPED_SCHEMA.format(base_dir=exam_dir_followup, modality="t1c")  # non-co-registered version
    tumor_seg_dir = TUMORSEG_SCHEMA.format(base_dir=exam_dir_preop)
    recurrence_seg_dir = RECURRENCE_SCHEMA.format(base_dir=exam_dir_followup)
    #recurrence_seg_dir = TUMORSEG_SCHEMA.format(base_dir=exam_dir_followup)  # non-co-registered version

    # Load images
    t1c_data_pre = load_mri_data(t1c_pre_dir)
    seg_data_pre = load_mri_data(tumor_seg_dir)
    t1c_data_post = load_mri_data(t1c_post_dir)
    seg_data_post = load_mri_data(recurrence_seg_dir)

    seg_data_post[seg_data_post==4] = 0  # ignore ressection cavity label

    # Compute tumor center of mass
    center = compute_center_of_mass(seg_data_pre, t1c_data_pre, classes_of_interest)

    # Create axial/coronal slices
    step_size = 10
    num_slices = 5
    patient_dim = t1c_data_pre.shape
    axial_slices, coronal_slices = get_slices(center, num_slices, step_size, patient_dim)

    # Tumor segmentation legend (1: non enhancing, 2: edema, 3: enhancing)
    cmap, norm, patches = get_cmap_norm_patches_tumorseg(classes_of_interest)

    # Titles
    col_titles = ["T1C (preop)", "T1C (preop)+Tumor", "T1C (follow up)", "T1C (follow up) + Recurrence"]
    row_titles = axial_slices + coronal_slices
    header = (
            f"Patient: {patient_identifier}\n"
            f"Exam (preop): {exam_identifier_pre}\n"
            f"Exam (follow up): {exam_identifier_followup}\n"
            f"CoM slice (axial/coronal): {center[2]}/{center[1]}\n"
            )

    # Build image tensor
    image_tensor = np.empty((n_layers, num_slices*2, 4), dtype=object)

    # Layer 1: T1c (pre), T1c (pre), T1c (post, T1c (post)
    layer_1_args = {"cmap": "gray", "interpolation": "none"}
    for ind, ax_slice, cor_slice in zip(range(num_slices), axial_slices, coronal_slices):

        image_tensor[0, ind, 0] = t1c_data_pre[:, :, ax_slice]
        image_tensor[0, ind, 1] = t1c_data_pre[:, :, ax_slice]
        image_tensor[0, ind, 2] = t1c_data_post[:, :, ax_slice]
        image_tensor[0, ind, 3] = t1c_data_post[:, :, ax_slice]

        image_tensor[0, ind+num_slices, 0] = t1c_data_pre[:, cor_slice, :]
        image_tensor[0, ind+num_slices, 1] = t1c_data_pre[:, cor_slice, :]
        image_tensor[0, ind+num_slices, 2] = t1c_data_post[:, cor_slice, :]
        image_tensor[0, ind+num_slices, 3] = t1c_data_post[:, cor_slice, :]

    # Layer 2: None, Tumorseg (pre), None, Tumorseg (post)
    layer_2_args = {"cmap": cmap, "norm": norm, "alpha": 0.9, "interpolation": "none"}
    for ind, ax_slice, cor_slice in zip(range(num_slices), axial_slices, coronal_slices):

        image_tensor[1, ind, 1] = seg_data_pre[:, :, ax_slice]
        image_tensor[1, ind, 3] = seg_data_post[:, :, ax_slice]
        image_tensor[1, ind+num_slices, 1] = seg_data_pre[:, cor_slice, :]
        image_tensor[1, ind+num_slices, 3] = seg_data_post[:, cor_slice, :]

    # Imshow arguments
    imshow_args = [layer_1_args, layer_2_args]

    grid_plot(
            image_tensor=image_tensor,
            imshow_args=imshow_args,
            header=header,
            col_titles=col_titles,
            row_titles=row_titles,
            outfile=outfile,
            legend_handles=patches
            )


def plot_pipeline(patient_identifier: str, exam_identifier_pre: str, exam_identifier_followup: str,
                  exam_dir_preop: Path, exam_dir_followup: Path, outfile: str,
                  classes_of_interest: List[int] = [1, 2, 3]) -> None:

    n_layers = 3    # one layer for each imshow config
    modalities = ["t1c", "t1", "t2", "flair"]
    tissues = ["gm", "wm", "csf"]

    # Paths
    preop_converted_files = {modality: MODALITY_CONVERTED_SCHEMA.format(base_dir=exam_dir_preop, modality=modality) for modality in modalities}
    followup_converted_files = {modality: MODALITY_CONVERTED_SCHEMA.format(base_dir=exam_dir_followup, modality=modality) for modality in modalities}

    preop_stripped_files = {modality: MODALITY_STRIPPED_SCHEMA.format(base_dir=exam_dir_preop, modality=modality) for modality in modalities}
    followup_stripped_files = {modality: MODALITY_STRIPPED_SCHEMA.format(base_dir=exam_dir_followup, modality=modality) for modality in modalities}

    tumor_seg_file = TUMORSEG_SCHEMA.format(base_dir=exam_dir_preop)
    recurrence_seg_file = TUMORSEG_SCHEMA.format(base_dir=exam_dir_followup)

    tissue_seg_file = TISSUE_SEG_SCHEMA.format(base_dir=exam_dir_preop)
    tissue_pbmaps_files = {tissue: TISSUE_PBMAP_SCHEMA.format(base_dir=exam_dir_preop, tissue=tissue) for tissue in tissues}

    brain_mask_file = BRAIN_MASK_SCHEMA.format(base_dir=exam_dir_preop)
    tumor_mask_file = TUMORSEG_CORE_SCHEMA.format(base_dir=exam_dir_preop)

    longitudinal_t1c_file = LONGITUDINAL_WARP_SCHEMA.format(base_dir=exam_dir_followup)
    longitudinal_rec_file = RECURRENCE_SCHEMA.format(base_dir=exam_dir_followup)

    model_output_file = PREDICTION_OUTPUT_SCHEMA.format(base_dir=exam_dir_preop, algo_id="sbtc")

    standard_plan_file = STANDARD_PLAN_SCHEMA.format(base_dir=exam_dir_preop)
    model_plan_file = MODEL_PLAN_SCHEMA.format(base_dir=exam_dir_preop, algo_id="sbtc")

    # Load images
    t1c_data_pre = load_mri_data(preop_stripped_files["t1c"])
    seg_data_pre = load_mri_data(tumor_seg_file)
    seg_data_post = load_mri_data(recurrence_seg_file)
    longitudinal_rec = load_mri_data(longitudinal_rec_file)
    model_data = load_and_resample_mri_data(model_output_file, resample_params=t1c_data_pre.shape, interp_type=1)

    # Compute tumor center of mass
    center = compute_center_of_mass(seg_data_pre, t1c_data_pre, classes_of_interest)
    ax_slice = center[2]

    # Tumor segmentation legend (1: non enhancing, 2: edema, 3: enhancing)
    cmap, norm, patches = get_cmap_norm_patches_tumorseg(classes_of_interest)

    # Titles
    col_titles = ["T1c", "T1", "T2", "Flair"]
    row_titles = ["Converted (preop)", "Converted (follow)", "Stripped (preop)", "Stripped (follow)", "Tumorseg", "Tissueseg", "Masks"]
    header = (
            f"Patient: {patient_identifier}\n"
            f"Exam (preop): {exam_identifier_pre}\n"
            f"Exam (postop): {exam_identifier_followup}\n"
            f"CoM slice (axial/coronal): {center[2]}/{center[1]}\n"
            )

    # Build image tensor
    image_tensor = np.empty((n_layers, 7, 4), dtype=object)

    # Layer 1: T1c, T1c, T1c, Tissueseg
    layer_1_args = {"cmap": "gray", "interpolation": "none"}

    #tmp = load_and_resample_mri_data(followup_converted_files["t1c"], resample_params=t1c_data_pre.shape, interp_type=1)[:, :, ax_slice]  #TODO pad and resample for nicer visualization before registration
    #tmp = tmp[::2, :]
    #t1c_converted_followup = np.zeros_like(t1c_data_pre[:,:,0])
    #t1c_converted_followup[60:180, :] = tmp
    t1c_converted_followup = load_and_resample_mri_data(followup_converted_files["t1c"], resample_params=t1c_data_pre.shape, interp_type=1)[:, :, ax_slice]
    
    image_tensor[0, 0, 0] = load_and_resample_mri_data(preop_converted_files["t1c"], resample_params=t1c_data_pre.shape, interp_type=1)[:, :, ax_slice]
    image_tensor[0, 0, 1] = load_and_resample_mri_data(preop_converted_files["t1"], resample_params=t1c_data_pre.shape, interp_type=1)[::-1, :, ax_slice]
    image_tensor[0, 0, 2] = load_and_resample_mri_data(preop_converted_files["t2"], resample_params=t1c_data_pre.shape, interp_type=1)[::-1, :, ax_slice]
    image_tensor[0, 0, 3] = load_and_resample_mri_data(preop_converted_files["flair"], resample_params=t1c_data_pre.shape, interp_type=1)[::-1, :, ax_slice]

    image_tensor[0, 1, 0] = t1c_converted_followup
    image_tensor[0, 1, 1] = load_and_resample_mri_data(followup_converted_files["t1"], resample_params=t1c_data_pre.shape, interp_type=1)[::-1, :, ax_slice]
    image_tensor[0, 1, 2] = load_and_resample_mri_data(followup_converted_files["t2"], resample_params=t1c_data_pre.shape, interp_type=1)[::-1, :, ax_slice]
    image_tensor[0, 1, 3] = load_and_resample_mri_data(followup_converted_files["flair"], resample_params=t1c_data_pre.shape, interp_type=1)[::-1, :, ax_slice]
    
    image_tensor[0, 2, 0] = load_mri_data(preop_stripped_files["t1c"])[:, :, ax_slice]
    image_tensor[0, 2, 1] = load_mri_data(preop_stripped_files["t1"])[:, :, ax_slice]
    image_tensor[0, 2, 2] = load_mri_data(preop_stripped_files["t2"])[:, :, ax_slice]
    image_tensor[0, 2, 3] = load_mri_data(preop_stripped_files["flair"])[:, :, ax_slice]

    image_tensor[0, 3, 0] = load_mri_data(followup_stripped_files["t1c"])[:, :, ax_slice]
    image_tensor[0, 3, 1] = load_mri_data(followup_stripped_files["t1"])[:, :, ax_slice]
    image_tensor[0, 3, 2] = load_mri_data(followup_stripped_files["t2"])[:, :, ax_slice]
    image_tensor[0, 3, 3] = load_mri_data(followup_stripped_files["flair"])[:, :, ax_slice]

    image_tensor[0, 4, 0] = load_mri_data(preop_stripped_files["t1c"])[:, :, ax_slice]
    image_tensor[0, 4, 1] = load_mri_data(followup_stripped_files["t1c"])[:, :, ax_slice]
    image_tensor[0, 4, 2] = load_mri_data(longitudinal_t1c_file)[:, :, ax_slice]
    image_tensor[0, 4, 3] = load_mri_data(tumor_mask_file)[:, :, ax_slice]

    image_tensor[0, 5, 0] = load_mri_data(tissue_seg_file)[:, :, ax_slice]
    image_tensor[0, 5, 1] = load_mri_data(tissue_pbmaps_files["gm"])[:, :, ax_slice]
    image_tensor[0, 5, 2] = load_mri_data(tissue_pbmaps_files["wm"])[:, :, ax_slice]
    image_tensor[0, 5, 3] = load_mri_data(tissue_pbmaps_files["csf"])[:, :, ax_slice]

    image_tensor[0, 6, 0] = load_mri_data(preop_stripped_files["t1c"])[:, :, ax_slice]
    image_tensor[0, 6, 1] = load_mri_data(preop_stripped_files["t1c"])[:, :, ax_slice]
    image_tensor[0, 6, 2] = load_mri_data(preop_stripped_files["t1c"])[:, :, ax_slice]
    image_tensor[0, 6, 3] = load_mri_data(preop_stripped_files["t1c"])[:, :, ax_slice]

    # Layer 2: None, Tumorseg, None, None
    layer_2_args = {"cmap": cmap, "norm": norm, "alpha": 0.9, "interpolation": "none"}
        
    image_tensor[1, 4, 0] = load_mri_data(tumor_seg_file)[:, :, ax_slice]
    image_tensor[1, 4, 1] = load_mri_data(recurrence_seg_file)[:, :, ax_slice]
    image_tensor[1, 4, 2] = load_mri_data(longitudinal_rec_file)[:, :, ax_slice]

    image_tensor[1, 6, 2] = load_mri_data(standard_plan_file)[:, :, ax_slice]
    image_tensor[1, 6, 3] = load_mri_data(model_plan_file)[:, :, ax_slice]

    # Layer 3: None, None, Model, None
    layer_3_args = {"cmap": "inferno", "alpha": 0.90, "vmin": 0.0, "vmax": 1.0, "interpolation": "none"}
        
    image_tensor[2, 6, 1] = model_data[:, :, ax_slice]

    # Imshow arguments
    imshow_args = [layer_1_args, layer_2_args, layer_3_args]

    grid_plot(
            image_tensor=image_tensor,
            imshow_args=imshow_args,
            header=header,
            col_titles=col_titles,
            row_titles=row_titles,
            outfile=outfile,
            legend_handles=patches
            )


def plot_plans(patient_identifier: str, exam_identifier_pre: str, exam_identifier_followup: str,
               exam_dir_preop: Path, exam_dir_followup: Path, outfile: str,
               classes_of_interest: List[int] = [1, 2, 3]) -> None:

    n_layers = 3    # one layer for each imshow config

    # Paths
    longitudinal_t1c_file = LONGITUDINAL_WARP_SCHEMA.format(base_dir=exam_dir_followup)
    longitudinal_rec_file = RECURRENCE_SCHEMA.format(base_dir=exam_dir_followup)
    model_output_file = PREDICTION_OUTPUT_SCHEMA.format(base_dir=exam_dir_preop, algo_id="sbtc")
    standard_plan_file = STANDARD_PLAN_SCHEMA.format(base_dir=exam_dir_preop)
    model_plan_file = MODEL_PLAN_SCHEMA.format(base_dir=exam_dir_preop, algo_id="sbtc")

    # Load images
    t1c_data_post = load_mri_data(longitudinal_t1c_file)
    longitudinal_rec = load_mri_data(longitudinal_rec_file)
    model_data = load_and_resample_mri_data(model_output_file, resample_params=t1c_data_post.shape, interp_type=1)
    standard_plan = load_mri_data(standard_plan_file)
    model_plan = load_mri_data(model_plan_file)

    # Ignore resection cavity label
    longitudinal_rec[longitudinal_rec==4] = 0

    # Compute tumor center of mass
    center = compute_center_of_mass(longitudinal_rec, t1c_data_post, classes_of_interest)
    step_size = 10
    num_slices = 5
    patient_dim = t1c_data_post.shape
    axial_slices, coronal_slices = get_slices(center, num_slices, step_size, patient_dim)

    # Tumor segmentation legend (1: non enhancing, 2: edema, 3: enhancing)
    cmap, norm, patches = get_cmap_norm_patches_tumorseg(classes_of_interest)

    # Titles
    col_titles = ["T1c", "Standard", "Model-based", "Model"]
    row_titles = axial_slices + coronal_slices
    header = (
            f"Patient: {patient_identifier}\n"
            f"Exam (preop): {exam_identifier_pre}\n"
            f"Exam (postop): {exam_identifier_followup}\n"
            f"CoM slice (axial/coronal): {center[2]}/{center[1]}\n"
            )

    # Build image tensor
    image_tensor = np.empty((n_layers, num_slices*2, 4), dtype=object)

    # Layer 1: T1c, T1c, T1c, Tissueseg
    layer_1_args = {"cmap": "gray", "interpolation": "none"}

    layer_1_args = {"cmap": "gray"}
    for ind, ax_slice, cor_slice in zip(range(num_slices), axial_slices, coronal_slices):
        image_tensor[0, ind, 0] = t1c_data_post[:, :, ax_slice]
        image_tensor[0, ind, 1] = t1c_data_post[:, :, ax_slice]
        image_tensor[0, ind, 2] = t1c_data_post[:, :, ax_slice]
        image_tensor[0, ind, 3] = t1c_data_post[:, :, ax_slice]

        image_tensor[0, ind+num_slices, 0] = t1c_data_post[:, cor_slice, :]
        image_tensor[0, ind+num_slices, 1] = t1c_data_post[:, cor_slice, :]
        image_tensor[0, ind+num_slices, 2] = t1c_data_post[:, cor_slice, :]
        image_tensor[0, ind+num_slices, 3] = t1c_data_post[:, cor_slice, :]

    layer_2_args = {"cmap": cmap, "norm": norm, "alpha": 0.9, "interpolation": "none"}
    for ind, ax_slice, cor_slice in zip(range(num_slices), axial_slices, coronal_slices):
        image_tensor[1, ind, 0] = longitudinal_rec[:, :, ax_slice]
        image_tensor[1, ind, 1] = standard_plan[:, :, ax_slice]
        image_tensor[1, ind, 2] = model_plan[:, :, ax_slice]

        image_tensor[1, ind+num_slices, 0] = longitudinal_rec[:, cor_slice, :]
        image_tensor[1, ind+num_slices, 1] = standard_plan[:, cor_slice, :]
        image_tensor[1, ind+num_slices, 2] = model_plan[:, cor_slice, :]

    layer_3_args = {"cmap": "inferno", "alpha": 0.90, "vmin": 0.0, "vmax": 1.0, "interpolation": "none"}
    for ind, ax_slice, cor_slice in zip(range(num_slices), axial_slices, coronal_slices):
        image_tensor[2, ind, 3] = model_data[:, :, ax_slice]
        image_tensor[2, ind+num_slices, 3] = model_data[:, cor_slice, :]

    # Imshow arguments
    imshow_args = [layer_1_args, layer_2_args, layer_3_args]

    grid_plot(
            image_tensor=image_tensor,
            imshow_args=imshow_args,
            header=header,
            col_titles=col_titles,
            row_titles=row_titles,
            outfile=outfile,
            legend_handles=patches
            )


def plot_full_brain(patient_identifier: str, exam_identifier_pre: str, exam_identifier_followup: str,
                    exam_dir_preop: Path, exam_dir_followup: Path, outfile: str,
                    classes_of_interest: List[int] = [1, 2, 3]) -> None:

    n_layers = 3    # one layer for each imshow config
    modalities = ["t1c", "t1", "t2", "flair"]

    # Paths
    preop_stripped_files = {modality: MODALITY_STRIPPED_SCHEMA.format(base_dir=exam_dir_preop, modality=modality) for modality in modalities}
    followup_stripped_files = {modality: MODALITY_STRIPPED_SCHEMA.format(base_dir=exam_dir_followup, modality=modality) for modality in modalities}
    tumor_seg_file = TUMORSEG_SCHEMA.format(base_dir=exam_dir_preop)
    recurrence_seg_file = TUMORSEG_SCHEMA.format(base_dir=exam_dir_followup)
    standard_plan_file = STANDARD_PLAN_SCHEMA.format(base_dir=exam_dir_preop)

    # Load images
    t1c_pre = load_mri_data(preop_stripped_files["t1c"])
    t1c_post = load_mri_data(followup_stripped_files["t1c"])
    
    try:
        flair_pre = load_mri_data(preop_stripped_files["flair"])
    except:
        flair_pre = np.ones(t1c_pre.shape)
        print(f"Preop FLAIR MRI not found. Conitnuing with empty image.")
    try:
        flair_post = load_mri_data(followup_stripped_files["flair"])
    except:
        flair_post = np.ones(t1c_post.shape)
        print(f"Followup FLAIR MRI not found. Conitnuing with empty image.")

    tumor_seg = load_mri_data(tumor_seg_file)
    recurrence_seg = load_mri_data(recurrence_seg_file)
    
    try:
        standard_plan = load_mri_data(standard_plan_file)
    except:
        standard_plan = np.zeros(t1c_pre.shape)
        print(f"Standard plan not found. Continuing with emtpy image.")

    # Generate projections
    tumor_projections = [get_segmentation_projection(tumor_seg, label=label, axis=2) for label in classes_of_interest]
    recurrence_projections = [get_segmentation_projection(recurrence_seg, label=label, axis=2) for label in classes_of_interest]
    radplan_projection = get_segmentation_projection(standard_plan, label=1, axis=2)

    # Ignore resection cavity label
    recurrence_seg[recurrence_seg==4] = 0  # ignore cavity

    # Compute tumor center of mass
    #center = compute_center_of_mass(longitudinal_rec, t1c_data_post, classes_of_interest)
    center = [d // 2 for d in t1c_post.shape]
    step_size = 10
    num_slices = 15
    patient_dim = t1c_post.shape
    axial_slices = [k*10 for k in range(0, 15)]
    coronal_slices = axial_slices
    #axial_slices, coronal_slices = get_slices(center, num_slices, step_size, patient_dim)

    # Tumor segmentation legend (1: non enhancing, 2: edema, 3: enhancing)
    cmap, norm, patches = get_cmap_norm_patches_tumorseg(classes_of_interest)

    # Titles
    col_titles = ["Projection"] + axial_slices
    row_titles = ["T1c\n(preop)", "FLAIR\n(preop)", "TumorSeg\n(preop)", "RecurrenceSeg\n(followup)", "FLAIR\n(followup)", "T1c\n(followup)", "StandardPlan\n(followup)"]
    header = (
            f"Patient: {patient_identifier}\n"
            f"Exam (preop): {exam_identifier_pre}\n"
            f"Exam (postop): {exam_identifier_followup}\n"
            f"CoM slice (axial/coronal): {center[2]}/{center[1]}\n"
            )

    # Build image tensor
    image_tensor = np.empty((n_layers, len(row_titles), num_slices+1), dtype=object)

    # Layer 1: T1c, T1c, T1c, Tissueseg
    layer_1_args = {"cmap": "gray", "interpolation": "none"}
    for ind, ax_slice, cor_slice in zip(range(num_slices), axial_slices, coronal_slices):
        image_tensor[0, 0, ind+1] = t1c_pre[:, :, ax_slice]
        image_tensor[0, 1, ind+1] = flair_pre[:, :, ax_slice]
        image_tensor[0, 2, ind+1] = t1c_pre[:, :, ax_slice]
        image_tensor[0, 3, ind+1] = t1c_post[:, :, ax_slice]
        image_tensor[0, 4, ind+1] = flair_post[:, :, ax_slice]
        image_tensor[0, 5, ind+1] = t1c_post[:, :, ax_slice]
        image_tensor[0, 6, ind+1] = standard_plan[:, :, ax_slice]

    # Layer 2: Tumor segmentations
    layer_2_args = {"cmap": cmap, "norm": norm, "alpha": 0.9, "interpolation": "none"}
    for ind, ax_slice, cor_slice in zip(range(num_slices), axial_slices, coronal_slices):
        image_tensor[1, 2, ind+1] = tumor_seg[:, :, ax_slice]
        image_tensor[1, 3, ind+1] = recurrence_seg[:, :, ax_slice]

    # Layer 3: Projections
    layer_3_args = {"cmap": "gray", "interpolation": "none"}
    image_tensor[2, 0, 0] = tumor_projections[0]
    image_tensor[2, 1, 0] = tumor_projections[1]
    image_tensor[2, 2, 0] = tumor_projections[2]
    image_tensor[2, 3, 0] = recurrence_projections[0]
    image_tensor[2, 4, 0] = recurrence_projections[1]
    image_tensor[2, 5, 0] = recurrence_projections[2]
    image_tensor[2, 6, 0] = radplan_projection

    # Imshow arguments
    imshow_args = [layer_1_args, layer_2_args, layer_3_args]

    grid_plot(
            image_tensor=image_tensor,
            imshow_args=imshow_args,
            header=header,
            col_titles=col_titles,
            row_titles=row_titles,
            outfile=outfile,
            legend_handles=patches
            )


def plot_difference(img1_file, img2_file, identifier, outfile) -> None:

    n_layers = 2

    # Load images
    img1 = load_mri_data(img1_file)
    img2 = load_mri_data(img2_file)
    diff = (img1 - img2)

    if img1.shape != img2.shape:
        raise ValueError(f"Dimension mismatch. Images need to be the same dimension.")

    center = [d // 2 for d in img1.shape]
    step_size = 10
    num_slices = 15
    axial_slices = [k*10 for k in range(0, 15)]
    coronal_slices = axial_slices

    # Titles
    col_titles = axial_slices
    row_titles = ["img1", "img2", "difference"]
    header = (
            f"Patient: {identifier}\n"
            f"Volume 1: {np.sum(img1 > 0)}\n"
            f"Volume 2: {np.sum(img2 > 0)}\n"
            f"Difference: {np.sum(diff > 0)}"
            )

    # Build image tensor
    image_tensor = np.empty((n_layers, len(row_titles), len(col_titles)), dtype=object)

    # Layer 1: T1c, T1c, T1c, Tissueseg
    layer_1_args = {"cmap": "gray", "interpolation": "none", "vmin": 0, "vmax": 1}
    for ind, ax_slice, cor_slice in zip(range(num_slices), axial_slices, coronal_slices):
        image_tensor[0, 0, ind] = img1[:, :, ax_slice]
        image_tensor[0, 1, ind] = img2[:, :, ax_slice]

    layer_2_args = {"cmap": "inferno", "vmin": -1.0, "vmax": 1.0, "interpolation": "none"}
    for ind, ax_slice, cor_slice in zip(range(num_slices), axial_slices, coronal_slices):
        image_tensor[1, 2, ind] = diff[:, :, ax_slice]

    # Imshow arguments
    imshow_args = [layer_1_args, layer_2_args]

    grid_plot(
            image_tensor=image_tensor,
            imshow_args=imshow_args,
            header=header,
            col_titles=col_titles,
            row_titles=row_titles,
            outfile=outfile
            )


def plot_tumor_volumes(recurrence_exam_paths: List[Path], outfile: Path, bins="auto") -> None:
    """
    Plot a histogram of tumor volumes.
    """
    if not recurrence_exam_paths:
        raise ValueError("The 'volumes' sequence is empty.")

    volumes = []
    print(f"Got {len(recurrence_exam_paths)} exam paths. Extracting volumes...")
    for ind, exam_path in enumerate(recurrence_exam_paths):
        print(f"{ind} / {len(recurrence_exam_paths)}")
        recurrence_seg_file = TUMORSEG_SCHEMA.format(base_dir=exam_path)
        if recurrence_seg_file.is_file():
            recurrence_seg = load_mri_data(TUMORSEG_SCHEMA.format(base_dir=exam_path))
            recurrence_seg[recurrence_seg==2] = 0  # ingores edema
            recurrence_seg[recurrence_seg==3] = 1
            recurrence_seg[recurrence_seg==4] = 0  # ignores cavity
            volumes.append(np.sum(recurrence_seg))
        else:
            print(f"{recurrence_seg_file} does not exist.")

    fig, ax = plt.subplots()
    ax.hist(volumes, bins=bins, edgecolor="black")
    ax.set_title("Distribution of Tumor Volumes")
    ax.set_xlabel("Volume (mm³)")
    ax.set_ylabel("Frequency")
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    Path(outfile).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile, format="pdf")
    print(f"Plot saved as {outfile}")
    plt.close(fig)


def scatter_plot(xvals: List[float], yvals: List[float], outfile: Path) -> None:
    """
    Create a scatter plot of paired numeric values.
    """
    if len(xvals) != len(yvals):
        raise ValueError("xvals and yvals must be the same length.")
    if len(xvals) == 0:
        raise ValueError("Input sequences are empty.")

    fig, ax = plt.subplots()
    ax.scatter(xvals, yvals)
    ax.set_title("")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    Path(outfile).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile, format="pdf")
    print(f"Plot saved as {outfile}")
    plt.close(fig)


if __name__ == "__main__":
    # Example:
    # python gbm_bench/utils/visualization.py

    """
    plot_model_multislice(
            patient_identifier="RHUH-0001",
            exam_identifier="01-25-2015",
            algorithm_identifier="LMI",
            exam_dir=Path("test_data/exam1/"),
            outfile="tmp_visualization/test_multislice.pdf"
            )

    plot_recurrence_multislice(
            patient_identifier="RHUH-0001",
            exam_identifier_pre="Pre",
            exam_identifier_followup="Post",
            exam_dir_preop=Path("test_data/exam1/"),
            exam_dir_followup=Path("test_data/exam3/"),
            outfile="tmp_visualization/test_longitudinal.pdf"
            )

    plot_pipeline(
            patient_identifier="RHUH-0030",
            exam_identifier_pre="Pre",
            exam_identifier_followup="Post",
            exam_dir_preop=Path("/mnt/Drive2/lucas/datasets/RHUH-GBM/Images/DICOM/RHUH-GBM/RHUH-0030/11-25-2013-NA-RM CEREBRALC-02749"),
            exam_dir_followup=Path("/mnt/Drive2/lucas/datasets/RHUH-GBM/Images/DICOM/RHUH-GBM/RHUH-0030/08-11-2014-NA-RM DE CEREBRO SINCON CONTRASTE-96321"),
            outfile="tmp_visualization/pipeline.pdf"
            )
    
    plot_plans(
            patient_identifier="RHUH-0011",
            exam_identifier_pre="Pre",
            exam_identifier_followup="Post",
            exam_dir_preop=Path("/mnt/Drive2/lucas/datasets/RHUH-GBM/Images/DICOM/RHUH-GBM/RHUH-0008/09-27-2015-NA-Craneo-26679"),
            exam_dir_followup=Path("/mnt/Drive2/lucas/datasets/RHUH-GBM/Images/DICOM/RHUH-GBM/RHUH-0008/04-05-2016-NA-RM CEREBRO-94961"),
            outfile="tmp_visualization/plans.pdf"
            )
    
    plot_full_brain(
            patient_identifier="RHUH-0024",
            exam_identifier_pre="Pre",
            exam_identifier_followup="Post",
            exam_dir_preop=Path("/mnt/Drive2/lucas/datasets/RHUH-GBM/Images/DICOM/RHUH-GBM/RHUH-0024/11-10-2013-NA-Craneo-58463"),
            exam_dir_followup=Path("/mnt/Drive2/lucas/datasets/RHUH-GBM/Images/DICOM/RHUH-GBM/RHUH-0024/01-29-2014-NA-RM CEREBRO-96283"),
            outfile="tmp_visualization/qualitycontrol.pdf"
            )
    """
    plot_difference(
            img1_file="/home/home/lucas/jonasplans/standardPlan.nii.gz",
            img2_file="/mnt/Drive2/lucas/datasets/RHUH-GBM/Images/NIfTI/RHUH-GBM/RHUH-0012/0/processed/tumor_segmentation/standard_plan.nii.gz",
            identifier="tgm016",
            outfile="tmp/standard_difference.pdf")
