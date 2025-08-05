import os
import json
import argparse
import numpy as np
import nibabel as nib
from pathlib import Path
from loguru import logger
from typing import Any, Dict
from scipy.ndimage import center_of_mass, distance_transform_edt
from gbm_bench.evaluation.metrics import coverage
from gbm_bench.utils.utils import load_mri_data, load_and_resample_mri_data, is_binary_array
from gbm_bench.utils.constants import (
        BRAIN_MASK_SCHEMA,
        METRICS_SCHEMA,
        MODALITY_STRIPPED_SCHEMA,
        MODEL_PLAN_SCHEMA,
        RECURRENCE_SCHEMA,
        STANDARD_PLAN_SCHEMA,
        TISSUE_SEG_SCHEMA,
        TISSUE_LABELS,
        TUMORSEG_SCHEMA,
        TUMORSEG_CORE_SCHEMA
        )


def create_standard_plan(core_segmentation: np.ndarray, ctv_margin: int) -> np.ndarray:
    """
    Creates a target volume mask by dilating the tumor core segmentation with ctv_margin
    
    Parameters:
        core_segmentation (np.ndarray): A NumPy array representing the core segmentation mask, where non-zero
            values indicate the region of interest.
        ctv_margin (int): The margin to dilate the core segmentation.

    Returns:
        np.ndarray: A binary NumPy array of the same shape as core_segmentation, where
            pixels within the ctv_margin from the core segmentation are True, and
            all other pixels are False.
    """
    if ctv_margin <= 0:
        raise ValueError("ctv_margin must be a positive int.")
    distance_transform = distance_transform_edt(~ (core_segmentation >0))
    dilated_core = distance_transform <= ctv_margin
    return dilated_core.astype(np.int32)


def find_threshold(volume: np.ndarray, target_volume: float, tolerance: float = 0.01, initial_threshold: float = 0.2, maxIter: int = 10000):
    """
    Compute the threshold that produces a specified volume after masking the input array using the threshold.

    Parameters:
        volume (np.ndarray): A NumPy array representing the input volume to be thresholded (tumor cell concentration).
        target_volume (float): The desired volume size.
        tolerance (float, optional): The acceptable relative difference between the target volume and the thresholded volume. Default is 0.01 (1%).
        initial_threshold (float, optional): The starting threshold value for the segmentation. Should be between 0 and 1.
        max_iter (int, optional): The maximum number of iterations to perform. If the threshold is not found within this number of iterations,
            the function will terminate.

    Returns:
        float: The threshold value that segments the volume to match the target volume within the specified tolerance.
            Returns 1.01 if the threshold exceeds the valid range (0, 1), indicating an "above the model" condition.
            Returns 0 if the maximum number of iterations is reached without finding a suitable threshold.
    """

    if np.sum(volume > 0) < target_volume:
        print("Volume is too small")
        return 0

    # Define the initial threshold, step, and previous direction
    threshold = initial_threshold
    step = 0.1
    previous_direction = None

    # Calculate the current volume
    current_volume = np.sum(volume > threshold)

    # Iterate until the current volume is within the tolerance of the target volume

    while abs(current_volume - target_volume) / target_volume > tolerance:
        # Determine the current direction
        if current_volume > target_volume:
            direction = 'increase'
        else:
            direction = 'decrease'

        # Adjust the threshold
        if direction == 'increase':
            threshold += step
        else:
            threshold -= step

        # Check if the threshold is out of bounds
        if threshold < 0 or threshold > 1:
            return 1.01 #above the model

        # Update the current volume
        current_volume = np.sum(volume > threshold)

        # Reduce the step size if the direction has alternated
        if previous_direction and previous_direction != direction:
            step *= 0.5

        # Update the previous direction
        previous_direction = direction

        maxIter -= 1
        if maxIter < 0:
            print("Max Iter reached, no threshold found")
            return 0

    return threshold


def recurrence_coverage(recurrence_segmentation: np.ndarray, target_volume: np.ndarray) -> float:
    """
    Calculate the coverage of tumor recurrence by the treatment plan volume.

    Parameters:
        recurrence_segmentation (np.ndarray): A (boolean) NumPy array indicating the presence of tumor recurrence.
        target_volume (np.ndarray): A (boolean) NumPy array indicating the area covered by the treatment plan.

    Returns:
        float: The coverage ratio of the treatment plan over the tumor recurrence (0-1.0).
            Returns 1.0 if there is no tumor recurrence.
    """
    if not is_binary_array(recurrence_segmentation):
        raise ValueError(f"recurrence_segmentation values have to be in (True, False, 0, 1, 0.0, 1.0).")
    if not is_binary_array(target_volume):
        raise ValueError(f"target_volume values have to be in (True, False, 0, 1, 0.0, 1.0).")
    if recurrence_segmentation.shape != target_volume.shape:
        raise ValueError(f"Dimension mismatch between recurrence_segmentation and target_volume.")
    
    # If there is no recurrence, return 1
    if np.sum(recurrence_segmentation) <=  0.00001:
        return 1

    # Calculate the intersection between the recurrence and the plan
    intersection = np.logical_and(recurrence_segmentation, target_volume)

    # Calculate the coverage as the ratio of the intersection to the recurrence
    coverage = np.sum(intersection) / np.sum(recurrence_segmentation)
    return coverage


def evaluate_tumor_model(preop_dir: Path, followup_dir: Path, pred_file: Path, model_id: str, ctv_margin: int = 15, csf_mask: bool = False) -> Dict[str, Any]:
    """
    Evaluate a tumor model by computing recurrence coverage for standard and 
    model-based radiotherapy plans using MRI segmentation data.

    Parameters:
        preop_dir (Path): Directory to the preoperative exam that has been preprocessed. Should contain the folder with the output.
        followup_dir (Path): Directory to the postoperative exam that has been preprocessed. Should contain the folder with the output.
        pred_file (Path): File path containing model prediction MRI data.
        model_id (str): Identifier for the model. Used for the name of the output file.
        ctv_margin (int, optional): Margin used to expand the clinical target volume for the standard plan in mm. Defaults to 15.
        csf_mask (bool, optional): If true, does not consider predictions/recurrences in CSF in any way by masking it out.

    Returns:
        Dict[str, Any]: Dictionary with computed metrics
    """
    results = {}

    # Load data
    brain_mask_dir = BRAIN_MASK_SCHEMA.format(base_dir=str(preop_dir))
    if brain_mask_dir.exists():
        brain_mask = np.rint(load_mri_data(str(brain_mask_dir))).astype(np.int32)
    else:
        t1c_file = MODALITY_STRIPPED_SCHEMA.format(base_dir=str(preop_dir), modality="t1c")
        t1c = load_mri_data(str(t1c_file))
        background = np.min(t1c)
        brain_mask = np.rint(t1c > background)

    if csf_mask:
        tissue_segmentation_dir = TISSUE_SEG_SCHEMA.format(base_dir=str(preop_dir))
        tissue_segmentation = np.rint(load_mri_data(str(tissue_segmentation_dir))).astype(np.int32)
        brain_mask[tissue_segmentation==TISSUE_LABELS["csf"]] = 0

    tumorseg_dir = TUMORSEG_SCHEMA.format(base_dir=str(preop_dir))
    core_segmentation = np.rint(load_mri_data(str(tumorseg_dir))).astype(np.int32)
    core_segmentation[core_segmentation==2] = 0  # ignore edma
    core_segmentation[core_segmentation==3] = 1

    recurrence_dir = RECURRENCE_SCHEMA.format(base_dir=str(followup_dir))
    #recurrence_dir = TUMORSEG_SCHEMA.format(base_dir=str(followup_dir))  # tests without longitudinal registration
    recurrence_segmentation = np.rint(load_mri_data(str(recurrence_dir))).astype(np.int32)
    recurrence_segmentation[recurrence_segmentation == 1] = 1  # TODO: include necrosis as recurrence?
    recurrence_segmentation[recurrence_segmentation == 2] = 0  # ignore edema
    recurrence_segmentation[recurrence_segmentation == 3] = 1
    recurrence_segmentation[recurrence_segmentation == 4] = 0  # ignore resection cavity 
    recurrence_segmentation_all = np.rint(load_mri_data(recurrence_dir)).astype(np.int32)
    recurrence_segmentation_all[recurrence_segmentation_all == 1] = 1  # TODO
    recurrence_segmentation_all[recurrence_segmentation_all == 2] = 1
    recurrence_segmentation_all[recurrence_segmentation_all == 3] = 1
    recurrence_segmentation_all[recurrence_segmentation_all == 4] = 0

    model_prediction = load_and_resample_mri_data(str(pred_file), resample_params=core_segmentation.shape, interp_type=0)

    # Create standard plan
    standard_plan = create_standard_plan(core_segmentation, ctv_margin)
    standard_plan[brain_mask==0] = 0
    standard_plan_volume = np.sum(standard_plan)
    standard_plan_coverage = recurrence_coverage(recurrence_segmentation, standard_plan)
    standard_plan_coverage_all = recurrence_coverage(recurrence_segmentation_all, standard_plan)

    # Create model based plan
    tumor_threshold = find_threshold(model_prediction, standard_plan_volume, initial_threshold=0.2)
    model_plan = (model_prediction > tumor_threshold).astype(np.int32)
    #model_plan[brain_mask == 0] = 0
    model_recurrence_coverage = recurrence_coverage(recurrence_segmentation, model_plan)
    model_recurrence_coverage_all = recurrence_coverage(recurrence_segmentation_all, model_plan)

    # Save plans
    outfile_standard = STANDARD_PLAN_SCHEMA.format(base_dir=str(preop_dir))
    outfile_model = MODEL_PLAN_SCHEMA.format(base_dir=str(preop_dir), algo_id=model_id)
    
    standard_plan_nifti = nib.Nifti1Image(standard_plan, affine=np.eye(4))
    nib.save(standard_plan_nifti, outfile_standard)

    model_img = nib.Nifti1Image(model_plan, affine=np.eye(4))
    nib.save(model_img, outfile_model)

    # Compute metrics
    results["recurrence_coverage_standard"] = standard_plan_coverage
    results["recurrence_coverage_standard_all"] = standard_plan_coverage_all
    results["recurrence_coverage_model"] = model_recurrence_coverage
    results["recurrence_coverage_model_all"] = model_recurrence_coverage_all
    
    # Save results
    save_file = METRICS_SCHEMA.format(base_dir=followup_dir, algo_id=model_id)
    with open(save_file, 'w', encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Finished evaluation of {preop_dir}. Saved results to {save_file}.")

    print(tumorseg_dir)
    print(recurrence_dir)
    print(outfile_standard)
    return results


if __name__ == "__main__":
    # Example:
    # python gbm_bench/evaluation/evaluate.py -preop_dir test_data/exam1 -followup_dir test_data/exam3 -pred_file test_data/exam1/processed/growth_models/sbtc/sbtc_pred.nii.gz
    # python gbm_bench/evaluation/evaluate.py -preop_dir /mnt/Drive2/lucas/datasets/RHUH-GBM/Images/NIfTI/RHUH-GBM/RHUH-0024/0 -followup_dir /mnt/Drive2/lucas/datasets/RHUH-GBM/Images/NIfTI/RHUH-GBM/RHUH-0024/2 -pred_file /mnt/Drive2/lucas/datasets/RHUH-GBM/Images/NIfTI/RHUH-GBM/RHUH-0024/0/processed/growth_models/sbtc/sbtc_pred.nii.gz
    # python gbm_bench/evaluation/evaluate.py -preop_dir /mnt/Drive2/lucas/datasets/RHUH-GBM/Images/NIfTI/RHUH-GBM/RHUH-0001/0 -followup_dir /mnt/Drive2/lucas/datasets/RHUH-GBM/Images/NIfTI/RHUH-GBM/RHUH-0001/2 -pred_file /mnt/Drive2/lucas/datasets/RHUH-GBM/Images/NIfTI/RHUH-GBM/RHUH-0001/0/processed/growth_models/sbtc/sbtc_pred.nii.gz
    parser = argparse.ArgumentParser()
    parser.add_argument("-preop_dir", type=str, help="Path.")
    parser.add_argument("-followup_dir", type=str, help="Path.")
    parser.add_argument("-pred_file", type=str, help="Algorithm identifier, should be the same as the folder for the algorithm in patient/exam/processed/.")
    args = parser.parse_args()

    results = evaluate_tumor_model(
            preop_dir=Path(args.preop_dir),
            followup_dir=Path(args.followup_dir),
            pred_file=Path(args.pred_file),
            model_id="sbtc"  # "test"
            )

    print(results)
