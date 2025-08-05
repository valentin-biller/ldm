import os
import time
import argparse
from pathlib import Path
from loguru import logger
from typing import Optional
from gbm_bench.preprocessing.dicom_to_nifti import convert_nifti
from gbm_bench.preprocessing.tumor_segmentation import run_brats
from gbm_bench.preprocessing.norm_ss_coregistration import normalize, norm_ss_coregister, register_recurrence
from gbm_bench.preprocessing.tissue_segmentation import generate_registration_mask, run_tissue_seg_registration
from gbm_bench.utils.utils import make_symlink
from gbm_bench.utils.constants import (
        BRAIN_MASK_SCHEMA,
        DCM2NIIX_LOCATION,
        HEALTHY_BRAIN_MASK_SCHEMA,
        LONGITUDINAL_WARP_SCHEMA,
        MODALITY_CONVERTED_SCHEMA,
        MODALITY_STRIPPED_SCHEMA,
        RECURRENCE_SCHEMA,
        REGISTRATION_MASK_SCHEMA,
        TUMOR_LABELS,
        TUMORSEG_SCHEMA
        )


def preprocess_dicom(t1_dir: Path, t1c_dir: Path, t2_dir: Path, flair_dir: Path, pre_treatment: bool, outdir: Path,
                     dcm2niix_location: Path = DCM2NIIX_LOCATION, cuda_device: str = "0") -> None:
    """
    Performs a multitude of processing steps to prepare DICOM inputs for tumor growth models.
    DICOM data is first converted to NIfTI, then preprocess_exam is called for the rest. 

    Parameters:
         t1_dir (Path): Path to the directory with the DICOM files for T1.
         t1c_dir (Path): Path to the directory with the DICOM files for T1c.
         t2_dir (Path): Path to the directory with the DICOM files for T2.
         flair_dir (Path): Path to the directory with the DICOM files for Flair.
         pre_treatment (bool): Wether the provided DICOM are preop (True) or postop (False).
             Causes the BRATS segmentation algorithm to choose different models.
         outdir (Path): Base directory for the output. Usually exam directory.
         dcm2niix_location (Path, optional): The location of the dcm2niix executable.
         cuda_device (str): GPU device to use.

    Returns:
        None
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
    
    start_time_conversion = time.time()
    logger.info(f"Starting DICOM to NIfTI conversion.")
    dicom_modalities = {
            "t1" : t1_dir,
            "t1c" : t1c_dir,
            "t2" : t2_dir,
            "flair" : flair_dir
            }

    for modality_name, dicom_dir in dicom_modalities.items():
        # Remove suffix since dicom2niix adds .nii.gz automatically
        outfile_tmp = MODALITY_CONVERTED_SCHEMA.format(base_dir=outdir, modality=modality_name).with_suffix("").with_suffix("")

        convert_nifti(
                input_dir=dicom_dir,
                outfile=outfile_tmp,
                dcm2niix_location=dcm2niix_location
                )

    time_spent_conversion = time.time() - start_time_conversion
    logger.info(f"Finished conversion step in {time_spent_conversion:.2f} seconds.")

    preprocess_exam(
            outdir=outdir,
            pre_treatment=pre_treatment,
            cuda_device=cuda_device,
            perform_coregistration=True,
            perform_tumorseg=True,
            perform_tissueseg=pre_treatment
            )


def preprocess_nifti(t1_file: Path, t1c_file: Path, t2_file: Path, flair_file: Path, pre_treatment: bool, outdir: Path,
                     is_coregistered: bool, is_skull_stripped: bool, tumorseg_file: Optional[Path] = None,
                     cuda_device: str = "0", registration_mask_file: str = None) -> None:
    """
    Performs a multitude of precessing steps to prepare nifti inputs for tumor growth models. Allows passing
    available intermediate results like tumor segmentation or already skull stripped images. Starts by
    preparing the directory structure and then calling preprocess_exam.

    Parameters:
        t1_file (Path): Path to the file with the t1 data.
        t1c_file (Path): Path to the file with the t1c data.
        t2_file (Path): Path to the file with the t2 data.
        flair_file (Path): Path to the file with the flair data.
        pre_treatment (bool): Wether the provided DICOM are preop (True) or postop (False).
            Causes the BRATS segmentation algorithm to choose different models.
        outdir (Path): Base directory for the output. Usually exam directory.
        cuda_device (str): GPU device to use.
        is_coregistered (bool): True if the provided data has already been co-registered to SRI-24 space and skull stripped.
        is_skull_stripped (bool): True if the provided data has already been normalized, skull stripped and co-registered.
        tumorseg_file (Optional, Path): Path to the tumor segmentation.

    Returns:
        None
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device

    modality_dict = {
            "t1": t1_file,
            "t1c": t1c_file,
            "t2": t2_file,
            "flair": flair_file
            }

    # Prepare directory structure using if intermediate results are available
    if is_coregistered:
        logger.info(f"Running with provided skull stripped modality images.")
        for modality, path in modality_dict.items():
            if str(path) == ".":
                continue  # ignore missing modalities
            normalize(img_file=path, outfile=MODALITY_STRIPPED_SCHEMA.format(base_dir=outdir, modality=modality))
    else:
        for modality, path in modality_dict.items():
            make_symlink(src=path, dst=MODALITY_CONVERTED_SCHEMA.format(base_dir=outdir, modality=modality))

    if tumorseg_file is not None:
        logger.warning(f"Running with provided tumor segmentation {tumorseg_file}. Expected labels: {TUMOR_LABELS}")
        make_symlink(src=tumorseg_file, dst=TUMORSEG_SCHEMA.format(base_dir=outdir))

    preprocess_exam(
            outdir=outdir,
            pre_treatment=pre_treatment,
            cuda_device=cuda_device,
            perform_coregistration=not is_coregistered,
            perform_skull_stripping=not is_skull_stripped,
            perform_tumorseg=False if tumorseg_file is not None else True,
            perform_tissueseg=pre_treatment,
            registration_mask_file=registration_mask_file
            )


def preprocess_exam(outdir: Path, pre_treatment: bool, perform_coregistration: bool = True, 
                    perform_skull_stripping: bool = True, perform_tumorseg: bool = True, perform_tissueseg: bool = True,
                    cuda_device: str = "0", registration_mask_file: str = None) -> None:
    """
    This function is called by preprocess_nifti and preprocess_dicom and is not meant for end users. It assumes a fixed
    directory structure that is built by the aforementioned functions. It then performs normalization, co-registration,
    skull stripping, tumor segmentation and tissue segmentation to prepare the input for tumor growth models.

    Parameters:
        outdir (Path): Output directory containing either nifti converted data or skull stripped niftis in the
            expected directory structure.
        pre_treatment (bool): Wether the provided DICOM are preop (True) or postop (False).
            Causes the BRATS segmentation algorithm to choose different models.
        mask_tissueseg (bool): If true, masks out the tumor region for the tissue segmentation step.
        perform_coregistration (bool): If true, performs normalization, skull stripping, co-registration
        perform_skull_stripping (bool): If true, performs skull stripping during co-registration step.
        perform_tumorseg (bool): If true, performs tumor segmentation via BRATS.
        perform_tissueseg (bool): If true, performs tissue segmentation.
        cuda_device (str): GPU device to use.

    Returns:
        None
    """
    start_time = time.time()
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
    logger.info("Starting preprocessing.")

    if registration_mask_file is None:  # valentin
        # Step 1: Normalization, co-registration, skull stripping
        if perform_coregistration:
            norm_ss_coregister(
                    t1_file=MODALITY_CONVERTED_SCHEMA.format(base_dir=outdir, modality="t1"),
                    t1c_file=MODALITY_CONVERTED_SCHEMA.format(base_dir=outdir, modality="t1c"),
                    t2_file=MODALITY_CONVERTED_SCHEMA.format(base_dir=outdir, modality="t2"),
                    flair_file=MODALITY_CONVERTED_SCHEMA.format(base_dir=outdir, modality="flair"),
                    skull_strip=perform_skull_stripping,
                    outdir=outdir
                    )

        # Step 2: Segment tumor
        if perform_tumorseg:
            run_brats(
                    t1_file=MODALITY_STRIPPED_SCHEMA.format(base_dir=outdir, modality="t1"),
                    t1c_file=MODALITY_STRIPPED_SCHEMA.format(base_dir=outdir, modality="t1c"),
                    t2_file=MODALITY_STRIPPED_SCHEMA.format(base_dir=outdir, modality="t2"),
                    flair_file=MODALITY_STRIPPED_SCHEMA.format(base_dir=outdir, modality="flair"),
                    outdir=outdir,
                    pre_treatment=pre_treatment,
                    cuda_device=cuda_device
                    )

        # Step 3: Segment tissue
        healthy_brain_mask_file = REGISTRATION_MASK_SCHEMA.format(base_dir=outdir)
        generate_registration_mask(
                tumor_seg_file=TUMORSEG_SCHEMA.format(base_dir=outdir),
                outfile=healthy_brain_mask_file
                )

    tissueseg_kwargs = {}
    if registration_mask_file is not None:
        tissueseg_kwargs["registration_mask_file"] = registration_mask_file

    if perform_tissueseg:
        # valentin
        if registration_mask_file is None:
            temp = MODALITY_STRIPPED_SCHEMA.format(base_dir=outdir, modality="t1c")
        else:
            temp = MODALITY_STRIPPED_SCHEMA.format(base_dir=outdir, modality="t1")
        # valentin
        run_tissue_seg_registration(
                # t1_file = MODALITY_STRIPPED_SCHEMA.format(base_dir=outdir, modality="t1c"),
                t1_file = temp,  # valentin
                outdir=outdir,
                **tissueseg_kwargs
                )
    time_spent = time.time() - start_time
    logger.info(f"Finished preprocessing in {time_spent:.2f} seconds. Results saved to {outdir}.")


def process_longitudinal(preop_exam_dir: Path, followup_exam_dir: Path, outdir: Path, is_coregistered: bool = False) -> None:
    """
    TODO
    """
    start_time = time.time()
    logger.info(f"Starting longitudinal processing.")

    # Prepare directories
    t1c_pre_file = MODALITY_STRIPPED_SCHEMA.format(base_dir=preop_exam_dir, modality="t1c")
    t1c_post_file = MODALITY_STRIPPED_SCHEMA.format(base_dir=followup_exam_dir, modality="t1c")
    recurrence_seg_file = TUMORSEG_SCHEMA.format(base_dir=followup_exam_dir)

    # Mask
    fixed_mask_file = REGISTRATION_MASK_SCHEMA.format(base_dir=preop_exam_dir)
    moving_mask_file = REGISTRATION_MASK_SCHEMA.format(base_dir=followup_exam_dir)

    if is_coregistered:
        make_symlink(src=t1c_post_file, dst=LONGITUDINAL_WARP_SCHEMA.format(base_dir=followup_exam_dir))
        make_symlink(src=recurrence_seg_file, dst=RECURRENCE_SCHEMA.format(base_dir=followup_exam_dir))
    else:
        register_recurrence(
                t1c_pre_file=t1c_pre_file,
                t1c_post_file=t1c_post_file,
                recurrence_seg_file=recurrence_seg_file,
                outdir=outdir,
                fixed_mask_file=fixed_mask_file,
                moving_mask_file=moving_mask_file
                )
    time_spent = time.time() - start_time
    logger.info(f"Finished longitudinal preprocessing in {time_spent:.2f} seconds. Results saved to {outdir}.")


if __name__ == "__main__":
    # Example:
    # python gbm_bench/preprocessing/preprocess.py -cuda_device 0
    parser = argparse.ArgumentParser()
    parser.add_argument("-cuda_device", type=str, default="0", help="GPU id to run on.")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

    outdir_tmp = Path("tmp_preprocessing")

    # Pre-treatment example
    preprocess_dicom(
            t1_dir=Path("test_data/exam1/t1"),
            t1c_dir=Path("test_data/exam1/t1c"),
            t2_dir=Path("test_data/exam1/t2"),
            flair_dir=Path("test_data/exam1/flair"),
            outdir=outdir_tmp / "preop",
            pre_treatment=True,
            cuda_device=args.cuda_device
            )
    
    # Post-treatment example
    """
    preprocess_dicom(
            t1_dir=Path("test_data/exam3/t1"),
            t1c_dir=Path("test_data/exam3/t1c"),
            t2_dir=Path("test_data/exam3/t2"),
            flair_dir=Path("test_data/exam3/flair"),
            outdir=outdir_tmp / "postop",
            pre_treatment=False,
            cuda_device=args.cuda_device
            )
    """
    
    # Nifti example
    preprocess_nifti(
            t1_file=Path("test_data/nifti/t1.nii.gz"),
            t1c_file=Path("test_data/nifti/t1c.nii.gz"),
            t2_file=Path("test_data/nifti/t2.nii.gz"),
            flair_file=Path("test_data/nifti/flair.nii.gz"),
            pre_treatment=True,
            outdir=outdir_tmp / "nifti_preop",
            is_coregistered=True,
            tumorseg_file=Path("test_data/nifti/tumorseg.nii.gz"),
            cuda_device=args.cuda_device
            )

    # Longitudinal example
    process_longitudinal(
            preop_exam_dir=Path("test_data/exam1"),
            followup_exam_dir=Path("test_data/exam3"),
            outdir=outdir_tmp / "longitudinal"
            )
