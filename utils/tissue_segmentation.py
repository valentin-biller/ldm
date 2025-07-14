import sys
sys.path.append('/vol/miltank/users/bilv/gbm_bench')

import numpy as np
if not hasattr(np, 'bool'):
    np.bool = np.bool_

import os
import argparse
from pathlib import Path
from gbm_bench.preprocessing.preprocess import preprocess_nifti
from gbm_bench.prediction.predict import predict_tumor_growth
from gbm_bench.utils.visualization import plot_model_multislice, plot_recurrence_multislice
import nibabel as nib


def runForPat(patstring: str):
    # Example:
    # python scripts/nifti_example.py -cuda_device 0
    parser = argparse.ArgumentParser()
    parser.add_argument("-cuda_device", type=str, default="0", help="GPU id to run on.")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

    # Example input, everything has to be pathlib.Path
    #"/mnt/Drive4/jonas/datasets/brats_2021_train2/BraTS2021_00000/preop/sub-BraTS2021_00000_ses-preop_space-sri_flair.nii.gz"
    # path = "/mnt/Drive4/jonas/datasets/brats_2021_train2/" + patstring + "/preop/"

    # valentin
    # image_path = Path('/vol/miltank/users/bilv/2025_challenge/data/BraTS2021_00000/BraTS2021_00000-t1n-voided-0000.nii.gz')
    # mask_path = Path('/vol/miltank/users/bilv/2025_challenge/data/BraTS2021_00000/BraTS2021_00000-mask-0000.nii.gz')
    patient = 'BraTS-GLI-01479-000'
    image_path = Path(f'/vol/miltank/datasets/glioma/brats_inpainting/ASNR-MICCAI-BraTS2023-Local-Synthesis-Challenge-Training/{patient}/{patient}-t1n-voided.nii.gz')
    mask_path = Path(f'/vol/miltank/datasets/glioma/brats_inpainting/ASNR-MICCAI-BraTS2023-Local-Synthesis-Challenge-Training/{patient}/{patient}-mask.nii.gz')
    image_nifti = nib.load(str(image_path))
    mask_nifti = nib.load(str(mask_path))
    image_data = image_nifti.get_fdata()
    mask_data = mask_nifti.get_fdata()

    non_zero_values = image_data[image_data > 0]
    mean_value = np.mean(non_zero_values) if len(non_zero_values) > 0 else 0

    print(f"Mean of non-zero values: {mean_value:.4f}")
    print(f"Number of non-zero voxels: {len(non_zero_values)}")

    # Create a copy of the image data
    filled_data = image_data.copy()

    # Replace black parts (value 0) that are covered by the mask with the mean value
    # Assuming mask value > 0 indicates the region to fill
    mask_indices = mask_data > 0
    black_indices = image_data == 0

    # Fill only the black voxels that are within the mask
    fill_indices = mask_indices & black_indices
    filled_data[fill_indices] = mean_value

    print(f"Number of voxels filled: {np.sum(fill_indices)}")

    # Create new NIfTI image with the same header/affine as original
    filled_nifti = nib.Nifti1Image(filled_data, affine=image_nifti.affine)

    # Save the filled image
    output_dir = '/vol/miltank/users/bilv/ldm/utils/preprocessing'
    os.makedirs(output_dir, exist_ok=True)
    output_path = f'{output_dir}/filled_image.nii.gz'
    nib.save(filled_nifti, str(output_path))
    print(f"Filled image saved to: {output_path}")
    # valentin

    t1_dir = '.'  # valentin
    t1c_dir = image_path  # valentin
    registration_mask_file = mask_path  # valentin
    t2_dir = '.'  # valentin
    flair_dir = '.'  # valentin


    # test_data_basedir = Path(path)
    # t1_dir = test_data_basedir / Path("sub-" + patstring + "_ses-preop_space-sri_t1.nii.gz")
    # t1c_dir = test_data_basedir / Path("sub-" + patstring + "_ses-preop_space-sri_t1c.nii.gz")
    # t2_dir = test_data_basedir / Path("sub-" + patstring + "_ses-preop_space-sri_t2.nii.gz")
    # flair_dir = test_data_basedir / Path("sub-" + patstring + "_ses-preop_space-sri_flair.nii.gz")

    # original_tumorseg_dir = test_data_basedir / Path("sub-" + patstring + "_ses-preop_space-sri_seg.nii.gz")

    # original_tumorseg = nib.load(original_tumorseg_dir).get_fdata()
    # affine = nib.load(original_tumorseg_dir).affine

    # roundedSeg = original_tumorseg.round().astype(float)

    # tempSeg = roundedSeg.copy() * 0.0
    # tempSeg[roundedSeg == 1] = 1.0  # Necrotic tumor core
    # tempSeg[roundedSeg == 2] = 2.0  # Edema
    # tempSeg[roundedSeg == 4] = 3.0  # Enhancing tumor


    #saveTemp
    # tempDir = Path("/mnt/Drive4/jonas/working_dir/brats_2021_train2_secondRun/" + patstring + "_seg.nii.gz")
    # nib.save(nib.Nifti1Image(tempSeg, affine), tempDir)


    # outputPath = "/mnt/Drive4/jonas/working_dir/brats_2021_train2_secondRun/" + patstring
    outputPath = '/vol/miltank/users/bilv/2025_challenge/data/BraTS2021_00000'  # valentin

    #outdir = Path("./tmp_testdata")  # This is where all output is stored, I usually set it to the exam directory
    #create folder
    os.makedirs(outputPath, exist_ok=True)
    outdir = Path(outputPath)  # Ensure outdir is a Path object

    model_id = "sbtc"  

    # Preprocessing
    preprocess_nifti(
                t1_file=t1_dir,
                t1c_file=t1c_dir,
                t2_file=t2_dir,
                flair_file=flair_dir,
                pre_treatment=True,
                outdir=outdir,
                is_coregistered=True,
                is_skull_stripped=True,
                # tumorseg_file=tempDir,
                cuda_device=args.cuda_device,
                registration_mask_file=registration_mask_file
            )
    # # Growth Model Inference
    predict_tumor_growth(
            preop_dir=outdir,
            model_id=model_id,
            cuda_device=args.cuda_device
            )

    # # Visualization
    # pdf_outfile = outdir / "multislice.pdf"
    # plot_model_multislice(
    #         patient_identifier="test",
    #         exam_identifier="test",
    #         algorithm_identifier=model_id,
    #         exam_dir=outdir,
    #         outfile=pdf_outfile
    #         )

    #delete tempfile
    # os.remove(tempDir)


if __name__ == "__main__":
    # patList = sorted(os.listdir("/mnt/Drive4/jonas/datasets/brats_2021_train2/"))
    # for pat in patList:
    #     print(pat)
    #     runForPat(pat)
    runForPat('')