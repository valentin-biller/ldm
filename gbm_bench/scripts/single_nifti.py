import os   
import argparse
import nibabel as nib
from pathlib import Path
from gbm_bench.utils.constants import GLIODIL_DIR
from gbm_bench.utils.parsing import LongitudinalDataset
from gbm_bench.preprocessing.preprocess import preprocess_nifti, process_longitudinal
from gbm_bench.prediction.predict import predict_tumor_growth
from gbm_bench.evaluation.evaluate import evaluate_tumor_model
from gbm_bench.utils.constants import PREDICTION_OUTPUT_SCHEMA


# for tgm
def convert_tumorseg_labels(inpath, outpath):
    img = nib.load(str(inpath))
    data = img.get_fdata()

    data[data == 4] = 3

    new_img = nib.Nifti1Image(data, affine=img.affine, header=img.header)
    outpath.parent.mkdir(exist_ok=True)
    nib.save(new_img, str(outpath))


if __name__ == "__main__":
    # Example:
    # python scripts/single_nifti.py -cuda_device 2
    # nohup python -u scripts/single_nifti.py -cuda_device 5 > tmp_single_nifti.out 2>&1 &
    parser = argparse.ArgumentParser()
    parser.add_argument("-cuda_device", type=str, default="0", help="GPU id to run on.")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

    # Read dataset
    patient_id = "tgm016"
    algo_id = "lmi"
    
    # tgm016 (ess tissues)
    """
    glio_root = Path("/mnt/Drive2/lucas/datasets/test_data/tgm016_ess_tissues/preop")
    t1_file = glio_root / "sub-tgm016_ses-preop_space-sri_t1.nii.gz"
    t1c_file = glio_root / "sub-tgm016_ses-preop_space-sri_t1c.nii.gz"
    t2_file = glio_root / "sub-tgm016_ses-preop_space-sri_t2.nii.gz"
    flair_file = glio_root / "sub-tgm016_ses-preop_space-sri_flair.nii.gz"
    tumorseg_file = glio_root / "sub-tgm016_ses-preop_space-sri_seg.nii.gz"
    t1c_followup_file = glio_root / "sub-tgm016_ses-preop_space-sri_t1c-rec.nii.gz"
    recurrenceseg_file = glio_root / "sub-tgm016_ses-preop_space-sri_seg-rec.nii.gz"
    exam_dir_preop = glio_root / "preop"
    exam_dir_followup = glio_root / "followup"

    # Convert tumor segmentations
    converted_tumorseg_file_preop = exam_dir_preop / "tumorseg_123.nii.gz"
    converted_tumorseg_file_followup = exam_dir_followup / "tumorseg_123.nii.gz"
    convert_tumorseg_labels(tumorseg_file, converted_tumorseg_file_preop)
    convert_tumorseg_labels(recurrenceseg_file, converted_tumorseg_file_followup)

    # tgm16 (gbmdata)
    """
    glio_root = Path("/mnt/Drive2/lucas/datasets/GLIODIL/tgm016/preop")
    t1_file = glio_root / "sub-tgm016_ses-preop_space-sri_t1.nii.gz"
    t1c_file = glio_root / "sub-tgm016_ses-preop_space-sri_t1c.nii.gz"
    t2_file = glio_root / "sub-tgm016_ses-preop_space-sri_t2.nii.gz"
    flair_file = glio_root / "sub-tgm016_ses-preop_space-sri_flair.nii.gz"
    tumorseg_file = glio_root / "sub-tgm016_ses-preop_space-sri_seg.nii.gz"
    t1c_followup_file = glio_root / "sub-tgm016_ses-preop_space-sri_t1c-rec.nii.gz"
    recurrenceseg_file = glio_root / "sub-tgm016_ses-preop_space-sri_seg-rec.nii.gz"
    exam_dir_preop = glio_root / "preop"
    exam_dir_followup = glio_root / "followup"

    # Convert tumor segmentations
    converted_tumorseg_file_preop = exam_dir_preop / "tumorseg_123.nii.gz"
    converted_tumorseg_file_followup = exam_dir_followup / "tumorseg_123.nii.gz"
    convert_tumorseg_labels(tumorseg_file, converted_tumorseg_file_preop)
    convert_tumorseg_labels(recurrenceseg_file, converted_tumorseg_file_followup)

    # rhuh0002
    """
    rhuh_root = Path("/mnt/Drive2/lucas/datasets/RHUH-GBM/Images/NIfTI/RHUH-GBM/RHUH-0002/")
    t1_file = rhuh_root / "0/RHUH-0002_0_t1.nii.gz"
    t1c_file = rhuh_root / "0/RHUH-0002_0_t1ce.nii.gz"
    t2_file = rhuh_root / "0/RHUH-0002_0_t2.nii.gz"
    flair_file = rhuh_root / "0/RHUH-0002_0_flair.nii.gz"
    tumorseg_file = rhuh_root / "0/RHUH-0002_0_segmentations.nii.gz"
    t1c_followup_file = rhuh_root / "2/RHUH-0002_2_t1ce.nii.gz"
    recurrenceseg_file = rhuh_root / "2/RHUH-0002_2_segmentations.nii.gz"
    exam_dir_preop = rhuh_root / "0"
    exam_dir_followup = rhuh_root / "2"

    converted_tumorseg_file_preop = tumorseg_file
    converted_tumorseg_file_followup = recurrenceseg_file
    
    # Preprocessing
    preprocess_nifti(
        t1_file=t1_file,
        t1c_file=t1c_file,
        t2_file=t2_file,
        flair_file=flair_file,
        tumorseg_file=converted_tumorseg_file_preop,
        pre_treatment=True,
        outdir=exam_dir_preop,
        is_skull_stripped=True,
        is_coregistered=True,
        cuda_device=args.cuda_device
        )

    preprocess_nifti(
        t1_file=Path(""),
        t1c_file=t1c_followup_file,
        t2_file=Path(""),
        flair_file=Path(""),
        tumorseg_file=converted_tumorseg_file_followup,
        pre_treatment=False,
        outdir=exam_dir_followup,
        is_skull_stripped=True,
        is_coregistered=True,
        cuda_device=args.cuda_device
        )

    # Longitudinal
    process_longitudinal(
            preop_exam_dir=exam_dir_preop,
            followup_exam_dir=exam_dir_followup,
            outdir=exam_dir_followup,
            is_coregistered=True    # change for tgm/rhuh
            )
    """
    
    # Predict
    predict_tumor_growth(
            preop_dir=exam_dir_preop,
            model_id=algo_id,
            cuda_device=args.cuda_device
            )

    # Evaluate
    prediction_dir = PREDICTION_OUTPUT_SCHEMA.format(base_dir=exam_dir_preop, algo_id=algo_id)
    results = evaluate_tumor_model(
            preop_dir=exam_dir_preop,
            followup_dir=exam_dir_followup,
            pred_file=prediction_dir,
            model_id=algo_id
            )

    print(f"Done: {results}")
