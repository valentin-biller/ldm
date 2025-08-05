import os   
import argparse
import nibabel as nib
from pathlib import Path
from gbm_bench.utils.constants import GLIODIL_DIR
from gbm_bench.utils.parsing import LongitudinalDataset
from gbm_bench.preprocessing.preprocess import preprocess_nifti, process_longitudinal


def convert_tumorseg_labels(inpath, outpath):
    img = nib.load(str(inpath))
    data = img.get_fdata()

    data[data == 4] = 3

    new_img = nib.Nifti1Image(data, affine=img.affine, header=img.header)
    nib.save(new_img, str(outpath))


if __name__ == "__main__":
    # Example:
    # python scripts/preprocess_gliodil.py -cuda_device 0
    # nohup python -u scripts/preprocess_gliodil.py -cuda_device 0 > tmp_gliodil_preproc1.out 2>&1 &
    parser = argparse.ArgumentParser()
    parser.add_argument("-cuda_device", type=str, default="0", help="GPU id to run on.")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

    # Read dataset
    gliodil_root = "/mnt/Drive2/lucas/datasets/GLIODIL"
    gliodil = LongitudinalDataset(dataset_id="GLIODIL", root_dir=gliodil_root)
    gliodil.load(GLIODIL_DIR)

    # Individual exams
    for patient_ind, patient in enumerate(gliodil.patients):
        print(f"Processing {patient_ind}/{len(gliodil.patients)}...")

        for exam in patient["exams"]:
            if exam["timepoint"] == "postop":  # skip postop
                continue

            is_preop = (exam["timepoint"] == "preop")
            print(f"{exam['t1c']}")

            patient_dir = exam["t1c"].parent
            outdir = patient_dir / f"{'preop' if is_preop else 'followup'}"
            outdir.mkdir(exist_ok=True)

            converted_tumorseg_file = outdir / "tumorseg_123.nii.gz"
            convert_tumorseg_labels(exam["tumorseg"], converted_tumorseg_file)

            preprocess_nifti(
                    t1_file=exam["t1"],
                    t1c_file=exam["t1c"],
                    t2_file=exam["t2"],
                    flair_file=exam["flair"],
                    tumorseg_file=converted_tumorseg_file,
                    pre_treatment=is_preop,
                    outdir=outdir,
                    is_skull_stripped=True,
                    is_coregistered=True,
                    cuda_device=args.cuda_device
                    )

    # Longitudinal registration
    for patient_ind, patient in enumerate(gliodil.patients):
        print(f"Performing longitudinal registration {patient_ind}/{len(gliodil.patients)}.")

        patient_id = patient["patient_id"]
        preop_exam = gliodil.get_patient_exams(patient_id=patient_id, timepoint="preop")[0]  # Find first preop exam
        preop_exam_dir = preop_exam["t1c"].parent / "preop"

        # Loop through followup exams
        followup_exams = gliodil.get_patient_exams(patient_id=patient_id, timepoint="followup")
        
        for followup_exam in followup_exams:
            followup_exam_dir = followup_exam["t1c"].parent / "followup"
            
            process_longitudinal(
                    preop_exam_dir=preop_exam_dir,
                    followup_exam_dir=followup_exam_dir,
                    outdir=followup_exam_dir,
                    is_coregistered=True
                    )
    
    print(f"Finished processing.")
