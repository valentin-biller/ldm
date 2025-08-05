import os   
import argparse
from pathlib import Path
from gbm_bench.utils.constants import RHUH_NIFTI_DIR, RECURRENCE_SCHEMA, TUMORSEG_SCHEMA
from gbm_bench.utils.parsing import LongitudinalDataset
from gbm_bench.preprocessing.preprocess import preprocess_nifti, process_longitudinal


if __name__ == "__main__":
    # Example:
    # python scripts/preprocess_rhuh_nifti.py -cuda_device 5 
    # nohup python -u scripts/preprocess_rhuh_nifti.py -cuda_device 5 > tmp_rhuh_nifti_preproc.out 2>&1 &
    parser = argparse.ArgumentParser()
    parser.add_argument("-cuda_device", type=str, default="0", help="GPU id to run on.")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

    # Read dataset
    rhuh_root = "/mnt/Drive2/lucas/datasets/RHUH-GBM/Images/NIfTI/RHUH-GBM"
    rhuh = LongitudinalDataset(dataset_id="RHUH", root_dir=rhuh_root)
    rhuh.load(RHUH_NIFTI_DIR)

    # Individual exams
    for patient_ind, patient in enumerate(rhuh.patients):
        print(f"Processing {patient_ind}/{len(rhuh.patients)}...")

        for exam in patient["exams"]:
            if exam["timepoint"] == "postop":  # skip postop
                continue

            is_preop = (exam["timepoint"] == "preop")
            print(f"{exam['t1']}")

            """
            preprocess_nifti(
                    t1_file=exam["t1"],
                    t1c_file=exam["t1c"],
                    t2_file=exam["t2"],
                    flair_file=exam["flair"],
                    pre_treatment=is_preop,
                    outdir=exam["t1"].parent,
                    is_skull_stripped=True,
                    is_coregistered=False,
                    cuda_device=args.cuda_device,
                    tumorseg_file=exam["tumorseg"]
                    )
            """

    # Longitudinal registration
    for patient_ind, patient in enumerate(rhuh.patients[5:]):  # hung at 5
        print(f"Performing longitudinal registration {patient_ind}/{len(rhuh.patients)}.")

        patient_id = patient["patient_id"]
        preop_exam = rhuh.get_patient_exams(patient_id=patient_id, timepoint="preop")[0]  # Find first preop exam
        preop_exam_dir = preop_exam["t1"].parent

        # Loop through followup exams
        followup_exams = rhuh.get_patient_exams(patient_id=patient_id, timepoint="followup")
        
        for followup_exam in followup_exams:
            followup_exam_dir = followup_exam["t1"].parent
            
            process_longitudinal(
                    preop_exam_dir=preop_exam_dir,
                    followup_exam_dir=followup_exam_dir,
                    outdir=followup_exam_dir
                    )
    
    print(f"Finished processing.")
