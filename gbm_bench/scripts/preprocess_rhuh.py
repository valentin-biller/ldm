import os   
import argparse
from pathlib import Path
from gbm_bench.utils.constants import RHUH_GBM_DIR
from gbm_bench.utils.parsing import LongitudinalDataset
from gbm_bench.preprocessing.preprocess import preprocess_dicom, process_longitudinal


if __name__ == "__main__":
    # Example:
    # python scripts/preprocess_rhuh.py -cuda_device 0
    # nohup python -u scripts/preprocess_rhuh.py -cuda_device 6 > tmp_rhuh_preproc.out 2>&1 &
    parser = argparse.ArgumentParser()
    parser.add_argument("-cuda_device", type=str, default="0", help="GPU id to run on.")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    dcm2niix_location = Path("/home/home/lucas/bin/dcm2niix")

    # Read dataset
    rhuh_root = "/home/home/lucas/data/RHUH-GBM/Images/DICOM/RHUH-GBM"
    rhuh_gbm = LongitudinalDataset(dataset_id="RHUH", root_dir=rhuh_root)
    rhuh_gbm.load(RHUH_GBM_DIR)

    # Individual exams
    for patient_ind, patient in enumerate(rhuh_gbm.patients):
        print(f"Processing {patient_ind}/{len(rhuh_gbm.patients)}...")

        for exam in patient["exams"]:
            if exam["timepoint"] == "postop":  # skip postop
                continue

            is_preop = (exam["timepoint"] == "preop")

            preprocess_dicom(
                    t1_dir=exam["t1"],
                    t1c_dir=exam["t1c"],
                    t2_dir=exam["t2"],
                    flair_dir=exam["flair"],
                    outdir=exam["t1"].parent,
                    dcm2niix_location=dcm2niix_location,
                    pre_treatment=is_preop,
                    cuda_device=args.cuda_device,
                    )

    # Longitudinal registration
    for patient_ind, patient in enumerate(rhuh_gbm.patients):
        print(f"Performing longitudinal registration {patient_ind}/{len(rhuh_gbm.patients)}.")

        patient_id = patient["patient_id"]
        preop_exam = rhuh_gbm.get_patient_exams(patient_id=patient_id, timepoint="preop")[0]  # Find first preop exam
        preop_exam_dir = preop_exam["t1"].parent

        # Loop through followup exams
        followup_exams = rhuh_gbm.get_patient_exams(patient_id=patient_id, timepoint="followup")
        
        for followup_exam in followup_exams:
            followup_exam_dir = followup_exam["t1"].parent
            
            process_longitudinal(
                    preop_exam_dir=preop_exam_dir,
                    followup_exam_dir=followup_exam_dir,
                    outdir=followup_exam_dir
                    )
    
    print(f"Finished processing.")
