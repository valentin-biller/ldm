import os
import argparse
from pathlib import Path
from gbm_bench.utils.constants import RHUH_NIFTI_DIR
from gbm_bench.utils.parsing import LongitudinalDataset
from gbm_bench.prediction.predict import predict_tumor_growth


if __name__ == "__main__":
    # Example:
    # python scripts/predict_rhuh_nifti.py -cuda_device 0
    # nohup python -u scripts/predict_rhuh_nifti.py -cuda_device 5 > tmp_rhuh_nifti_pred.out 2>&1 &
    parser = argparse.ArgumentParser()
    parser.add_argument("-cuda_device", type=str, default="0", help="GPU id to run on.")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    
    # Read dataset
    rhuh_root = "/mnt/Drive2/lucas/datasets/RHUH-GBM/Images/NIfTI/RHUH-GBM"
    rhuh = LongitudinalDataset(dataset_id="RHUH", root_dir=rhuh_root)
    rhuh.load(RHUH_NIFTI_DIR)

    # Predict on preop exams
    for patient_ind, patient in enumerate(rhuh.patients):
        print(f"Predicting {patient_ind}/{len(rhuh.patients)}...")
        
        for exam in patient["exams"]:
            if exam["timepoint"] != "preop":
                continue

            print(exam["t1"].parent)

            predict_tumor_growth(
                    preop_dir=exam["t1"].parent,
                    model_id="sbtc", # lmi, sbtc, gliodil
                    cuda_device=args.cuda_device
                    )
    print("Done.")
