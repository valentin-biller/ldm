import os
import argparse
from pathlib import Path
from gbm_bench.utils.constants import LUMIERE_DIR
from gbm_bench.utils.parsing import LongitudinalDataset
from gbm_bench.prediction.predict import predict_tumor_growth


if __name__ == "__main__":
    # Example:
    # python scripts/predict_lumiere.py -cuda_device 0
    # nohup python -u scripts/predict_lumiere.py -cuda_device 5 > tmp_lumiere_pred.out 2>&1 &
    parser = argparse.ArgumentParser()
    parser.add_argument("-cuda_device", type=str, default="0", help="GPU id to run on.")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    
    # Read dataset
    lumiere_root = "/mnt/Drive2/lucas/datasets/LUMIERE/Imaging"
    lumiere = LongitudinalDataset(dataset_id="LUMIERE", root_dir=lumiere_root)
    lumiere.load(LUMIERE_DIR)

    # Predict on preop exams
    for patient_ind, patient in enumerate(lumiere.patients):
        print(f"Predicting {patient_ind}/{len(lumiere.patients)}...")
        
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
