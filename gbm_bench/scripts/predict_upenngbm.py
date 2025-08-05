import os
import argparse
from pathlib import Path
from gbm_bench.utils.constants import UPENN_GBM_DIR
from gbm_bench.utils.parsing import LongitudinalDataset
from gbm_bench.prediction.predict import predict_tumor_growth


if __name__ == "__main__":
    # Example:
    # python scripts/predict_upenngbm.py -cuda_device 0
    # nohup python -u scripts/predict_upenngbm.py -cuda_device 2 > tmp_upenn_pred.out 2>&1 &
    parser = argparse.ArgumentParser()
    parser.add_argument("-cuda_device", type=str, default="0", help="GPU id to run on.")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    
    # Read dataset
    upenn_gbm_root = "/home/home/lucas/data/UPENN-GBM/UPENN-GBM"
    upenn_gbm = LongitudinalDataset(dataset_id="UPENN_GBM", root_dir=upenn_gbm_root)
    upenn_gbm.load(UPENN_GBM_DIR)

    # Predict on preop exams
    for patient_ind, patient in enumerate(upenn_gbm.patients):
        print(f"Predicting {patient_ind}/{len(upenn_gbm.patients)}...")
        
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
