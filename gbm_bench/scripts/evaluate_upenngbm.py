import os
import shutil
import pickle
import argparse
import numpy as np
from scipy import stats
from pathlib import Path
from gbm_bench.utils.constants import UPENN_GBM_DIR
from gbm_bench.utils.parsing import LongitudinalDataset
from gbm_bench.evaluation.evaluate import evaluate_tumor_model
from gbm_bench.utils.constants import PREDICTION_OUTPUT_SCHEMA


if __name__ == "__main__":
    # Example:
    # python scripts/evaluate_upenngbm.py
    # nohup python -u scripts/evaluate_upenngbm.py > tmp_upenn_eval.out 2>&1 &
    
    # Read dataset
    upenn_gbm_root = "/home/home/lucas/data/UPENN-GBM/UPENN-GBM"
    upenn_gbm = LongitudinalDataset(dataset_id="UPENN_GBM", root_dir=upenn_gbm_root)
    upenn_gbm.load(UPENN_GBM_DIR)

    all_results = []

    for patient_ind, patient in enumerate(upenn_gbm.patients):
        print(f"Processing {patient_ind}/{len(upenn_gbm.patients)}...")

        patient_identifier = patient["patient_id"]
        preop_exams = upenn_gbm.get_patient_exams(patient_id=patient_identifier, timepoint="preop")
        followup_exams = upenn_gbm.get_patient_exams(patient_id=patient_identifier, timepoint="followup")

        if len(preop_exams) > 1:
            print(f"Warning: found {len(preop_exams)} preop exams for patient {patiend_ind, patiend}. Using first exam for evaluation.")

        algo_id = "gliodil" # sbtc, gliodil
        preop_exam_dir = preop_exams[0]["t1"].parent
        prediction_dir = PREDICTION_OUTPUT_SCHEMA.format(base_dir=preop_exam_dir, algo_id=algo_id)
        
        for followup_exam in followup_exams:
            followup_exam_dir = followup_exam["t1"].parent

            results = evaluate_tumor_model(
                    preop_dir=preop_exam_dir,
                    followup_dir=followup_exam_dir,
                    pred_file=prediction_dir,
                    model_id=algo_id
                    )
            all_results.append(results)
            print(f"{patient_identifier}: {results}")

    recurrence_coverage_standard = [r["recurrence_coverage_standard"] for r in all_results]
    recurrence_coverage_standard_all = [r["recurrence_coverage_standard_all"] for r in all_results]
    recurrence_coverage_model = [r["recurrence_coverage_model"] for r in all_results]
    recurrence_coverage_model_all = [r["recurrence_coverage_model_all"] for r in all_results]

    print(f"Finished evaluation.")
    print(f"Standard plan coverge: {100*np.mean(recurrence_coverage_standard):.2f} \u00B1 {100*stats.sem(recurrence_coverage_standard):.2f}")
    print(f"Standard plan coverge (all): {100*np.mean(recurrence_coverage_standard_all):.2f} \u00B1 {100*stats.sem(recurrence_coverage_standard_all):.2f}")
    print(f"Model plan coverge: {100*np.mean(recurrence_coverage_model):.2f} \u00B1 {100*stats.sem(recurrence_coverage_model):.2f}")
    print(f"Model plan coverge (all): {100*np.mean(recurrence_coverage_model_all):.2f} \u00B1 {100*stats.sem(recurrence_coverage_model_all):.2f}")
