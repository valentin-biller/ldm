import os
import json
import shutil
import argparse
from pathlib import Path
from gbm_bench.utils.constants import RHUH_GBM_DIR, LUMIERE_DIR, UPENN_GBM_DIR, GLIODIL_DIR, METRICS_SCHEMA
from gbm_bench.utils.parsing import LongitudinalDataset
from gbm_bench.utils.visualization import scatter_plot


def load_performances(performance_dict_path):
    with open(performance_dict_path, "r") as f:
        data = json.load(f)
    return data


if __name__ == "__main__":
    # Example:
    # python scripts/plot_performances.py

    # Read datasets
    rhuh_root = "/home/home/lucas/data/RHUH-GBM/Images/DICOM/RHUH-GBM"
    rhuh_gbm = LongitudinalDataset(dataset_id="RHUH", root_dir=rhuh_root)
    rhuh_gbm.load(RHUH_GBM_DIR)

    lumiere_root = "/mnt/Drive2/lucas/datasets/LUMIERE/Imaging"
    lumiere = LongitudinalDataset(dataset_id="LUMIERE", root_dir=lumiere_root)
    lumiere.load(LUMIERE_DIR)

    upenn_gbm_root = "/home/home/lucas/data/UPENN-GBM/UPENN-GBM"
    upenn_gbm = LongitudinalDataset(dataset_id="UPENN_GBM", root_dir=upenn_gbm_root)
    upenn_gbm.load(UPENN_GBM_DIR)

    gliodil_root = "/mnt/Drive2/lucas/datasets/GLIODIL"
    gliodil = LongitudinalDataset(dataset_id="GLIODIL", root_dir=gliodil_root)
    gliodil.load(GLIODIL_DIR)

    datasets = [rhuh_gbm, lumiere, upenn_gbm, gliodil]

    algo_id = "sbtc"
    standard_performances = []
    model_performances = []

    for dataset in datasets:
        for patient_ind, patient in enumerate(dataset.patients):
            for exam in patient["exams"]:
                if exam["timepoint"] == "followup":
                    performance_path = METRICS_SCHEMA.format(base_dir=exam["t1"].parent, algo_id=algo_id)
                    if performance_path.is_file():    
                        performance = load_performances(performance_path)
                        standard_performances.append(performance["recurrence_coverage_standard"])
                        model_performances.append(performance["recurrence_coverage_model"])
                    else:
                        print(f"{performance_path} is not a valid json file.")

    outfile = Path("tmp_visualization/performances.pdf")
    scatter_plot(standard_performances, model_performances, outfile=outfile)
