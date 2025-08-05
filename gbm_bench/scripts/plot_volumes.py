import os
import shutil
import argparse
from pathlib import Path
from gbm_bench.utils.constants import RHUH_GBM_DIR, LUMIERE_DIR, UPENN_GBM_DIR, GLIODIL_DIR
from gbm_bench.utils.parsing import LongitudinalDataset
from gbm_bench.utils.visualization import plot_tumor_volumes


if __name__ == "__main__":
    # Example:
    # python scripts/plot_volumes.py

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
    followup_exams = []

    for dataset in datasets:
        for patient_ind, patient in enumerate(dataset.patients):
            for exam in patient["exams"]:
                if exam["timepoint"] == "followup":
                    if exam["t1"].parent != ".":  # happens when an exam has no t1
                        followup_exams.append(exam["t1"].parent)

    outfile = Path("tmp_visualization/volumes.pdf")
    plot_tumor_volumes(recurrence_exam_paths=followup_exams, outfile=str(outfile))
