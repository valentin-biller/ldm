import os   
import argparse
from pathlib import Path
from gbm_bench.utils.constants import LUMIERE_DIR, RHUH_DIR, UPENN_DIR
from gbm_bench.utils.parsing import LongitudinalDataset


ROOT_DIRS = {
        "rhuh": "/home/home/lucas/data/RHUH-GBM/Images/DICOM/RHUH-GBM",
        "lumiere": "/mnt/Drive2/lucas/datasets/LUMIERE/Imaging",
        "upenn": "/home/home/lucas/data/UPENN-GBM/UPENN-GBM"
        }

DATASET_DIRS = {
        "rhuh": RHUH_DIR,
        "lumiere": LUMIERE_DIR,
        "upenn": UPENN_DIR
        }


def load_dataset(dataset_id):
    root_dir = ROOT_DIRS[dataset_id]
    dataset_dir = DATASET_DIRS[dataset_id]
    dataset = LongitudinalDataset(dataset_id=dataset_id.upper(), root_dir=root_dir)
    dataset.load(dataset_dir)
    return dataset


if __name__ == "__main__":
    # Example:
    # python scripts/completion_report.py -dataset rhuh
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset_id", type=str)
    args = parser.parse_args()

    # Read dataset
    dataset = load_dataset(args.dataset_id)

    # Individual exams
    for patient_ind, patient in enumerate(dataset.patients):

        for exam in patient["exams"]:
            if exam["timepoint"] == "postop":  # skip postop
                continue

            is_preop = (exam["timepoint"] == "preop")
            print(f"{exam['t1']}")

            #TODO
