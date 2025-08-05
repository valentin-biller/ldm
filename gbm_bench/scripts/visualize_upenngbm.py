import os
import shutil
import argparse
from pathlib import Path
from gbm_bench.utils.utils import merge_pdfs
from gbm_bench.utils.constants import UPENN_GBM_DIR
from gbm_bench.utils.parsing import LongitudinalDataset
from gbm_bench.utils.visualization import plot_model_multislice, plot_recurrence_multislice


if __name__ == "__main__":
    # Example:
    # python scripts/visualize_upenngbm.py

    # Read dataset
    upenn_gbm_root = "/home/home/lucas/data/UPENN-GBM/UPENN-GBM"
    upenn_gbm = LongitudinalDataset(dataset_id="UPENN_GBM", root_dir=upenn_gbm_root)
    upenn_gbm.load(UPENN_GBM_DIR)

    outfiles_model, outfiles_recurrences = [], []
    tmp_dir_model, tmp_dir_rec = "./tmp/model", "./tmp/recurrence"
    os.makedirs(tmp_dir_model, exist_ok=True)
    os.makedirs(tmp_dir_rec, exist_ok=True)
    
    for patient_ind, patient in enumerate(upenn_gbm.patients):
        print(f"Visualizing {patient_ind}/{len(upenn_gbm.patients)}...")
        
        patient_identifier = patient["patient_id"]
        exam_dir_preopop = upenn_gbm.get_patient_exams(patient_id=patient_identifier, timepoint="preop")[0]["t1"].parent
        exam_dir_followup = upenn_gbm.get_patient_exams(patient_id=patient_identifier, timepoint="followup")[0]["t1"].parent
        exam_identifier_preop = str(exam_dir_preopop.name)
        exam_identifier_followup = str(exam_dir_followup.name)

        algorithm_identifier = "sbtc"                       # LMI, SBTC, GLIODIL
        
        # Model plot
        outfile_model = os.path.join(tmp_dir_model, f"{patient_identifier}_{algorithm_identifier}.pdf")

        try:
            plot_model_multislice(
                    patient_identifier=patient_identifier,
                    exam_identifier=exam_identifier_preop,
                    algorithm_identifier=algorithm_identifier,
                    exam_dir=exam_dir_preopop,
                    outfile=outfile_model
                    )
            outfiles_model.append(outfile_model)
        except Exception as e:
            raise e
            #print(f"Plotting failed for {exam_identifier_preop}, method {algorithm_identifier}. Possibly file not found. Continuing...")
        
        # Recurrences
        outfile_recurrence = os.path.join(tmp_dir_rec, f"{patient_identifier}_recurrence.pdf")
        outfiles_recurrences.append(outfile_recurrence)
        
        plot_recurrence_multislice(
            patient_identifier=patient_identifier,
            exam_identifier_pre=exam_identifier_preop,
            exam_identifier_followup=exam_identifier_followup,
            exam_dir_preop=exam_dir_preopop,
            exam_dir_followup=exam_dir_followup,
            outfile=outfile_recurrence
            )

    # Merge PDFs
    outfiles_model.sort()
    outfiles_recurrences.sort()
    merge_pdfs(outfiles_model, f"./tmp/UPENN_{algorithm_identifier}.pdf")
    merge_pdfs(outfiles_recurrences, f"./tmp/UPENN_recurrences.pdf")

    # Delete temporary files
    shutil.rmtree(tmp_dir_model)
    shutil.rmtree(tmp_dir_rec)
