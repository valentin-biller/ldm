import os
import shutil
import argparse
from pathlib import Path
from gbm_bench.utils.utils import merge_pdfs
from gbm_bench.utils.constants import RHUH_GBM_DIR
from gbm_bench.utils.parsing import LongitudinalDataset
from gbm_bench.utils.visualization import plot_model_multislice, plot_recurrence_multislice, plot_plans


if __name__ == "__main__":
    # Example:
    # python scripts/visualize_rhuh.py

    # Read dataset
    rhuh_root = "/home/home/lucas/data/RHUH-GBM/Images/DICOM/RHUH-GBM"
    rhuh_gbm = LongitudinalDataset(dataset_id="RHUH", root_dir=rhuh_root)
    rhuh_gbm.load(RHUH_GBM_DIR)

    outfiles_model, outfiles_recurrences, outfile_plans = [], [], []
    tmp_dir_model, tmp_dir_rec, tmp_dir_plans = "./tmp/model", "./tmp/recurrence", "./tmp/plans"
    os.makedirs(tmp_dir_model, exist_ok=True)
    os.makedirs(tmp_dir_rec, exist_ok=True)
    os.makedirs(tmp_dir_plans, exist_ok=True)
    
    for patient_ind, patient in enumerate(rhuh_gbm.patients):
        print(f"Visualizing {patient_ind}/{len(rhuh_gbm.patients)}...")
        
        patient_identifier = patient["patient_id"]
        exam_dir_preopop = rhuh_gbm.get_patient_exams(patient_id=patient_identifier, timepoint="preop")[0]["t1c"].parent
        exam_dir_followup = rhuh_gbm.get_patient_exams(patient_id=patient_identifier, timepoint="followup")[0]["t1c"].parent
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

        # Radiation plans
        outfile_plan = os.path.join(tmp_dir_plans, f"{patient_identifier}_plans.pdf")
        outfile_plans.append(outfile_plan)

        plot_plans(
            patient_identifier=patient_identifier,
            exam_identifier_pre=exam_identifier_preop,
            exam_identifier_followup=exam_identifier_followup,
            exam_dir_preop=exam_dir_preopop,
            exam_dir_followup=exam_dir_followup,
            outfile=outfile_plan
            )

    # Merge PDFs
    outfiles_model.sort()
    outfiles_recurrences.sort()
    outfile_plans.sort()
    merge_pdfs(outfiles_model, f"./tmp/RHUH_{algorithm_identifier}.pdf")
    merge_pdfs(outfiles_recurrences, f"./tmp/RHUH_recurrences.pdf")
    merge_pdfs(outfile_plans, f"./tmp/RHUH_plans.pdf")

    # Delete temporary files
    shutil.rmtree(tmp_dir_model)
    shutil.rmtree(tmp_dir_rec)
    shutil.rmtree(tmp_dir_plans)
