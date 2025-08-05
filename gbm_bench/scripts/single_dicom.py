import os   
import argparse
from pathlib import Path
from gbm_bench.utils.constants import RHUH_GBM_DIR
from gbm_bench.utils.parsing import LongitudinalDataset
from gbm_bench.preprocessing.preprocess import preprocess_dicom, process_longitudinal
from gbm_bench.prediction.predict import predict_tumor_growth
from gbm_bench.evaluation.evaluate import evaluate_tumor_model
from gbm_bench.utils.constants import PREDICTION_OUTPUT_SCHEMA


if __name__ == "__main__":
    # Example:
    # python scripts/single_dicom.py -cuda_device 0
    # nohup python -u scripts/single_dicom.py -cuda_device 0 > tmp_single_dicom.out 2>&1 &
    parser = argparse.ArgumentParser()
    parser.add_argument("-cuda_device", type=str, default="0", help="GPU id to run on.")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    dcm2niix_location = Path("/home/home/lucas/bin/dcm2niix")

    # Read dataset
    rhuh_root = "/mnt/Drive4/lucas"
    rhuh_gbm = LongitudinalDataset(dataset_id="RHUH", root_dir=rhuh_root)
    rhuh_gbm.load(RHUH_GBM_DIR)

    patient_id = "RHUH-0024"
    algo_id ="sbtc"
    preop_exams = rhuh_gbm.get_patient_exams(patient_id=patient_id, timepoint="preop")[0]
    followup_exams = rhuh_gbm.get_patient_exams(patient_id=patient_id, timepoint="followup")[0]
    preop_exam_dir = preop_exams["t1"].parent
    followup_exam_dir = followup_exams["t1"].parent

    print(f"Processing patient {patient_id}, preop exam: {preop_exam_dir}, followup exam: {followup_exam_dir}")

    # Preprocessing
    preprocess_dicom(
            t1_dir=preop_exams["t1"],
            t1c_dir=preop_exams["t1c"],
            t2_dir=preop_exams["t2"],
            flair_dir=preop_exams["flair"],
            outdir=preop_exam_dir,
            dcm2niix_location=dcm2niix_location,
            pre_treatment=True,
            cuda_device=args.cuda_device
            )

    preprocess_dicom(
            t1_dir=followup_exams["t1"],
            t1c_dir=followup_exams["t1c"],
            t2_dir=followup_exams["t2"],
            flair_dir=followup_exams["flair"],
            outdir=followup_exam_dir,
            dcm2niix_location=dcm2niix_location,
            pre_treatment=False,
            cuda_device=args.cuda_device
            )

    # Longitudinal
    process_longitudinal(
            preop_exam_dir=preop_exam_dir,
            followup_exam_dir=followup_exam_dir,
            outdir=followup_exam_dir
            )

    # Predict
    predict_tumor_growth(
            preop_dir=preop_exam_dir,
            model_id=algo_id,
            cuda_device=args.cuda_device
            )
    
    # Evaluate
    prediction_dir = PREDICTION_OUTPUT_SCHEMA.format(base_dir=preop_exam_dir, algo_id=algo_id)
    results = evaluate_tumor_model(
            preop_dir=preop_exam_dir,
            followup_dir=followup_exam_dir,
            pred_file=prediction_dir,
            model_id=algo_id
            )

    print(results)

    print(f"Finished processing.")
