import json
import shutil
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

import nibabel as nib

import torch
import pytorch_lightning as pl
from utils.data import DataModule, create_conditioning
from utils.trainer import LatentDiffusion


dir_output_model = Path('/vol/miltank/users/bilv/ldm/output')
dir_metrics = dir_output_model / "metrics"

path_data = "/vol/miltank/users/bilv/data"
path_data_challenge = "/vol/miltank/datasets/glioma/brats_inpainting/ASNR-MICCAI-BraTS2023-Local-Synthesis-Challenge-Validation"
path_autoencoder = "/vol/miltank/users/bilv/ldm/maisi/maisi_vae.pt"
path_diffusion = "/vol/miltank/users/bilv/master-thesis/models/ldm-diffusion/output/12.07.2025-12:34:41_3/checkpoints/last.ckpt"


def inference(mode):
    # Initialize data module
    datamodule = DataModule(
        mode=mode,
        path_data=path_data,
        path_data_challenge=path_data_challenge,
        latent_shape=(60, 60, 40),
        batch_size=2,
        num_workers=4
    )
    datamodule.setup()

    # Load model from checkpoint
    model = LatentDiffusion.load_from_checkpoint(
        path_diffusion,
        challenge_mode=True,  # Set to True for challenge mode
        dir_output_model=dir_output_model,
    )

    # Initialize trainer
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        logger=False,
        enable_progress_bar=True
    )

    # Run inference on test set
    trainer.test(model, datamodule=datamodule)

def calculate_metrics(dir_metrics):
    files_csv = list(dir_metrics.glob("*.csv"))
    if not files_csv:
        print("No CSV files found in the metrics directory.")
        return
    
    stats = {}
    
    for file_csv in files_csv:
        metric_name = file_csv.stem
        
        df = pd.read_csv(file_csv)
        values = df['value'].values

        stats[metric_name] = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)), 
            "std": float(np.std(values))
        }
        
    # Save to JSON
    json_output = dir_metrics / "stats.json"
    with open(json_output, 'w') as f:
        json.dump(stats, f, indent=4)

def generate_conditioning():
    import sys
    sys.path.append('/vol/miltank/users/bilv/gbm_bench')
    from gbm_bench.preprocessing.preprocess import preprocess_nifti

    dir_output_model = Path('/vol/miltank/users/bilv/ldm/output')  # muss weg
    path_data_challenge = "/vol/miltank/datasets/glioma/brats_inpainting/ASNR-MICCAI-BraTS2023-Local-Synthesis-Challenge-Validation"  # muss weg

    temp = Path(dir_output_model) / f"temp_{random.randint(10000, 99999)}"
    temp.mkdir(parents=True, exist_ok=True)

    for dir_patient in tqdm(sorted(list(Path(path_data_challenge).iterdir()))):
        path_conditioning = dir_patient / 'conditioning.pt'
        if path_conditioning.exists():
            continue

        patient = dir_patient.name
        path_original_t1_voided = dir_patient / f"{patient}-t1n-voided.nii.gz"
        path_original_mask = dir_patient / f"{patient}-mask.nii.gz"

        temp_patient = temp / patient
        temp_patient.mkdir(parents=True, exist_ok=True)

        preprocess_nifti(
            t1_file=path_original_t1_voided,
            t1c_file='.',
            t2_file='.',
            flair_file='.',
            pre_treatment=True,
            outdir=temp_patient,
            is_coregistered=True,
            is_skull_stripped=True,
            # tumorseg_file=Path(temp_dir),
            cuda_device='0',
            registration_mask_file=path_original_mask
        )

        path_original_tissue_segmentation = temp_patient / 'processed' / 'tissue_segmentation' / 'tissue_seg.nii.gz'
        original_tissue_segmentation = nib.load(path_original_tissue_segmentation).get_fdata()  # 240, 240, 155
        original_tissue_segmentation = torch.as_tensor(original_tissue_segmentation).float()
        original_growth_model = torch.zeros_like(original_tissue_segmentation)
        original_conditioning = create_conditioning(original_growth_model, original_tissue_segmentation)
        original_conditioning = torch.as_tensor(original_conditioning).float()  # 4, 240, 240, 155

        torch.save(original_conditioning, path_conditioning)

        shutil.rmtree(temp_patient)


if __name__ == "__main__":
    mode = 'inference_challenge'  # 'inference' or 'inference_challenge'
    inference(mode)
    # calculate_metrics(dir_metrics)
    # generate_conditioning()
