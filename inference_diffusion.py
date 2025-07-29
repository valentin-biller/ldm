import csv
import json
import shutil
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

import seaborn as sns
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch
import pytorch_lightning as pl
from utils.data import DataModule, create_conditioning
from utils.trainer import LatentDiffusion


def inference():
    # Initialize data module
    datamodule = DataModule(
        debug=debug,
        mode=mode,
        oversampling=True,
        path_data=path_data,
        path_data_challenge=path_data_challenge,
        dir_output_model=dir_output_model,
        latent_shape=(60, 60, 40),
        batch_size=4,
        num_workers=4,
    )
    datamodule.setup()

    # Load model from checkpoint
    model = LatentDiffusion.load_from_checkpoint(
        path_diffusion,
        path_autoencoder=path_autoencoder,
        dir_output_model=dir_output_model,
        model_=model_,
        scheduler_=scheduler_,
        denoising=denoising,
        num_inference_steps=100,
    )

    # Initialize trainer
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=torch.cuda.device_count(),
        logger=False,
        enable_progress_bar=True
    )

    # Run inference on test set
    trainer.test(model, datamodule=datamodule)

def calculate_metrics():
    files_csv = list(dir_metrics.glob("*.csv"))
    if not files_csv:
        print("No CSV files found in the metrics directory.")
        return
    
    stats = {}
    metrics = {}
    
    for file_csv in files_csv:
        metric_name = file_csv.stem
        file_csv_filtered = file_csv.with_name(f"filtered_{file_csv.name}")
        
        df = pd.read_csv(file_csv)
        df = df.drop_duplicates(['patient', 'mask'], keep='first').sort_values(by=['patient', 'mask'])
        if len(df) != 2163:
            print(f"WARNING: Expected 2163 unique (patient, mask) entries, found {len(df)} in {file_csv}")
        df.to_csv(file_csv_filtered, index=False)

        df = pd.read_csv(file_csv_filtered)
        values = df['value'].values

        stats[metric_name] = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)), 
            "std": float(np.std(values))
        }
        metrics[metric_name] = values
        
    # Save to JSON
    json_output = dir_metrics / "stats.json"
    with open(json_output, 'w') as f:
        json.dump(stats, f, indent=4)

    def plot_custom_violin(ax, data, metric_name, ylims, yticks):
        # Violin
        # parts = sns.violinplot(x=[1.18]*len(data), y=data, ax=ax, color='#cccccc', inner=None, linewidth=2)
        # df = pd.DataFrame({'x': [1.18]*len(data), 'y': data})
        # parts = sns.violinplot(data=df, x='x', y='y', ax=ax, color='#cccccc', inner=None, linewidth=2, cut=0)
        parts = sns.violinplot(y=data, ax=ax, color='#cccccc', inner=None, linewidth=2)
        for pc in ax.collections:
            pc.set_edgecolor('black')
            pc.set_linewidth(2)

        # Calculate stats
        q1, q3 = np.percentile(data, [25, 75])
        median = np.median(data)
        mean = np.mean(data)
        iqr = q3 - q1
        whisker_low = q1 - 1.5 * iqr
        whisker_high = q3 + 1.5 * iqr
        outliers = data[(data < whisker_low) | (data > whisker_high)]

        xlims = ax.get_xlim()
        width = 0.075 * (xlims[1] - xlims[0])
        width_rhombus = 0.5 * width

        # Calculate height in data units so that it matches the width visually
        # Get axis transformation
        trans = ax.transData.transform

        # Center point in data coordinates
        center_x = 0
        center_y = mean

        # Calculate half-width in display units
        p1 = trans((center_x - width_rhombus/2, center_y))
        p2 = trans((center_x + width_rhombus/2, center_y))
        display_width = abs(p2[0] - p1[0])

        # Now, find the height in data units that gives the same display height
        # Start with an initial guess
        guess_height = width_rhombus
        p_top = trans((center_x, center_y - guess_height/2))
        p_bottom = trans((center_x, center_y + guess_height/2))
        display_height = abs(p_bottom[1] - p_top[1])

        # Scale guess_height so display_height == display_width
        height_rhombus = guess_height * (display_width / display_height)

        # Whiskers
        whisker_min = np.min(data[data >= whisker_low])
        whisker_max = np.max(data[data <= whisker_high])
        ax.plot([0, 0], [whisker_min, whisker_max], color='black', linewidth=1, zorder=1)

        # Box (quartiles)
        box = patches.Rectangle((-width / 2, q1), width, q3-q1, linewidth=1, edgecolor='black', facecolor='none', zorder=5)
        ax.add_patch(box)

        # Median line
        ax.plot([-width/2, width/2], [median, median], color='black', linewidth=2, zorder=5)

        # Mean rhombus (diamond)
        rhombus_x = 0  # center x, adjust if needed to match box
        rhombus = patches.Polygon([
            [rhombus_x, mean + height_rhombus / 2],           # top
            [rhombus_x - width_rhombus / 2, mean],            # left
            [rhombus_x, mean - height_rhombus / 2],           # bottom
            [rhombus_x + width_rhombus / 2, mean]             # right
        ], closed=True, facecolor='none', edgecolor='black', linewidth=1, zorder=10)
        ax.add_patch(rhombus)

        # Outlier dots (above and below box/violin)
        out_above = outliers[outliers > whisker_max]
        out_below = outliers[outliers < whisker_min]
        ax.scatter(np.full_like(out_above, 0), out_above, color='black', s=30, zorder=10)
        ax.scatter(np.full_like(out_below, 0), out_below, color='black', s=30, zorder=10)

        # Style
        ax.set_ylim(*ylims)
        ax.set_yticks(yticks)
        ax.yaxis.set_tick_params(labelsize=36)
        ax.yaxis.grid(True, linestyle='--', linewidth=1.0, color='gray', alpha=0.6)
        ax.set_axisbelow(True)
        ax.set_xticks([])
        ax.set_xlabel(metric_name, fontsize=40)
        ax.set_title('')
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    dict_metrics = {
        'ssim': 'SSIM',
        'psnr': 'PSNR',
        'psnr_01': 'PSNR (0-1)',
        'psnr_eps': 'PSNR (eps)',
        'psnr_01_eps': 'PSNR (0-1 / eps)',
        'mse': 'MSE',
        'rmse': 'RMSE',
        'mae': 'MAE',
        'msle': 'MSLE'
    }

    # All metrics in a 3x3 grid
    all_metrics = ['psnr', 'psnr_01', 'psnr_eps', 'psnr_01_eps', 'mse', 'rmse', 'mae', 'msle', 'ssim']
    fig, axes = plt.subplots(3, 4, figsize=(24, 18), sharey=False)
    axes_flat = axes.flatten()
    for i, metric in enumerate(all_metrics):
        plot_custom_violin(
            axes_flat[i], metrics[metric], dict_metrics[metric],
            ylims=(-2, 42) if metric.startswith('psnr') else (0.0, 1.1) if metric == 'ssim' else (-0.05, 0.55),
            yticks=[0, 10, 20, 30, 40] if metric.startswith('psnr') else [0, 0.25, 0.5, 0.75, 1.0] if metric == 'ssim' else [0, 0.1, 0.2, 0.3, 0.4, 0.5]
        )
        if i % 4 != 0:  # 4 columns per row
            axes_flat[i].set_yticklabels([])
            # axes_flat[i].set_yticks([])
            axes_flat[i].spines['left'].set_visible(False)
            # axes_flat[i].yaxis.set_visible(False)
    # Hide unused axes (last 3 if only 9 metrics)
    for j in range(len(all_metrics), 12):
        fig.delaxes(axes_flat[j])
    plt.tight_layout()
    plt.savefig(dir_metrics / 'all_metrics.svg', format='svg', dpi=300)
    plt.close(fig)

def generate_conditioning():
    import sys
    sys.path.append('/vol/miltank/users/bilv/gbm_bench')
    from gbm_bench.preprocessing.preprocess import preprocess_nifti
    
    temp = Path(dir_output_model) / f"temp_{random.randint(10000, 99999)}"
    temp.mkdir(parents=True, exist_ok=True)

    for dir_patient in tqdm(sorted(list(path_data_challenge.iterdir()))):
        path_conditioning = dir_patient / 'conditioning.pt'
        path_conditioning_old = dir_patient / 'conditioning_old.pt'

        if path_conditioning.exists():
            continue

        patient = dir_patient.name

        path_original_t1_voided = dir_patient / f"{patient}-t1n-voided.nii.gz"
        path_original_mask = dir_patient / f"{patient}-mask.nii.gz"
        path_inverted_mask = dir_patient / f"{patient}-mask-inverted.nii.gz"

        img_inverted_mask = nib.load(path_original_mask)
        data_inverted_mask = img_inverted_mask.get_fdata()
        affine_inverted_mask = img_inverted_mask.affine
        inverted_mask = (data_inverted_mask == 0).astype(np.float32)
        nib.save(nib.Nifti1Image(inverted_mask, affine_inverted_mask), path_inverted_mask)

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
            registration_mask_file=path_inverted_mask
        )

        ### temp
        path_original_tissue_segmentation = temp_patient / 'processed' / 'tissue_segmentation' / 'tissue_seg.nii.gz'
        path_temp_tissue_segmentation = dir_patient / 'tissue_segmentation.nii.gz'
        shutil.copy(path_original_tissue_segmentation, path_temp_tissue_segmentation)
        ### temp

        path_original_tissue_segmentation = temp_patient / 'processed' / 'tissue_segmentation' / 'tissue_seg.nii.gz'
        original_tissue_segmentation = nib.load(path_original_tissue_segmentation).get_fdata()  # 240, 240, 155
        original_tissue_segmentation = torch.as_tensor(original_tissue_segmentation).float()
        original_growth_model = torch.zeros_like(original_tissue_segmentation)
        original_conditioning = create_conditioning(original_growth_model, original_tissue_segmentation)
        original_conditioning = torch.as_tensor(original_conditioning).float()  # 4, 240, 240, 155

        torch.save(original_conditioning, path_conditioning)

        shutil.rmtree(temp_patient)

def create_slices():

    def save_slice_with_bbox(image, mask, filename, draw_bbox=False):
        fig, ax = plt.subplots()
        ax.imshow(np.rot90(image), cmap='gray')
        if draw_bbox:
            mask = np.rot90(mask)
            coords = np.argwhere(mask == 1)
            if coords.size > 0:
                margin = 5  # pixels to expand the bounding box
                y_min, x_min = coords.min(axis=0)
                y_max, x_max = coords.max(axis=0)
                y_min = max(y_min - margin, 0)
                x_min = max(x_min - margin, 0)
                y_max = min(y_max + margin, mask.shape[0] - 1)
                x_max = min(x_max + margin, mask.shape[1] - 1)
                rect = patches.Rectangle(
                    (x_min, y_min),
                    x_max - x_min,
                    y_max - y_min,
                    linewidth=1,
                    edgecolor='red',
                    facecolor='none'
                )
                ax.add_patch(rect)
        ax.axis('off')
        fig.savefig(filename, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    with open(path_psnr, newline='') as f:
        reader = csv.DictReader(f)
        rows = sorted(reader, key=lambda x: float(x['value']), reverse=True)

    start = 0
    end = 25
    patients = 5

    row_volumes = []
    for row in rows[start:end]:
        path_mask = path_data / row['patient'] / 'masks' / f"mask-healthy-{row['mask']}.nii.gz"
        mask_data = nib.load(str(path_mask)).get_fdata()
        volume = np.sum(mask_data == 1)
        row_volumes.append((row, volume))
        print(f"{row['patient']:25} {row['mask']:5} {row['value']:25} {volume:5}")

    row_volumes.sort(key=lambda x: x[1], reverse=True)

    print('-' * 63)
    patients_masks_reconstructed = []
    for row_volume in row_volumes[:patients]:
        patients_masks_reconstructed.append((row_volume[0]['patient'], row_volume[0]['mask']))
        print(f"{row_volume[0]['patient']:25} {row_volume[0]['mask']:5} {row_volume[0]['value']:25} {row_volume[1]:5}")

    for patient, mask in patients_masks_reconstructed:
        path_mask = path_data / patient / 'masks' / f'mask-healthy-{mask}.nii.gz'
        path_original = path_data / patient / 't1.nii.gz'
        path_voided = path_data / patient / 'voided' / f't1-voided-{mask}.nii.gz'
        path_reconstructed = dir_output_model / 'inference' / 'reconstructed' / f"{patient}_{mask}.nii.gz"

        # Load mask and reconstructed image
        mask_data = nib.load(str(path_mask)).get_fdata()
        original_data = nib.load(str(path_original)).get_fdata()
        voided_data = nib.load(str(path_voided)).get_fdata()
        reconstructed_data = nib.load(str(path_reconstructed)).get_fdata()

        # Find center of mask (where mask == 1)
        coords = np.argwhere(mask_data == 1)
        center = coords.mean(axis=0).astype(int)  # (z, y, x) or (x, y, z) depending on orientation
        x, y, z = center

        # Extract slices
        axial_slice_mask = mask_data[:, :, z]
        coronal_slice_mask = mask_data[:, y, :]
        sagittal_slice_mask = mask_data[x, :, :]

        axial_slice_original = original_data[:, :, z]
        coronal_slice_original = original_data[:, y, :]
        sagittal_slice_original = original_data[x, :, :]

        axial_slice_voided = voided_data[:, :, z]
        coronal_slice_voided = voided_data[:, y, :]
        sagittal_slice_voided = voided_data[x, :, :]

        axial_slice_reconstructed = reconstructed_data[:, :, z]
        coronal_slice_reconstructed = reconstructed_data[:, y, :]
        sagittal_slice_reconstructed = reconstructed_data[x, :, :]

        # Prepare output directory
        dir_out = dir_output_model / 'slices' / f'{patient}_{mask}'
        dir_out.mkdir(parents=True, exist_ok=True)

        dir_out_original = dir_out / 'original'
        dir_out_voided = dir_out / 'voided'
        dir_out_reconstructed = dir_out / 'reconstructed'
        dir_out_original.mkdir(parents=True, exist_ok=True)
        dir_out_voided.mkdir(parents=True, exist_ok=True)
        dir_out_reconstructed.mkdir(parents=True, exist_ok=True)

        # Save original slices with bounding box if requested
        save_slice_with_bbox(axial_slice_original, axial_slice_mask, dir_out_original / 'axial.png', True)
        save_slice_with_bbox(coronal_slice_original, coronal_slice_mask, dir_out_original / 'coronal.png', True)
        save_slice_with_bbox(sagittal_slice_original, sagittal_slice_mask, dir_out_original / 'sagittal.png', True)

        save_slice_with_bbox(axial_slice_voided, axial_slice_mask, dir_out_voided / 'axial.png', False)
        save_slice_with_bbox(coronal_slice_voided, coronal_slice_mask, dir_out_voided / 'coronal.png', False)
        save_slice_with_bbox(sagittal_slice_voided, sagittal_slice_mask, dir_out_voided / 'sagittal.png', False)

        # Save reconstructed slices without bounding box
        save_slice_with_bbox(axial_slice_reconstructed, axial_slice_mask, dir_out_reconstructed / 'axial.png', True)
        save_slice_with_bbox(coronal_slice_reconstructed, coronal_slice_mask, dir_out_reconstructed / 'coronal.png', True)
        save_slice_with_bbox(sagittal_slice_reconstructed, sagittal_slice_mask, dir_out_reconstructed / 'sagittal.png', True)

def create_failure():
    paths_failures = [
        ('/vol/miltank/users/bilv/ldm/output/inference/reconstructed/BraTS2021_00012_0000.nii.gz', 100, 'BraTS2021_00012_0000'),
        ('/vol/miltank/users/bilv/ldm/output/inference/reconstructed/egd-0692_0000.nii.gz', 128, 'egd-0692_0000'),
        ('/vol/miltank/users/bilv/ldm/output/inference/reconstructed/egd-0615_0002.nii.gz', 125, 'egd-0615_0002')
    ]

    for path_failure, slice, patient_mask in paths_failures:
        failure_data = nib.load(path_failure).get_fdata()
        failure_slice = failure_data[:, slice, :]

        dir_out = dir_output_model / 'slices'
        dir_out.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots()
        ax.imshow(np.rot90(failure_slice), cmap='gray')
        ax.axis('off')
        fig.savefig(dir_out / f"failure_{patient_mask}.png", bbox_inches='tight', pad_inches=0)
        plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference  Diffusion")
    parser.add_argument('--function', type=str, default='inference', choices=['inference', 'calculate_metrics', 'generate_conditioning', 'create_slices', 'create_failure'], help='Function')
    parser.add_argument('--debug', action='store_true', help='Debug Mode')
    parser.add_argument('--model', type=str, default='big', choices=['small', 'big', 'big_old'], help='Model')
    parser.add_argument('--scheduler', type=str, default='ddpm', choices=['ddpm', 'ddim'], help='Scheduler')
    parser.add_argument('--version', type=int, default=1, help='Version')
    parser.add_argument('--mode', type=str, default='inference', choices=['inference', 'inference_challenge', 'inference_conditioning'], help='Mode')
    parser.add_argument('--denoising', type=str, default='repaint', choices=['repaint', 'own'], help='Denoising')
    args = parser.parse_args()

    debug = args.debug
    model_ = args.model
    scheduler_ = args.scheduler
    version_ = args.version
    mode = args.mode
    denoising = args.denoising

    print(f"Debug: {debug}, Model: {model_}, Scheduler: {scheduler_}, Version: {version_}, Mode: {mode}, Denoising: {denoising}")

    dir_output_model = Path(f"/vol/miltank/users/bilv/ldm/output{'_debug' if debug else ''}")
    dir_output_model = dir_output_model / f'{model_}_{scheduler_}_v{version_}' / mode / denoising
    dir_metrics = dir_output_model / "metrics"

    path_data = Path("/vol/miltank/users/bilv/data")
    path_data_challenge = Path("/vol/miltank/datasets/glioma/brats_inpainting/ASNR-MICCAI-BraTS2023-Local-Synthesis-Challenge-Validation")
    path_data_pseudo = Path("/vol/miltank/users/bilv/ldm/pseudo/reconstructed_challenge")

    dir_current = Path(__file__).resolve().parent
    path_autoencoder = dir_current / 'maisi' / 'maisi_vae.pt'
    path_diffusion = dir_current / 'models' / f'diffusion_{model_}_{scheduler_}_v{version_}.ckpt'

    path_psnr = dir_metrics / "psnr.csv"

    if args.function == 'inference':
        inference()
    elif args.function == 'calculate_metrics':
        calculate_metrics()
    elif args.function == 'generate_conditioning':
        generate_conditioning()
    elif args.function == 'create_slices':
        create_slices()
    elif args.function == 'create_failure':
        create_failure()