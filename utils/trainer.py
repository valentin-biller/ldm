import sys
from pathlib import Path
dir_current = Path(__file__).resolve().parent
path_maisi = dir_current.parent / 'maisi'
path_gbm_bench = dir_current.parent / 'gbm_bench'
sys.path.append(str(path_maisi))
sys.path.append(str(path_gbm_bench))

import numpy as np
if not hasattr(np, 'bool'):
    np.bool = np.bool_

import torch
import torch.nn as nn
import pytorch_lightning as pl

import os
import csv
import fcntl
import shutil
import random
import subprocess
import nibabel as nib
from pathlib import Path

from monai import transforms
from generative.networks.schedulers import DDIMScheduler, DDPMScheduler

from .data import create_conditioning
from .maisi_autoencoder import MaisiAutoencoder
from .challenge_metrics import generate_metrics

from controlnet_maisi import ControlNetMaisi
from diffusion_model_unet_maisi import DiffusionModelUNetMaisi

class LatentDiffusion(pl.LightningModule):
    """
    PyTorch Lightning module for latent diffusion inpainting
    """
    
    def __init__(
        self,
        path_autoencoder=None,
        dir_output_model=None,
        small_model=False,
        learning_rate=1e-4,
        num_train_timesteps=1000,
        num_inference_steps=100,
        beta_start=0.0001,
        beta_end=0.02,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.path_autoencoder = path_autoencoder
        self.dir_output_model = dir_output_model
        self.small_model = small_model
        
        small_num_channels = (32, 64, 128)  # (64, 128, 256)
        small_num_head_channels = (32, 64, 128)  # (0, 128, 256)
        big_num_channels = (128, 256, 512)
        big_num_head_channels = (0, 256, 512)
        
        if self.small_model:
            config_unet = {
                "spatial_dims": 3,
                "in_channels": 4,  # latent shape: (B, 4, 60, 60, 40)
                "out_channels": 4,  # latent shape: (B, 4, 60, 60, 40)
                "num_res_blocks": [2, 2, 2],
                "num_channels": small_num_channels if self.small_model else big_num_channels,  # 64, 128, 256 or 128, 256, 512
                "attention_levels": (False, True, True),
                "norm_num_groups": 32,
                "resblock_updown": True,
                "num_head_channels": small_num_head_channels if self.small_model else big_num_head_channels,  # 0, 128, 256 or 0, 256, 512
                "transformer_num_layers": 8,
                "use_flash_attention": True,
                "with_conditioning": False,
                "cross_attention_dim": None
            }
            config_controlnet = {
                "spatial_dims": 3,
                "in_channels": 4,
                "num_res_blocks": [2, 2, 2],
                "num_channels": small_num_channels if self.small_model else big_num_channels,  # 64, 128, 256 or 128, 256, 512
                "attention_levels": (False, True, True),
                "norm_num_groups": 32,
                "resblock_updown": True,
                "num_head_channels": small_num_head_channels if self.small_model else big_num_head_channels,  # 0, 128, 256 or 0, 256, 512
                "transformer_num_layers": 8,
                "use_flash_attention": True,
                "with_conditioning": False,
                "cross_attention_dim": None
            }
            config_conditioning_embedding_num_channels = (16,)
        else:
            config_unet = {
                "spatial_dims": 3,
                "in_channels": 4,  # latent shape: (B, 4, 60, 60, 40)
                "out_channels": 4,  # latent shape: (B, 4, 60, 60, 40)
                "num_res_blocks": [2, 2, 2],
                "num_channels": small_num_channels if small_model else big_num_channels,  # 64, 128, 256 or 128, 256, 512
                "attention_levels": (False, True, True),
                "num_head_channels": small_num_head_channels if small_model else big_num_head_channels,  # 0, 128, 256 or 0, 256, 512
            }
            config_controlnet = {
                "spatial_dims": 3,
                "in_channels": 4,
                "num_res_blocks": [2, 2, 2],
                "num_channels": small_num_channels if small_model else big_num_channels,  # 64, 128, 256 or 128, 256, 512
                "attention_levels": (False, True, True),
                "num_head_channels": small_num_head_channels if small_model else big_num_head_channels,  # 0, 128, 256 or 0, 256, 512
            }
            config_conditioning_embedding_num_channels = (128,)

        # Initialize UNet and ControlNet
        self.unet = DiffusionModelUNetMaisi(**config_unet)
        self.controlnet = ControlNetMaisi(
            **config_controlnet, 
            conditioning_embedding_in_channels=4,
            conditioning_embedding_num_channels=config_conditioning_embedding_num_channels,
        )
        # Initialize ControlNet weights from UNet
        self.controlnet.load_state_dict(self.unet.state_dict(), strict=False)
        
        # Initialize scheduler and inferer
        self.scheduler = DDIMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            schedule="linear_beta",
            prediction_type="epsilon",
            clip_sample=False,
        )
        
        # Loss function
        self.loss_fn = nn.MSELoss()
        self.autoencoder_crop = transforms.CenterSpatialCrop(roi_size=(1, 240, 240, 155))

    def setup(self, stage=None):
        self.autoencoder = MaisiAutoencoder(path_autoencoder=self.path_autoencoder, device=self.device)

    def _get_noise_prediction(self, sample, timesteps, conditioning):
        """
        Modular function to get noise prediction from ControlNet + UNet
        """
        # Get ControlNet conditioning residuals
        down_block_res_samples, mid_block_res_sample = self.controlnet(
            x=sample,
            timesteps=timesteps,
            controlnet_cond=conditioning,
            context=None
        )
        # Predict noise with ControlNet residuals
        noise_pred = self.unet(
            x=sample,
            timesteps=timesteps,
            context=None,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample
        )
        return noise_pred

    def _get_encoded_autoencoder(self, original_autoencoder):
        latent = self.autoencoder.encode(original_autoencoder)  # (B, 4, 60, 60, 40)
        return latent

    def _get_decoded_autoencoder(self, latent):
        reconstructed_autoencoder = self.autoencoder.decode(latent).squeeze(0)
        reconstructed_autoencoder = torch.clamp(reconstructed_autoencoder, 0.0, 1.0)  # B, 1, 240, 240, 160
        if latent.shape[0] == 1:
            reconstructed_autoencoder = reconstructed_autoencoder.unsqueeze(0)
        reconstructed = self.autoencoder_crop(reconstructed_autoencoder)  # B, 1, 240, 240, 155
        return reconstructed

    def training_step(self, batch, batch_idx):
        """Training step"""
        original_t1_autoencoder = batch['original_t1_autoencoder']  # (B, 1, 240, 240, 160)
        latent_t1 = self._get_encoded_autoencoder(original_t1_autoencoder)  # (B, 4, 60, 60, 40)
        latent_conditioning = batch['latent_conditioning']  # (B, 4, 60, 60, 40)

        # Sample random timesteps
        timesteps = torch.randint(self.scheduler.num_train_timesteps, (latent_t1.shape[0],), device=self.device)

        # Standard inpainting approach: Add noise to entire clean image
        noise = torch.randn_like(latent_t1)
        noisy_latent = self.scheduler.add_noise(latent_t1, noise, timesteps)

        noise_pred = self._get_noise_prediction(noisy_latent, timesteps, latent_conditioning)
        loss = self.loss_fn(noise_pred, noise)

        # Log training metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        original_t1 = batch['original_t1']  # (B, 1, 240, 240, 155)
        latent_conditioning = batch['latent_conditioning']  # (B, 4, 60, 60, 40)
        
        # Generate inpainted image
        with torch.no_grad():
            denoised_t1 = self._generate_denoising( 
                latent_conditioning
            ).float()

            reconstructed_t1 = self._get_decoded_autoencoder(denoised_t1)  # (B, 1, 240, 240, 155)

        if batch_idx == 0:
            self.validation_outputs_for_saving = {
                'original_t1': original_t1.cpu(),
                'reconstructed_t1': reconstructed_t1.cpu(),
                'patient': batch['patient'],
            }

        # Compute validation loss (MSE in image space)
        val_loss = self.loss_fn(reconstructed_t1, original_t1)
        
        # Log validation metrics
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return {
            'val_loss': val_loss,
        }
    
    def test_step(self, batch, batch_idx):
        """Test step"""
        original_t1_voided = batch['original_t1_voided']  # (B, 1, 240, 240, 155)
        original_t1_voided_autoencoder = batch['original_t1_voided_autoencoder']  # (B, 1, 240, 240, 160)
        latent_t1_voided = self._get_encoded_autoencoder(original_t1_voided_autoencoder)  # (B, 4, 60, 60, 40)
        original_mask = batch['original_mask']  # (B, 1, 240, 240, 155)
        latent_mask = batch['latent_mask']  # (B, 4, 60, 60, 40)
        if 'original_t1' in batch:
            challenge = False  # inference
            original_t1 = batch['original_t1']  # (B, 1, 240, 240, 155)
            original_mask_healthy = batch['original_mask_healthy']  # (B, 1, 240, 240, 155)
            original_conditioning = batch['original_conditioning']  # (B, 1, 240, 240, 155)
            latent_conditioning = batch['latent_conditioning']  # (B, 4, 60, 60, 40)
        else:
            challenge = True  # inference_challenge)

        with torch.no_grad():
            if challenge:
                '''
                While generating your predictions, keep in mind the following:
                    - All individual files must be NIfTI format and use the .nii.gz file extension
                    - All individual files have a dimension of 240x240x155 and origin at [0, -239, 0]. You may use CaPTk to verify and/or visualize this.
                    - Filenames must end with the 5-digit case ID, followed by the 3-digit timepoint then by the word "t1n-inference", all delimited by a single dash (-) -- the case ID and timepoint information are provided by the input filenames. The format should look something like this:
                        - *{ID}-{timepoint}-t1n-inference.nii.gz
                        - For example, given:
                            BraTS-GLI-12345-000/
                            ├─ BraTS-GLI-12345-000-mask.nii.gz
                            └─ BraTS-GLI-12345-000-t1n-voided.nii.gz
                            A valid output filename could be: BraTS-GLI-12345-000-t1n-inference.nii.gz
                '''
                conditioning_exists = all(batch['exists_conditioning'])
                if not conditioning_exists:
                    original_conditioning = self._generate_conditioning(
                        batch['path_original_t1_voided'],
                        batch['path_original_mask']
                    ).float()  # B, 4, 240, 240, 155
                    latent_conditioning = torch.nn.functional.interpolate(
                        original_conditioning,
                        size=(latent_t1_voided.shape[2], latent_t1_voided.shape[3], latent_t1_voided.shape[4]),
                        mode='nearest'
                    ).float()  # B, 4, 60, 60, 40
                else:
                    original_conditioning = batch['original_conditioning']
                    latent_conditioning = batch['latent_conditioning']

            # Generate inpainted latent
            inpainted_t1 = self._generate_denoising(
                latent_conditioning,
                latent_voided=latent_t1_voided,
                latent_mask=latent_mask,
            ).float()

            reconstructed_t1 = self._get_decoded_autoencoder(inpainted_t1)  # (B, 1, 240, 240, 155)
            
            # Inject voided image in known regions
            reconstructed_t1 = (reconstructed_t1 * original_mask + original_t1_voided * (1 - original_mask))
        
            # Evaluate metrics if not in challenge mode
            if not challenge:
                self._evaluate_metrics(
                    batch['patient'],
                    batch['mask'],
                    reconstructed_t1,  # (B, 1, 240, 240, 155)
                    original_t1,  # (B, 1, 240, 240, 155)
                    original_mask_healthy,  # (B, 1, 240, 240, 155)
                    original_t1_voided,  # (B, 1, 240, 240, 155)
                )

            # Save images
            if not challenge:
                dir_output = Path(self.dir_output_model) / 'inference'
                masks = batch['mask']
            else:
                dir_output = Path(self.dir_output_model) / 'inference_challenge'
            dir_output_original = dir_output / 'original'
            dir_output_reconstructed = dir_output / 'reconstructed'
            dir_output_original.mkdir(parents=True, exist_ok=True)
            dir_output_reconstructed.mkdir(parents=True, exist_ok=True)
            
            patients = batch['patient']
            affines = batch['affine']
            
            for i, (patient, affine) in enumerate(zip(patients, affines)):
                if not challenge:
                    mask = masks[i]
                    original_t1_ = original_t1[i, 0].cpu().float().numpy()
                    file_name = f"{patient}_{mask}.nii.gz"
                    path_original_t1 = dir_output_original / file_name
                    path_reconstructed_t1 = dir_output_reconstructed / file_name
                else:
                    original_t1_ = original_t1_voided[i, 0].cpu().float().numpy()
                    file_name_original = f"{patient}.nii.gz"
                    file_name_reconstructed = f"{patient}-t1n-inference.nii.gz"
                    path_original_t1 = dir_output_original / file_name_original
                    path_reconstructed_t1 = dir_output_reconstructed / file_name_reconstructed
                reconstructed_t1_ = reconstructed_t1[i, 0].cpu().float().numpy()
                affine = affine.cpu().float().numpy()

                nib.save(nib.Nifti1Image(original_t1_, affine), path_original_t1)
                nib.save(nib.Nifti1Image(reconstructed_t1_, affine), path_reconstructed_t1)

    def _generate_denoising(self, latent_conditioning, latent_voided=None, latent_mask=None):
        """Generate denoised latent from pure noise using ControlNet conditioning"""
        batch_size = latent_conditioning.shape[0]
        
        if latent_mask is not None:
            repeated_mask = latent_mask.repeat(1, 4, 1, 1, 1)

        # Initialize with pure noise
        sample = torch.randn(
            latent_conditioning.shape,
            device=self.device
        )
        
        # Set inference timesteps
        self.scheduler.set_timesteps(self.hparams.num_inference_steps)
        
        # Denoising loop
        for i, t in enumerate(self.scheduler.timesteps):
            t_device = t.type_as(sample)
            
            noise_pred = self._get_noise_prediction(
                sample,
                t_device.unsqueeze(0).repeat(batch_size),
                latent_conditioning
            )
            
            # Denoising step
            sample = self.scheduler.step(noise_pred, t, sample)[0]

            # inpainting
            if latent_voided is not None and latent_mask is not None:
                # Pixel injection: preserve known regions (add noise to ground truth based on timestep)
                if i < len(self.scheduler.timesteps) - 1:
                    noise_gt = torch.randn(
                        latent_conditioning.shape,
                        device=self.device
                    )
                    noisy_gt = self.scheduler.add_noise(latent_voided, noise_gt, t)
                    sample = sample * repeated_mask + noisy_gt * (1 - repeated_mask)
                else:
                    sample = sample * repeated_mask + latent_voided * (1 - repeated_mask)

        return sample

    def _generate_conditioning(self, paths_original_t1_voided, paths_original_mask):
        from gbm_bench.preprocessing.preprocess import preprocess_nifti

        temp = Path(self.dir_output_model) / f"temp_{random.randint(10000, 99999)}"
        temp.mkdir(parents=True, exist_ok=True)

        original_conditionings = []
        for i in range(len(paths_original_t1_voided)):
            path_original_t1_voided = Path(paths_original_t1_voided[i])
            path_original_mask = Path(paths_original_mask[i])

            temp_patient = temp / path_original_t1_voided.parent.name
            temp_patient.mkdir(parents=True, exist_ok=True)

            if torch.cuda.is_available():
                device_str = str(torch.cuda.current_device())
            else:
                device_str = 'cpu'

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
                cuda_device=device_str,
                registration_mask_file=path_original_mask
            )

            path_original_tissue_segmentation = temp_patient / 'processed' / 'tissue_segmentation' / 'tissue_seg.nii.gz'
            original_tissue_segmentation = nib.load(path_original_tissue_segmentation).get_fdata()  # 240, 240, 155
            original_tissue_segmentation = torch.as_tensor(original_tissue_segmentation).float()
            original_growth_model = torch.zeros_like(original_tissue_segmentation)
            original_conditioning = create_conditioning(original_growth_model, original_tissue_segmentation)
            original_conditioning = torch.as_tensor(original_conditioning).float()  # 4, 240, 240, 155

            original_conditionings.append(original_conditioning)

        shutil.rmtree(temp)

        return torch.stack(original_conditionings, dim=0).to(self.device)

    def _evaluate_metrics(self, patients, masks, reconstructed_t1, original_t1, original_mask_healthy, original_t1_voided):
        dir_metrics = Path(self.dir_output_model) / "metrics"
        dir_metrics.mkdir(parents=True, exist_ok=True)

        for i, (patient, mask) in enumerate(zip(patients, masks)):
            # Compute metrics
            metrics_dict = generate_metrics(
                prediction=torch.tensor(reconstructed_t1[i].cpu().numpy()).unsqueeze(0),
                target=torch.tensor(original_t1[i].cpu().numpy()).unsqueeze(0),
                mask=torch.tensor(original_mask_healthy[i].cpu().numpy()).unsqueeze(0).bool(),
                normalization_tensor=torch.tensor(original_t1_voided[i].cpu().numpy()).unsqueeze(0)
            )
                
            # Save each metric to its own CSV file
            for metric_name, metric_value in metrics_dict.items():
                file_csv = dir_metrics / f"{metric_name}.csv"
                for _ in range(10):
                    try:
                        with open(file_csv, 'a', newline='') as f:
                            # Try to acquire exclusive lock and check if file is empty
                            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                            f.seek(0, 2)  # Seek to end
                            file_size = f.tell()
                            
                            writer = csv.writer(f)
                            # Write header if file is empty
                            if file_size == 0:
                                writer.writerow(['patient', 'mask', 'value'])
                            # Write data
                            writer.writerow([patient, mask, metric_value])
                            break  
                    except Exception:
                        pass

    def on_validation_epoch_end(self):
        """Save sample images at the end of validation epoch"""

        if self.dir_output_model is None:
            print("No output directory specified for saving sample images.")
            return

        original_t1 = self.validation_outputs_for_saving['original_t1'][:, 0, :, :, :].float().numpy()  # (4, 240, 240, 155)
        reconstructed_t1 = self.validation_outputs_for_saving['reconstructed_t1'][:, 0, :, :, :].float().numpy()  # (4, 240, 240, 155)
        patients = self.validation_outputs_for_saving['patient']

        # Create output directory
        output_dir = Path(self.dir_output_model).parent / 'images' / f'epoch_{self.current_epoch+1:04d}'
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, patient in enumerate(patients):
            # Save as NIfTI files
            reconstructed_t1_nii = nib.Nifti1Image(reconstructed_t1[i], np.eye(4))
            original_t1_nii = nib.Nifti1Image(original_t1[i], np.eye(4))
            
            nib.save(reconstructed_t1_nii, output_dir / f"{patient}_reconstructed_t1.nii.gz")
            nib.save(original_t1_nii, output_dir / f"{patient}_original_t1.nii.gz")
                
    def configure_optimizers(self):
        """Configure optimizer"""

        unet_params = list(self.unet.parameters())
        controlnet_params = list(self.controlnet.parameters())
        all_params = unet_params + controlnet_params

        optimizer = torch.optim.AdamW(
            all_params,
            lr=self.hparams.learning_rate,
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=1e-6
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }