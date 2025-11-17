import sys
from pathlib import Path
dir_current = Path(__file__).resolve().parent
dir_autoencoder = dir_current.parent / 'autoencoder'
dir_gbm_bench = dir_current.parent / 'gbm_bench'
sys.path.append(str(dir_current))
sys.path.append(str(dir_autoencoder))
sys.path.append(str(dir_gbm_bench))

import numpy as np
if not hasattr(np, 'bool'):
    np.bool = np.bool_

import torch
import lightning.pytorch as L

import time
import pickle
import shutil
import random
import nibabel as nib
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import pietorch
import skimage.exposure
from scipy.ndimage import binary_dilation, gaussian_filter

from monai import transforms

import helpers
import schedulers
from model_unet import UNet
from model_dit import DiT_Custom
from maisi_autoencoder import MaisiAutoencoder
from f8d16_autoencoder import F8D16Autoencoder


class LatentDiffusion(L.LightningModule):
    """
    PyTorch Lightning module for latent diffusion inpainting
    """
    
    def __init__(
        self,
        path_autoencoder=None,
        dir_output_model=None,
        dir_ema=None,
        use_latents=True,
        use_distribution_shift=False,
        model_='unet',  # unet, dit
        mask_conditioning=64,  # 64, 32, None
        modality_conditioning=True,  # True, False
        denoising='own', # own, repaint
        scheduler_='ddpm',  # ddpm, iddpm
        latent_shape=None,
        learning_rate=1e-4,
        num_train_timesteps=1000,  # TODO 4000 for iddpm (fixed in schedulers.py)
        num_inference_steps=100,  # TODO
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.modality_class_mapping = {'t1': 0, 't1c': 1, 't2': 2, 'flair': 3}

        self.debugging = True

        self.path_autoencoder = Path(path_autoencoder)
        self.dir_output_model = Path(dir_output_model) if dir_output_model else None

        self.dir_ema = Path(dir_ema) if dir_ema else None
        
        self.use_latents = use_latents
        self.use_distribution_shift = use_distribution_shift

        self.model_ = model_
        self.mask_conditioning = mask_conditioning
        self.modality_conditioning = modality_conditioning
        self.denoising = denoising
        self.scheduler_ = scheduler_

        self.latent_shape = latent_shape
        self.learning_rate = learning_rate
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_steps = num_inference_steps

        # Model
        if self.model_ == 'unet':
            self.model = UNet(self.mask_conditioning, self.modality_conditioning, self.scheduler_, self.latent_shape)
            self.parameters_ = self.model.parameters_unet_controlnet
        elif self.model_ == 'dit':
            self.model = DiT_Custom(self.scheduler_, self.latent_shape)
            self.parameters_ = self.model.parameters()

        # Scheduler
        self._scheduler_ = schedulers.Scheduler(self.scheduler_, self.num_train_timesteps, self.num_inference_steps)
        if self.scheduler_ == 'ddpm':
            self.scheduler_training = self._scheduler_.scheduler_training
            self.scheduler_inference = self._scheduler_.scheduler_inference
        elif self.scheduler_ == 'iddpm':
            self.scheduler = self._scheduler_.scheduler
            self.diffusion = self._scheduler_.diffusion

        # Autoencoders latent mean and std
        if self.use_distribution_shift:
            path_ae_latent = self.path_autoencoder.parent.parent / 'ae_latent.pkl'
            with path_ae_latent.open("rb") as f:
                ae_latent = pickle.load(f)
            self.ae_latent_mean = torch.as_tensor(list(ae_latent["mean"].values()), dtype=torch.float32).mean().to(self.device)
            self.ae_latent_std = torch.as_tensor(list(ae_latent["std"].values()), dtype=torch.float32).mean().to(self.device)
            print('Autoencoder Latent Mean:', self.ae_latent_mean)
            print('Autoencoder Latent Std:', self.ae_latent_std)

        # Postprocessing
        self.autoencoder_crop = transforms.CenterSpatialCrop(roi_size=(240, 240, 155))

        # Experiment 'volume_temporal'
        self.volume_temporal_previous = None

    def setup(self, stage=None):
        # Autoencoder
        if self.latent_shape == (4, 64, 64, 40):
            self.autoencoder = MaisiAutoencoder(path_autoencoder=str(self.path_autoencoder), device=self.device)
        elif self.latent_shape == (4, 32, 32, 20):
            self.autoencoder = F8D16Autoencoder(path_autoencoder=str(self.path_autoencoder), device=self.device)

        # Scheduler
        def scheduler_to_device(self, scheduler_to_move):
            for key, value in scheduler_to_move.__dict__.items():
                if isinstance(value, torch.Tensor):
                    scheduler_to_move.__dict__[key] = value.to(self.device)
        if self.scheduler_ == 'ddpm':
            scheduler_to_device(self, self.scheduler_training)
            scheduler_to_device(self, self.scheduler_inference)
        elif self.scheduler_ == 'iddpm':
            scheduler_to_device(self, self.scheduler)
    
    def _predict_noise(self, sample, timesteps, modality=None, conditioning=None):
        """
        Modular function to get noise prediction from (ControlNet +) UNet
        """
        modality = torch.tensor([self.modality_class_mapping[m] for m in modality], device=self.device, dtype=torch.long)
        noise_pred = self.model.forward(sample, timesteps, modality, conditioning)

        if self.scheduler_ == 'ddpm':
            _debugging_noise_pred = noise_pred
        elif self.scheduler_ == 'iddpm':
            _debugging_noise_pred = noise_pred[:, :4]  # only epsilon head

        # Debugging
        if self.training:
            helpers._debugging(self, _debugging_noise_pred, 'train/noise_pred', logging_=True, distribution_=True)
        else:
            t = timesteps[0].item()
            if t % 100 == 0 or t <= 10:
                helpers._debugging(self, _debugging_noise_pred, f'denoising/noise_pred/t_{t}', logging_=True, distribution_=True)

        return noise_pred

    def _generate_denoising(self, patients, modality, conditioning, affines, latent_voided=None, latent_mask=None, volume_temporal=False, volume_temporal_continue=None):
        helpers._swap_ema(self, apply_ema=True)
        
        batch_size = len(modality)

        if latent_mask is not None:
            latent_mask_np = latent_mask.cpu().numpy()
            dilated_mask_np = binary_dilation(latent_mask_np, iterations=1)
            dilated_mask = torch.from_numpy(dilated_mask_np).to(latent_mask.device).float().clamp(0, 1)

        # Initialize with pure noise
        sample = torch.randn(
            (batch_size, *self.latent_shape),
            device=self.device
        )

        if volume_temporal:  # experiment 'volume_temporal' and 'volume_temporal_continue'
            if volume_temporal_continue is not None and self.volume_temporal_previous is None:
                print('DEBUGGING Setting self.volume_temporal_previous (should only be done once)')
                self.volume_temporal_previous = self._get_encoded(volume_temporal_continue)
            if self.volume_temporal_previous is not None:
                print('DEBUGGING Adding noise to self.volume_temporal_previous')
                sample = self.scheduler_training.add_noise(self.volume_temporal_previous, sample, torch.tensor([100], device=self.device))

        # Denoising loop
        for t in list(range(self.num_inference_steps))[::-1]:
            t = torch.tensor([t], device=self.device, dtype=torch.long)
            
            # inpainting
            if latent_voided is not None and latent_mask is not None:
                # Pixel injection: preserve known regions (add noise to ground truth based on timestep)
                noise_gt = torch.randn(
                    (batch_size, *self.latent_shape),
                    device=self.device
                )
                noisy_gt = self.scheduler_training.add_noise(latent_voided, noise_gt, t)
                sample = (sample * dilated_mask + noisy_gt * (1 - dilated_mask)).float()

            if self.scheduler_ == 'ddpm':
                noise_pred = self._predict_noise(
                    sample,
                    t.expand(batch_size),
                    modality=modality,
                    conditioning=conditioning,
                )
                # Denoising step
                pred_prev_sample, pred_original_sample = self.scheduler_inference.step(noise_pred, t, sample)

            elif self.scheduler_ == 'iddpm':
                out = self.diffusion.p_sample(
                    self._predict_noise,
                    sample,
                    t.expand(batch_size),
                    clip_denoised=False,
                    model_kwargs={"modality": modality, "conditioning": conditioning},
                )
                pred_prev_sample, pred_original_sample = out["sample"], out["pred_xstart"]
                
            sample = pred_prev_sample

            # Debugging
            if t % 100 == 0 or t <= 10:
                helpers._debugging(self, sample, f'denoising/sample/t_{t.item()}', logging_=True, distribution_=True)
                if self.debugging:
                    generate_denoising_outputs = {
                        'timestep': t,
                        'patients': patients,
                        'affines': affines,
                        'modality': modality,
                        'pred_prev_sample': self._get_decoded(pred_prev_sample),
                        'pred_original_sample': self._get_decoded(pred_original_sample),
                    }
                    helpers._save_generate_denoising_outputs(self, generate_denoising_outputs)

        if latent_voided is not None and latent_mask is not None:
            sample = sample * dilated_mask + latent_voided * (1 - dilated_mask)

        if volume_temporal:
            self.volume_temporal_previous = sample

        if self.use_distribution_shift:
            sample = helpers._distribution_shift(sample, self.ae_latent_mean, self.ae_latent_std)

        helpers._swap_ema(self, apply_ema=False)

        return sample

    def _get_encoded(self, autoencoder_modality):
        # batch size for autoencoder can't be as big as for diffusion model
        # autoencoder_modality: B, 1, 256, 256, 160
        batch_size = autoencoder_modality.shape[0]
        latent_modality = []
        for i in range(batch_size):
            autoencoder_modality_i = autoencoder_modality[i].unsqueeze(0)  # 1, 1, 256, 256, 160
            # ====================================================================================================
            latent_modality_i = self.autoencoder.encode(autoencoder_modality_i)  # 1, 4, 64, 64, 40
            # ====================================================================================================
            latent_modality_i = latent_modality_i.squeeze(0)  # 4, 64, 64, 40
            latent_modality.append(latent_modality_i)
        latent_modality = torch.stack(latent_modality, dim=0)  # B, 4, 64, 64, 40
        return latent_modality.float()

    def _get_decoded(self, latent_modality):
        # batch size for autoencoder can't be as big as for diffusion model
        # latent_modality: B, 4, 64, 64, 40
        batch_size = latent_modality.shape[0]
        reconstructed_modality = []
        for i in range(batch_size):
            latent_modality_i = latent_modality[i].unsqueeze(0)  # 1, 4, 64, 64, 40
            # ====================================================================================================
            reconstructed_modality_i = self.autoencoder.decode(latent_modality_i)  # 1, 1, 256, 256, 160
            # ====================================================================================================
            reconstructed_modality_i = reconstructed_modality_i.squeeze(0)  # 1, 256, 256, 160
            # TODO
            # if self.latent_shape == (4, 64, 64, 40):
            #     reconstructed_modality_i = torch.clamp(reconstructed_modality_i, 0.0, 1.0)  # 1, 256, 256, 160
            # elif self.latent_shape == (4, 32, 32, 20):
            #     reconstructed_modality_i = torch.clamp(reconstructed_modality_i, -1.0, 1.0)  # 1, 256, 256, 160 
            # TODO
            reconstructed_modality_i = self.autoencoder_crop(reconstructed_modality_i)  # 1, 240, 240, 155
            reconstructed_modality.append(reconstructed_modality_i)
        reconstructed_modality = torch.stack(reconstructed_modality, dim=0)  # B, 1, 240, 240, 155
        return reconstructed_modality.float()

    def training_step(self, batch, batch_idx):
        """Training step"""
        self.batch_id = batch_idx

        modality = batch['modality']  # (B,)
        latent_modality = batch['latent_modality']  # (B, 4, 64, 64, 40)
        if self.mask_conditioning is not None: 
            conditioning = batch['conditioning']  # (B, 8, 64, 64, 40) or (B, 8, 32, 32, 20)
        else:
            conditioning = None
        
        batch_size = len(modality)

        if self.scheduler_ == 'ddpm':
            # Sampling
            timesteps = torch.randint(self.scheduler_training.num_train_timesteps, (batch_size,), device=self.device)
            
            noise = torch.randn_like(latent_modality)
            noisy_latent = self.scheduler_training.add_noise(latent_modality, noise, timesteps)
            noise_pred = self._predict_noise(noisy_latent, timesteps, modality=modality, conditioning=conditioning)

            # Loss
            loss = torch.nn.functional.mse_loss(noise_pred, noise)
        
        elif self.scheduler_ == 'iddpm':
            # Sampling
            timesteps, weights = self.scheduler.sample(batch_size, self.device)  # both: B,
            
            # Loss
            losses = self.diffusion.training_losses(
                self._predict_noise,
                latent_modality,
                timesteps,
                model_kwargs={"modality": modality, "conditioning": conditioning},
            )
            self.scheduler.update_with_local_losses(timesteps, losses["loss"].detach())
            loss = (losses["loss"] * weights).mean()

        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        patients = batch['patient']  # (B,)

        affines = batch['affine']  # (B,)
        normalized = batch['normalized_modality']  # (B, 1, 240, 240, 155)

        modality = batch['modality']  # (B,)
        if self.mask_conditioning is not None: 
            conditioning = batch['conditioning']  # (B, 8, 64, 64, 40) or (B, 8, 32, 32, 20)
        else:
            conditioning = None

        # Generate inpainted image
        with torch.no_grad():
            denoised = self._generate_denoising( 
                patients,
                modality,
                conditioning,
                affines
            ).float()  # (B, 4, 64, 64, 40)

            reconstructed = self._get_decoded(denoised)  # (B, 1, 240, 240, 155)

        # Loss
        loss_mse = torch.nn.functional.mse_loss(reconstructed, normalized)
        loss_l1 = torch.nn.functional.l1_loss(reconstructed, normalized)

        # Logging & Debugging
        self.log('val/loss_mse', loss_mse, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val/loss_l1', loss_l1, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        helpers._debugging(self, denoised, 'val/denoised', logging_=True, distribution_=True)
        helpers._debugging(self, reconstructed, 'val/reconstructed', logging_=True)

        validation_step_outputs = {
            'patients': patients,
            'modality': modality,
            'affines': affines,
            'normalized': normalized.cpu(),
            'reconstructed': reconstructed.cpu(),
        }
        helpers._save_validation_step_outputs(self, validation_step_outputs)

        return {
            'val/loss_l1': loss_l1,
        }

    def test_step(self, batch, batch_idx, volume_temporal=False, volume_temporal_continue=None):
        """Test step"""
        mode = batch['mode'][0]

        patients = batch['patient']  # (B,)

        modality = batch['modality']  # (B,)
        conditioning = batch['conditioning']  # (B, 8, 64, 64, 40) or (B, 8, 32, 32, 20)
        
        affines = batch['affine']  # (B,)

        if mode in ['inpainting_inference', 'inpainting_inference_conditioning', 'inpainting_inference_challenge']:

            original_t1_voided = batch['original_t1_voided']  # (B, 1, 240, 240, 155)
            autoencoder_t1_voided = batch['autoencoder_t1_voided']  # (B, 1, 240, 240, 160)
            latent_t1_voided = self._get_encoded(autoencoder_t1_voided)  # (B, 4, 60, 60, 40)
            
            original_mask = batch['original_mask']  # (B, 1, 240, 240, 155)
            latent_mask = batch['latent_mask']  # (B, 1, 60, 60, 40)

            original_conditioning = batch['original_conditioning']  # (B, 4, 240, 240, 155)

            if mode in ['inpainting_inference', 'inpainting_inference_conditioning']:
                masks = batch['mask']  # (B,)
            elif mode == 'inpainting_inference_challenge':
                masks = None
                exists_conditioning = all(batch['exists_conditioning'])
                paths_original_t1_voided = batch['path_original_t1_voided']  # (B,)
                paths_original_mask = batch['path_original_mask']  # (B,)

        with torch.no_grad():
            if mode == 'inpainting_inference_challenge':
                if not exists_conditioning:
                    original_conditioning = self._generate_conditioning(
                        paths_original_t1_voided,
                        paths_original_mask
                    ).float()  # B, 4, 240, 240, 155
                    latent_conditioning = torch.nn.functional.interpolate(
                        original_conditioning,
                        size=(latent_t1_voided.shape[2], latent_t1_voided.shape[3], latent_t1_voided.shape[4]),
                        mode='nearest'
                    ).float()  # B, 4, 60, 60, 40

            # Generate inpainted latent
            denoisings = {
                'repaint': self._repaint_generate_denoising,
                'own': self._generate_denoising,
            }

            # Generation
            if mode == 'validation':
                denoised = self._generate_denoising( 
                    patients,
                    modality,
                    conditioning,
                    affines,
                    volume_temporal=volume_temporal,
                    volume_temporal_continue=volume_temporal_continue
                ).float()  # (B, 4, 64, 64, 40)
                reconstructed = self._get_decoded(denoised)  # (B, 1, 240, 240, 155)

                return reconstructed

            # Inpainting
            elif mode in ['inpainting_inference', 'inpainting_inference_conditioning', 'inpainting_inference_challenge']:

                # modality = 't1' ?
                inpainted_t1 = denoisings[self.denoising](
                    latent_conditioning,
                    latent_voided=latent_t1_voided,
                    latent_mask=latent_mask,
                ).float()
                reconstructed_t1 = self._get_decoded(inpainted_t1)  # (B, 1, 240, 240, 155)
                
                original_min, original_max = original_t1_voided.min(), original_t1_voided.max()
                reconstructed_t1 = reconstructed_t1 * (original_max - original_min) + original_min
                helpers._save_reconstruction(self, reconstructed_t1, patients, masks, affines, identifier='inpainted', mode=mode)
                
                ########## Histogram Equalization

                reconstructed_t1_he = self._histogram_equalization(reconstructed_t1, original_t1_voided)  # (B, 1, 240, 240, 155)
                helpers._save_reconstruction(self, reconstructed_t1_he, patients, masks, affines, identifier='histogram_equalization', mode=mode)

                ########## Poisson Blending

                reconstructed_t1_pb = self._poisson_blending(reconstructed_t1_he, original_t1_voided, original_mask)  # (B, 1, 240, 240, 155)
                helpers._save_reconstruction(self, reconstructed_t1_pb, patients, masks, affines, identifier='poisson_blending', mode=mode)

                ########## Pixel Injection
                
                reconstructed_t1_pi = self._pixel_injection(reconstructed_t1_pb, original_t1_voided, original_mask)  # (B, 1, 240, 240, 155)
                helpers._save_reconstruction(self, reconstructed_t1_pi, patients, masks, affines, identifier='pixel_injection', mode=mode)

    def on_save_checkpoint(self, checkpoint):
        helpers._save_ema(self)

    def on_after_backward(self):
        pass
        # ~ 2 seconds
        # gradients_norm = helpers._gradients_compute_norm(self.parameters())
        # self.log(f'gradients_norm', gradients_norm, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)
        # if gradients_norm > 1.0: tqdm.write(f"[DEBUGGING] Gradients Warning (total_norm={gradients_norm:.3f})")

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        helpers._update_ema(self)

    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(
            self.parameters_,
            lr=self.learning_rate,
            # fused=True,  # would halve the time needed but then gradient clipping doesn't work anymore
        )
        helpers._load_ema(self)

        return self.optimizer