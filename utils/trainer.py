import sys
from pathlib import Path
dir_current = Path(__file__).resolve().parent
dir_maisi = dir_current.parent / 'maisi'
dir_gbm_bench = dir_current.parent / 'gbm_bench'
sys.path.append(str(dir_current))
sys.path.append(str(dir_maisi))
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
from collections import defaultdict

import pietorch
import skimage.exposure
from scipy.ndimage import binary_dilation, gaussian_filter

from monai import transforms

import saving
import helpers
import schedulers
from model_dit import DiT
from model_unet import UNet
from maisi_autoencoder import MaisiAutoencoder
# from controlnet_maisi import ControlNetMaisi
# from diffusion_model_unet_maisi import DiffusionModelUNetMaisi


class LatentDiffusion(L.LightningModule):
    """
    PyTorch Lightning module for latent diffusion inpainting
    """
    
    def __init__(
        self,
        path_autoencoder=None,
        dir_output_model=None,
        use_latents=True,
        use_distribution_shift=False,
        model_='unet',  # unet, dit  # TODO
        mask_conditioning=256,  # 256, 64, None
        modality_conditioning=True,  # True, False
        denoising='own', # own, repaint
        scheduler_='ddpm',  # ddpm, ddim, iddpm  # TODO
        latent_shape=None,
        learning_rate=1e-4,
        num_train_timesteps=1000,
        num_inference_steps=1, # TODO
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.modality_class_mapping = {'t1': 0, 't1c': 1, 't2': 2, 'flair': 3}

        self.debugging = True

        self.path_autoencoder = Path(path_autoencoder)
        self.dir_output_model = Path(dir_output_model) if dir_output_model else None
        
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
            self.model = UNet(self.mask_conditioning, self.modality_conditioning)
            self.parameters_ = self.model.parameters_unet_controlnet
        elif self.model_ == 'dit':
            self.model = DiT(
                input_size=[64, 64, 40], 
                patch_size=[2, 2, 2],  # DiT_B_2: [2]*dims
                hidden_size=768,  # DiT_B_2: 768, Default: 1152
                depth=12,  # DiT_B_2: 12, Default: 28
                num_heads=12,  # DiT_B_2: 12, Default: 16
                num_classes=4,
            )
            self.parameters_ = self.model.parameters()

        # Scheduler
        self.scheduler__ = schedulers.Scheduler(self.scheduler_, self.num_train_timesteps, self.num_inference_steps, self.device)
        self.diffusion = self.scheduler__.diffusion
        self.scheduler = self.scheduler__.scheduler

        # Autoencoders latent mean and std
        if self.use_distribution_shift:
            path_ae_latent = self.path_autoencoder.parent / 'ae_latent.pkl'
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
        self.autoencoder = MaisiAutoencoder(path_autoencoder=str(self.path_autoencoder), device=self.device)

    def _predict_noise_wrapper(self, sample, timesteps, modality=None, conditioning=None):
        return self._predict_noise(sample, timesteps, modality, conditioning, split_dit=False)

    def _predict_noise(self, sample, timesteps, modality, conditioning, split_dit=True):
        """
        Modular function to get noise prediction from (ControlNet +) UNet
        """
        modality = torch.tensor([self.modality_class_mapping[m] for m in modality], device=self.device, dtype=torch.long)
        noise_pred = self.model.forward(sample, timesteps, modality, conditioning)
        if self.model_ == 'dit' and split_dit:
            sigma = noise_pred[:, 4:]  # TODO
            noise_pred = noise_pred[:, :4]  # TODO
        print('Noise Pred Shape', noise_pred.shape) # TODO
        return noise_pred

    def _generate_denoising(self, patients, modality, conditioning, affines, latent_voided=None, latent_mask=None, volume_temporal=False, volume_temporal_continue=None):
        batch_size = conditioning.shape[0]

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
                sample = self.scheduler.add_noise(self.volume_temporal_previous, sample, torch.tensor([100], device=self.device))

        # Denoising loop
        for t in list(range(self.num_inference_steps))[::-1]:
            t = t.floor().long().to(self.device)
            
            # inpainting
            if latent_voided is not None and latent_mask is not None:
                # Pixel injection: preserve known regions (add noise to ground truth based on timestep)
                noise_gt = torch.randn(
                    (batch_size, *self.latent_shape),
                    device=self.device
                )
                noisy_gt = self.scheduler.add_noise(latent_voided, noise_gt, t)
                sample = (sample * dilated_mask + noisy_gt * (1 - dilated_mask)).float()

            if self.scheduler_ in ['ddpm', 'ddim']:
                noise_pred = self._predict_noise(
                    sample,
                    t.expand(batch_size),
                    modality,
                    conditioning
                )
                # Denoising step
                pred_prev_sample, pred_original_sample = self.scheduler.step(noise_pred, t, sample)

            elif self.scheduler_ == 'iddpm':
                out = self.diffusion.p_sample(
                    self._predict_noise_wrapper,
                    sample,
                    t.expand(batch_size),
                    clip_denoised=False,
                    model_kwargs={"modality": modality, "conditioning": conditioning},
                )
                pred_prev_sample, pred_original_sample = out["sample"], out["pred_xstart"]
                
            sample = pred_prev_sample

            # Debugging
            if t % 100 == 0 or t <= 10:
                # helpers._debugging(self, noise_pred, f'denoising/noise_pred/t_{t}', logging_=True, distribution_=True)
                helpers._debugging(self, sample, f'denoising/sample/t_{t}', logging_=True, distribution_=True)
                if self.debugging:
                    generate_denoising_outputs = {
                        'timestep': t,
                        'patients': patients,
                        'affines': affines,
                        'modality': modality,
                        'pred_prev_sample': self._get_decoded(pred_prev_sample),
                        'pred_original_sample': self._get_decoded(pred_original_sample),
                    }
                    saving._save_generate_denoising_outputs(self, generate_denoising_outputs)

        if latent_voided is not None and latent_mask is not None:
            sample = sample * dilated_mask + latent_voided * (1 - dilated_mask)

        if volume_temporal:
            self.volume_temporal_previous = sample

        if self.use_distribution_shift:
            sample = helpers._distribution_shift(sample, self.ae_latent_mean, self.ae_latent_std)

        return sample

    def _get_encoded(self, autoencoder_modality):
        # batch size for autoencoder can't be as big as for diffusion model
        # autoencoder_modality: B, 1, 256, 256, 160
        batch_size = latent_modality.shape[0]
        latent_modality = []
        for i in range(batch_size):
            autoencoder_modality_i = autoencoder_modality[i].unsqueeze(0)  # 1, 1, 256, 256, 160
            # ====================================================================================================
            latent_modality = self.autoencoder.encode(autoencoder_modality)  # 1, 4, 64, 64, 40
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
            # reconstructed_modality_i = torch.clamp(reconstructed_modality_i, 0.0, 1.0)  # 1, 256, 256, 160 # TODO
            reconstructed_modality_i = self.autoencoder_crop(reconstructed_modality_i)  # 1, 240, 240, 155
            reconstructed_modality.append(reconstructed_modality_i)
        reconstructed_modality = torch.stack(reconstructed_modality, dim=0)  # B, 1, 240, 240, 155
        return reconstructed_modality.float()

    def training_step(self, batch, batch_idx):
        """Training step"""
        modality = batch['modality']  # (B,)
        latent_modality = batch['latent_modality']  # (B, 4, 64, 64, 40)
        if self.mask_conditioning is not None: 
            conditioning = batch['conditioning']  # (B, 4, 256, 256, 160)
        else:
            conditioning = None
        
        batch_size = len(modality)

        if self.scheduler_ in ['ddpm', 'ddim']:
            # Sampling
            timesteps = torch.randint(self.scheduler.num_train_timesteps, (batch_size,), device=self.device)
            
            noise = torch.randn_like(latent_modality)
            noisy_latent = self.scheduler.add_noise(latent_modality, noise, timesteps)
            noise_pred = self._predict_noise(noisy_latent, timesteps, modality, conditioning)

            # Loss
            loss_mse = torch.nn.functional.mse_loss(noise_pred, noise)
            loss_l1 = torch.nn.functional.l1_loss(noise_pred, noise)
            loss = loss_mse

            # Logging & Debugging
            self.log('train/loss_mse', loss_mse, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log('train/loss_l1', loss_l1, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            helpers._debugging(self, noise, 'train/noise', logging_=True, distribution_=True)
            helpers._debugging(self, noisy_latent, 'train/noisy_latent', logging_=True, distribution_=True)
            helpers._debugging(self, noise_pred, 'train/noise_pred', logging_=True, distribution_=True)
        
        elif self.scheduler_ == 'iddpm':
            # Sampling
            timesteps, weights = self.scheduler.sample(batch_size)
            print(timesteps.shape, weights.shape)
            
            # Loss
            # losses = self.diffusion.training_losses(self.ddp_model, micro, timesteps, model_kwargs=micro_cond)
            losses = self.diffusion.training_losses(
                self._predict_noise_wrapper,
                latent_modality,
                timesteps,
                model_kwargs={"modality": modality, "conditioning": conditioning},
            )
            self.scheduler.update_with_local_losses(timesteps, losses["loss"].detach())
            loss = (losses["loss"] * weights).mean()

            # Logging
            self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        patients = batch['patient']  # (B,)

        modality = batch['modality']  # (B,)
        conditioning = batch['conditioning']  # (B, 4, 256, 256, 160)
        
        affines = batch['affine']  # (B,)
        normalized = batch['normalized_modality']  # (B, 1, 240, 240, 155)

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
        saving._save_validation_step_outputs(self, validation_step_outputs)

        return {
            'val/loss_l1': loss_l1,
        }

    def test_step(self, batch, batch_idx, volume_temporal=False, volume_temporal_continue=None):
        """Test step"""
        mode = batch['mode'][0]

        patients = batch['patient']  # (B,)

        modality = batch['modality']  # (B,)
        conditioning = batch['conditioning']  # (B, 4, 256, 256, 160)
        
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
                saving._save_reconstruction(self, reconstructed_t1, patients, masks, affines, identifier='inpainted', mode=mode)
                
                ########## Histogram Equalization

                reconstructed_t1_he = self._histogram_equalization(reconstructed_t1, original_t1_voided)  # (B, 1, 240, 240, 155)
                saving._save_reconstruction(self, reconstructed_t1_he, patients, masks, affines, identifier='histogram_equalization', mode=mode)

                ########## Poisson Blending

                reconstructed_t1_pb = self._poisson_blending(reconstructed_t1_he, original_t1_voided, original_mask)  # (B, 1, 240, 240, 155)
                saving._save_reconstruction(self, reconstructed_t1_pb, patients, masks, affines, identifier='poisson_blending', mode=mode)

                ########## Pixel Injection
                
                reconstructed_t1_pi = self._pixel_injection(reconstructed_t1_pb, original_t1_voided, original_mask)  # (B, 1, 240, 240, 155)
                saving._save_reconstruction(self, reconstructed_t1_pi, patients, masks, affines, identifier='pixel_injection', mode=mode)


    def on_after_backward(self):
        gradients_norm = helpers._gradients_compute_norm(self.parameters())
        self.log(f'gradients_norm', gradients_norm, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)
        if gradients_norm > 1.0: tqdm.write(f"[DEBUGGING] Gradients Warning (total_norm={gradients_norm:.3f})")


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters_,
            lr=self.learning_rate,
            # fused=True,  # would halve the time needed but then gradient clipping doesn't work anymore
        )

        return optimizer