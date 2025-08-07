import sys
from pathlib import Path
dir_current = Path(__file__).resolve().parent
dir_maisi = dir_current.parent / 'maisi'
dir_gbm_bench = dir_current.parent / 'gbm_bench'
sys.path.append(str(dir_maisi))
sys.path.append(str(dir_gbm_bench))

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
import nibabel as nib
from pathlib import Path
from collections import defaultdict

import pietorch
import skimage.exposure
from scipy.ndimage import binary_dilation, gaussian_filter

from monai import transforms
from generative.networks.schedulers import DDIMScheduler, DDPMScheduler

from .data import create_conditioning

from maisi_autoencoder import MaisiAutoencoder
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
        model_='big',  # small, big, big_old
        scheduler_='ddpm',  # ddpm, ddim
        learning_rate=1e-4,
        num_train_timesteps=1000,
        denoising='repaint', # repaint, own
        num_inference_steps=100,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.path_autoencoder = Path(path_autoencoder)
        self.dir_output_model = Path(dir_output_model)
        self.denoising = denoising
        self.model_ = model_
        self.scheduler_ = scheduler_
        
        if self.model_ == 'small':
            config_unet = {
                "spatial_dims": 3,
                "in_channels": 4,  # latent shape: (B, 4, 60, 60, 40)
                "out_channels": 4,  # latent shape: (B, 4, 60, 60, 40)
                "num_res_blocks": [2, 2, 2],
                "num_channels": (32, 64, 128),
                "attention_levels": (False, True, True),
                "norm_num_groups": 32,
                "resblock_updown": True,
                "num_head_channels": (32, 64, 128),
                "transformer_num_layers": 8,
                "use_flash_attention": True,
                "with_conditioning": False,
                "cross_attention_dim": None
            }
            config_controlnet = {
                "spatial_dims": 3,
                "in_channels": 4,
                "num_res_blocks": [2, 2, 2],
                "num_channels": (32, 64, 128),
                "attention_levels": (False, True, True),
                "norm_num_groups": 32,
                "resblock_updown": True,
                "num_head_channels": (32, 64, 128),
                "transformer_num_layers": 8,
                "use_flash_attention": True,
                "with_conditioning": False,
                "cross_attention_dim": None
            }
            config_conditioning_embedding_num_channels = (16,)
        elif self.model_ in ['big', 'big_old']:
            config_unet = {
                "spatial_dims": 3,
                "in_channels": 4,  # latent shape: (B, 4, 60, 60, 40)
                "out_channels": 4,  # latent shape: (B, 4, 60, 60, 40)
                "num_res_blocks": [2, 2, 2],
                "num_channels": (64, 128, 256) if self.model_ == 'big' else (128, 256, 512),
                "attention_levels": (False, True, True),
                "resblock_updown": True if self.model_ == 'big' else False,
                "num_head_channels": (64, 128, 256) if self.model_ == 'big' else (0, 256, 512),
                "transformer_num_layers": 8 if self.model_ == 'big' else 1,
            }
            config_controlnet = {
                "spatial_dims": 3,
                "in_channels": 4,
                "num_res_blocks": [2, 2, 2],
                "num_channels": (64, 128, 256) if self.model_ == 'big' else (128, 256, 512),
                "attention_levels": (False, True, True),
                "resblock_updown": True if self.model_ == 'big' else False,
                "num_head_channels": (64, 128, 256) if self.model_ == 'big' else (0, 256, 512),
                "transformer_num_layers": 8 if self.model_ == 'big' else 1,
            }
            config_conditioning_embedding_num_channels = (64,) if self.model_ == 'big' else (128,)

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
        if self.scheduler_ == 'ddpm':
            self.scheduler = DDPMScheduler(
                num_train_timesteps=num_train_timesteps,
                schedule="linear_beta",
                prediction_type="epsilon",
                clip_sample=False,
            )
        elif self.scheduler_ == 'ddim':
            self.scheduler = DDIMScheduler(
                num_train_timesteps=num_train_timesteps,
                schedule="linear_beta",
                prediction_type="epsilon",
                clip_sample=False,
            )
        
        # Loss function
        self.loss_fn = nn.MSELoss()
        self.autoencoder_crop = transforms.CenterSpatialCrop(roi_size=(1, 240, 240, 155))

        # Repaint
        self.betas = self._repaint_get_named_beta_schedule('linear', self.hparams.num_train_timesteps, use_scale=True)
        self.betas = np.array(self.betas, dtype=np.float64)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)

    def setup(self, stage=None):
        self.autoencoder = MaisiAutoencoder(path_autoencoder=str(self.path_autoencoder), device=self.device)

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

    def _generate_denoising(self, latent_conditioning, latent_voided=None, latent_mask=None):
        """Generate denoised latent from pure noise using ControlNet conditioning"""
        batch_size = latent_conditioning.shape[0]
        
        if latent_mask is not None:
            latent_mask_np = latent_mask.cpu().numpy()
            dilated_mask_np = binary_dilation(latent_mask_np, iterations=1)
            dilated_mask = torch.from_numpy(dilated_mask_np).to(latent_mask.device).float().clamp(0, 1)

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

            # inpainting
            if latent_voided is not None and latent_mask is not None:
                # Pixel injection: preserve known regions (add noise to ground truth based on timestep)
                noise_gt = torch.randn(
                    latent_conditioning.shape,
                    device=self.device
                )
                noisy_gt = self.scheduler.add_noise(latent_voided, noise_gt, t)
                sample = (sample * dilated_mask + noisy_gt * (1 - dilated_mask)).float()
               
            noise_pred = self._get_noise_prediction(
                sample,
                t_device.unsqueeze(0).repeat(batch_size),
                latent_conditioning
            )

            # Denoising step
            sample = self.scheduler.step(noise_pred, t, sample)[0]

        if latent_voided is not None and latent_mask is not None:
            sample = sample * dilated_mask + latent_voided * (1 - dilated_mask)

        return sample

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
        autoencoder_t1 = batch['autoencoder_t1']  # (B, 1, 240, 240, 160)
        latent_t1 = self._get_encoded_autoencoder(autoencoder_t1)  # (B, 4, 60, 60, 40)
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
        patients = batch['patient']  # (B,)
        affines = batch['affine']  # (B,)
        normalized_t1 = batch['normalized_t1']  # (B, 1, 240, 240, 155)
        latent_conditioning = batch['latent_conditioning']  # (B, 4, 60, 60, 40)
        
        # Generate inpainted image
        with torch.no_grad():
            denoised_t1 = self._generate_denoising( 
                latent_conditioning
            ).float()

            reconstructed_t1 = self._get_decoded_autoencoder(denoised_t1)  # (B, 1, 240, 240, 155)

        if batch_idx == 0:
            self.validation_outputs_for_saving = {
                'patients': patients,
                'affines': affines,
                'normalized_t1': normalized_t1.cpu(),
                'reconstructed_t1': reconstructed_t1.cpu(),
            }

        # Compute validation loss (MSE in image space)
        val_loss = self.loss_fn(reconstructed_t1, normalized_t1)
        
        # Log validation metrics
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return {
            'val_loss': val_loss,
        }
    
    def test_step(self, batch, batch_idx):
        """Test step"""
        mode = batch['mode'][0]

        patients = batch['patient']  # (B,)
        affines = batch['affine']  # (B,)

        original_t1_voided = batch['original_t1_voided']  # (B, 1, 240, 240, 155)
        autoencoder_t1_voided = batch['autoencoder_t1_voided']  # (B, 1, 240, 240, 160)
        latent_t1_voided = self._get_encoded_autoencoder(autoencoder_t1_voided)  # (B, 4, 60, 60, 40)
        
        original_mask = batch['original_mask']  # (B, 1, 240, 240, 155)
        latent_mask = batch['latent_mask']  # (B, 1, 60, 60, 40)

        original_conditioning = batch['original_conditioning']  # (B, 1, 240, 240, 155)
        latent_conditioning = batch['latent_conditioning']  # (B, 1, 60, 60, 40)

        if mode in ['inference', 'inference_conditioning']:
            masks = batch['mask']  # (B,)
        elif mode == 'inference_challenge':
            masks = None
            exists_conditioning = all(batch['exists_conditioning'])
            paths_original_t1_voided = batch['path_original_t1_voided']  # (B,)
            paths_original_mask = batch['path_original_mask']  # (B,)

        with torch.no_grad():
            if mode == 'inference_challenge':
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
            if mode == 'only_generation':  # not implemented yet, also naming will be changed

                denoised_t1 = denoisings[self.denoising]( 
                    latent_conditioning
                ).float()
                reconstructed_t1 = self._get_decoded_autoencoder(denoised_t1)  # (B, 1, 240, 240, 155)

            # Inpainting
            elif mode in ['inference', 'inference_conditioning', 'inference_challenge']:

                inpainted_t1 = denoisings[self.denoising](
                    latent_conditioning,
                    latent_voided=latent_t1_voided,
                    latent_mask=latent_mask,
                ).float()
                reconstructed_t1 = self._get_decoded_autoencoder(inpainted_t1)  # (B, 1, 240, 240, 155)
                
                original_min, original_max = original_t1_voided.min(), original_t1_voided.max()
                reconstructed_t1 = reconstructed_t1 * (original_max - original_min) + original_min
                self._save_reconstruction(reconstructed_t1, patients, masks, affines, identifier='inpainted', mode=mode)
                
                ########## Histogram Equalization

                reconstructed_t1_he = self._histogram_equalization(reconstructed_t1, original_t1_voided)  # (B, 1, 240, 240, 155)
                self._save_reconstruction(reconstructed_t1_he, patients, masks, affines, identifier='histogram_equalization', mode=mode)

                ########## Poisson Blending

                reconstructed_t1_pb = self._poisson_blending(reconstructed_t1_he, original_t1_voided, original_mask)  # (B, 1, 240, 240, 155)
                self._save_reconstruction(reconstructed_t1_pb, patients, masks, affines, identifier='poisson_blending', mode=mode)

                ########## Pixel Injection
                
                reconstructed_t1_pi = self._pixel_injection(reconstructed_t1_pb, original_t1_voided, original_mask)  # (B, 1, 240, 240, 155)
                self._save_reconstruction(reconstructed_t1_pi, patients, masks, affines, identifier='pixel_injection', mode=mode)

    def _repaint_generate_denoising(self, latent_conditioning, latent_voided=None, latent_mask=None):
        self.scheduler.set_timesteps(self.hparams.num_inference_steps)

        final = None
        for sample in self._repaint_p_sample_loop_progressive(latent_conditioning, latent_voided, latent_mask):
            final = sample
        return final["sample"]

    def _repaint_p_sample_loop_progressive(self, latent_conditioning, latent_voided=None, latent_mask=None):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        shape = latent_conditioning.shape

        image_after_step = torch.randn(
            shape,
            device=self.device
        )
        
        self.gt_noises = None  # reset for next image

        pred_xstart = None

        idx_wall = -1
        
        sample_idxs = defaultdict(lambda: 0)

        # schedule_jump_params
        t_T = 250
        n_sample = 1
        jump_length = 10
        jump_n_sample = 10

        times = self._repaint_get_schedule_jump(t_T=t_T, n_sample=n_sample, jump_length=jump_length, jump_n_sample=jump_n_sample)

        time_pairs = list(zip(times[:-1], times[1:]))

        for t_last, t_cur in time_pairs:
            idx_wall += 1
            t_last_t = torch.tensor([t_last] * shape[0],  # pylint: disable=not-callable
                                    device=self.device)

            if t_cur < t_last:  # reverse
                with torch.no_grad():
                    image_before_step = image_after_step.clone()
                    out = self._repaint_p_sample(
                        image_after_step,
                        t_last_t,
                        latent_conditioning=latent_conditioning,
                        latent_voided=latent_voided,
                        latent_mask=latent_mask,
                    )
                    image_after_step = out["sample"]
                    pred_xstart = out["pred_xstart"]

                    sample_idxs[t_cur] += 1

                    yield out

            else:
                t_shift = 1

                image_before_step = image_after_step.clone()
                image_after_step = self._repaint_undo(
                    image_before_step, image_after_step,
                    est_x_0=out['pred_xstart'], t=t_last_t+t_shift, debug=False)
                pred_xstart = out["pred_xstart"]
        
    def _repaint_p_sample(
        self,
        x,
        t,
        latent_conditioning,
        latent_voided,
        latent_mask,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """

        mask_np = latent_mask.cpu().numpy()
        dilated_mask_np = binary_dilation(mask_np, iterations=1)
        gt_keep_mask = 1 - torch.from_numpy(dilated_mask_np).to(latent_mask.device).float().clamp(0, 1)
        gt = latent_voided  # model_kwargs['gt']

        alpha_cumprod = self._repaint_extract_into_tensor(
            self.alphas_cumprod, t, x.shape)

        gt_weight = torch.sqrt(alpha_cumprod)
        gt_part = gt_weight * gt

        noise_weight = torch.sqrt((1 - alpha_cumprod))
        noise_part = noise_weight * torch.randn_like(x)

        weighed_gt = gt_part + noise_part

        x = gt_keep_mask * weighed_gt + (1 - gt_keep_mask) * x

        noise_pred = self._get_noise_prediction(
            x,  # x is the current sample
            t,  # t is the current timestep
            latent_conditioning
        )
        step_result = self.scheduler.step(noise_pred, t[0], x)

        sample = step_result[0]
        pred_xstart = step_result[1]

        result = {
            "sample": sample,
            "pred_xstart": pred_xstart,
            "gt": latent_voided
        }
        
        return result

    def _repaint_get_schedule_jump(self, t_T, n_sample, jump_length, jump_n_sample,
                      jump2_length=1, jump2_n_sample=1,
                      jump3_length=1, jump3_n_sample=1,
                      start_resampling=100000000):

        jumps = {}
        for j in range(0, t_T - jump_length, jump_length):
            jumps[j] = jump_n_sample - 1

        jumps2 = {}
        for j in range(0, t_T - jump2_length, jump2_length):
            jumps2[j] = jump2_n_sample - 1

        jumps3 = {}
        for j in range(0, t_T - jump3_length, jump3_length):
            jumps3[j] = jump3_n_sample - 1

        t = t_T
        ts = []

        while t >= 1:
            t = t-1
            ts.append(t)

            if (
                t + 1 < t_T - 1 and
                t <= start_resampling
            ):
                for _ in range(n_sample - 1):
                    t = t + 1
                    ts.append(t)

                    if t >= 0:
                        t = t - 1
                        ts.append(t)

            if (
                jumps3.get(t, 0) > 0 and
                t <= start_resampling - jump3_length
            ):
                jumps3[t] = jumps3[t] - 1
                for _ in range(jump3_length):
                    t = t + 1
                    ts.append(t)

            if (
                jumps2.get(t, 0) > 0 and
                t <= start_resampling - jump2_length
            ):
                jumps2[t] = jumps2[t] - 1
                for _ in range(jump2_length):
                    t = t + 1
                    ts.append(t)
                jumps3 = {}
                for j in range(0, t_T - jump3_length, jump3_length):
                    jumps3[j] = jump3_n_sample - 1

            if (
                jumps.get(t, 0) > 0 and
                t <= start_resampling - jump_length
            ):
                jumps[t] = jumps[t] - 1
                for _ in range(jump_length):
                    t = t + 1
                    ts.append(t)
                jumps2 = {}
                for j in range(0, t_T - jump2_length, jump2_length):
                    jumps2[j] = jump2_n_sample - 1

                jumps3 = {}
                for j in range(0, t_T - jump3_length, jump3_length):
                    jumps3[j] = jump3_n_sample - 1

        ts.append(-1)

        self._repaint_check_times(ts, -1, t_T)

        return ts

    def _repaint_check_times(self, times, t_0, t_T):
        # Check end
        assert times[0] > times[1], (times[0], times[1])

        # Check beginning
        assert times[-1] == -1, times[-1]

        # Steplength = 1
        for t_last, t_cur in zip(times[:-1], times[1:]):
            assert abs(t_last - t_cur) == 1, (t_last, t_cur)

        # Value range
        for t in times:
            assert t >= t_0, (t, t_0)
            assert t <= t_T, (t, t_T)

    def _repaint_extract_into_tensor(self, arr, timesteps, broadcast_shape):
        """
        Extract values from a 1-D numpy array for a batch of indices.

        :param arr: the 1-D numpy array.
        :param timesteps: a tensor of indices into the array to extract.
        :param broadcast_shape: a larger shape of K dimensions with the batch
                                dimension equal to the length of timesteps.
        :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
        """
        res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)

    def _repaint_get_named_beta_schedule(self, schedule_name, num_diffusion_timesteps, use_scale):
        """
        Get a pre-defined beta schedule for the given name.

        The beta schedule library consists of beta schedules which remain similar
        in the limit of num_diffusion_timesteps.
        Beta schedules may be added, but should not be removed or changed once
        they are committed to maintain backwards compatibility.
        """
        if schedule_name == "linear":
            # Linear schedule from Ho et al, extended to work for any number of
            # diffusion steps.

            if use_scale:
                scale = 1000 / num_diffusion_timesteps
            else:
                scale = 1

            beta_start = scale * 0.0001
            beta_end = scale * 0.02
            return np.linspace(
                beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
            )

    def _repaint_undo(self, image_before_step, img_after_model, est_x_0, t, debug=False):
        return self._repaint__undo(img_after_model, t)

    def _repaint__undo(self, img_out, t):
        beta = self._repaint_extract_into_tensor(self.betas, t, img_out.shape)

        img_in_est = torch.sqrt(1 - beta) * img_out + \
            torch.sqrt(beta) * torch.randn_like(img_out)

        return img_in_est

    def _generate_conditioning(self, paths_original_t1_voided, paths_original_mask):
        from gbm_bench.preprocessing.preprocess import preprocess_nifti

        temp_dir = Path(self.dir_output_model) / f"temp_{random.randint(10000, 99999)}"
        temp_dir.mkdir(parents=True, exist_ok=True)

        original_conditionings = []
        for i in range(len(paths_original_t1_voided)):
            path_original_t1_voided = Path(paths_original_t1_voided[i])
            path_original_mask = Path(paths_original_mask[i])

            temp_patient = path_original_t1_voided.parent.name
            temp_dir_patient = temp_dir / temp_patient
            temp_dir_patient.mkdir(parents=True, exist_ok=True)

            path_inverted_mask = temp_dir_patient / f"{temp_patient}-mask-inverted.nii.gz"
            image_original_mask = nib.load(path_original_mask)
            data_original_mask = image_original_mask.get_fdata()
            affine_original_mask = image_original_mask.affine
            inverted_mask = (data_original_mask == 0).astype(np.float32)
            nib.save(nib.Nifti1Image(inverted_mask, affine_original_mask), path_inverted_mask)

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
                outdir=temp_dir_patient,
                is_coregistered=True,
                is_skull_stripped=True,
                # tumorseg_file=Path(temp_dir),
                cuda_device=device_str,
                registration_mask_file=path_inverted_mask
            )

            path_original_tissue_segmentation = temp_dir_patient / 'processed' / 'tissue_segmentation' / 'tissue_seg.nii.gz'
            original_tissue_segmentation = nib.load(path_original_tissue_segmentation).get_fdata()  # 240, 240, 155
            original_tissue_segmentation = torch.as_tensor(original_tissue_segmentation).float()
            original_growth_model = torch.zeros_like(original_tissue_segmentation)
            original_conditioning = create_conditioning(original_growth_model, original_tissue_segmentation)
            original_conditioning = torch.as_tensor(original_conditioning).float()  # 4, 240, 240, 155

            original_conditionings.append(original_conditioning)

        shutil.rmtree(temp_dir)

        return torch.stack(original_conditionings, dim=0).to(self.device)

    def _histogram_equalization(self, reconstructed_t1, original_t1_voided):

        reconstructed_t1_np = reconstructed_t1.cpu().numpy()  # (B, 1, 240, 240, 155)
        original_t1_voided_np = original_t1_voided.cpu().numpy()  # (B, 1, 240, 240, 155)

        reconstructed_t1_he = []
        for i in range(reconstructed_t1_np.shape[0]):
            threshold = 10
            reconstructed_t1_flat = reconstructed_t1_np[i, 0][reconstructed_t1_np[i, 0] > threshold]
            original_t1_voided_flat = original_t1_voided_np[i, 0][original_t1_voided_np[i, 0] > threshold]

            reconstructed_t1_he_flat = skimage.exposure.match_histograms(
                reconstructed_t1_flat,
                original_t1_voided_flat
            )

            reconstructed_t1_he_matched = reconstructed_t1_np[i, 0].copy()
            mask = reconstructed_t1_np[i, 0] > threshold
            reconstructed_t1_he_matched[mask] = reconstructed_t1_he_flat
            reconstructed_t1_he_matched[~mask] = 0

            reconstructed_t1_he.append(reconstructed_t1_he_matched)

        reconstructed_t1_he = np.stack(reconstructed_t1_he, axis=0)
        reconstructed_t1_he = torch.from_numpy(reconstructed_t1_he).unsqueeze(1).type_as(original_t1_voided)

        return reconstructed_t1_he  # (B, 1, 240, 240, 155)

    def _poisson_blending(self, reconstructed_t1, original_t1_voided, original_mask):
        reconstructed_t1_pb = reconstructed_t1.clone()  # (B, 1, 240, 240, 155)
        for i in range(reconstructed_t1.shape[0]):
            reconstructed_t1_ = reconstructed_t1[i, 0].to('cpu')
            original_t1_voided_ = (reconstructed_t1 * original_mask + original_t1_voided * (1 - original_mask))[i, 0].to('cpu')
            original_mask_ = original_mask[i, 0].to('cpu')
            corner_coord = torch.tensor([0, 0, 0]).to('cpu')
            reconstructed_t1_pb_ = pietorch.blend(
                source = reconstructed_t1_,
                target = original_t1_voided_,
                mask = original_mask_,
                corner_coord = corner_coord,
                mix_gradients = True,
            )
            reconstructed_t1_pb[i, 0] = reconstructed_t1_pb_  
            
        return reconstructed_t1_pb  # (B, 1, 240, 240, 155)

    def _pixel_injection(self, reconstructed_t1, original_t1_voided, original_mask):
        original_mask_np = original_mask.cpu().numpy()
        dilated_mask_np = binary_dilation(original_mask_np, iterations=1)
        dilated_mask = torch.from_numpy(dilated_mask_np).to(original_mask.device).float().clamp(0, 1)

        reconstructed_t1_pi = reconstructed_t1 * dilated_mask + original_t1_voided * (1 - dilated_mask)
        reconstructed_t1_pi[reconstructed_t1_pi < 0.01] = 0

        return reconstructed_t1_pi  # (B, 1, 240, 240, 155)

    def _save_reconstruction(self, reconstructed_t1, patients, masks, affines, identifier, mode):
        # for i in range(reconstructed_t1.shape[0]):
        #     path_temp = Path(self.dir_output_model) / 'temp_autoencoder' / f'{batch["patient"][i]}_{batch["mask"][i]}.nii.gz'
        #     path_temp.parent.mkdir(parents=True, exist_ok=True)
        #     nib.save(nib.Nifti1Image(reconstructed_t1[i, 0].cpu().float().numpy(), batch['affine'][i].cpu().float().numpy()), path_temp)
        
        if mode == 'inference_challenge' and identifier != 'pixel_injection':
            return

        if mode in ['inference', 'inference_conditioning']:
            dir_output = self.dir_output_model / identifier
        elif mode == 'inference_challenge':
            dir_output = self.dir_output_model
        dir_output.mkdir(parents=True, exist_ok=True)

        # Save images
        for i, patient in enumerate(patients):
            if mode in ['inference', 'inference_conditioning']:
                mask = masks[i]
                file_name = f"{patient}_{mask}.nii.gz"
            elif mode == 'inference_challenge':
                file_name = f"{patient}-t1n-inference.nii.gz"
            path_reconstructed_t1 = dir_output / file_name

            reconstructed_t1_ = reconstructed_t1[i, 0].cpu().float().numpy()
            affine = affines[i].cpu().float().numpy()

            nib.save(nib.Nifti1Image(reconstructed_t1_, affine), path_reconstructed_t1)

    def on_validation_epoch_end(self):
        """Save sample images at the end of validation epoch"""

        if self.dir_output_model is None:
            print("No output directory specified for saving sample images.")
            return

        patients = self.validation_outputs_for_saving['patients']
        affines = self.validation_outputs_for_saving['affines']
        normalized_t1 = self.validation_outputs_for_saving['normalized_t1'][:, 0, :, :, :].float().numpy()  # (4, 240, 240, 155)
        reconstructed_t1 = self.validation_outputs_for_saving['reconstructed_t1'][:, 0, :, :, :].float().numpy()  # (4, 240, 240, 155)

        # Create output directory
        output_dir = self.dir_output_model.parent / 'images' / f'epoch_{self.current_epoch+1:04d}'
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, patient in enumerate(patients):
            # Save as NIfTI files
            affine = affines[i].cpu().float().numpy()
            normalized_t1_nii = nib.Nifti1Image(normalized_t1[i], affine)
            reconstructed_t1_nii = nib.Nifti1Image(reconstructed_t1[i], affine)

            nib.save(normalized_t1_nii, output_dir / f"{patient}_normalized_t1.nii.gz")
            nib.save(reconstructed_t1_nii, output_dir / f"{patient}_reconstructed_t1.nii.gz")
                
    def configure_optimizers(self):
        """Configure optimizer"""

        unet_params = list(self.unet.parameters())
        controlnet_params = list(self.controlnet.parameters())
        all_params = unet_params + controlnet_params

        optimizer = torch.optim.AdamW(
            all_params,
            lr=self.hparams.learning_rate,
        )
        
        return optimizer

        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer,
        #     T_max=self.trainer.max_epochs,
        #     eta_min=1e-6
        # )
        
        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": {
        #         "scheduler": scheduler,
        #         "monitor": "val_loss"
        #     }
        # }