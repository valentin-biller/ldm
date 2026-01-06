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

import pickle
from pathlib import Path
from scipy.ndimage import binary_dilation

from monai import transforms
from flow_matching.solver import ODESolver

import helpers
import schedulers
import inpainting_repaint
import inpainting_postprocessing
from model_unet import UNet
from model_dit import DiT_Custom
from maisi_autoencoder import MaisiAutoencoder
from maisi_f8_autoencoder import MaisiF8Autoencoder
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
        mask_conditioning=64,  # 64, 32, scalar, None
        modality_conditioning=True,  # True, False
        denoising='own', # own, repaint
        scheduler_='ddpm',  # ddpm, iddpm, flow_matching
        latent_shape=None,
        learning_rate=1e-4,
        num_train_timesteps=1000,  # 4000 for iddpm (fixed in schedulers.py)
        num_inference_steps=100,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.modality_class_mapping = {'t1': 0, 't1c': 1, 't2': 2, 'flair': 3}

        self.debugging = False

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
            if self.num_inference_steps <= 100:
                print('========== USING DDIM FOR INFERENCE =============')
                self.scheduler_inference = self._scheduler_.scheduler_inference
            else:
                print('========== USING DDPM FOR INFERENCE =============')
                self.scheduler_inference = self.scheduler_training
        elif self.scheduler_ == 'iddpm':
            self.scheduler = self._scheduler_.scheduler
            self.diffusion = self._scheduler_.diffusion
        elif self.scheduler_ == 'flow_matching':
            self.path = self._scheduler_.path
            self.solver_config = self._scheduler_.solver_config

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

        # Experiment 'spatio_temporal'
        self.spatio_temporal_previous = None

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
    
    def _predict_noise(self, sample, timesteps, modality=None, conditioning=None, latent_modality=None, latent_mask=None, noise_gt=None):
        """
        Modular function to get noise prediction from (ControlNet +) UNet
        """
        if self.scheduler_ == 'flow_matching':
            batch_size = sample.shape[0]
            # timesteps
            timesteps_max = 1000
            timesteps_inpainting =  timesteps.clone()
            timesteps = timesteps * (timesteps_max - 1)
            timesteps = timesteps.floor().long()
            if timesteps.dim() == 0:
                timesteps = timesteps.expand(batch_size)

            # inpainting
            if latent_modality is not None and latent_mask is not None:
                noisy_gt = (1 - timesteps_inpainting) * noise_gt + timesteps_inpainting * latent_modality  # could have also used path.sample
                sample = (sample * latent_mask + noisy_gt * (1 - latent_mask)).float()  # latent_mask = dilated_mask     
                
        ##################################################

        modality = torch.tensor([self.modality_class_mapping[m] for m in modality], device=self.device, dtype=torch.long)

        noise_pred = self.model.forward(sample, timesteps, modality, conditioning)

        if self.scheduler_ == 'ddpm':
            _debugging_noise_pred = noise_pred
        elif self.scheduler_ == 'iddpm':
            _debugging_noise_pred = noise_pred[:, :4]  # only epsilon head
        elif self.scheduler_ == 'flow_matching':
            _debugging_noise_pred = noise_pred

        # Debugging
        if self.training:
            helpers._debugging(self, _debugging_noise_pred, 'train/noise_pred', logging_=True, distribution_=True)
        else:
            t = timesteps[0].item()
            if t % 100 == 0 or t <= 10:
                helpers._debugging(self, _debugging_noise_pred, f'denoising/noise_pred/t_{t}', logging_=True, distribution_=True)

        return noise_pred

    def _generate_denoising(self, patients, modality, conditioning, affines, latent_modality=None, latent_mask=None, spatio_temporal=None):
        helpers._swap_ema(self, apply_ema=True)
        
        batch_size = len(modality)

        # Initialize with pure noise
        sample = torch.randn(
            (batch_size, *self.latent_shape),
            device=self.device
        )

        # inpainting
        if latent_modality is not None and latent_mask is not None:
            latent_mask_np = latent_mask.cpu().numpy()
            dilated_mask_np = binary_dilation(latent_mask_np, iterations=1)
            dilated_mask = torch.from_numpy(dilated_mask_np).to(latent_mask.device).float().clamp(0, 1)
            noise_gt = torch.randn(
                (batch_size, *self.latent_shape),
                device=self.device
            )
        else:
            dilated_mask = None
            noise_gt = None

        # spatio_temporal
        if spatio_temporal is not None:
            spatio_temporal_time_ddpm = 375
            spatio_temporal_time_flow_matching = 375  # TODO
            if self.spatio_temporal_previous is None:
                temp = spatio_temporal.to(self.device)
            else:
                temp = self.spatio_temporal_previous.to(self.device)
            assert temp.shape == sample.shape

            if self.scheduler_ == 'ddpm':
                sample = self.scheduler_training.add_noise(temp, sample, torch.tensor([spatio_temporal_time_ddpm], device=self.device))
            elif self.scheduler_ == 'flow_matching':
                t_start = 1.0 - (spatio_temporal_time_flow_matching / 1000)
                t_start_timesteps = torch.full((batch_size,), t_start, device=self.device)

                sample_info = self.path.sample(t=t_start_timesteps, x_0=sample, x_1=temp)
                sample = sample_info.x_t
                t_start_timesteps = sample_info.t

        if self.scheduler_ in ['ddpm', 'iddpm']:
            # Denoising loop
            if spatio_temporal is not None:
                num_steps = spatio_temporal_time_ddpm
            else:
                num_steps = self.num_inference_steps

            for t in list(range(num_steps))[::-1]:
                t = torch.tensor([t], device=self.device, dtype=torch.long)
                
                # inpainting
                if latent_modality is not None and latent_mask is not None:
                    # Pixel injection: preserve known regions (add noise to ground truth based on timestep)
                    noisy_gt = self.scheduler_training.add_noise(latent_modality, noise_gt, t)
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
                    if self.debugging and self.dir_output_model is not None:
                        generate_denoising_outputs = {
                            'timestep': t,
                            'patients': patients,
                            'affines': affines,
                            'modality': modality,
                            'pred_prev_sample': self._get_decoded(pred_prev_sample),
                            'pred_original_sample': self._get_decoded(pred_original_sample),
                        }
                        helpers._save_generate_denoising_outputs(self, generate_denoising_outputs)

        elif self.scheduler_ == 'flow_matching':
            if spatio_temporal is not None:
                t_start = t_start
            else:
                t_start = 0.0

            solver = ODESolver(
                velocity_model=lambda x, t, **kwargs: 
                self._predict_noise(x, t, **kwargs)
            )

            time_grid = torch.linspace(t_start, 1, self.solver_config["time_points"], device=self.device)

            sample = solver.sample(
                x_init=sample,
                time_grid=time_grid,
                method=self.solver_config["method"],
                step_size=self.solver_config["step_size"],
                return_intermediates=False,
                modality=modality,
                conditioning=conditioning,
                latent_modality = latent_modality,
                latent_mask = dilated_mask,
                noise_gt = noise_gt,
            )

        if latent_modality is not None and latent_mask is not None:
            sample = sample * dilated_mask + latent_modality * (1 - dilated_mask)

        if spatio_temporal is not None:
            self.spatio_temporal_previous = sample

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
            # TODO clamping?
            # if self.latent_shape == (4, 64, 64, 40):
            #     reconstructed_modality_i = torch.clamp(reconstructed_modality_i, 0.0, 1.0)  # 1, 256, 256, 160
            # elif self.latent_shape == (4, 32, 32, 20):
            #     reconstructed_modality_i = torch.clamp(reconstructed_modality_i, -1.0, 1.0)  # 1, 256, 256, 160 
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

        elif self.scheduler_ == 'flow_matching':
            noise = torch.randn_like(latent_modality)
            timesteps = torch.rand((batch_size,), device=self.device)

            sample_info = self.path.sample(t=timesteps, x_0=noise, x_1=latent_modality)

            velocity_pred = self._predict_noise(
                sample=sample_info.x_t,
                timesteps=sample_info.t,
                modality=modality,
                conditioning=conditioning,
            )
            loss = torch.nn.functional.mse_loss(velocity_pred, sample_info.dx_t)

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

    def test_step(self, batch, batch_idx, spatio_temporal=None, inpainting_output=None):
        """Test step"""
        mode = batch['mode'][0]
        patients = batch['patient']  # (B,)

        affines = batch['affine']  # (B,)
        
        modality = batch['modality']  # (B,)
        conditioning = batch['conditioning']  # (B, 8, 64, 64, 40) or (B, 8, 32, 32, 20)

        if mode.startswith('inpainting'):
            normalized_modality = batch['normalized_modality']  # (B, 1, 240, 240, 155)
            latent_modality = batch['latent_modality']  # (B, 4, 64, 64, 40) or (B, 4, 32, 32, 20)

            masks = batch['mask']  # (B,)
            original_mask = batch['original_mask']  # (B, 1, 240, 240, 155)
            latent_mask = batch['latent_mask']  # (B, 1, 64, 64, 40) or (B, 1, 32, 32, 20)

            voided_modality = torch.zeros_like(normalized_modality)
            voided_modality = voided_modality * original_mask + normalized_modality * (1 - original_mask)

            step = batch['step']

        ########## Conditioning (& Masks) ##########
        # Experiments 'spatio_temporal' / 'spatio_temporal_scalar' / 'inpainting_spatio_temporal' (latents have to be generated)
        if conditioning.shape[1] == 2:
            # batch size
            batch_size = conditioning.shape[0]
            assert batch_size == 1

            # masks for 'inpainting_spatio_temporal'
            if mode == 'inpainting_spatio_temporal':
                growth_model = self.autoencoder_crop(conditioning[:, 0, ...]).unsqueeze(1)  # (B, 1, 240, 240, 155)
                tissue_segmentation = self.autoencoder_crop(conditioning[:, 1, ...]).unsqueeze(1)  # (B, 1, 240, 240, 155)
                step = (step, growth_model, tissue_segmentation)
                original_mask, latent_mask = helpers._get_inpainting_masks(self, growth_model, original_mask)

            # latent for tissue segmentation
            latent_tissue_segmentation = self._get_encoded(conditioning[:, 1, ...].unsqueeze(1))            
            
            # latent for growth model
            scalar = np.unique(conditioning[:, 0, ...].cpu().numpy())
            if len(scalar) != 1:  # experiment 'spatio_temporal' / 'inpainting_spatio_temporal'
                latent_growth_model = self._get_encoded(conditioning[:, 0, ...].unsqueeze(1))
            else:  # experiment 'spatio_temporal_scalar'
                latent_growth_model = torch.full_like(latent_tissue_segmentation, float(scalar))

            conditioning = torch.cat([latent_growth_model, latent_tissue_segmentation], dim=1)

        ########## Inference ##########
        with torch.no_grad():
            # Generation
            if mode == 'validation':
                denoised = self._generate_denoising( 
                    patients,
                    modality,
                    conditioning,
                    affines,
                    spatio_temporal=spatio_temporal
                ).float()  # (B, 4, 64, 64, 40)
                reconstructed = self._get_decoded(denoised)  # (B, 1, 240, 240, 155)

                # TODO post-processing?
                reconstructed[reconstructed < 0.001] = 0
                reconstructed = transforms.ScaleIntensity(minv=0.0, maxv=1.0)(reconstructed)

                return reconstructed

            # Inpainting
            elif mode in ['inpainting_healthy_tissue', 'inpainting_tumorous_tissue', 'inpainting_spatio_temporal']:
                
                if self.denoising == 'own':
                    inpainted = self._generate_denoising(
                        patients,
                        modality,
                        conditioning,
                        affines,
                        latent_modality=latent_modality,
                        latent_mask=latent_mask,
                        spatio_temporal=spatio_temporal,
                    ).float()
                
                elif self.denoising == 'repaint':
                    inpainting_repaint._repaint_init(self)
                    inpainted = inpainting_repaint._repaint_start(
                        self,
                        patients,
                        modality,
                        conditioning,
                        affines,
                        latent_modality=latent_modality,
                        latent_mask=latent_mask,
                        spatio_temporal=spatio_temporal,
                    ).float()

                reconstructed = self._get_decoded(inpainted)  # (B, 1, 240, 240, 155)
                
                # original_min, original_max = voided_modality.min(), voided_modality.max()
                # reconstructed = reconstructed * (original_max - original_min) + original_min
                helpers._save_inpainting(self, inpainting_output, mode, 'inpainted', conditioning, original_mask, normalized_modality, reconstructed, affines, patients, modality, masks, step=step)

                ########## Histogram Equalization OR Poisson Blending

                reconstructed_he = inpainting_postprocessing._histogram_equalization(self, reconstructed, voided_modality)  # (B, 1, 240, 240, 155)
                helpers._save_inpainting(self, inpainting_output, mode, 'histogram_equalization', conditioning, original_mask, normalized_modality, reconstructed_he, affines, patients, modality, masks, step=step)

                reconstructed_pb = inpainting_postprocessing._poisson_blending(self, reconstructed, voided_modality, original_mask)  # (B, 1, 240, 240, 155)
                helpers._save_inpainting(self, inpainting_output, mode, 'poisson_blending', conditioning, original_mask, normalized_modality, reconstructed_pb, affines, patients, modality, masks, step=step)

                ########## Pixel Injection (on reconstructed_pb)
                
                reconstructed_pi = inpainting_postprocessing._pixel_injection(self, reconstructed_pb, voided_modality, original_mask)  # (B, 1, 240, 240, 155)
                helpers._save_inpainting(self, inpainting_output, mode, 'pixel_injection', conditioning, original_mask, normalized_modality, reconstructed_pi, affines, patients, modality, masks, step=step)

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