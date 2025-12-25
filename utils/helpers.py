import time
import torch
import numpy as np
import nibabel as nib
from tqdm import tqdm
from scipy.ndimage import binary_dilation

# --------- Gradients ---------  (doesn't slow down the training too much)
def _gradients_compute_norm(parameters):
    sqsum = 0.0
    for p in parameters:
        if p.grad is None:
            continue
        sqsum += (p.grad ** 2).sum().item()
    return np.sqrt(sqsum)

# --------- Distribution Shift ---------
def _distribution_shift(z, ae_latent_mean, ae_latent_std):
    batch_mean = z.mean()
    batch_std = z.std(unbiased=False)
    z = (z - batch_mean) / (batch_std + 1e-6)
    z = z * ae_latent_std + ae_latent_mean
    return z

# --------- Debugging ---------
@torch.no_grad()
def _debugging(self, tensor, tag, print_=False, logging_=False, distribution_=False, target_mean=0.0, target_std=1.0, target_tolerance=0.25):  # print_: False, 'scientific', 'float'
    if not self.debugging:
        return
    
    tensor = tensor.detach().float()
    std, mean = torch.std_mean(tensor, unbiased=False)
    std, mean = std.item(), mean.item()
    if print_ is not False:
        var = std.pow(2).item()
        min, max = torch.aminmax(tensor)

    if print_ == 'scientific':
        msg = f'[DEBUGGING] {tag}: mean={mean:.4e}, std={std:.4e}, var={var:.4e}, min={min:.4e}, max={max:.4e}'
    elif print_ == 'float':
        msg = f'[DEBUGGING] {tag}: mean={mean:.2f}, std={std:.2f}, var={var:.2f}, min={min:.2f}, max={max:.2f}'
    if print_ is not False:    
        tqdm.write(msg)

    if logging_:
        self.log(f'{tag}_mean', mean, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)
        self.log(f'{tag}_std', std, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)

    if distribution_:
        if not (abs(mean - target_mean) < target_tolerance and abs(std - target_std) < target_tolerance):
            tqdm.write(f'[DEBUGGING] {tag}: Distribution Warning (mean≈{mean:.4f}, std≈{std:.4f})')

    return  # {'mean': mean, 'std': std, 'var': var, 'min': min, 'max': max}

# --------- EMA ---------
def _load_ema(self, decay=0.999):
    self.model_params = [p for group in self.optimizer.param_groups for p in group['params']]
    path_ema = self.dir_ema / "ema.pt" if self.dir_ema is not None else None
    if path_ema is not None and path_ema.exists():
        ema = torch.load(path_ema, map_location="cpu")
        self.ema_decay = ema['decay']
        self.ema_params = [ema_param.to(self.device) for ema_param in ema['params']]
    else:
        self.ema_decay = decay
        self.ema_params = [p.detach().clone() for p in self.model_params]
    
    self.ema_params_backup = None

def _save_ema(self):
    if self.dir_output_model is not None:
        path_ema = self.dir_output_model / "ema.pt"
        path_ema.parent.mkdir(parents=True, exist_ok=True)
        ema = {
            'decay': self.ema_decay,
            'params': [ema_param.detach().cpu() for ema_param in self.ema_params],
        }
        torch.save(ema, path_ema)

def _update_ema(self):
    with torch.no_grad():
        for ema_p, model_p in zip(self.ema_params, self.model_params):
            ema_p.mul_(self.ema_decay).add_(model_p, alpha=1 - self.ema_decay)

def _swap_ema(self, apply_ema=True):
    if apply_ema:
        # Backup original model weights and apply EMA weights
        self.ema_params_backup = [p.detach().clone() for p in self.model_params]
        for model_p, ema_p in zip(self.model_params, self.ema_params):
            model_p.data.copy_(ema_p.data)
    else:
        # Restore original weights from backup
        for model_p, backup_p in zip(self.model_params, self.ema_params_backup):
            model_p.data.copy_(backup_p.data)
        self.ema_params_backup = None

# --------- Saving ---------
def _nib_save(img, file_path):
    retries = 10
    delay = 3
    exception = None
    for i in range(retries):
        try:
            nib.save(img, file_path)
            return
        except Exception as e:
            exception = e
            tqdm.write(f"[WARNING]: {e}. Retrying ({i+1}/{retries}) in {delay}s...")
            time.sleep(delay)
    tqdm.write(f"[WARNING] nib.save failed after {retries} retries for {file_path}. Skipping.")
    return
    # raise exception

def _save_generate_denoising_outputs(self, generate_denoising_outputs):
    
    if self.dir_output_model is None:
        print("No output directory specified for saving sample images.")
        return

    # Create output directory
    dir_output = self.dir_output_model.parent / 'images' / f'epoch_{self.current_epoch+1:04d}' / 'denoising'

    timestep = generate_denoising_outputs['timestep'].cpu().item()
    patients = generate_denoising_outputs['patients']
    affines = generate_denoising_outputs['affines'].cpu().float().numpy()
    modality = generate_denoising_outputs['modality']
    pred_prev_sample = generate_denoising_outputs['pred_prev_sample'].cpu().float().numpy()  # (B, 1, 240, 240, 155)
    pred_original_sample = generate_denoising_outputs['pred_original_sample'].cpu().float().numpy()  # (B, 1, 240, 240, 155)

    # Save as NIfTI files
    for i, (patient, modality_) in enumerate(zip(patients, modality)):
        dir_patient_modality = dir_output / patient / modality_
        dir_patient_modality.mkdir(parents=True, exist_ok=True)

        affine = affines[i]

        pred_prev_sample_patient = pred_prev_sample[i][0]  # (240, 240, 155)
        pred_original_sample_patient = pred_original_sample[i][0]  # (240, 240, 155)

        pred_prev_sample_nii = nib.Nifti1Image(pred_prev_sample_patient, affine)
        pred_original_sample_nii = nib.Nifti1Image(pred_original_sample_patient, affine)

        _nib_save(pred_prev_sample_nii, dir_patient_modality / f"pred_prev_sample_{timestep}.nii.gz")
        _nib_save(pred_original_sample_nii, dir_patient_modality / f"pred_original_sample_{timestep}.nii.gz")

def _save_validation_step_outputs(self, validation_step_outputs):

    if self.dir_output_model is None:
        print("No output directory specified for saving sample images.")
        return

    # Create output directory
    dir_output = self.dir_output_model.parent / 'images' / f'epoch_{self.current_epoch+1:04d}' / 'validation'

    patients = validation_step_outputs['patients']
    affines = validation_step_outputs['affines'].cpu().float().numpy()
    modality = validation_step_outputs['modality']
    normalized = validation_step_outputs['normalized'].cpu().float().numpy()  # (B, 1, 240, 240, 155)
    reconstructed = validation_step_outputs['reconstructed'].cpu().float().numpy()  # (B, 1, 240, 240, 155)

    # Save as NIfTI files
    for i, (patient, modality_) in enumerate(zip(patients, modality)):
        dir_patient = dir_output / patient
        dir_patient.mkdir(parents=True, exist_ok=True)

        affine = affines[i]
        
        normalized_modality = normalized[i][0]  # (240, 240, 155)
        reconstructed_modality = reconstructed[i][0]  # (240, 240, 155)

        normalized_modality_nii = nib.Nifti1Image(normalized_modality, affine)
        reconstructed_modality_nii = nib.Nifti1Image(reconstructed_modality, affine)

        _nib_save(normalized_modality_nii, dir_patient / f"normalized_{modality_}.nii.gz")
        _nib_save(reconstructed_modality_nii, dir_patient / f"reconstructed_{modality_}.nii.gz")

def _save_inpainting(self, inpainting_output, mode, identfier, conditioning, original_mask, normalized_modality, reconstructed, affines, patients, modality, masks, step=None):
    if step is not None:
        step, growth_model, tissue_segmentation = step
    
    dir_output = inpainting_output / 'inpainting' / mode
    for i, patient in enumerate(patients):
        dir_output_patient = dir_output / patient
        dir_output_patient.mkdir(parents=True, exist_ok=True)

        if step is not None:
            dir_output_patient = dir_output_patient / f"{step:.2f}"
            dir_output_patient.mkdir(parents=True, exist_ok=True)
            growth_model_ = growth_model[i].cpu().float().numpy()  # (1, 240, 240, 155)
            tissue_segmentation_ = tissue_segmentation[i].cpu().float().numpy()  # (1, 240, 240, 155)

        conditioning_ = conditioning[i].cpu().float().numpy()
        original_mask_ = original_mask[i].cpu().float().numpy()
        normalized_modality_ = normalized_modality[i].cpu().float().numpy()
        reconstructed_ = reconstructed[i].cpu().float().numpy()
        affine = affines[i].cpu().float().numpy()

        if identfier == 'inpainted':
            # conditioning
            dir_conditioning = dir_output_patient / 'conditioning'
            dir_conditioning.mkdir(exist_ok=True, parents=True)
            if step is None:
                for num, condition in enumerate(['growth_model', 'tissue_segmentation']):
                    for layer in range(4):
                        # print('Saving layer', num*4 + layer)
                        layer_ = conditioning_[num*4 + layer]
                        layer_nii = nib.Nifti1Image(layer_, affine)
                        nib.save(layer_nii, dir_conditioning / f"{condition}_{layer}.nii.gz")
            else:
                growth_model_nii = nib.Nifti1Image(growth_model_[0], affine)
                nib.save(growth_model_nii, dir_conditioning / f"growth_model.nii.gz")
                tissue_segmentation_nii = nib.Nifti1Image(tissue_segmentation_[0], affine)
                nib.save(tissue_segmentation_nii, dir_conditioning / f"tissue_segmentation.nii.gz")

            # masks
            dir_masks = dir_output_patient / 'masks'
            dir_masks.mkdir(exist_ok=True, parents=True)

            original_masks_nii = nib.Nifti1Image(original_mask_[0], affine)
            nib.save(original_masks_nii, dir_masks / f"original_mask_{masks[i]}.nii.gz")

            # normalized
            dir_normalized = dir_output_patient / 'normalized'
            dir_normalized.mkdir(exist_ok=True, parents=True)
            
            normalized_nii = nib.Nifti1Image(normalized_modality_[0], affine)
            nib.save(normalized_nii, dir_normalized / f"normalized_{modality[i]}_{masks[i]}.nii.gz")

        # images
        dir_images = dir_output_patient / identfier
        dir_images.mkdir(parents=True, exist_ok=True)

        reconstructed_nii = nib.Nifti1Image(reconstructed_[0], affine)
        nib.save(reconstructed_nii, dir_images / f"reconstructed_{modality[i]}_{masks[i]}.nii.gz")

def _get_inpainting_masks(self, growth_model, threshold=0.001, margin=1):
    # growth_model: (1, 1, 240, 240, 155)
    original_mask = growth_model > threshold
    if margin > 0:
        original_mask = original_mask.cpu().numpy()
        original_mask = binary_dilation(original_mask, iterations=margin)
        original_mask = torch.from_numpy(original_mask).to(self.device).float().clamp(0, 1)

    latent_mask = torch.nn.functional.interpolate(
        original_mask,
        size=(self.latent_shape[1], self.latent_shape[2], self.latent_shape[3]),
        mode='nearest'
    )
    return original_mask, latent_mask