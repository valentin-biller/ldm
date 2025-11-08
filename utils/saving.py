import time
import nibabel as nib
from tqdm import tqdm


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


def _save_reconstruction(self, reconstructed_t1, patients, masks, affines, identifier, mode):
    if mode == 'inpainting_inference_challenge' and identifier != 'pixel_injection':
        return

    if mode in ['inpainting_inference', 'inpainting_inference_conditioning']:
        dir_output = self.dir_output_model / identifier
    elif mode == 'inpainting_inference_challenge':
        dir_output = self.dir_output_model
    dir_output.mkdir(parents=True, exist_ok=True)

    # Save images
    for i, patient in enumerate(patients):
        if mode in ['inpainting_inference', 'inpainting_inference_conditioning']:
            mask = masks[i]
            file_name = f"{patient}_{mask}.nii.gz"
        elif mode == 'inpainting_inference_challenge':
            file_name = f"{patient}-t1n-inference.nii.gz"
        path_reconstructed_t1 = dir_output / file_name

        reconstructed_t1_ = reconstructed_t1[i, 0].cpu().float().numpy()
        affine = affines[i].cpu().float().numpy()

        _nib_save(nib.Nifti1Image(reconstructed_t1_, affine), path_reconstructed_t1)