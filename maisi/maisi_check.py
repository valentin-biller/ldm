import json
import torch
import pickle
import numpy as np
import nibabel as nib
from tqdm import tqdm
from pathlib import Path
from monai import transforms
from maisi_autoencoder import MaisiAutoencoder
from torchmetrics.image import PeakSignalNoiseRatio



dir_data = Path('/vol/miltank/users/bilv/data')
path_autoencoder = "/vol/miltank/users/bilv/ldm/maisi/maisi_vae.pt"
# path_autoencoder = "/vol/miltank/users/bilv/ldm/maisi/maisi_vae_finetuned.pt"

path_ae_latent = Path("/vol/miltank/users/bilv/ldm/maisi/ae_latent.pkl")
path_ae_latent_patients = Path("/vol/miltank/users/bilv/ldm/maisi/ae_latent_patients.pkl")

domain = 'modality'  # 'modality' or 'condition'  # TODO
mode = ['psnr']  # 'ae_latent' or 'psnr'  # TODO

shape_pad = (256, 256, 160)
intensity = transforms.ScaleIntensity(minv=0.0, maxv=1.0)
autoencoder_pad = transforms.SpatialPad(spatial_size=shape_pad)
autoencoder_crop = transforms.CenterSpatialCrop(roi_size=(240, 240, 155))


MODALITIES = ['t1', 't1c', 't2', 'flair'] if domain == 'modality' else ['growth_model', 'tissue_segmentation']


def psnr(reconstructed, original):
    psnr = PeakSignalNoiseRatio()
    return psnr(preds=reconstructed, target=original)


# latent stats
with path_ae_latent.open("rb") as f:
    ae_latent = pickle.load(f)
print(json.dumps(ae_latent, indent=4))


# autoencoder
device = torch.device("cuda")
autoencoder = MaisiAutoencoder(path_autoencoder=str(path_autoencoder), device=device)
def _get_encoded(autoencoder, autoencoder_modality):
    latent_modality = autoencoder.encode(autoencoder_modality)  # (B, 4, 64, 64, 40)
    return latent_modality
def _get_decoded(autoencoder, latent_modality):
    reconstructed_modality = autoencoder.decode(latent_modality).squeeze(0)
    reconstructed_modality = torch.clamp(reconstructed_modality, 0.0, 1.0)  # B, 256, 256, 160
    reconstructed_modality = autoencoder_crop(reconstructed_modality)  # B, 240, 240, 155
    return reconstructed_modality


# data
def _get_data(file_path):
    img = nib.load(file_path)
    return torch.as_tensor(img.get_fdata())
def _process_normalized_modality(data):
    normalized_modality = intensity(data)
    return torch.as_tensor(normalized_modality).unsqueeze(0)  # 1, 240, 240, 155
def _process_autoencoder_modality(normalized_modality):
    autoencoder_modality = autoencoder_pad(normalized_modality)
    return torch.as_tensor(autoencoder_modality)  # 1, 240, 240, 160
def _get_file_growth_model(patient):
        return dir_data / patient / 'processed' / 'growth_model.nii.gz'
def _get_file_tissue_segmentation(patient):
    return dir_data / patient / 'processed' / 'tissue_segmentation.nii.gz'
def create_conditioning(growth_model, tissue_segmentation, threshold=0.001):
    """
    Create 4-channel conditioning:
    - Channel 0: Growth model (threshold)
    - Channels 1-3: Tissue segmentation one-hot
    """
    # Apply threshold: values < threshold → 0, values >= threshold → keep
    growth_model = np.where(growth_model >= threshold, growth_model, 0.0)
    # Create one-hot encoding for tissue segmentation
    tissue_segmentation_1 = np.where(tissue_segmentation == 1, 1.0, 0.0)  # Tissue type 1
    tissue_segmentation_2 = np.where(tissue_segmentation == 2, 1.0, 0.0)  # Tissue type 2  
    tissue_segmentation_3 = np.where(tissue_segmentation == 3, 1.0, 0.0)  # Tissue type 3

    conditioning = np.stack([growth_model, tissue_segmentation_1, tissue_segmentation_2, tissue_segmentation_3], axis=0)  # 4, 240, 240, 155
    conditioning = torch.as_tensor(conditioning)

    return conditioning
def _process_interpolation(original, size=(64, 64, 40), mode='nearest'):
    latent = torch.nn.functional.interpolate(
        original.unsqueeze(0),
        size=size,
        mode=mode
    )[0]
    return latent

count_patients = 0

count_psnr_below_threshold = 0
psnr_normalized_total = []

if path_ae_latent_patients.exists():
    with path_ae_latent_patients.open("rb") as f:
        ae_latent_patients = pickle.load(f)
else:
    ae_latent_patients = {stat: {modality: {} for modality in MODALITIES} for stat in ('mean', 'std', 'min', 'max')}


for folder in tqdm(sorted(dir_data.iterdir())):
    if not folder.is_dir():
        continue
    count_patients += 1
    patient = folder.name

    path_autoencoder_conditioning = dir_data / patient / 'latents_64_64_40' / '256_conditioning.pt'
    path_latent_conditioning = dir_data / patient / 'latents_64_64_40' / '64_conditioning.pt'
    if not (path_autoencoder_conditioning.exists() and path_latent_conditioning.exists()):
        data_growth_model = _get_data(_get_file_growth_model(patient))  # 240, 240, 155
        data_tissue_segmentation = _get_data(_get_file_tissue_segmentation(patient))  # 240, 240, 155
        
        original_conditioning = create_conditioning(data_growth_model, data_tissue_segmentation)  # 4, 240, 240, 155
        assert original_conditioning.shape == (4, 240, 240, 155), f"Expected shape (4, 240, 240, 155), got {original_conditioning.shape}"
        
        autoencoder_conditioning = _process_autoencoder_modality(original_conditioning)  # 4, 256, 256, 160
        assert autoencoder_conditioning.shape == (4, shape_pad[0], shape_pad[1], shape_pad[2]), f"Expected shape (4, {shape_pad[0]}, {shape_pad[1]}, {shape_pad[2]}), got {autoencoder_conditioning.shape}"
        
        data_growth_model_latent = _process_interpolation(data_growth_model.unsqueeze(0), mode='trilinear')[0]  # 64, 64, 40
        data_tissue_segmentation_latent = _process_interpolation(data_tissue_segmentation.unsqueeze(0), mode='nearest')[0]  # 64, 64, 40
        latent_conditioning = create_conditioning(data_growth_model_latent, data_tissue_segmentation_latent)  # 4, 64, 64, 40
        assert latent_conditioning.shape == (4, 64, 64, 40), f"Expected shape (4, 64, 64, 40), got {latent_conditioning.shape}"

        path_autoencoder_conditioning.parent.mkdir(parents=True, exist_ok=True)
        path_latent_conditioning.parent.mkdir(parents=True, exist_ok=True)
        torch.save(autoencoder_conditioning, path_autoencoder_conditioning)
        torch.save(latent_conditioning, path_latent_conditioning)

    for modality in MODALITIES:
        path_latent_modality = dir_data / patient / 'latents_64_64_40' / f'latent_{modality}.pt'
        if 'ae_latent' in mode and patient in ae_latent_patients['mean'][modality].keys() and path_latent_modality.exists():
            continue
        
        if domain == 'modality':
            path_file = f'{modality}.nii.gz'
        elif domain == 'condition':
            path_file = f'processed/{modality}.nii.gz'

        data_modality = _get_data(folder / path_file)  # 240, 240, 155

        normalized_modality = _process_normalized_modality(data_modality)  # 1, 240, 240, 155
        assert normalized_modality.min() == 0.0 and normalized_modality.max() == 1.0, "Intensity values should be in the range [0, 1]"

        autoencoder_modality = _process_autoencoder_modality(normalized_modality).unsqueeze(0).to("cuda")  # 1, 1, 256, 256, 256
        assert autoencoder_modality.shape == (1, 1, shape_pad[0], shape_pad[1], shape_pad[2]), f"Expected shape (1, 1, {shape_pad[0]}, {shape_pad[1]}, {shape_pad[2]}), got {autoencoder_modality.shape}"
        assert autoencoder_modality.min() == 0.0 and autoencoder_modality.max() == 1.0, "Intensity values should be in the range [0, 1]"

        latent_modality = _get_encoded(autoencoder, autoencoder_modality)  # B, 4, 64, 64, 40
        # torch.save(latent_modality, path_latent_modality)  # TODO
       
        if 'ae_latent' in mode:
            latent_temp = latent_modality.cpu()[0]  # 4, 64, 64, 40
            ae_latent_patients['mean'][modality][patient] = float(latent_temp.mean().item())
            ae_latent_patients['std'][modality][patient] = float(latent_temp.std().item())
            ae_latent_patients['min'][modality][patient] = float(latent_temp.min().item())
            ae_latent_patients['max'][modality][patient] = float(latent_temp.max().item())
            with path_ae_latent_patients.open("wb") as f:
                pickle.dump(ae_latent_patients, f, protocol=pickle.HIGHEST_PROTOCOL)
            
        if 'psnr' in mode:
            # nib.save(nib.Nifti1Image(latent_modality[0][0].cpu().float().numpy(), np.eye(4)), f'/vol/miltank/users/bilv/ldm/maisi/temp/{patient}_{modality}_latent.nii.gz')  # TODO

            reconstructed_modality = _get_decoded(autoencoder, latent_modality)  # B, 240, 240, 155
            assert reconstructed_modality.min() >= 0.0 and reconstructed_modality.max() <= 1.0, "Intensity values should be in the range [0, 1]"

            # nib.save(nib.Nifti1Image(reconstructed_modality.squeeze(0).cpu().float().numpy(), np.eye(4)), f'/vol/miltank/users/bilv/ldm/maisi/temp/{patient}_{modality}_reconstructed.nii.gz')  # TODO
            # nib.save(nib.Nifti1Image(normalized_modality.squeeze(0).cpu().float().numpy(), np.eye(4)), f'/vol/miltank/users/bilv/ldm/maisi/temp/{patient}_{modality}_normalized.nii.gz')  # TODO

            assert reconstructed_modality.shape == normalized_modality.shape
            psnr_normalized = psnr(reconstructed_modality.to('cpu'), torch.as_tensor(normalized_modality).to('cpu'))
            tqdm.write(f'PSNR Normalized: {psnr_normalized}')

            if psnr_normalized < 30:
                count_psnr_below_threshold += 1
            tqdm.write(f'PSNR too low for {count_psnr_below_threshold} images')

            psnr_normalized_total.append(psnr_normalized.item())
            tqdm.write(f'PSNR Normalized Total: {sum(psnr_normalized_total) / len(psnr_normalized_total)}')

            # dir_temp = Path(f'/vol/miltank/users/bilv/ldm/maisi/output/{patient}')
            # dir_temp.mkdir(parents=True, exist_ok=True)
            # nib.save(nib.Nifti1Image(original.float().numpy(), np.eye(4)), dir_temp / 't1_original.nii.gz')
            # nib.save(nib.Nifti1Image(reconstruction.float().numpy(), np.eye(4)), dir_temp / 't1_reconstruction.nii.gz')

print(f'Total patients processed: {count_patients} (should be 3801)')

if 'ae_latent' in mode:
    mean_, std_, min_, max_ = {}, {}, {}, {}

    for modality in MODALITIES:
        modality_means = np.array(list(ae_latent_patients['mean'][modality].values()), dtype=np.float64)
        modality_stds = np.array(list(ae_latent_patients['std'][modality].values()), dtype=np.float64)
        modality_mins = np.array(list(ae_latent_patients['min'][modality].values()), dtype=np.float64)
        modality_maxs = np.array(list(ae_latent_patients['max'][modality].values()), dtype=np.float64)

        modality_mean = float(modality_means.mean())

        temp_ex2 = float(np.mean(modality_stds**2 + modality_means**2))
        temp_var = max(temp_ex2 - modality_mean * modality_mean, 1e-12)

        mean_[modality] = modality_mean
        std_[modality] = temp_var**0.5
        min_[modality] = float(modality_mins.mean())
        max_[modality] = float(modality_maxs.mean())

    ae_latent = {
        'mean': mean_,
        'std': std_,
        'min': min_,
        'max': max_,
    }
    print(json.dumps(ae_latent, indent=4))

    path_ae_latent.parent.mkdir(parents=True, exist_ok=True)
    with path_ae_latent.open("wb") as f:
        pickle.dump(ae_latent, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved latent stats to: {path_ae_latent}")