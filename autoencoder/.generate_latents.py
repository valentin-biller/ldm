import sys
from pathlib import Path
dir_current = Path(__file__).resolve().parent
dir_metrics = dir_current.parent.parent / '.metrics'  # image metrics
dir_tumor_growth = dir_current.parent.parent / '.tumor_growth'  # tumor growth
sys.path.append(str(dir_metrics))
sys.path.append(str(dir_tumor_growth))

import json
import shutil
import pickle
import numpy as np
import pandas as pd
import nibabel as nib
from tqdm import tqdm
from pathlib import Path
from monai import transforms
from torchmetrics.image import PeakSignalNoiseRatio

import torch
import torch_fidelity
from torch.utils.data import Dataset

from maisi_autoencoder import MaisiAutoencoder
from maisi_f8_autoencoder import MaisiF8Autoencoder
from f8d16_autoencoder import F8D16Autoencoder

from TumorGrowthToolkit.FK import Solver


dir_data = Path('/vol/miltank/users/bilv/data_lumiere')  # TODO remove lumiere
dir_autoencoder = Path('/vol/miltank/users/bilv/ldm/autoencoder/checkpoints')

dir_temp = dir_current / 'temp' 
dir_reconstructed = dir_temp / 'reconstructed'

path_ae_latent = Path("/vol/miltank/users/bilv/ldm/autoencoder/ae_latent.pkl")
path_ae_latent_patients = Path("/vol/miltank/users/bilv/ldm/autoencoder/ae_latent_patients.pkl")


models = {
    'maisi_autoencoder': (4, 64, 64, 40),
    'maisi_f8_autoencoder': (4, 32, 32, 20),
    'f8d16_autoencoder': (4, 32, 32, 20),
}

model = 'maisi_autoencoder'  # 'maisi_autoencoder', 'maisi_f8_autoencoder' or 'f8d16_autoencoder'
domain = ['condition', 'modality']  # 'modality' and/or 'condition' and/or 'tumor_concentrations'  # TODO
mode = []  # 'ae_latent' and/or 'psnr' and/or 'image_metrics'  # TODO
save_latent_modality = True  # TODO


# initialization
MODALITIES = ['t1', 't1c', 't2', 'flair']

tumor_concentrations = {
    'random_1': [0.35, 0.35, 0.35],
    'random_2': [0.35, 0.65, 0.65],
    'random_3': [0.65, 0.35, 0.65],
    'spatio_temporal': [0.5, 0.5, 0.5],
}

shape_pad = (256, 256, 160)
autoencoder_pad = transforms.SpatialPad(spatial_size=shape_pad)
autoencoder_crop = transforms.CenterSpatialCrop(roi_size=(240, 240, 155))

device = torch.device("cuda")
if model == 'maisi_autoencoder':
    latent_shape = models['maisi_autoencoder']
    intensity = transforms.ScaleIntensity(minv=0.0, maxv=1.0)
    path_autoencoder = dir_autoencoder / 'maisi_vae.pt'
    autoencoder = MaisiAutoencoder(path_autoencoder=str(path_autoencoder), device=device)

    # latent stats
    with path_ae_latent.open("rb") as f:
        ae_latent = pickle.load(f)
    print(json.dumps(ae_latent, indent=4))
    if path_ae_latent_patients.exists():
        with path_ae_latent_patients.open("rb") as f:
            ae_latent_patients = pickle.load(f)
    else:
        ae_latent_patients = {stat: {modality: {} for modality in MODALITIES} for stat in ('mean', 'std', 'min', 'max')}

elif model == 'maisi_f8_autoencoder':
    latent_shape = models['maisi_f8_autoencoder']
    intensity = transforms.ScaleIntensity(minv=0.0, maxv=1.0)
    path_autoencoder = dir_autoencoder / 'maisi_f8_vae.pt'
    autoencoder = MaisiF8Autoencoder(path_autoencoder=str(path_autoencoder), device=device)

    assert 'ae_latent' not in mode, "Latent stats not supported for MAISI F8 autoencoder yet."

elif model == 'f8d16_autoencoder':
    latent_shape = models['f8d16_autoencoder']
    intensity = transforms.ScaleIntensity(minv=-1.0, maxv=1.0)
    path_autoencoder = dir_autoencoder / 'f8d16_vae.pt'
    autoencoder = F8D16Autoencoder(path_autoencoder=str(path_autoencoder), device=device)

    assert 'ae_latent' not in mode, "Latent stats not supported for F8D16 autoencoder yet."

else:
    raise ValueError(f"Unsupported model: {model}")

# psnr
def psnr(reconstructed, original):
    psnr = PeakSignalNoiseRatio()
    return psnr(preds=reconstructed, target=original)

# autoencoder
def _get_encoded(autoencoder, autoencoder_modality):
    latent_modality = autoencoder.encode(autoencoder_modality)  # (B, 4, 64, 64, 40)
    return latent_modality
def _get_decoded(autoencoder, latent_modality):
    reconstructed_modality = autoencoder.decode(latent_modality).squeeze(0)
    if model in ['maisi_autoencoder', 'maisi_f8_autoencoder']:
        reconstructed_modality = torch.clamp(reconstructed_modality, 0.0, 1.0)  # B, 256, 256, 160
    elif model == 'f8d16_autoencoder':
        reconstructed_modality = torch.clamp(reconstructed_modality, -1.0, 1.0)  # 1, 256, 256, 160 
    else:
        raise ValueError("Unsupported model.")
    reconstructed_modality = autoencoder_crop(reconstructed_modality)  # B, 240, 240, 155
    return reconstructed_modality
def _encode(data, normalize=True):
    if normalize:
        normalized = _normalize(data)  # 1, 240, 240, 155
        assert (normalized.min() == 0.0 or normalized.min() == -1.0) and normalized.max() == 1.0, "Intensity values should be in the range [0, 1] or [-1, 1]"
    else:
        normalized = torch.as_tensor(data).unsqueeze(0).float()  # 1, 240, 240, 155

    autoencoder_ = _pad(normalized).unsqueeze(0).to("cuda")  # 1, 1, 256, 256, 256
    assert autoencoder_.shape == (1, 1, shape_pad[0], shape_pad[1], shape_pad[2]), f"Expected shape (1, 1, {shape_pad[0]}, {shape_pad[1]}, {shape_pad[2]}), got {autoencoder_.shape}"
    assert (autoencoder_.min() >= 0.0 or autoencoder_.min() >= -1.0) and autoencoder_.max() <= 1.0, "Intensity values should be in the range [0, 1] or [-1, 1]"

    latent = _get_encoded(autoencoder, autoencoder_)  # B, 4, 64, 64, 40
    return latent, normalized


# loading with nibabel
def _get_data(file_path):
    img = nib.load(file_path)
    return torch.as_tensor(img.get_fdata())
def _get_affine(file_path):
    img = nib.load(file_path)
    return img.affine

# get files default
def _get_file_growth_model(patient):
    return dir_data / patient / 'processed' / 'growth_model.nii.gz'
def _get_file_tissue_segmentation(patient):
    return dir_data / patient / 'processed' / 'tissue_segmentation.nii.gz'

# processing
def _normalize(data):
    normalized = intensity(data)
    return torch.as_tensor(normalized).unsqueeze(0)  # 1, 240, 240, 155
def _pad(data):
    padded = autoencoder_pad(data)
    return torch.as_tensor(padded)  # 1, 256, 256, 160

# patients val
with open('/vol/miltank/users/bilv/ldm/utils/.patients_val.txt', "r") as f:
    patients_val = [line.strip() for line in f if line.strip()]


count_patients = 0

count_psnr_below_threshold = 0
psnr_normalized_total = []

latent_shape_string = f"{latent_shape[1]}_{latent_shape[2]}_{latent_shape[3]}"

pbar = tqdm(sorted(dir_data.iterdir()))
# TODO remove
temp = []
for folder in sorted(dir_data.iterdir()):
    if folder.is_dir() and not folder.name.startswith('.'):
        for sub in folder.iterdir():
            if sub.is_dir():
                temp.append(sub)
pbar = tqdm(sorted(temp, reverse=True))
# TODO remove
for folder in pbar:
    if not folder.is_dir():
        continue
    count_patients += 1
    # patient = folder.name  # TODO comment in
    patient = f'{folder.parent.name}/{folder.name}'  # TODO remove
    pbar.set_description(f"Patient: {patient}")

    if 'tumor_concentrations' in domain:
        if patient in patients_val:
            data_growth_model = _get_data(_get_file_growth_model(patient))  # 240, 240, 155
            
            path_grey_matter = folder / 'processed/processed/tissue_segmentation/gm_pbmap.nii.gz'
            path_white_matter = folder / 'processed/processed/tissue_segmentation/wm_pbmap.nii.gz'
            data_grey_matter = nib.load(path_grey_matter).get_fdata()
            data_white_matter = nib.load(path_white_matter).get_fdata()

            for key, value in tumor_concentrations.items():
                path_result = folder / 'tumor_concentrations' / f'{key}.pt'
                path_result.parent.mkdir(parents=True, exist_ok=True)

                if not path_result.exists():

                    def run_solver(key, values, data_growth_model):

                        NxT1_pct, NyT1_pct, NzT1_pct = values

                        parameters = {
                            'Dw': 1.0,          # Diffusion coefficient for white matter
                            'rho': 0.10,         # Proliferation rate
                            'RatioDw_Dg': 10,  # Ratio of diffusion coefficients in white and grey matter
                            'gm': data_grey_matter,      # Grey matter data
                            'wm': data_white_matter,      # White matter data
                            'NxT1_pct': NxT1_pct,  # 0.3   # tumor position [%]
                            'NyT1_pct': NyT1_pct,  # 0.7
                            'NzT1_pct': NzT1_pct,  # 0.5
                            'init_scale': 1., #scale of the initial gaussian
                            'resolution_factor': 0.5, #resultion scaling for calculations
                            'th_matter': 0.1, #when to stop diffusing: at th_matter > gm+wm
                            'verbose': True, #printing timesteps 
                            'time_series_solution_Nt': 64 # number of timesteps in the output  
                        }
                        fk_solver = Solver(parameters)
                        result = fk_solver.solve(original_growth_model=data_growth_model if key == 'spatio_temporal' else None)
                    
                        return result

                    result = run_solver(key, tumor_concentrations[key], data_growth_model)

                    if result['success'] == False:
                        values = tumor_concentrations[key]
                        values[2] = 0.5
                        result = run_solver(key, values, data_growth_model)
                        if result['success'] == False:
                            values[1] = 0.5
                            result = run_solver(key, values, data_growth_model)
                            if result['success'] == False:
                                values[0] = 0.5
                                result = run_solver(key, values, data_growth_model)

                    result = torch.as_tensor(result['time_series'])  # 64, 240, 240, 155
                    
                    # nib.save(nib.Nifti1Image(result[0].cpu().float().numpy(), np.eye(4)), folder / 'tumor_concentrations' / f'{key}_0.nii.gz')
                    # nib.save(nib.Nifti1Image(result[63].cpu().float().numpy(), np.eye(4)), folder / 'tumor_concentrations' / f'{key}_63.nii.gz')
                    
                    torch.save(result, path_result)

    if 'condition' in domain:
        path_latent_conditioning = dir_data / patient / f'latents_{latent_shape_string}' / f'latent_conditioning.pt'  # TODO
        path_latent_conditioning = dir_data / patient / f'latents_{latent_shape_string}' / f'latent_conditioning.pt'

        path_latent_conditioning.parent.mkdir(parents=True, exist_ok=True)
        if not path_latent_conditioning.exists():
            data_growth_model = _get_data(_get_file_growth_model(patient))  # 240, 240, 155
            data_tissue_segmentation = _get_data(_get_file_tissue_segmentation(patient))  # 240, 240, 155

            latent_growth_model = _encode(data_growth_model, normalize=False)[0].squeeze(0)  # 4, 64, 64, 40
            latent_tissue_segmentation = _encode(data_tissue_segmentation)[0].squeeze(0)  # 4, 64, 64, 40

            latent_conditioning = torch.cat((latent_growth_model, latent_tissue_segmentation), dim=0)
            assert latent_conditioning.shape == (2*latent_shape[0], latent_shape[1], latent_shape[2], latent_shape[3]), f"Expected shape {(2*latent_shape[0], latent_shape[1], latent_shape[2], latent_shape[3])}, got {latent_conditioning.shape}"

            torch.save(latent_conditioning, path_latent_conditioning)

    if 'modality' in domain:
        for modality in MODALITIES:
            path_latent_modality = dir_data / patient / f'latents_{latent_shape_string}' / f'latent_{modality}.pt'
            if 'ae_latent' in mode and patient in ae_latent_patients['mean'][modality].keys() and path_latent_modality.exists():
                continue

            path_reconstructed = dir_reconstructed / patient / f'{modality}.nii.gz'
            path_psnr = dir_temp / 'psnr.csv'
            path_psnr_average = dir_temp / 'psnr.txt'
            if 'psnr' in mode and 'image_metrics' in mode:
                if path_psnr.exists():
                    df_psnr = pd.read_csv(path_psnr)
                    if ((df_psnr['patient'] == patient) & (df_psnr['modality'] == modality)).any():
                        continue

            if save_latent_modality == True and path_latent_modality.exists():
                continue

            data_modality = _get_data(folder / f'{modality}.nii.gz')  # 240, 240, 155
            affine_modality = _get_affine(folder / f'{modality}.nii.gz')
            latent_modality, normalized_modality = _encode(data_modality)  # B, 4, 64, 64, 40

            if save_latent_modality == True:
                torch.save(latent_modality, path_latent_modality)
        
            if 'ae_latent' in mode:
                latent_temp = latent_modality.cpu()[0]  # 4, 64, 64, 40
                ae_latent_patients['mean'][modality][patient] = float(latent_temp.mean().item())
                ae_latent_patients['std'][modality][patient] = float(latent_temp.std().item())
                ae_latent_patients['min'][modality][patient] = float(latent_temp.min().item())
                ae_latent_patients['max'][modality][patient] = float(latent_temp.max().item())
                with path_ae_latent_patients.open("wb") as f:
                    pickle.dump(ae_latent_patients, f, protocol=pickle.HIGHEST_PROTOCOL)
                
            if 'psnr' in mode:
                # nib.save(nib.Nifti1Image(latent_modality[0][0].cpu().float().numpy(), np.eye(4)), f'/vol/miltank/users/bilv/ldm/autoencoder/temp/{patient}_{modality}_latent.nii.gz')

                reconstructed_modality = _get_decoded(autoencoder, latent_modality)  # B, 240, 240, 155
                reconstructed_modality = transforms.ScaleIntensity(minv=0.0, maxv=1.0)(reconstructed_modality)
                assert (reconstructed_modality.min() >= 0.0 or reconstructed_modality.min() >= -1.0) and reconstructed_modality.max() <= 1.0, "Intensity values should be in the range [0, 1] or [-1, 1]"

                if 'image_metrics' in mode:
                    path_reconstructed.parent.mkdir(parents=True, exist_ok=True)
                    nib.save(nib.Nifti1Image(reconstructed_modality.squeeze(0).cpu().float().numpy(), affine_modality), path_reconstructed)

                assert reconstructed_modality.shape == normalized_modality.shape
                psnr_normalized = psnr(reconstructed_modality.to('cpu'), torch.as_tensor(normalized_modality).to('cpu'))

                path_psnr.parent.mkdir(parents=True, exist_ok=True)
                if path_psnr.exists():
                    df_psnr = pd.read_csv(path_psnr)
                    assert not ((df_psnr['patient'] == patient) & (df_psnr['modality'] == modality)).any(), f"PSNR for patient {patient} and modality {modality} already exists in {path_psnr}"
                else:
                    df_psnr = pd.DataFrame(columns=['patient', 'modality', 'psnr'])
                df_psnr = pd.concat([df_psnr, pd.DataFrame([{
                    'patient': patient,
                    'modality': modality,
                    'psnr': psnr_normalized.item(),
                }])], ignore_index=True)
                df_psnr.to_csv(path_psnr, index=False)

                # assert len(df_psnr) == (count_patients-1) * len(MODALITIES) + MODALITIES.index(modality) + 1
                assert len(df_psnr) == len(df_psnr[['patient', 'modality']].drop_duplicates())

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

if 'psnr' in mode:
    df_psnr = pd.read_csv(path_psnr)
    df_psnr = df_psnr.sort_values(by=['patient', 'modality']).reset_index(drop=True)
    assert len(df_psnr) == count_patients * len(MODALITIES)
    with open(path_psnr_average, 'w') as f:
        f.write(f'{df_psnr["psnr"].mean()}\n')

### FID / KID / SSIM ###
if 'image_metrics' in mode:

    class DatasetMetrics(Dataset):
        def __init__(self, dir_root, modality):
            self.dir_root = Path(dir_root)
            self.modality = modality

            self.patients = []
            for folder in sorted(self.dir_root.iterdir()):
                if folder.is_dir():
                    self.patients.append(folder.name)
            assert len(self.patients) == 3801, f"Expected 3801 patients, got {len(self.patients)}"

            self.intensity = transforms.ScaleIntensity(minv=0.0, maxv=1.0)

        def __len__(self):
            return len(self.patients)
        
        def __getitem__(self, idx):
            patient = self.patients[idx]
            file_name = f'{self.modality}.nii.gz'
            data = torch.as_tensor(nib.load(self.dir_root / patient / file_name).get_fdata())

            normalized = self.intensity(data).unsqueeze(0)
            normalized = normalized.permute(0, 3, 1, 2)  # c, d, h, w
            return normalized.to(torch.float64)

    '''
    EXAMPLE
    metrics = torch_fidelity.calculate_metrics(
        input1=torch_fidelity.GenerativeModelModuleWrapper(G, args.z_size, args.z_type, num_classes),
        input1_model_num_samples=args.num_samples_for_metrics,
        input2="cifar10-train",
        isc=True,
        fid=True,
        kid=True,
        ppl=True,
        ppl_epsilon=1e-2,
        ppl_sample_similarity_resize=64,
        prc=True,
    )
    '''

    image_metrics_rows = []
    for modality in MODALITIES:
        print(f"========== Processing modality: {modality}")

        dataset_reconstruction = DatasetMetrics(dir_root=dir_reconstructed, modality=modality)
        dataset_ground_truth = DatasetMetrics(dir_root=dir_data, modality=modality)
        assert len(dataset_reconstruction) == len(dataset_ground_truth)

        # .../.metrics/torch_fidelity/utils.py
        # line: 146 - 'if batch.dim() == 5:'
        # number of images: 16 !!!
        metrics_view = torch_fidelity.calculate_metrics(
            vol_mode=None,
            input1=dataset_reconstruction,
            input2=dataset_ground_truth,
            # input1=dir_reconstruction_modality,
            # input2=dir_data_modality,
            isc=True,
            msssim=False,
            fid=True,
            kid=True,
            kid_subset_size=min(len(dataset_reconstruction), 1000),
            prc=True,
            # only with generator model:
            # ppl=True,  
            # ppl_epsilon=1e-2,
            # ppl_sample_similarity_resize=64,
            batch_size=4,
            feature_extractor='med3dnet-50',
            feature_extractor_weights_path=str(dir_metrics / "resnet_50.pth"),
            feature_layer_isc='256',
            feature_layer_fid='256',
            feature_layer_kid='256',
            feature_layer_prc='256',
            cuda=True if torch.cuda.is_available() else False,
            verbose=False,
        )

        image_metrics_rows.append({
            "modality": modality,
            **metrics_view
        })
    
    path_metrics = dir_temp / 'image_metrics.csv'
    path_metrics_average = dir_temp / 'image_metrics.txt'
    path_metrics.parent.mkdir(parents=True, exist_ok=True)

    metrics = pd.DataFrame(image_metrics_rows)
    metrics.to_csv(path_metrics, index=False)

    metrics = pd.read_csv(path_metrics)
    with open(path_metrics_average, 'w') as f:
        for col in metrics.columns:
            if col != 'modality':
                f.write(f'{col}: {metrics[col].mean()}\n')
