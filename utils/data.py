import os
import numpy as np
from pathlib import Path

import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split

import nibabel as nib
from monai import transforms


def create_conditioning(growth_model, tissue_segmentation):
    """
    Create 4-channel conditioning:
    - Channel 0: Growth model (threshold at 0.2)
    - Channels 1-3: Tissue segmentation one-hot
    """
    # Apply threshold: values < 0.2 → 0, values >= 0.2 → keep
    growth_model = np.where(growth_model >= 0.2, growth_model, 0.0)
    # Create one-hot encoding for tissue segmentation
    tissue_segmentation_1 = np.where(tissue_segmentation == 1, 1.0, 0.0)  # Tissue type 1
    tissue_segmentation_2 = np.where(tissue_segmentation == 2, 1.0, 0.0)  # Tissue type 2  
    tissue_segmentation_3 = np.where(tissue_segmentation == 3, 1.0, 0.0)  # Tissue type 3

    return np.stack([growth_model, tissue_segmentation_1, tissue_segmentation_2, tissue_segmentation_3], axis=0)  # 4, 240, 240, 155

class DataModule(pl.LightningDataModule):
    """
    PyTorch Lightning data module for brain MRI data
    """
    
    def __init__(
        self,
        debug=False,
        mode='training',  # training, inference, inference_challenge
        path_data=None,
        path_data_challenge=None,
        latent_shape=None,
        batch_size=2,
        num_workers=4,
        train_val_split=0.8,
        **kwargs
    ):
        super().__init__()
        self.debug = debug
        self.mode = mode
        self.path_data = path_data
        self.path_data_challenge = path_data_challenge
        self.latent_shape = latent_shape
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_split = train_val_split

        assert self.mode in ['training', 'inference', 'inference_challenge'], f"Invalid mode: {self.mode}. Choose from: training, inference, inference_challenge"

        pl.seed_everything(42)
    
    def setup(self, stage=None): 

        if self.path_data_challenge is None:
            challenge_patients = []
            path_challenge_patients_identifier = str(Path(__file__).resolve().parent / 'challenge_patients_identifier.txt')
            with open(path_challenge_patients_identifier, 'r') as f:
                challenge_patients_identifier = set([line.strip() for line in f.readlines()])
        else:
            challenge_patients = sorted(list(os.listdir(self.path_data_challenge)))
            challenge_patients_identifier = set([folder.split('-')[2] for folder in challenge_patients])

        patients = []
        for folder in sorted(os.listdir(self.path_data)):
            path_growth_model = os.path.join(self.path_data, folder, 'processed', 'growth_model.nii.gz')
            path_tissue_segmentation = os.path.join(self.path_data, folder, 'processed_voided', 'tissue_segmentation.nii.gz')
            if os.path.exists(path_growth_model) and os.path.exists(path_tissue_segmentation):
                # For Challenge
                if folder.startswith('BraTS2021') and folder.split('_')[1] in challenge_patients_identifier:
                    continue
                patients.append(folder)
        
        # Split by patient IDs
        if self.debug:
            patients = patients[:10]
        n_train = int(len(patients) * self.train_val_split)
        n_val = len(patients) - n_train
        patients_train, patients_val = random_split(patients, [n_train, n_val])
        patients_train = [patients[i] for i in patients_train.indices]
        patients_val = [patients[i] for i in patients_val.indices]

        if self.mode in ['training', 'inference']:
            print(f"Total patients found: {len(patients)}")
            print(f"Train patients: {len(patients_train)}, Val/Test patients: {len(patients_val)}")
        elif self.mode == 'inference_challenge':
            print(f"Total challenge patients found: {len(challenge_patients)}")
        
        self.dataset_train = DataSet(
            mode=self.mode,
            path_data=self.path_data,
            path_data_challenge=self.path_data_challenge,
            latent_shape=self.latent_shape,
            patients=patients_train,
        )
        
        self.dataset_val = DataSet(
            mode=self.mode,
            path_data=self.path_data,
            path_data_challenge=self.path_data_challenge,
            latent_shape=self.latent_shape,
            patients=patients_val,
        )

        self.dataset_challenge = DataSet(
            mode=self.mode,
            path_data=self.path_data,
            path_data_challenge=self.path_data_challenge,
            latent_shape=self.latent_shape,
            patients=challenge_patients,
        )
        
        self.dataloader_train = DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        self.dataloader_val = DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

        self.dataloader_challenge = DataLoader(
            self.dataset_challenge,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def train_dataloader(self):
        return self.dataloader_train
    
    def val_dataloader(self):
        return self.dataloader_val
    
    def test_dataloader(self):
        if self.mode == 'inference':
            return self.dataloader_val
        elif self.mode == 'inference_challenge':
            return self.dataloader_challenge
        

class DataSet(Dataset):
    """
    Dataset for brain MRI inpainting with conditioning
    
    Returns:
        - voided_latent: Encoded voided input image (6, 32, 32, 20)
        - conditioning: 4-channel conditioning (4, 32, 32, 20)
        - mask_latent: Inpainting mask in latent space (1, 32, 32, 20)
        - target_image: Ground truth image (1, 240, 240, 155)
        - patient_id: Patient identifier
        - mask_id: Mask identifier (0000, 0001, 0002)
    """
    
    def __init__(
        self,
        mode='training',
        path_data=None,
        path_data_challenge=None,
        latent_shape=None,
        patients=None,
    ):
        self.mode = mode
        self.path_data = path_data
        self.path_data_challenge = path_data_challenge
        self.latent_shape = latent_shape
        self.patients = patients
    
        if self.mode in ['training', 'inference_challenge']:
            self.samples = self.patients

        elif self.mode == 'inference':
            self.samples = []
            for patient_id in self.patients:
                # Check for mask files 0000, 0001, 0002
                for mask_id in ["0000", "0001", "0002"]:
                    self.samples.append((patient_id, mask_id))

        self.intensity = transforms.ScaleIntensity(minv=0.0, maxv=1.0)
        self.autoencoder_pad = transforms.SpatialPad(spatial_size=(240, 240, 160))
        self.autoencoder_crop = transforms.CenterSpatialCrop(roi_size=(240, 240, 155))
    
    def _get_data(self, file_path, affine=False):
        img = nib.load(file_path)
        if affine:
            return img.get_fdata(), img.affine
        else:
            return img.get_fdata()
    
    def _get_file_modality(self, patient, modality):
        return os.path.join(self.path_data, patient, f'{modality}.nii.gz')
    def _get_file_growth_model(self, patient):
        return os.path.join(self.path_data, patient, 'processed', 'growth_model.nii.gz')
    def _get_file_tissue_segmentation(self, patient):
        return os.path.join(self.path_data, patient, 'processed_voided', 'tissue_segmentation.nii.gz')
    
    def _get_file_modality_voided(self, patient, mask, modality):
        return Path(self.path_data) / patient / 'voided' / f"{modality}-voided-{mask}.nii.gz"
    def _get_file_mask(self, patient, mask, healthy=False):
        if healthy:
            return Path(self.path_data) / patient / 'masks' / f"mask-healthy-{mask}.nii.gz"
        else:
            return Path(self.path_data) / patient / 'masks' / f"mask-{mask}.nii.gz"

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        if self.mode == 'training':
            patient = self.samples[idx]

            original_t1 = self._get_data(self._get_file_modality(patient, 't1'))
            original_t1 = self.intensity(original_t1)  
            original_t1 = torch.as_tensor(original_t1).unsqueeze(0)  # 1, 240, 240, 155

            original_t1_autoencoder = self.autoencoder_pad(original_t1)
            original_t1_autoencoder = torch.as_tensor(original_t1_autoencoder)  # 1, 240, 240, 160

            original_growth_model = self._get_data(self._get_file_growth_model(patient))
            original_tissue_segmentation = self._get_data(self._get_file_tissue_segmentation(patient))
            original_conditioning = create_conditioning(original_growth_model, original_tissue_segmentation)
            original_conditioning = torch.as_tensor(original_conditioning)  # 4, 240, 240, 155
            
            latent_conditioning = torch.nn.functional.interpolate(
                original_conditioning.unsqueeze(0),
                size=(self.latent_shape[0], self.latent_shape[1], self.latent_shape[2]),
                mode='nearest'
            )[0]  # 4, 60, 60, 40

            return {
                'patient': patient,
                'original_t1': original_t1.float(),
                'original_t1_autoencoder': original_t1_autoencoder.float(),
                'latent_conditioning': latent_conditioning.float(),
            }

        elif self.mode == 'inference':
            patient, mask = self.samples[idx]

            original_t1, affine = self._get_data(self._get_file_modality(patient, 't1'), affine=True)
            original_t1 = self.intensity(original_t1)  
            original_t1 = torch.as_tensor(original_t1).unsqueeze(0)  # 1, 240, 240, 155

            original_tissue_segmentation = self._get_data(self._get_file_tissue_segmentation(patient))  # 240, 240, 155
            original_growth_model = torch.zeros_like(torch.as_tensor(original_tissue_segmentation))  # 240, 240, 155
            original_conditioning = create_conditioning(original_growth_model, original_tissue_segmentation)
            original_conditioning = torch.as_tensor(original_conditioning)  # 4, 240, 240, 155
            
            latent_conditioning = torch.nn.functional.interpolate(
                original_conditioning.unsqueeze(0),
                size=(self.latent_shape[0], self.latent_shape[1], self.latent_shape[2]),
                mode='nearest'
            )[0]  # 4, 60, 60, 40

            original_t1_voided = self._get_data(self._get_file_modality_voided(patient, mask, 't1'))
            original_t1_voided = self.intensity(original_t1_voided)  
            original_t1_voided = torch.as_tensor(original_t1_voided).unsqueeze(0)  # 1, 240, 240, 155

            original_t1_voided_autoencoder = self.autoencoder_pad(original_t1_voided)
            original_t1_voided_autoencoder = torch.as_tensor(original_t1_voided_autoencoder)  # 1, 240, 240, 160

            original_mask = self._get_data(self._get_file_mask(patient, mask))
            original_mask = torch.as_tensor(original_mask).unsqueeze(0)  # 1, 240, 240, 155

            latent_mask = torch.nn.functional.interpolate(
                original_mask.unsqueeze(0),
                size=(self.latent_shape[0], self.latent_shape[1], self.latent_shape[2]),
                mode='nearest'
            )[0]  # 1, 60, 60, 40

            original_mask_healty = self._get_data(self._get_file_mask(patient, mask, healthy=True))
            original_mask_healty = torch.as_tensor(original_mask_healty).unsqueeze(0)  # 1, 240, 240, 155

            return {
                'patient': patient,
                'mask': mask,
                'original_t1': original_t1.float(),
                'original_conditioning': original_conditioning.float(),
                'latent_conditioning': latent_conditioning.float(),
                'original_t1_voided': original_t1_voided.float(),
                'original_t1_voided_autoencoder': original_t1_voided_autoencoder.float(),
                'original_mask': original_mask.float(),
                'latent_mask': latent_mask.float(),
                'original_mask_healthy': original_mask_healty.float(),
                'affine': affine,
            }

        elif self.mode == 'inference_challenge':
            patient = self.samples[idx]

            path_data_challenge_t1_voided = Path(self.path_data_challenge) / patient / f"{patient}-t1n-voided.nii.gz"
            path_data_challenge_mask = Path(self.path_data_challenge) / patient / f"{patient}-mask.nii.gz"

            original_t1_voided, affine = self._get_data(path_data_challenge_t1_voided, affine=True)
            original_t1_voided = self.intensity(original_t1_voided)  
            original_t1_voided = torch.as_tensor(original_t1_voided).unsqueeze(0)  # 1, 240, 240, 155

            original_t1_voided_autoencoder = self.autoencoder_pad(original_t1_voided)
            original_t1_voided_autoencoder = torch.as_tensor(original_t1_voided_autoencoder)  # 1, 240, 240, 160

            original_mask = self._get_data(path_data_challenge_mask)
            original_mask = torch.as_tensor(original_mask).unsqueeze(0)  # 1, 240, 240, 155

            latent_mask = torch.nn.functional.interpolate(
                original_mask.unsqueeze(0),
                size=(self.latent_shape[0], self.latent_shape[1], self.latent_shape[2]),
                mode='nearest'
            )[0]  # 1, 60, 60, 40

            path_original_conditioning = Path(self.path_data_challenge) / patient / 'conditioning.pt'
            if path_original_conditioning.exists():
                exists_conditioning = True
                original_conditioning = torch.load(path_original_conditioning).float()  # 4, 240, 240, 155
                latent_conditioning = torch.nn.functional.interpolate(
                    original_conditioning.unsqueeze(0),
                    size=(self.latent_shape[0], self.latent_shape[1], self.latent_shape[2]),
                    mode='nearest'
                )[0].float()  # 4, 60, 60, 40
            else:
                exists_conditioning = False
                original_conditioning = torch.zeros(4, 240, 240, 155).float()
                latent_conditioning = torch.zeros(4, self.latent_shape[0], self.latent_shape[1], self.latent_shape[2]).float()
            
            return {
                'patient': patient,
                'original_t1_voided': original_t1_voided.float(),
                'original_t1_voided_autoencoder': original_t1_voided_autoencoder.float(),
                'original_mask': original_mask.float(),
                'latent_mask': latent_mask.float(),
                'path_original_t1_voided': str(path_data_challenge_t1_voided),
                'path_original_mask': str(path_data_challenge_mask),
                'affine': affine,
                'original_conditioning': original_conditioning,
                'latent_conditioning': latent_conditioning,
                'exists_conditioning': exists_conditioning,
            }