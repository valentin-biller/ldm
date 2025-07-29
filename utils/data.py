import os
import random
import numpy as np
from pathlib import Path

import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split

import nibabel as nib
from monai import transforms

from scipy.ndimage import distance_transform_edt, label, binary_dilation


def create_conditioning(growth_model, tissue_segmentation, threshold=0.001, tensor=False):
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

    if tensor:
        conditioning = torch.as_tensor(conditioning)

    return conditioning

class DataModule(pl.LightningDataModule):
    """
    PyTorch Lightning data module for brain MRI data
    """
    
    def __init__(
        self,
        debug=False,
        mode='training',  # training, inference, inference_challenge, inference_conditioning
        oversampling=True,
        path_data=None,
        path_data_challenge=None,
        dir_output_model=None,
        latent_shape=None,
        batch_size=2,
        num_workers=4,
        train_val_split=0.8,
        **kwargs
    ):
        super().__init__()
        self.debug = debug
        self.mode = mode
        self.oversampling = oversampling
        self.path_data = path_data
        self.path_data_challenge = path_data_challenge
        self.dir_output_model = dir_output_model
        self.latent_shape = latent_shape
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_split = train_val_split

        assert self.mode in ['training', 'inference', 'inference_challenge', 'inference_conditioning'], f"Invalid mode: {self.mode}. Choose from: training, inference, inference_challenge"
    
        self.print_length = 25
        pl.seed_everything(42)
    
    def setup(self, stage=None): 

        if self.path_data_challenge is None:
            patients_challenge = []
            path_patients_challenge_identifier = str(Path(__file__).resolve().parent / '.patients_challenge_identifier.txt')
            with open(path_patients_challenge_identifier, 'r') as f:
                patients_challenge_identifier = set([line.strip() for line in f.readlines()])
        else:
            patients_challenge = sorted(list(os.listdir(self.path_data_challenge)))
            patients_challenge_identifier = set([folder.split('-')[2] for folder in patients_challenge])

        patients = []
        for folder in sorted(os.listdir(self.path_data)):
            path_growth_model = os.path.join(self.path_data, folder, 'processed', 'growth_model.nii.gz')
            path_tissue_segmentation = os.path.join(self.path_data, folder, 'processed_voided', 'tissue_segmentation.nii.gz')
            if os.path.exists(path_growth_model) and os.path.exists(path_tissue_segmentation):
                # For Challenge
                if folder.startswith('BraTS2021') and folder.split('_')[1] in patients_challenge_identifier:
                    continue
                patients.append(folder)
        
        # Split by patient IDs
        n_train = int(len(patients) * self.train_val_split)
        n_val = len(patients) - n_train
        patients_train, patients_val = random_split(patients, [n_train, n_val])
        patients_train = [patients[i] for i in patients_train.indices]
        patients_val = [patients[i] for i in patients_val.indices]

        # For Inference filter patients that are not in the dir_output_model
        if self.dir_output_model is not None:
            files_completed = list((self.dir_output_model / 'reconstructed').iterdir())
            if self.mode in ['inference', 'inference_conditioning']:
                patient_masks = {}
                for file in files_completed:
                    name = file.name
                    patient_mask = name[:-7]
                    patient = patient_mask[:-5]
                    mask = patient_mask[-4:]
                    patient_masks.setdefault(patient, set()).add(mask)
                patients_completed = [patient for patient, masks in patient_masks.items() if {'0000', '0001', '0002'}.issubset(masks)]
                patients_val = [patient for patient in patients_val if patient not in patients_completed]
                self._print_numbers('Completed', patients_completed)
            elif self.mode == 'inference_challenge':
                patients_completed = [file.name[:-21] for file in files_completed]
                patients_challenge = [patient for patient in patients_challenge if patient not in patients_completed]
                self._print_numbers('Completed', patients_completed)

        print(self.print_length * '=')
        if self.mode in ['training', 'inference', 'inference_conditioning']:
            self._print_numbers('Total', patients)
            self._print_numbers('Train', patients_train)
            self._print_numbers('Val', patients_val)
        elif self.mode == 'inference_challenge':
            self._print_numbers('Challenge', patients_challenge)

        ### Counting Prefixes, Oversampling and Debugging
        if self.mode in ['training', 'inference', 'inference_conditioning']:
            prefixes = ["900", "BraTS", "egd", "glioma", "hf", "Patient", "tcga", "ucsf", "upenn"]
            groups_train = self._count_prefixes('Train', patients_train, prefixes)
            groups_val = self._count_prefixes('Val', patients_val, prefixes)

            if self.mode == 'training' and self.oversampling:
                patients_train = self._oversample_prefixes(groups_train)
                groups_train = self._count_prefixes('Train Oversampling', patients_train, prefixes)
                self._print_numbers('Train', patients_train)

            if self.mode in ['inference', 'inference_conditioning'] and self.debug:
                sampled = []
                for prefix in prefixes:
                    group = [p for p in patients_val if p.startswith(prefix)]
                    sampled += random.sample(group, min(3, len(group)))
                patients_val = sampled
                groups_val = self._count_prefixes('Val Debug', patients_val, prefixes)
                self._print_numbers('Val', patients_val)

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
            patients=patients_challenge,
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
        if self.mode in ['inference', 'inference_conditioning']:
            return self.dataloader_val
        elif self.mode == 'inference_challenge':
            return self.dataloader_challenge
        
    def _print_numbers(self, identifier, patients):
        length = self.print_length - 5
        print(f"{'Patients ' + identifier + ':':<{length}}{len(patients):>5}")

    def _count_prefixes(self, identifier, patients, prefixes):
        groups = {prefix: [] for prefix in prefixes}
        for patient in patients:
            for prefix in prefixes:
                if patient.startswith(prefix):
                    groups[prefix].append(patient)
                    break
        identifier_str = str(identifier)
        side = (self.print_length - len(identifier_str) - 2) // 2
        extra = (self.print_length - len(identifier_str) - 2) % 2
        print(f"{'=' * (side + extra)} {identifier_str} {'=' * side}")
        for prefix in prefixes:
            count = len(groups[prefix])
            print(f"{prefix:<10}{count:>5}{count / len(patients) * 100:9.2f}%")
        return groups

    def _oversample_prefixes(self, groups):
        max_size = max(len(group) for group in groups.values())
        oversampled = []
        for prefix, group in groups.items():
            if group:
                oversampled += random.choices(group, k=max_size)
        return oversampled


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

        elif self.mode in ['inference', 'inference_conditioning']:
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
        return os.path.join(self.path_data, patient, 'processed', 'tissue_segmentation.nii.gz')
    
    def _get_file_modality_voided(self, patient, mask, modality):
        return Path(self.path_data) / patient / 'voided' / f"{modality}-voided-{mask}.nii.gz"
    def _get_file_mask(self, patient, mask, healthy=False):
        if healthy:
            return Path(self.path_data) / patient / 'masks' / f"mask-healthy-{mask}.nii.gz"
        else:
            return Path(self.path_data) / patient / 'masks' / f"mask-{mask}.nii.gz"
        
    def _process_mask(self, data):
        return torch.as_tensor(data).unsqueeze(0)  # 1, 240, 240, 155
    def _process_modality(self, data):
        original_modality = self.intensity(data)  
        original_modality = torch.as_tensor(original_modality).unsqueeze(0)
        return original_modality  # 1, 240, 240, 155
    def _process_modality_autoencoder(self, original_modality):
        original_modality_autoencoder = self.autoencoder_pad(original_modality)
        original_modality_autoencoder = torch.as_tensor(original_modality_autoencoder)  
        return original_modality_autoencoder  # 1, 240, 240, 160
    def _process_interpolation(self, original):
        latent = torch.nn.functional.interpolate(
            original.unsqueeze(0),
            size=(self.latent_shape[0], self.latent_shape[1], self.latent_shape[2]),
            mode='nearest'
        )[0]
        return latent  # 4, 60, 60, 40

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        if self.mode == 'training':
            patient = self.samples[idx]

            original_t1 = self._process_modality(self._get_data(self._get_file_modality(patient, 't1')))  # 1, 240, 240, 155
            original_t1_autoencoder = self._process_modality_autoencoder(original_t1)  # 1, 240, 240, 160

            original_growth_model = self._get_data(self._get_file_growth_model(patient))
            original_tissue_segmentation = self._get_data(self._get_file_tissue_segmentation(patient))
            original_conditioning = create_conditioning(original_growth_model, original_tissue_segmentation, tensor=True)  # 4, 240, 240, 155
            latent_conditioning = self._process_interpolation(original_conditioning)  # 4, 60, 60, 40

            return {
                'patient': patient,
                'original_t1': original_t1.float(),
                'original_t1_autoencoder': original_t1_autoencoder.float(),
                'latent_conditioning': latent_conditioning.float(),
            }

        elif self.mode in ['inference', 'inference_conditioning']:
            patient, mask = self.samples[idx]

            original_t1, affine = self._get_data(self._get_file_modality(patient, 't1'), affine=True)
            original_t1 = self._process_modality(original_t1)  # 1, 240, 240, 155

            original_t1_voided = self._process_modality(self._get_data(self._get_file_modality_voided(patient, mask, 't1')))  # 1, 240, 240, 155
            original_t1_voided_autoencoder = self._process_modality_autoencoder(original_t1_voided)  # 1, 240, 240, 160
            
            original_mask_healty = self._process_mask(self._get_data(self._get_file_mask(patient, mask, healthy=True)))  # 1, 240, 240, 155
            original_mask = self._process_mask(self._get_data(self._get_file_mask(patient, mask)))  # 1, 240, 240, 155
            latent_mask = self._process_interpolation(original_mask)  # 1, 60, 60, 40
            
            # original_t1_voided_mirrored_margin, original_mask_mirrored_margin = self._mirror_known_region(original_t1_voided, original_mask, mode='margin')
            # original_t1_voided_mirrored_margin_autoencoder = self._process_modality_autoencoder(original_t1_voided_mirrored_margin)  # 1, 240, 240, 160
            # latent_mask_mirrored_margin = self._process_interpolation(original_mask_mirrored_margin)  # 1, 60, 60, 40

            original_tissue_segmentation = self._get_data(self._get_file_tissue_segmentation(patient))  # 240, 240, 155
            if self.mode == 'inference':
                original_growth_model = torch.zeros_like(torch.as_tensor(original_tissue_segmentation))  # 240, 240, 155
            elif self.mode == 'inference_conditioning':
                original_growth_model = self._get_data(self._get_file_growth_model(patient))  # 240, 240, 155
            original_conditioning = create_conditioning(original_growth_model, original_tissue_segmentation, tensor=True)  # 4, 240, 240, 155
            latent_conditioning = self._process_interpolation(original_conditioning)  # 4, 60, 60, 40

            return {
                'mode': self.mode,
                'patient': patient,
                'mask': mask,
                'affine': affine,

                'original_t1': original_t1.float(),
                'original_t1_voided': original_t1_voided.float(),
                'original_t1_voided_autoencoder': original_t1_voided_autoencoder.float(),
                # 'original_t1_voided_mirrored_margin': original_t1_voided_mirrored_margin.float(),
                # 'original_t1_voided_mirrored_margin_autoencoder': original_t1_voided_mirrored_margin_autoencoder.float(),

                'original_mask_healthy': original_mask_healty.float(),
                'original_mask': original_mask.float(),
                'latent_mask': latent_mask.float(),
                # 'original_mask_mirrored_margin': original_mask_mirrored_margin.float(),
                # 'latent_mask_mirrored_margin': latent_mask_mirrored_margin.float(),

                'original_conditioning': original_conditioning.float(),
                'latent_conditioning': latent_conditioning.float(),
            }            

        elif self.mode == 'inference_challenge':
            patient = self.samples[idx]

            path_data_challenge_t1_voided = Path(self.path_data_challenge) / patient / f"{patient}-t1n-voided.nii.gz"
            path_data_challenge_mask = Path(self.path_data_challenge) / patient / f"{patient}-mask.nii.gz"

            original_t1_voided, affine = self._get_data(path_data_challenge_t1_voided, affine=True)
            original_t1_voided = self._process_modality(original_t1_voided)  # 1, 240, 240, 155
            original_t1_voided_autoencoder = self._process_modality_autoencoder(original_t1_voided)  # 1, 240, 240, 160

            original_mask = self._process_mask(self._get_data(path_data_challenge_mask))  # 1, 240, 240, 155
            latent_mask = self._process_interpolation(original_mask)  # 1, 60, 60, 40

            # original_t1_voided_mirrored_margin, original_mask_mirrored_margin = self._mirror_known_region(original_t1_voided, original_mask, mode='margin')
            # original_t1_voided_mirrored_margin_autoencoder = self._process_modality_autoencoder(original_t1_voided_mirrored_margin)  # 1, 240, 240, 160
            # latent_mask_mirrored_margin = self._process_interpolation(original_mask_mirrored_margin)  # 1, 60, 60, 40

            # original_t1_voided_mirrored_whole, original_mask_mirrored_whole = self._mirror_known_region(original_t1_voided, original_mask, mode='whole')
            # path_original_t1_voided_mirrored_whole = Path(self.path_data_challenge) / patient / f"{patient}-t1n-voided-mirrored-whole.nii.gz"
            # path_original_mask_mirrored_whole = Path(self.path_data_challenge) / patient / f"{patient}-mask-mirrored-whole.nii.gz"
            # if not path_original_t1_voided_mirrored_whole.exists():
            #     nib.save(nib.Nifti1Image(original_t1_voided_mirrored_whole[0].numpy(), affine), path_original_t1_voided_mirrored_whole)
            # if not path_original_mask_mirrored_whole.exists():
            #     nib.save(nib.Nifti1Image(original_mask_mirrored_whole[0].numpy(), affine), path_original_mask_mirrored_whole)

            path_original_conditioning = Path(self.path_data_challenge) / patient / 'conditioning.pt'
            if path_original_conditioning.exists():
                exists_conditioning = True
                original_conditioning = torch.load(path_original_conditioning)  # 4, 240, 240, 155
                latent_conditioning = self._process_interpolation(original_conditioning)  # 4, 60, 60, 40
            else:
                exists_conditioning = False
                original_conditioning = torch.zeros(4, 240, 240, 155)
                latent_conditioning = torch.zeros(4, self.latent_shape[0], self.latent_shape[1], self.latent_shape[2])

            return {
                'mode': self.mode,
                'patient': patient,
                'affine': affine,

                'original_t1_voided': original_t1_voided.float(),
                'original_t1_voided_autoencoder': original_t1_voided_autoencoder.float(),
                # 'original_t1_voided_mirrored_margin': original_t1_voided_mirrored_margin.float(),
                # 'original_t1_voided_mirrored_margin_autoencoder': original_t1_voided_mirrored_margin_autoencoder.float(),

                'original_mask': original_mask.float(),
                'latent_mask': latent_mask.float(),
                # 'original_mask_mirrored_margin': original_mask_mirrored_margin.float(),
                # 'latent_mask_mirrored_margin': latent_mask_mirrored_margin.float(),

                'original_conditioning': original_conditioning.float(),
                'latent_conditioning': latent_conditioning.float(),
                'exists_conditioning': exists_conditioning,

                'path_original_t1_voided': str(path_data_challenge_t1_voided),
                'path_original_mask': str(path_data_challenge_mask),
                # 'path_original_t1_voided_mirrored_whole': str(path_original_t1_voided_mirrored_whole),
                # 'path_original_mask_mirrored_whole': str(path_original_mask_mirrored_whole),
            }
        
    def _mirror_known_region(self, original_voided, original_mask, margin=3, mode='margin'):

        original_voided_mirrored = np.copy(original_voided)  # 1, 240, 240, 155
        original_mask_mirrored = np.copy(original_mask)  # 1, 240, 240, 155
        
        original_voided_c = original_voided_mirrored[0]
        original_mask_c = original_mask_mirrored[0]
        
        S, _, _ = original_voided_c.shape

        sagittal_nonzero = np.where(np.any(original_voided_c != 0, axis=(1, 2)))[0]
        assert len(sagittal_nonzero) > 0, "No non-zero slices found in the sagittal plane."
        sagittal_first = sagittal_nonzero[0]
        sagittal_last = sagittal_nonzero[-1]
        sagittal_midline = (sagittal_first + sagittal_last) / 2.0

        # Find connected "clouds" of 1s in the mask
        labeled_mask_unfiltered, num_features_unfiltered = label(original_mask_c == 1)
        min_voxels = 500
        num_features = []
        for i in range(1, num_features_unfiltered + 1):
            feature_mask = (labeled_mask_unfiltered == i).astype(np.uint8)
            if feature_mask.sum() >= min_voxels:
                num_features.append(i)
        labeled_mask = labeled_mask_unfiltered * np.isin(labeled_mask_unfiltered, num_features)
        if len(num_features) != 2:
            for i in range(1, num_features_unfiltered + 1):
                feature_mask = (labeled_mask_unfiltered == i).astype(np.uint8)
                print(f"Feature {i}: {feature_mask.sum()} voxels")
        # assert len(num_features) == 2, f"Expected 2 valid features, got {len(num_features)}"

        # A region is on the border if it's adjacent to the background (latent == 0).
        background_mask = (original_voided_c == 0)
        background_mask_combined = (original_voided_c == 0) & (original_mask_c == 0)

        for i in num_features:
            feature_mask = (labeled_mask == i)
            
            # Dilate feature mask slightly to check its immediate neighborhood.
            feature_mask_dilated = binary_dilation(feature_mask)
            
            # If the dilated feature mask touches any background voxels, it's a border mask.
            is_border_mask = np.any(feature_mask_dilated & background_mask_combined)

            # --- Step 4: Process only the border masks ---
            if is_border_mask:
                # Get the coordinates of the current border mask.
                feature_mask_coords = np.argwhere(feature_mask)

                # Get the mirrored sagittal coordinates.
                feature_mask_coords_mirrored_sagittal = (2 * sagittal_midline - feature_mask_coords[:, 0]).astype(int)
                
                # Get the full set of mirrored coordinates (replace old sagittal coordinates with the mirrored ones).
                feature_mask_coords_mirrored = feature_mask_coords.copy()
                assert np.all((feature_mask_coords_mirrored_sagittal >= 0) & (feature_mask_coords_mirrored_sagittal < S)), \
                    f"Mirrored sagittal coordinates out of bounds: min={feature_mask_coords_mirrored_sagittal.min()}, max={feature_mask_coords_mirrored_sagittal.max()}, S={S}"
                feature_mask_coords_mirrored[:, 0] = feature_mask_coords_mirrored_sagittal

                if mode == 'margin':
                    # In the mirrored region, identify the brain's surface.
                    # Calculate the distance from every brain voxel to the nearest background voxel.
                    distance_transform = distance_transform_edt(~background_mask_combined)

                    # The brain surface is the region where the distance is > 0 and <= margin.
                    brain_surface_mask = (distance_transform > 0) & (distance_transform <= margin)

                    # Now, find which of our mirrored coordinates fall within this brain surface.
                    those_in_brain_surface = brain_surface_mask[tuple(feature_mask_coords_mirrored.T)]  # Boolean array

                    # Filter both original and mirrored coordinates to only include those in the brain surface.
                    # Both contain the exact same coordinates, just with mirrored sagittal values!
                    feature_mask_coords_to_update = feature_mask_coords[those_in_brain_surface]
                    feature_mask_coords_mirrored_to_copy = feature_mask_coords_mirrored[those_in_brain_surface]

                    best_shift = self._find_best_shift(
                        feature_mask,
                        brain_surface_mask,
                        feature_mask_coords_to_update,
                    )
                    # print("Best Shift", best_shift)
                    feature_mask_coords_to_update = feature_mask_coords_to_update + best_shift
                elif mode == 'whole':
                    feature_mask_coords_to_update = feature_mask_coords
                    feature_mask_coords_mirrored_to_copy = feature_mask_coords_mirrored

                feature_mask_coords_to_update_tuple = tuple(feature_mask_coords_to_update.T)
                feature_mask_coords_mirrored_to_copy_tuple = tuple(feature_mask_coords_mirrored_to_copy.T)

                # Update the voided and the mask
                original_voided_c[feature_mask_coords_to_update_tuple] = original_voided_c[feature_mask_coords_mirrored_to_copy_tuple]

                feature_mask_coords_to_update_mask = original_voided_c[feature_mask_coords_to_update_tuple] != 0
                feature_mask_coords_to_update_tuple = tuple(feature_mask_coords_to_update[feature_mask_coords_to_update_mask].T)
                original_mask_c[feature_mask_coords_to_update_tuple] = 0

            # Place the updated 3D volumes back into the 4D batch tensor.
            original_voided_mirrored[0] = original_voided_c
            original_mask_mirrored[0] = original_mask_c

        return torch.as_tensor(original_voided_mirrored), torch.as_tensor(original_mask_mirrored)

    def _find_best_shift(self, feature_mask, brain_surface_mask, feature_mask_coords_to_update, max_shift=3):
        """
        Finds the best translation vector to align mirrored coordinates with the brain surface,
        such that the translated coordinates are direct neighbors to the brain surface near the feature_mask border.
        """
        # Filter brain surface to only include voxels outside the feature mask
        brain_surface_mask_filtered = brain_surface_mask & ~feature_mask

        # Create a set of brain surface coordinates
        brain_surface_coords_filtered = np.argwhere(brain_surface_mask_filtered)
        brain_surface_set = set(map(tuple, brain_surface_coords_filtered))

        # Try all translations within the given range
        best_score = -1
        best_shift = np.array([0, 0, 0])
        shifts = range(-max_shift, max_shift + 1)
        offsets = np.array([
            [-1,0,0], [1,0,0], [0,-1,0], [0,1,0], [0,0,-1], [0,0,1]
        ])

        for dx in shifts:
            for dy in shifts:
                for dz in shifts:
                    shift = np.array([dx, dy, dz])
                    shifted_coords = feature_mask_coords_to_update + shift

                    # Ensure coordinates are within bounds
                    valid_mask = np.all((shifted_coords >= 0) & (shifted_coords < feature_mask.shape), axis=1)
                    shifted_coords_valid = shifted_coords[valid_mask]

                    # Vectorized neighbor calculation
                    if shifted_coords_valid.shape[0] == 0:
                        continue
                    neighbors = shifted_coords_valid[:, None, :] + offsets[None, :, :]
                    neighbors = neighbors.reshape(-1, 3)
                    # Remove out-of-bounds neighbors
                    neighbors_valid = neighbors[
                        np.all((neighbors >= 0) & (neighbors < feature_mask.shape), axis=1)
                    ]
                    # Convert to tuples for set lookup
                    neighbor_tuples = set(map(tuple, neighbors_valid))
                    score = len(neighbor_tuples & brain_surface_set)
                    if score > best_score:
                        best_score = score
                        best_shift = shift

        return best_shift