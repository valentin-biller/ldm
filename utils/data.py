import time
import random
import numpy as np
from pathlib import Path

import torch
import lightning.pytorch as L
from torch.utils.data import Dataset, DataLoader, random_split

import nibabel as nib
from monai import transforms


MODALITIES = ['t1', 't1c', 't2', 'flair']


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


def get_baseline_slices(data_growth_model, baseline_shape, threshold=0.001):
    baseline_center = np.mean(np.argwhere(data_growth_model.numpy() >= threshold), axis=0).astype(int)
    start = np.clip(baseline_center - np.array(baseline_shape) // 2, 0, np.array(data_growth_model.shape) - np.array(baseline_shape))
    end = start + np.array(baseline_shape)
    baseline_slices = [slice(start[0], end[0]), slice(start[1], end[1]), slice(start[2], end[2])]
    return baseline_slices


class DataModule(L.LightningDataModule):
    """
    PyTorch Lightning data module for brain MRI data
    """
    
    def __init__(
        self,
        dir_data=None,
        dir_utils=None,
        dir_output_model=None,
        use_latents=True,
        mask_conditioning=64,  # 64, 32, scalar, None
        modality_conditioning=True,  # True, False
        latent_shape=None,
        mode='training',
        oversampling=True,
        undersampling=False,
        batch_size=16,
        num_workers=16,
        train_val_split=0.8,
        **kwargs
    ):
        super().__init__()

        self.dir_data = None if dir_data is None else Path(dir_data)
        self.dir_utils = Path(__file__).resolve().parent if dir_utils is None else Path(dir_utils)
        self.dir_output_model = None if dir_output_model is None else Path(dir_output_model)
        
        self.use_latents = use_latents
        self.mask_conditioning = mask_conditioning
        self.modality_conditioning = modality_conditioning

        self.mode = mode
        self.oversampling = oversampling
        self.undersampling = undersampling

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.latent_shape = latent_shape
        self.train_val_split = train_val_split

        assert self.mode in ['autoencoder', 'training', 'baseline', 'inpainting_healthy_tissue', 'inpainting_tumorous_tissue', 'inpainting_spatio_temporal'], f"Invalid mode: {self.mode}."
    
        self.print_length = 25
        L.seed_everything(42)
    
    def setup(self, stage=None): 
        path_patients_challenge_identifier = str(self.dir_utils / '.patients_challenge_identifier.txt')
        with open(path_patients_challenge_identifier, 'r') as f:
            patients_challenge_identifier = set([line.strip() for line in f.readlines()])

        # Splitting
        '''
        patients = []
        for folder in sorted(os.listdir(self.dir_data)):
            path_growth_model = os.path.join(self.dir_data, folder, 'processed', 'growth_model.nii.gz')
            path_tissue_segmentation = os.path.join(self.dir_data, folder, 'processed', 'tissue_segmentation.nii.gz')
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

        with open(str(self.dir_utils / '.patients_train.txt'), "w") as f:
            for patient in sorted(patients_train):
                f.write(f"{patient}\n")

        with open(str(self.dir_utils / '.patients_val.txt'), "w") as f:
            for patient in sorted(patients_val):
                f.write(f"{patient}\n")
        '''

        with open(str(self.dir_utils / '.patients_train.txt'), "r") as f:
            patients_train = [line.strip() for line in f if line.strip()]

        with open(str(self.dir_utils / '.patients_val.txt'), "r") as f:
            patients_val = [line.strip() for line in f if line.strip()]

        patients = patients_train + patients_val

        # # Only do inference for the patients that haven't been inferred
        # if self.mode in ['inpainting_healthy_tissue', 'inpainting_tumorous_tissue'] and self.dir_output_model is not None:
        #     dir_output = self.dir_output_model / 'pixel_injection'
        #     if dir_output.exists():
        #         files_completed = list(dir_output.iterdir())
        #         patient_masks = {}
        #         for file in files_completed:
        #             name = file.name
        #             patient_mask = name[:-7]
        #             patient = patient_mask[:-5]
        #             mask = patient_mask[-4:]
        #             patient_masks.setdefault(patient, set()).add(mask)
        #         # patients_completed = [patient for patient, masks in patient_masks.items() if {'0000', '0001', '0002'}.issubset(masks)]
        #         patients_completed = [patient for patient, masks in patient_masks.items() if '0000' in masks]
        #         patients_val = [patient for patient in patients_val if patient not in patients_completed]
        #         self._print_numbers('Completed', patients_completed)

        # Summary
        print(self.print_length * '=')
        self._print_numbers('Total', patients)
        self._print_numbers('Train', patients_train)
        self._print_numbers('Val', patients_val)

        # Counting Prefixes
        prefixes = ["900", "BraTS", "egd", "glioma", "hf", "Patient", "tcga", "ucsf", "upenn"]
        groups_train = self._count_prefixes('Train', patients_train, prefixes)
        groups_val = self._count_prefixes('Val', patients_val, prefixes)

        # Oversampling
        if self.oversampling and self.mode in ['autoencoder', 'training', 'baseline']:
            patients_train = self._oversample_prefixes(groups_train)
            groups_train = self._count_prefixes('Train Oversampling', patients_train, prefixes)
            self._print_numbers('Train', patients_train)

        # Undersampling
        if self.undersampling: 
            sampled = []
            for prefix in prefixes:
                group = [p for p in patients_val if p.startswith(prefix)]
                sampled += random.sample(group, min(1, len(group)))
            patients_val = sampled
            groups_val = self._count_prefixes('Val Undersampling', patients_val, prefixes)
            self._print_numbers('Val', patients_val)
              
        self.dataset_train = DataSet(
            dir_data=self.dir_data,
            use_latents=self.use_latents,
            mask_conditioning=self.mask_conditioning,
            modality_conditioning=self.modality_conditioning,
            latent_shape=self.latent_shape,
            mode=self.mode,
            patients=patients_train,
        )
        
        self.dataset_val = DataSet(
            dir_data=self.dir_data,
            use_latents=self.use_latents,
            mask_conditioning=self.mask_conditioning,
            modality_conditioning=self.modality_conditioning,
            latent_shape=self.latent_shape,
            mode=self.mode if self.mode != 'training' else 'validation',
            patients=patients_val,
        )

        self.dataloader_train = DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            persistent_workers=True,  # prefetch_factor: no speed improvements / degradations (if any, then degradations)
        )
        
        self.dataloader_val = DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True,  # prefetch_factor: no speed improvements / degradations (if any, then degradations)
        )

    def train_dataloader(self):
        return self.dataloader_train
    
    def val_dataloader(self):
        return self.dataloader_val
    
    def test_dataloader(self):
        return self.dataloader_val
        
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
            print(f"{prefix:<10}{count:>5}{(count / len(patients) * 100) if len(patients) > 0 else 0:9.2f}%")
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
    """
    
    def __init__(
        self,
        dir_data=None,
        use_latents=True,
        mask_conditioning=64,  # 64, 32, scalar, None
        modality_conditioning=True,  # True, False
        latent_shape=None,
        mode='training',
        patients=None,
    ):
        self.dir_data = dir_data

        self.use_latents = use_latents
        self.mask_conditioning = mask_conditioning
        self.modality_conditioning = modality_conditioning

        self.mode = mode
        self.patients = patients
        self.latent_shape = latent_shape
        self.latent_shape_string = f"{latent_shape[1]}_{latent_shape[2]}_{latent_shape[3]}" if latent_shape is not None else None
    
        if self.mode in ['autoencoder']:
            self.samples = []
            for patient_id in self.patients:
                for modality in MODALITIES:
                    self.samples.append((patient_id, modality))
        elif self.mode in ['training', 'validation']:
            self.samples = []
            modalities = MODALITIES if self.modality_conditioning else ['t1']
            for patient_id in self.patients:
                for modality in modalities:
                    self.samples.append((patient_id, modality))
        elif self.mode in ['baseline']:
            self.samples = self.patients
        elif self.mode in ['inpainting_healthy_tissue', 'inpainting_tumorous_tissue', 'inpainting_spatio_temporal']:
            self.samples = []
            modalities = MODALITIES if self.modality_conditioning else ['t1']
            for patient_id in self.patients:
                for modality in modalities:
                    # Check for mask files 0000, 0001, 0002
                    for mask_id in ["0000"]:  # "0001", "0002"
                        self.samples.append((patient_id, modality, mask_id))

        if self.mode in ['autoencoder']:
            self.intensity = transforms.ScaleIntensity(minv=0.0, maxv=1.0)
        else:
            if self.latent_shape == (4, 64, 64, 40):
                self.intensity = transforms.ScaleIntensity(minv=0.0, maxv=1.0)
            elif self.latent_shape == (4, 32, 32, 20):
                self.intensity = transforms.ScaleIntensity(minv=-1.0, maxv=1.0)
            else:
                self.intensity = transforms.ScaleIntensity(minv=0.0, maxv=1.0)
        self.autoencoder_pad = transforms.SpatialPad(spatial_size=(256, 256, 160))
        self.autoencoder_crop = transforms.CenterSpatialCrop(roi_size=(240, 240, 155))

    # loading with nibabel
    def _nib_load(self, file_path):
        retries = 10
        delay = 3
        exception = None
        for i in range(retries):
            try:
                return nib.load(file_path)
            except Exception as e:
                exception = e
                print(f"[WARNING]: {e}. Retrying ({i+1}/{retries}) in {delay}s...")
                time.sleep(delay)
        raise exception
    def _get_affine(self, file_path):
        img = self._nib_load(file_path)
        return img.affine
    def _get_data(self, file_path):
        img = self._nib_load(file_path)
        return torch.as_tensor(img.get_fdata())
    
    # get files default
    def _get_file_modality(self, patient, modality):
        return self.dir_data / patient / f"{modality}.nii.gz"
    def _get_file_growth_model(self, patient):
        return self.dir_data / patient / 'processed' / 'growth_model.nii.gz'
    def _get_file_tissue_segmentation(self, patient):
        return self.dir_data / patient / 'processed' / 'tissue_segmentation.nii.gz'
    def _get_file_tumor_segmentation(self, patient):
        return self.dir_data / patient / 'processed' / 'tumor_segmentation.nii.gz'
    # get files latents
    def _get_file_latent_modality(self, patient, modality):
        return self.dir_data / patient / f'latents_{self.latent_shape_string}' / f'latent_{modality}.pt'
    def _get_file_latent_conditioning(self, patient):
        return self.dir_data / patient / f'latents_{self.latent_shape_string}' / f'latent_conditioning.pt'
    # get files inpainting
    def _get_file_modality_voided(self, patient, modality, mask):
        return self.dir_data / patient / 'voided' / f"{modality}-voided-{mask}.nii.gz"
    def _get_file_mask(self, patient, mask, healthy=False):
        if healthy:
            return self.dir_data / patient / 'masks' / f"mask-healthy-{mask}.nii.gz"
        else:
            return self.dir_data / patient / 'masks' / f"mask-{mask}.nii.gz"
        
    # processing
    def _to_torch(self, data):
        return torch.as_tensor(data).unsqueeze(0)  # 1, 240, 240, 155
    def _load_torch(self, path_latent):
        return torch.load(path_latent, map_location="cpu").squeeze(0)
    def _normalize(self, data):
        normalized = self.intensity(data)
        return torch.as_tensor(normalized).unsqueeze(0)  # 1, 240, 240, 155
    def _pad(self, data):
        padded = self.autoencoder_pad(data)
        return torch.as_tensor(padded)  # 1, 256, 256, 160
    def _process_interpolation(self, original, mode='nearest'):
        latent = torch.nn.functional.interpolate(
            original.unsqueeze(0),
            size=(self.latent_shape[1], self.latent_shape[2], self.latent_shape[3]),
            mode=mode
        )[0]
        return latent

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        if self.mode == 'autoencoder':
            patient, modality = self.samples[idx]
            affine = self._get_affine(self._get_file_modality(patient, modality))
            normalized_modality = self._normalize(self._get_data(self._get_file_modality(patient, modality)))  # 1, 240, 240, 155   
            padded_modality = self._pad(normalized_modality)  # 1, 256, 256, 256

            return {
                'patient': patient,
                    
                'affine': affine,
                'normalized_modality': normalized_modality.float(),

                'modality': modality,
                'padded_modality': padded_modality.float(),
            }

        elif self.mode == 'training':
            patient, modality = self.samples[idx]

            if self.use_latents:
                latent_modality = self._load_torch(self._get_file_latent_modality(patient, modality))  # 4, 64, 64, 40 or 4, 32, 32, 20
                return_ = {
                    'modality': modality,
                    'latent_modality': latent_modality.float(),
                }

                if self.mask_conditioning is not None:
                    conditioning = self._load_torch(self._get_file_latent_conditioning(patient))  # 8, 64, 64, 40 or 8, 32, 32, 20
                    
                    if self.mask_conditioning == 'scalar':
                        data_tumor_segmentation = self._get_data(self._get_file_tumor_segmentation(patient))
                        scalar_tumor_segmentation = (data_tumor_segmentation > 0).sum().item() / data_tumor_segmentation.numel()
                        conditioning[:4, :, :, :] = scalar_tumor_segmentation

                    return_['conditioning'] = conditioning.float()

                return return_
            else:
                raise NotImplementedError("Training without latents is not implemented.")
        
        elif self.mode == 'validation':
            patient, modality = self.samples[idx]

            affine = self._get_affine(self._get_file_modality(patient, modality))

            if self.use_latents:
                normalized_modality = self._normalize(self._get_data(self._get_file_modality(patient, modality)))  # 1, 240, 240, 155                
                latent_modality = self._load_torch(self._get_file_latent_modality(patient, modality))  # 4, 64, 64, 40 or 4, 32, 32, 20
                
                return_ = {
                    'mode': self.mode,
                    'patient': patient,
                    
                    'affine': affine,
                    'normalized_modality': normalized_modality.float(),

                    'modality': modality,
                    'latent_modality': latent_modality.float(),
                }

                if self.mask_conditioning is not None:
                    conditioning = self._load_torch(self._get_file_latent_conditioning(patient))  # 8, 64, 64, 40 or 8, 32, 32, 20
                    
                    if self.mask_conditioning == 'scalar':
                        data_tumor_segmentation = self._get_data(self._get_file_tumor_segmentation(patient))
                        scalar_tumor_segmentation = (data_tumor_segmentation > 0).sum().item() / data_tumor_segmentation.numel()
                        conditioning[:4, :, :, :] = scalar_tumor_segmentation

                    return_['conditioning'] = conditioning.float()

                return return_
            else:
                raise NotImplementedError("Validation without latents is not implemented.")

        elif self.mode in ['inpainting_healthy_tissue', 'inpainting_tumorous_tissue', 'inpainting_spatio_temporal']:            
            patient, modality, mask = self.samples[idx]

            affine = self._get_affine(self._get_file_modality(patient, modality))

            normalized_modality = self._normalize(self._get_data(self._get_file_modality(patient, modality)))  # 1, 240, 240, 155                
            latent_modality = self._load_torch(self._get_file_latent_modality(patient, modality))  # 4, 64, 64, 40 or 4, 32, 32, 20

            # always use combined mask -> only for evaluation relevant!
            original_mask = self._to_torch(self._get_data(self._get_file_mask(patient, mask)))  # 1, 240, 240, 155
            latent_mask = self._process_interpolation(original_mask)  # 1, 64, 64, 40

            conditioning = self._load_torch(self._get_file_latent_conditioning(patient))  # 8, 64, 64, 40 or 8, 32, 32, 20
            if self.mode == 'inpainting_healthy_tissue':
                assert conditioning.shape[0] == 8
                conditioning[:4, :, :, :] = torch.zeros_like(conditioning[:4, :, :, :])
            
            return {
                'mode': self.mode,
                'patient': patient,
                
                'affine': affine,
                'normalized_modality': normalized_modality.float(),

                'modality': modality,
                'latent_modality': latent_modality.float(),

                'mask': mask,
                'original_mask': original_mask.float(),
                'latent_mask': latent_mask.float(),

                'conditioning': conditioning.float(),
            }

        elif self.mode == 'baseline':
            self.baseline_shape = [128, 128, 64]

            patient = self.samples[idx]

            data_growth_model = self._get_data(self._get_file_growth_model(patient))
            data_tissue_segmentation = self._get_data(self._get_file_tissue_segmentation(patient))
            original_conditioning = create_conditioning(data_growth_model, data_tissue_segmentation)  # 4, 240, 240, 155

            baseline_slices = get_baseline_slices(data_growth_model, self.baseline_shape)
            baseline_conditioning = self._process_baseline_conditioning(data_growth_model, data_tissue_segmentation, baseline_slices)

            data_modalities = self._collect_data_modalities(patient)
            normalized_modalities = self._collect_normalized_modalities(data_modalities)
            baseline_modalities = self._collect_baseline_modalities(normalized_modalities, baseline_slices)
            baseline_modalities = torch.cat([baseline_modalities[key] for key in baseline_modalities.keys()], dim=0)

            affine = self._get_affine(self._get_file_modality(patient, 't1'))
            affine = self._get_baseline_affine(affine, baseline_slices)

            return {
                'patient': patient,
                'affine': affine,
                'original_conditioning': original_conditioning.float(),
                'baseline_conditioning': baseline_conditioning.float(),
                'baseline_modalities': baseline_modalities.float(),
                # 'baseline_slices': torch.as_tensor([(s.start, s.stop) for s in baseline_slices])
            }

    def _get_baseline_affine(self, affine, baseline_slices):
        R = affine[:3, :3]
        start = np.array([baseline_slices[0].start, baseline_slices[1].start, baseline_slices[2].start], float)
        new_affine = affine.copy()
        new_affine[:3, 3] = affine[:3, 3] + R @ start
        return new_affine
    def _process_baseline_modality(self, normalized_modality, baseline_slices):  
        baseline_modality = normalized_modality.squeeze(0)[baseline_slices].unsqueeze(0)
        assert baseline_modality.shape == (1, *self.baseline_shape)
        return torch.as_tensor(baseline_modality)
    def _process_baseline_conditioning(self, data_growth_model, data_tissue_segmentation, baseline_slices):
        baseline_growth_model = data_growth_model[baseline_slices]
        baseline_tissue_segmentation = data_tissue_segmentation[baseline_slices]
        baseline_conditioning = create_conditioning(baseline_growth_model, baseline_tissue_segmentation)
        assert baseline_conditioning.shape == (4, *self.baseline_shape)
        return baseline_conditioning
    def _collect_data_modalities(self, patient):
        data_t1 = self._get_data(self._get_file_modality(patient, 't1'))
        data_t1c = self._get_data(self._get_file_modality(patient, 't1c'))
        data_t2 = self._get_data(self._get_file_modality(patient, 't2'))
        data_flair = self._get_data(self._get_file_modality(patient, 'flair'))
        return {
            'data_t1': data_t1.float(),
            'data_t1c': data_t1c.float(),
            'data_t2': data_t2.float(),
            'data_flair': data_flair.float()
        }
    def _collect_normalized_modalities(self, data_modalities):
        normalized_t1 = self._normalize(data_modalities['data_t1'])  # 1, 240, 240, 155
        normalized_t1c = self._normalize(data_modalities['data_t1c'])  # 1, 240, 240, 155
        normalized_t2 = self._normalize(data_modalities['data_t2'])  # 1, 240, 240, 155
        normalized_flair = self._normalize(data_modalities['data_flair'])  # 1, 240, 240, 155
        return {
            'normalized_t1': normalized_t1.float(),
            'normalized_t1c': normalized_t1c.float(),
            'normalized_t2': normalized_t2.float(),
            'normalized_flair': normalized_flair.float()
        }
    def _collect_baseline_modalities(self, normalized_modalities, baseline_crop):
        baseline_t1 = self._process_baseline_modality(normalized_modalities['normalized_t1'], baseline_crop)
        baseline_t1c = self._process_baseline_modality(normalized_modalities['normalized_t1c'], baseline_crop)
        baseline_t2 = self._process_baseline_modality(normalized_modalities['normalized_t2'], baseline_crop)
        baseline_flair = self._process_baseline_modality(normalized_modalities['normalized_flair'], baseline_crop)
        return {
            'baseline_t1': baseline_t1,
            'baseline_t1c': baseline_t1c,
            'baseline_t2': baseline_t2,
            'baseline_flair': baseline_flair
        }
