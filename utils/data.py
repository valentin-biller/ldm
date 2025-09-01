import random
import numpy as np
from pathlib import Path

import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split

import nibabel as nib
from monai import transforms


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

class DataModule(pl.LightningDataModule):
    """
    PyTorch Lightning data module for brain MRI data
    """
    
    def __init__(
        self,
        debug=False,
        mode='training',  # training, inpainting_inference, inpainting_inference_conditioning, inpainting_inference_challenge, baseline
        oversampling=True,
        dir_data=None,
        dir_data_challenge=None,
        dir_output_model=None,
        dir_utils=None,
        latent_shape=(60, 60, 40),
        batch_size=2,
        num_workers=4,
        train_val_split=0.8,
        **kwargs
    ):
        super().__init__()
        self.debug = debug
        self.mode = mode
        self.oversampling = oversampling
        self.dir_data = None if dir_data is None else Path(dir_data)
        self.dir_data_challenge = None if dir_data_challenge is None else Path(dir_data_challenge)
        self.dir_output_model = None if dir_output_model is None else Path(dir_output_model)
        self.dir_utils = Path(__file__).resolve().parent if dir_utils is None else Path(dir_utils)
        self.latent_shape = latent_shape
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_split = train_val_split

        assert self.mode in ['training', 'inpainting_inference', 'inpainting_inference_conditioning', 'inpainting_inference_challenge', 'baseline'], f"Invalid mode: {self.mode}."
    
        self.print_length = 25
        pl.seed_everything(42)
    
    def setup(self, stage=None): 

        if self.dir_data_challenge is None:
            patients_challenge = []
            path_patients_challenge_identifier = str(self.dir_utils / '.patients_challenge_identifier.txt')
            with open(path_patients_challenge_identifier, 'r') as f:
                patients_challenge_identifier = set([line.strip() for line in f.readlines()])
        else:
            patients_challenge = []
            if self.dir_data_challenge.exists():
                for folder in sorted(self.dir_data_challenge.iterdir()):
                    if folder.is_dir():
                        patient = folder.name
                        path_mask = folder / f"{patient}-mask.nii.gz"
                        path_t1_voided = folder / f"{patient}-t1n-voided.nii.gz"
                        if path_mask.exists() and path_t1_voided.exists():
                            patients_challenge.append(patient)
            patients_challenge_identifier = set([folder.split('-')[2] for folder in patients_challenge])
            if self.mode == 'training':
                with open(str(self.dir_utils / '.patients_challenge_identifier.txt'), "w") as f:
                    for patient in sorted(patients_challenge_identifier):
                        f.write(f"{patient}\n")

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

        # Only do inference for the patients that haven't been inferred
        if self.dir_output_model is not None:
            if self.mode in ['inpainting_inference', 'inpainting_inference_conditioning']:
                dir_output = self.dir_output_model / 'pixel_injection'
            elif self.mode == 'inpainting_inference_challenge':
                dir_output = self.dir_output_model
            if dir_output.exists():
                files_completed = list(dir_output.iterdir())
                if self.mode in ['inpainting_inference', 'inpainting_inference_conditioning']:
                        patient_masks = {}
                        for file in files_completed:
                            name = file.name
                            patient_mask = name[:-7]
                            patient = patient_mask[:-5]
                            mask = patient_mask[-4:]
                            patient_masks.setdefault(patient, set()).add(mask)
                        # patients_completed = [patient for patient, masks in patient_masks.items() if {'0000', '0001', '0002'}.issubset(masks)]
                        patients_completed = [patient for patient, masks in patient_masks.items() if '0000' in masks]
                        patients_val = [patient for patient in patients_val if patient not in patients_completed]
                        self._print_numbers('Completed', patients_completed)
                elif self.mode == 'inpainting_inference_challenge':
                    patients_completed = ['-'.join(file.name.split('-')[:-2]) for file in files_completed]
                    patients_challenge = [patient for patient in patients_challenge if patient not in patients_completed]
                    self._print_numbers('Completed', patients_completed)

        print(self.print_length * '=')
        if self.mode == 'inpainting_inference_challenge':
            self._print_numbers('Challenge', patients_challenge)
        else:
            self._print_numbers('Total', patients)
            self._print_numbers('Train', patients_train)
            self._print_numbers('Val', patients_val)

        ## Counting Prefixes, Oversampling and Debugging
        if self.mode in ['training', 'inpainting_inference', 'inpainting_inference_conditioning', 'baseline']:
            prefixes = ["900", "BraTS", "egd", "glioma", "hf", "Patient", "tcga", "ucsf", "upenn"]
            groups_train = self._count_prefixes('Train', patients_train, prefixes)
            groups_val = self._count_prefixes('Val', patients_val, prefixes)

            if self.mode in ['training', 'baseline'] and self.oversampling:
                patients_train = self._oversample_prefixes(groups_train)
                groups_train = self._count_prefixes('Train Oversampling', patients_train, prefixes)
                self._print_numbers('Train', patients_train)

            if self.mode in ['training', 'inpainting_inference', 'inpainting_inference_conditioning', 'baseline'] and self.debug:
                sampled = []
                for prefix in prefixes:
                    group = [p for p in patients_val if p.startswith(prefix)]
                    sampled += random.sample(group, min(3, len(group)))
                patients_val = sampled
                groups_val = self._count_prefixes('Val Debug', patients_val, prefixes)
                self._print_numbers('Val', patients_val)

        self.dataset_train = DataSet(
            mode=self.mode,
            dir_data=self.dir_data,
            dir_data_challenge=self.dir_data_challenge,
            latent_shape=self.latent_shape,
            patients=patients_train,
        )
        
        self.dataset_val = DataSet(
            mode=self.mode,
            dir_data=self.dir_data,
            dir_data_challenge=self.dir_data_challenge,
            latent_shape=self.latent_shape,
            patients=patients_val,
        )

        self.dataset_challenge = DataSet(
            mode=self.mode,
            dir_data=self.dir_data,
            dir_data_challenge=self.dir_data_challenge,
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
        if self.mode in ['training', 'inpainting_inference', 'inpainting_inference_conditioning', 'baseline']:
            return self.dataloader_val
        elif self.mode == 'inpainting_inference_challenge':
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
        mode='training',
        dir_data=None,
        dir_data_challenge=None,
        latent_shape=None,
        patients=None,
    ):
        self.mode = mode
        self.dir_data = dir_data
        self.dir_data_challenge = dir_data_challenge

        self.latent_shape = latent_shape
        self.patients = patients
    
        if self.mode in ['training', 'inpainting_inference_challenge', 'baseline']:
            self.samples = self.patients
        elif self.mode in ['inpainting_inference', 'inpainting_inference_conditioning']:
            self.samples = []
            for patient_id in self.patients:
                # Check for mask files 0000, 0001, 0002
                for mask_id in ["0000"]:  # "0001", "0002"
                    self.samples.append((patient_id, mask_id))

        self.intensity = transforms.ScaleIntensity(minv=0.0, maxv=1.0)
        self.autoencoder_pad = transforms.SpatialPad(spatial_size=(240, 240, 160))
        self.autoencoder_crop = transforms.CenterSpatialCrop(roi_size=(240, 240, 155))

    def _get_affine(self, file_path):
        img = nib.load(file_path)
        return img.affine
    def _get_data(self, file_path):
        img = nib.load(file_path)
        return torch.as_tensor(img.get_fdata())
    
    def _get_file_modality(self, patient, modality):
        return self.dir_data / patient / f"{modality}.nii.gz"
    def _get_file_growth_model(self, patient):
        return self.dir_data / patient / 'processed' / 'growth_model.nii.gz'
    def _get_file_tissue_segmentation(self, patient):
        return self.dir_data / patient / 'processed' / 'tissue_segmentation.nii.gz'
    def _get_file_modality_voided(self, patient, mask, modality):
        return self.dir_data / patient / 'voided' / f"{modality}-voided-{mask}.nii.gz"
    def _get_file_mask(self, patient, mask, healthy=False):
        if healthy:
            return self.dir_data / patient / 'masks' / f"mask-healthy-{mask}.nii.gz"
        else:
            return self.dir_data / patient / 'masks' / f"mask-{mask}.nii.gz"
        
    def _process_mask(self, data):
        return torch.as_tensor(data).unsqueeze(0)  # 1, 240, 240, 155
    def _process_original_modality(self, data):
        return torch.as_tensor(data).unsqueeze(0)  # 1, 240, 240, 155
    def _process_normalized_modality(self, data):
        normalized_modality = self.intensity(data)
        return torch.as_tensor(normalized_modality).unsqueeze(0)  # 1, 240, 240, 155
    def _process_autoencoder_modality(self, normalized_modality):
        autoencoder_modality = self.autoencoder_pad(normalized_modality)
        return torch.as_tensor(autoencoder_modality)  # 1, 240, 240, 160
    def _process_interpolation(self, original):
        latent = torch.nn.functional.interpolate(
            original.unsqueeze(0),
            size=(self.latent_shape[0], self.latent_shape[1], self.latent_shape[2]),
            mode='nearest'
        )[0]
        return latent  # 4, 60, 60, 40

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
    def _collect_original_modalities(self, data_modalities):
        original_t1 = self._process_original_modality(data_modalities['data_t1'])  # 1, 240, 240, 155
        original_t1c = self._process_original_modality(data_modalities['data_t1c'])  # 1, 240, 240, 155
        original_t2 = self._process_original_modality(data_modalities['data_t2'])  # 1, 240, 240, 155
        original_flair = self._process_original_modality(data_modalities['data_flair'])  # 1, 240, 240, 155
        return {
            'original_t1': original_t1.float(),
            'original_t1c': original_t1c.float(),
            'original_t2': original_t2.float(),
            'original_flair': original_flair.float()
        }
    def _collect_normalized_modalities(self, data_modalities):
        normalized_t1 = self._process_normalized_modality(data_modalities['data_t1'])  # 1, 240, 240, 155
        normalized_t1c = self._process_normalized_modality(data_modalities['data_t1c'])  # 1, 240, 240, 155
        normalized_t2 = self._process_normalized_modality(data_modalities['data_t2'])  # 1, 240, 240, 155
        normalized_flair = self._process_normalized_modality(data_modalities['data_flair'])  # 1, 240, 240, 155
        return {
            'normalized_t1': normalized_t1.float(),
            'normalized_t1c': normalized_t1c.float(),
            'normalized_t2': normalized_t2.float(),
            'normalized_flair': normalized_flair.float()
        }
    def _collect_autoencoder_modalities(self, normalized_modalities):
        autoencoder_t1 = self._process_autoencoder_modality(normalized_modalities['normalized_t1'])  # 1, 240, 240, 160
        autoencoder_t1c = self._process_autoencoder_modality(normalized_modalities['normalized_t1c'])  # 1, 240, 240, 160
        autoencoder_t2 = self._process_autoencoder_modality(normalized_modalities['normalized_t2'])  # 1, 240, 240, 160
        autoencoder_flair = self._process_autoencoder_modality(normalized_modalities['normalized_flair'])  # 1, 240, 240, 160
        return {
            'autoencoder_t1': autoencoder_t1.float(),
            'autoencoder_t1c': autoencoder_t1c.float(),
            'autoencoder_t2': autoencoder_t2.float(),
            'autoencoder_flair': autoencoder_flair.float()
        }

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        if self.mode == 'training':
            patient = self.samples[idx]

            affine = self._get_affine(self._get_file_modality(patient, 't1'))

            # data_t1 = self._get_data(self._get_file_modality(patient, 't1'))
            # original_t1 = self._process_original_modality(data_t1)  # 1, 240, 240, 155
            # normalized_t1 = self._process_normalized_modality(data_t1)  # 1, 240, 240, 155
            # autoencoder_t1 = self._process_autoencoder_modality(normalized_t1)  # 1, 240, 240, 160

            data_modalities = self._collect_data_modalities(patient)
            original_modalities = self._collect_original_modalities(data_modalities)
            normalized_modalities = self._collect_normalized_modalities(data_modalities)
            autoencoder_modalities = self._collect_autoencoder_modalities(normalized_modalities)

            data_growth_model = self._get_data(self._get_file_growth_model(patient))
            data_tissue_segmentation = self._get_data(self._get_file_tissue_segmentation(patient))
            original_conditioning = create_conditioning(data_growth_model, data_tissue_segmentation)  # 4, 240, 240, 155
            latent_conditioning = self._process_interpolation(original_conditioning)  # 4, 60, 60, 40

            return {
                'mode': self.mode,
                'patient': patient,
                'affine': affine,
                # 'original_t1': original_t1.float(),
                # 'normalized_t1': normalized_t1.float(),
                # 'autoencoder_t1': autoencoder_t1.float(),
                'original_modalities': original_modalities,
                'normalized_modalities': normalized_modalities,
                'autoencoder_modalities': autoencoder_modalities,
                'latent_conditioning': latent_conditioning.float()
            }

        elif self.mode in ['inpainting_inference', 'inpainting_inference_conditioning']:
            patient, mask = self.samples[idx]

            affine = self._get_affine(self._get_file_modality(patient, 't1'))

            data_t1 = self._get_data(self._get_file_modality(patient, 't1'))
            original_t1 = self._process_original_modality(data_t1)  # 1, 240, 240, 155

            data_t1_voided = self._get_data(self._get_file_modality_voided(patient, mask, 't1'))
            original_t1_voided = self._process_original_modality(data_t1_voided)  # 1, 240, 240, 155
            normalized_t1_voided = self._process_normalized_modality(data_t1_voided)  # 1, 240, 240, 155
            autoencoder_t1_voided = self._process_autoencoder_modality(normalized_t1_voided)  # 1, 240, 240, 160
            
            original_mask = self._process_mask(self._get_data(self._get_file_mask(patient, mask)))  # 1, 240, 240, 155
            latent_mask = self._process_interpolation(original_mask)  # 1, 60, 60, 40
            
            original_tissue_segmentation = self._get_data(self._get_file_tissue_segmentation(patient))  # 240, 240, 155
            if self.mode == 'inpainting_inference':
                original_growth_model = torch.zeros(240, 240, 155)
            elif self.mode == 'inpainting_inference_conditioning':
                original_growth_model = self._get_data(self._get_file_growth_model(patient))  # 240, 240, 155
            original_conditioning = create_conditioning(original_growth_model, original_tissue_segmentation)  # 4, 240, 240, 155
            latent_conditioning = self._process_interpolation(original_conditioning)  # 4, 60, 60, 40

            return {
                'mode': self.mode,
                'patient': patient,
                'mask': mask,
                'affine': affine,

                'original_t1': original_t1.float(),
                'original_t1_voided': original_t1_voided.float(),
                'normalized_t1_voided': normalized_t1_voided.float(),
                'autoencoder_t1_voided': autoencoder_t1_voided.float(),

                'original_mask': original_mask.float(),
                'latent_mask': latent_mask.float(),

                'original_conditioning': original_conditioning.float(),
                'latent_conditioning': latent_conditioning.float(),
            }            

        elif self.mode == 'inpainting_inference_challenge':
            patient = self.samples[idx]

            path_data_challenge_t1_voided = self.dir_data_challenge / patient / f"{patient}-t1n-voided.nii.gz"
            path_data_challenge_mask = self.dir_data_challenge / patient / f"{patient}-mask.nii.gz"

            affine = self._get_affine(path_data_challenge_t1_voided)

            data_t1_voided = self._get_data(path_data_challenge_t1_voided)
            original_t1_voided = self._process_original_modality(data_t1_voided)  # 1, 240, 240, 155
            normalized_t1_voided = self._process_normalized_modality(data_t1_voided)  # 1, 240, 240, 155
            autoencoder_t1_voided = self._process_autoencoder_modality(normalized_t1_voided)  # 1, 240, 240, 160

            original_mask = self._process_mask(self._get_data(path_data_challenge_mask))  # 1, 240, 240, 155
            latent_mask = self._process_interpolation(original_mask)  # 1, 60, 60, 40

            path_original_conditioning = self.dir_data_challenge / patient / 'conditioning.pt'
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
                'normalized_t1_voided': normalized_t1_voided.float(),
                'autoencoder_t1_voided': autoencoder_t1_voided.float(),
                
                'original_mask': original_mask.float(),
                'latent_mask': latent_mask.float(),

                'original_conditioning': original_conditioning.float(),
                'latent_conditioning': latent_conditioning.float(),
                'exists_conditioning': exists_conditioning,

                'path_original_t1_voided': str(path_data_challenge_t1_voided),
                'path_original_mask': str(path_data_challenge_mask),
            }

        elif self.mode == 'baseline':
            self.baseline_shape = [128, 128, 64]

            patient = self.samples[idx]

            data_growth_model = self._get_data(self._get_file_growth_model(patient))
            data_tissue_segmentation = self._get_data(self._get_file_tissue_segmentation(patient))
            
            baseline_slices = self._get_baseline_slices(data_growth_model)
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
                'baseline_conditioning': baseline_conditioning.float(),
                'baseline_modalities': baseline_modalities.float()
            }

    def _get_baseline_slices(self, data_growth_model, threshold=0.001):
        baseline_center = np.mean(np.argwhere(data_growth_model.numpy() >= threshold), axis=0).astype(int)
        start = np.clip(baseline_center - np.array(self.baseline_shape) // 2, 0, np.array(data_growth_model.shape) - np.array(self.baseline_shape))
        end = start + np.array(self.baseline_shape)
        baseline_slices = [slice(start[0], end[0]), slice(start[1], end[1]), slice(start[2], end[2])]
        return baseline_slices
    def _get_baseline_affine(self, affine, baseline_slices):
        R = affine[:3, :3]
        start = np.array([baseline_slices[0].start, baseline_slices[1].start, baseline_slices[2].start], float)
        new_affine = affine.copy()
        new_affine[:3, 3] = affine[:3, 3] + R @ start
        return new_affine
    def _process_baseline_conditioning(self, data_growth_model, data_tissue_segmentation, baseline_slices):
        baseline_growth_model = data_growth_model[baseline_slices]
        baseline_tissue_segmentation = data_tissue_segmentation[baseline_slices]
        baseline_conditioning = create_conditioning(baseline_growth_model, baseline_tissue_segmentation)
        assert baseline_conditioning.shape == (4, *self.baseline_shape)
        return baseline_conditioning
    def _process_baseline_modality(self, normalized_modality, baseline_slices):  
        baseline_modality = normalized_modality.squeeze(0)[baseline_slices].unsqueeze(0)
        assert baseline_modality.shape == (1, *self.baseline_shape)
        return torch.as_tensor(baseline_modality)
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
