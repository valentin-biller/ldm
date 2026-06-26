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
VALID_MODES = [
    'autoencoder', 'training', 'validation', 'baseline',
    'med_ddpm_pretrained', 'lumiere', 'lumiere_med_ddpm_pretrained',
    'inpainting_healthy_tissue', 'inpainting_tumorous_tissue', 'inpainting_spatio_temporal',
]
INPAINTING_MODES = ['inpainting_healthy_tissue', 'inpainting_tumorous_tissue', 'inpainting_spatio_temporal']
BASELINE_SHAPE = [128, 128, 64]
MED_DDPM_SHAPE = (None, 192, 192, 144)

# Tumor label mapping
LABEL_NECROTIC = 1
LABEL_EDEMA    = 2
LABEL_ENHANCING = 3


def create_conditioning(growth_model, tissue_segmentation, threshold=0.001):
    """
    Create 4-channel conditioning tensor:
      - Channel 0: Thresholded growth model
      - Channels 1-3: One-hot tissue segmentation
    """
    growth_model = np.where(growth_model >= threshold, growth_model, 0.0)
    tissue_channels = [
        np.where(tissue_segmentation == label, 1.0, 0.0)
        for label in [1, 2, 3]
    ]
    conditioning = np.stack([growth_model, *tissue_channels], axis=0)  # 4, H, W, D
    return torch.as_tensor(conditioning)


def get_baseline_slices(data_growth_model, baseline_shape, threshold=0.001):
    """Compute crop slices centered on the tumor region."""
    center = np.mean(np.argwhere(data_growth_model.numpy() >= threshold), axis=0).astype(int)
    start  = np.clip(center - np.array(baseline_shape) // 2, 0, np.array(data_growth_model.shape) - np.array(baseline_shape))
    end    = start + np.array(baseline_shape)
    return [slice(s, e) for s, e in zip(start, end)]


def process_interpolation(original, latent_shape, mode='nearest'):
    """Interpolate a tensor to the given spatial shape."""
    return torch.nn.functional.interpolate(
        original.unsqueeze(0),
        size=(latent_shape[1], latent_shape[2], latent_shape[3]),
        mode=mode,
    )[0]


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------

class DataModule(L.LightningDataModule):
    """PyTorch Lightning data module for brain MRI data."""

    def __init__(
        self,
        dir_data=None,
        dir_utils=None,
        dir_output_model=None,
        use_latents=True,
        mask_conditioning=64,       # 64 | 32 | 'scalar' | None
        modality_conditioning=True,
        latent_shape=None,
        mode='training',
        oversampling=True,
        undersampling=False,
        batch_size=16,
        num_workers=16,
        train_val_split=0.8,
        **kwargs,
    ):
        super().__init__()

        self.dir_data         = None if dir_data is None else Path(dir_data)
        self.dir_utils        = Path(__file__).resolve().parent if dir_utils is None else Path(dir_utils)
        self.dir_output_model = None if dir_output_model is None else Path(dir_output_model)

        self.use_latents           = use_latents
        self.latent_shape          = latent_shape
        self.mask_conditioning     = mask_conditioning
        self.modality_conditioning = modality_conditioning

        self.mode          = mode
        self.oversampling  = oversampling
        self.undersampling = undersampling

        self.batch_size      = batch_size
        self.num_workers     = num_workers
        self.train_val_split = train_val_split

        self._print_width = 25
        L.seed_everything(42)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    
    def setup(self, stage=None): 
        # Splitting
        '''
        path_patients_challenge_identifier = str(self.dir_utils / '.patients_challenge_identifier.txt')
        with open(path_patients_challenge_identifier, 'r') as f:
            patients_challenge_identifier = set([line.strip() for line in f.readlines()])

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

        patients_train, patients_val = self._load_patient_lists()

        self._print_summary(patients_train, patients_val)

        # Counting Prefixes
        prefixes = ["900", "BraTS", "egd", "glioma", "hf", "Patient", "tcga", "ucsf", "upenn"]
        groups_train = self._count_prefixes('Train', patients_train, prefixes)
        groups_val   = self._count_prefixes('Val',   patients_val,   prefixes)

        # Oversampling
        if self.oversampling and self.mode in ['autoencoder', 'training', 'baseline']:
            patients_train = self._oversample_prefixes(groups_train)
            self._count_prefixes('Train Oversampling', patients_train, prefixes)
            self._print_numbers('Train', patients_train)

        # Undersampling
        if self.undersampling:
            patients_val = self._undersample_prefixes(patients_val, prefixes)
            self._count_prefixes('Val Undersampling', patients_val, prefixes)
            self._print_numbers('Val', patients_val)
              
        # Datasets
        shared_dataset_kwargs = dict(
            dir_data=self.dir_data,
            use_latents=self.use_latents,
            mask_conditioning=self.mask_conditioning,
            modality_conditioning=self.modality_conditioning,
            latent_shape=self.latent_shape,
        )

        self.dataset_train = DataSet(mode=self.mode,                                              patients=patients_train, **shared_dataset_kwargs)
        self.dataset_val   = DataSet(mode=self.mode if self.mode != 'training' else 'validation', patients=patients_val,   **shared_dataset_kwargs)

        # Dataloader
        shared_loader_kwargs = dict(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )  # prefetch_factor: no speed improvements / degradations (if any, then degradations)

        self.dataloader_train = DataLoader(self.dataset_train, shuffle=True,  **shared_loader_kwargs)
        self.dataloader_val   = DataLoader(self.dataset_val,   shuffle=False, **shared_loader_kwargs)

    def _load_patient_lists(self):
        with open(self.dir_utils / '.patients_train.txt') as f:
            patients_train = [l.strip() for l in f if l.strip()]
        with open(self.dir_utils / '.patients_val.txt') as f:
            patients_val = [l.strip() for l in f if l.strip()]
        return patients_train, patients_val

    def _print_summary(self, patients_train, patients_val):
        patients = patients_train + patients_val
        print(self._print_width * '=')
        self._print_numbers('Total', patients)
        self._print_numbers('Train', patients_train)
        self._print_numbers('Val',   patients_val)

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------

    def train_dataloader(self): return self.dataloader_train
    def val_dataloader(self):   return self.dataloader_val
    def test_dataloader(self):  return self.dataloader_val
    
        
    def _print_numbers(self, identifier, patients):
        length = self._print_width - 5
        print(f"{'Patients ' + identifier + ':':<{length}}{len(patients):>5}")

    def _count_prefixes(self, identifier, patients, prefixes):
        groups = {prefix: [] for prefix in prefixes}
        for patient in patients:
            for prefix in prefixes:
                if patient.startswith(prefix):
                    groups[prefix].append(patient)
                    break
        identifier_str = str(identifier)
        side = (self._print_width - len(identifier_str) - 2) // 2
        extra = (self._print_width - len(identifier_str) - 2) % 2
        print(f"{'=' * (side + extra)} {identifier_str} {'=' * side}")
        for prefix in prefixes:
            count = len(groups[prefix])
            print(f"{prefix:<10}{count:>5}{(count / len(patients) * 100) if len(patients) > 0 else 0:9.2f}%")
        return groups

    def _oversample_prefixes(self, groups):
        max_size = max(len(g) for g in groups.values())
        return [
            patient
            for group in groups.values() if group
            for patient in random.choices(group, k=max_size)
        ]

    def _undersample_prefixes(self, patients, prefixes):
        sampled = []
        for prefix in prefixes:
            group = [p for p in patients if p.startswith(prefix)]
            sampled += random.sample(group, min(1, len(group)))
        return sampled


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DataSet(Dataset):
    """
    Dataset for brain MRI with conditioning.

    Modes
    -----
    autoencoder                           : raw modality + padded volume for AE training
    training / validation                 : latents + conditioning for LGM
    baseline                              : 128³ cropped patches
    med_ddpm_pretrained                   : resized to 192³ for med-ddpm
    lumiere / lumiere_med_ddpm_pretrained : longitudinal data variants
    inpainting_*                          : masked inputs for inpainting tasks
    """

    def __init__(
        self,
        dir_data=None,
        use_latents=True,
        mask_conditioning=64,
        modality_conditioning=True,
        latent_shape=None,
        mode='training',
        patients=None,
    ):
        self.dir_data             = dir_data
        self.use_latents          = use_latents
        self.mask_conditioning    = mask_conditioning
        self.modality_conditioning = modality_conditioning
        self.mode                 = mode
        self.patients             = patients
        self.latent_shape         = latent_shape
        self.baseline_shape       = BASELINE_SHAPE

        self.latent_shape_string = (
            f"{latent_shape[1]}_{latent_shape[2]}_{latent_shape[3]}"
            if latent_shape is not None else None
        )

        self.samples = self._build_samples()
        self.intensity, self.autoencoder_pad, self.autoencoder_crop = self._build_transforms()

    # ------------------------------------------------------------------
    # Sample list construction
    # ------------------------------------------------------------------

    def _build_samples(self):
        mode = self.mode

        if mode == 'autoencoder':
            return [(p, m) for p in self.patients for m in MODALITIES]

        if mode in ['training', 'validation']:
            modalities = MODALITIES if self.modality_conditioning else ['t1']
            return [(p, m) for p in self.patients for m in modalities]

        if mode in ['baseline', 'med_ddpm_pretrained']:
            return list(self.patients)

        if mode == 'lumiere':
            self.dir_data = Path(str(self.dir_data) + '_lumiere')
            self.mode = 'validation'
            return [
                (f'{folder.name}/preop', m)
                for folder in sorted(self.dir_data.iterdir())
                if folder.is_dir() and not folder.name.startswith('.')
                for m in MODALITIES
            ]

        if mode == 'lumiere_med_ddpm_pretrained':
            self.dir_data = Path(str(self.dir_data) + '_lumiere')
            self.mode = 'med_ddpm_pretrained'
            return [
                f'{folder.name}/preop'
                for folder in sorted(self.dir_data.iterdir())
                if folder.is_dir() and not folder.name.startswith('.')
            ]

        if mode in INPAINTING_MODES:
            modalities = MODALITIES if self.modality_conditioning else ['t1']
            mask_ids = ["0000"]  # extend to ["0000", "0001", "0002"] if needed
            return [(p, m, mask) for p in self.patients for m in modalities for mask in mask_ids]

        raise ValueError(f"Unknown mode: {mode!r}. Valid modes: {VALID_MODES}")

    def _build_transforms(self):
        if self.mode == 'autoencoder' or self.latent_shape not in [(4, 32, 32, 20)]:
            intensity = transforms.ScaleIntensity(minv=0.0, maxv=1.0)
        else:
            intensity = transforms.ScaleIntensity(minv=-1.0, maxv=1.0)

        pad  = transforms.SpatialPad(spatial_size=(256, 256, 160))
        crop = transforms.CenterSpatialCrop(roi_size=(240, 240, 155))
        return intensity, pad, crop

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    def _nib_load(self, file_path, retries=10, delay=3):
        last_exc = None
        for attempt in range(retries):
            try:
                return nib.load(file_path)
            except Exception as exc:
                last_exc = exc
                print(f"[WARNING] {exc}. Retrying ({attempt+1}/{retries}) in {delay}s…")
                time.sleep(delay)
        raise last_exc

    def _get_affine(self, file_path):
        return self._nib_load(file_path).affine

    def _get_data(self, file_path):
        return torch.as_tensor(self._nib_load(file_path).get_fdata())
    
    # ------------------------------------------------------------------
    # File path helpers
    # ------------------------------------------------------------------

    def _file_modality(self, patient, modality):
        return self.dir_data / patient / f"{modality}.nii.gz"

    def _file_growth_model(self, patient):
        return self.dir_data / patient / 'processed' / 'growth_model.nii.gz'

    def _file_tumor_segmentation(self, patient):
        return self.dir_data / patient / 'processed' / 'tumor_segmentation.nii.gz'

    def _file_tissue_segmentation(self, patient):
        return self.dir_data / patient / 'processed' / 'tissue_segmentation.nii.gz'

    def _file_latent_modality(self, patient, modality):
        return self.dir_data / patient / f'latents_{self.latent_shape_string}' / f'latent_{modality}.pt'

    def _file_latent_conditioning(self, patient):
        return self.dir_data / patient / f'latents_{self.latent_shape_string}' / 'latent_conditioning.pt'

    def _file_modality_voided(self, patient, modality, mask):
        return self.dir_data / patient / 'voided' / f"{modality}-voided-{mask}.nii.gz"

    def _file_mask(self, patient, mask, healthy=False):
        label = 'healthy-' if healthy else ''
        return self.dir_data / patient / 'masks' / f"mask-{label}{mask}.nii.gz"
        
    # ------------------------------------------------------------------
    # Tensor transforms
    # ------------------------------------------------------------------

    def _to_torch(self, data):
        return torch.as_tensor(data).unsqueeze(0)                  # 1, H, W, D

    def _load_torch(self, path):
        return torch.load(path, map_location='cpu').squeeze(0)     # 1, H, W, D

    def _normalize(self, data):
        return torch.as_tensor(self.intensity(data)).unsqueeze(0)  # 1, H, W, D

    def _pad(self, data):
        return torch.as_tensor(self.autoencoder_pad(data))         # 1, 256, 256, 160

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.mode == 'autoencoder':
            return self._getitem_autoencoder(idx)
        if self.mode == 'training':
            return self._getitem_training(idx)
        if self.mode == 'validation':
            return self._getitem_validation(idx)
        if self.mode in INPAINTING_MODES:
            return self._getitem_inpainting(idx)
        if self.mode == 'med_ddpm_pretrained':
            return self._getitem_med_ddpm_pretrained(idx)
        if self.mode == 'baseline':
            return self._getitem_baseline(idx)
        raise ValueError(f"Unknown mode: {self.mode!r}")
    
    # ------------------------------------------------------------------
    # Mode-specific __getitem__ implementations
    # ------------------------------------------------------------------

    def _getitem_autoencoder(self, idx):
        patient, modality = self.samples[idx]
        path = self._file_modality(patient, modality)

        affine              = self._get_affine(path)
        normalized_modality = self._normalize(self._get_data(path))     # 1, 240, 240, 155
        padded_modality     = self._pad(normalized_modality)            # 1, 256, 256, 160

        return {
            'patient':            patient,
            'affine':             affine,
            'normalized_modality': normalized_modality.float(),
            'modality':           modality,
            'padded_modality':    padded_modality.float(),
        }

    def _getitem_training(self, idx):
        if not self.use_latents:
            raise NotImplementedError("Training without latents is not implemented.")

        patient, modality = self.samples[idx]
        latent_modality   = self._load_torch(self._file_latent_modality(patient, modality))

        result = {
            'modality':        modality,
            'latent_modality': latent_modality.float(),
        }

        if self.mask_conditioning is not None:
            conditioning = self._load_latent_conditioning(patient)
            result['conditioning'] = conditioning.float()

        return result

    def _getitem_validation(self, idx):
        if not self.use_latents:
            raise NotImplementedError("Validation without latents is not implemented.")

        patient, modality   = self.samples[idx]
        affine              = self._get_affine(self._file_modality(patient, modality))
        normalized_modality = self._normalize(self._get_data(self._file_modality(patient, modality)))
        latent_modality     = self._load_torch(self._file_latent_modality(patient, modality))

        result = {
            'mode':               self.mode,
            'patient':            patient,
            'affine':             affine,
            'normalized_modality': normalized_modality.float(),
            'modality':           modality,
            'latent_modality':    latent_modality.float(),
        }

        if self.mask_conditioning is not None:
            conditioning = self._load_latent_conditioning(patient)
            result['conditioning'] = conditioning.float()

        return result

    def _getitem_inpainting(self, idx):
        patient, modality, mask = self.samples[idx]

        affine              = self._get_affine(self._file_modality(patient, modality))
        normalized_modality = self._normalize(self._get_data(self._file_modality(patient, modality)))
        latent_modality     = self._load_torch(self._file_latent_modality(patient, modality))

        original_mask = self._to_torch(self._get_data(self._file_mask(patient, mask)))   # 1, H, W, D
        latent_mask   = process_interpolation(original_mask, self.latent_shape)          # 1, 64, 64, 40

        conditioning = self._load_torch(self._file_latent_conditioning(patient))
        if self.mode == 'inpainting_healthy_tissue':
            assert conditioning.shape[0] == 8
            conditioning[:4] = 0.0  # zero out tumor channel

        return {
            'mode':               self.mode,
            'patient':            patient,
            'affine':             affine,
            'normalized_modality': normalized_modality.float(),
            'modality':           modality,
            'latent_modality':    latent_modality.float(),
            'mask':               mask,
            'original_mask':      original_mask.float(),
            'latent_mask':        latent_mask.float(),
            'conditioning':       conditioning.float(),
        }
    
    def _getitem_med_ddpm_pretrained(self, idx):
        patient = self.samples[idx]
        shape   = MED_DDPM_SHAPE

        affine                = self._get_affine(self._file_modality(patient, 't1'))
        normalized_modalities = self._collect_normalized_modalities(self._collect_data_modalities(patient))
        baseline_modalities   = torch.cat([
            process_interpolation(v, shape, mode='trilinear').float()
            for v in normalized_modalities.values()
        ], dim=0)  # 4, 192, 192, 144

        original_conditioning = self._build_med_ddpm_conditioning(patient, shape)                    # 4, 240, 240, 155
        conditioning          = process_interpolation(original_conditioning, shape, mode='nearest')  # 4, 192, 192, 144

        return {
            'patient':              patient,
            'affine':               affine,
            'original_conditioning': original_conditioning.float(),
            'baseline_modalities':  baseline_modalities,
            'baseline_conditioning': conditioning.float(),
        }
    
    def _getitem_baseline(self, idx):
        patient = self.samples[idx]

        data_growth_model        = self._get_data(self._file_growth_model(patient))
        data_tissue_segmentation = self._get_data(self._file_tissue_segmentation(patient))

        original_conditioning = create_conditioning(data_growth_model, data_tissue_segmentation)  # 4, 240, 240, 155
        baseline_slices       = get_baseline_slices(data_growth_model, self.baseline_shape)
        baseline_conditioning = self._process_baseline_conditioning(data_growth_model, data_tissue_segmentation, baseline_slices)

        normalized_modalities = self._collect_normalized_modalities(self._collect_data_modalities(patient))
        baseline_modalities   = torch.cat([
            self._process_baseline_modality(v, baseline_slices)
            for v in normalized_modalities.values()
        ], dim=0)

        affine = self._get_baseline_affine(self._get_affine(self._file_modality(patient, 't1')), baseline_slices)

        return {
            'patient':              patient,
            'affine':               affine,
            'original_conditioning': original_conditioning.float(),
            'baseline_conditioning': baseline_conditioning.float(),
            'baseline_modalities':  baseline_modalities.float(),
        }
        
    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _load_latent_conditioning(self, patient):
        """Load latent conditioning, optionally replacing with scalar tumor size."""
        conditioning = self._load_torch(self._file_latent_conditioning(patient))
        if self.mask_conditioning == 'scalar':
            data_tumor_seg = self._get_data(self._file_tumor_segmentation(patient))
            scalar = (data_tumor_seg > 0).sum().item() / data_tumor_seg.numel()
            conditioning[:4] = scalar
        return conditioning

    def _build_med_ddpm_conditioning(self, patient, shape):
        """Build the 4-channel med-ddpm conditioning volume at original resolution."""
        data_growth_model        = self._get_data(self._file_growth_model(patient))
        data_tumor_segmentation  = self._get_data(self._file_tumor_segmentation(patient))
        data_tissue_segmentation = self._get_data(self._file_tissue_segmentation(patient))

        conditioning = torch.zeros(4, *data_growth_model.shape).float()  # 4, 240, 240, 155
        conditioning[0][data_tumor_segmentation == LABEL_NECROTIC]                           = 1.0
        conditioning[1][(data_growth_model >= 0.2) & (data_growth_model < 0.6)]              = 1.0
        conditioning[2][data_growth_model >= 0.6]                                            = 1.0
        conditioning[3][data_tissue_segmentation > 0]                                        = 1.0
        return conditioning

    def _get_baseline_affine(self, affine, baseline_slices):
        R     = affine[:3, :3]
        start = np.array([s.start for s in baseline_slices], float)
        new_affine = affine.copy()
        new_affine[:3, 3] = affine[:3, 3] + R @ start
        return new_affine

    def _process_baseline_modality(self, normalized_modality, baseline_slices):
        cropped = normalized_modality.squeeze(0)[baseline_slices].unsqueeze(0)
        assert cropped.shape == (1, *self.baseline_shape)
        return torch.as_tensor(cropped)

    def _process_baseline_conditioning(self, data_growth_model, data_tissue_segmentation, baseline_slices):
        conditioning = create_conditioning(
            data_growth_model[baseline_slices],
            data_tissue_segmentation[baseline_slices],
        )
        assert conditioning.shape == (4, *self.baseline_shape)
        return conditioning

    def _collect_data_modalities(self, patient):
        return {
            m: self._get_data(self._file_modality(patient, m)).float()
            for m in MODALITIES
        }

    def _collect_normalized_modalities(self, data_modalities):
        return {
            m: self._normalize(data_modalities[m]).float()
            for m in MODALITIES
        }