import torch
import importlib
import numpy as np
import nibabel as nib
from tqdm import tqdm
from pathlib import Path
from monai import transforms
from autoencoderkl_maisi import AutoencoderKlMaisi
from torchmetrics.image import PeakSignalNoiseRatio
from omegaconf import OmegaConf, DictConfig, ListConfig

def get_obj_from_str(string: str, reload: bool = False):
    module, cls = string.rsplit(".", 1)
    module_imp = importlib.import_module(module)
    if reload:
        importlib.reload(module_imp)
    return getattr(module_imp, cls)

def instantiate_from_config(config, *args, **kwargs):
    if isinstance(config, (DictConfig, ListConfig)):
        config = OmegaConf.to_container(config, resolve=True)

    if config is None:
        return None

    if isinstance(config, list):
        return [instantiate_from_config(item) for item in config]

    if isinstance(config, dict):
        if "_target_" in config:
            cls = get_obj_from_str(config["_target_"])
            init_args = {
                k: instantiate_from_config(v) for k, v in config.items() if k != "_target_"
            }
            return cls(*args, **init_args, **kwargs)
        else:
            return {k: instantiate_from_config(v) for k, v in config.items()}

    return config

def psnr(reconstructed, original):
    psnr = PeakSignalNoiseRatio()
    return psnr(preds=reconstructed, target=original)

config_dict = {
    "_target_": "autoencoderkl_maisi.AutoencoderKlMaisi", 
    "ckpt_path": "/vol/miltank/users/bilv/ldm/maisi/maisi_vae.pt",
    "spatial_dims": 3,
    "in_channels": 1,
    "out_channels": 1,
    "latent_channels": 4,
    "num_channels": [64, 128, 256],
    "num_res_blocks": [2, 2, 2],
    "norm_num_groups": 32,
    "norm_eps": 1e-06,
    "attention_levels": [False, False, False],
    "with_encoder_nonlocal_attn": False,
    "with_decoder_nonlocal_attn": False,
    "use_checkpointing": False,
    "use_convtranspose": False,
    "norm_float16": True,
    "num_splits": 1,
    "dim_split": 1, 
}

config = OmegaConf.create(config_dict)
model = instantiate_from_config(config)
model = model.to("cuda")
model.eval()

dir_root = Path('/vol/miltank/users/bilv/data')
paths_t1 = sorted(list(dir_root.rglob("**/t1.nii.gz")))
assert len(paths_t1) == 3801, f"Expected 3801 T1 files, found {len(paths_t1)}"

count = 0
for path_t1 in tqdm(paths_t1):
    folder = path_t1.parent.name
    tqdm.write(f'Processing folder: {folder}')

    original = nib.load(path_t1)
    original = original.get_fdata()

    transform_intensity = transforms.ScaleIntensity(minv=0.0, maxv=1.0)
    # transform_intensity = transforms.ScaleIntensityRangePercentiles(lower=0.5, upper=99.5, b_min=0.0, b_max=1.0, clip=True)
    transform_original = transforms.SpatialPad(spatial_size=(240, 240, 160))
    transform_reconstruction = transforms.CenterSpatialCrop(roi_size=(240, 240, 155))

    normalized = transform_intensity(original)
    # normalized = torch.clamp(normalized, 0.0, 1.0)
    assert normalized.min() == 0.0 and normalized.max() == 1.0, "Intensity values should be in the range [0, 1]"
    
    normalized_processed = transform_original(normalized.unsqueeze(0))
    normalized_processed = torch.as_tensor(normalized_processed).unsqueeze(0).to("cuda")
    assert normalized_processed.shape == (1, 1, 240, 240, 160), f"Expected shape (1, 1, 240, 240, 160), got {normalized_processed.shape}"
    assert normalized_processed.min() == 0.0 and normalized_processed.max() == 1.0, "Intensity values should be in the range [0, 1]"

    with torch.no_grad(), torch.autocast("cuda"):
        latent = model.sampling(*model.encode(normalized_processed)) * 1.0 # latents variance needs to be scaled, but already is 1.0 for maisi
        reconstruction_processed = model.decode(latent)

        ### reconstruction_processed = transform_intensity(reconstruction_processed)
        reconstruction_processed = torch.clamp(reconstruction_processed, 0.0, 1.0)
        assert reconstruction_processed.min() >= 0.0 and reconstruction_processed.max() <= 1.0, "Intensity values should be in the range [0, 1]"
        
        reconstruction = transform_reconstruction(reconstruction_processed.squeeze(0))
        reconstruction = reconstruction.squeeze(0).cpu()
        
        orig_min, orig_max = original.min(), original.max()
        reconstruction = reconstruction * (orig_max - orig_min) + orig_min
        recon_min, recon_max = reconstruction.min(), reconstruction.max()

        tqdm.write(f'Original Min: {orig_min}, Max: {orig_max}')
        tqdm.write(f'Reconstruction Min: {recon_min}, Max: {recon_max}')
        tqdm.write(f'Original Mean: {original.mean()}, Std: {original.std()}')
        tqdm.write(f'Reconstruction Mean: {reconstruction.mean()}, Std: {reconstruction.std()}')

        psnr_padded = psnr(reconstruction_processed.to('cpu'), normalized_processed.to('cpu'))
        psnr_original = psnr(reconstruction.to('cpu'), torch.as_tensor(original).to('cpu'))

        tqdm.write(f'PSNR Padded: {psnr_padded}')
        tqdm.write(f'PSNR Original: {psnr_original}')

        if psnr_padded < 30 or psnr_original < 30:
            count += 1
        tqdm.write(f'PSNR too low for {count} folder')
        
        # dir_temp = Path(f'/vol/miltank/users/bilv/ldm/maisi/output/{folder}')
        # dir_temp.mkdir(parents=True, exist_ok=True)
        # nib.save(nib.Nifti1Image(original.float().numpy(), np.eye(4)), dir_temp / 't1_original.nii.gz')
        # nib.save(nib.Nifti1Image(reconstruction.float().numpy(), np.eye(4)), dir_temp / 't1_reconstruction.nii.gz')