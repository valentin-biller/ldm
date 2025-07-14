import sys
import torch
import importlib
from omegaconf import OmegaConf, DictConfig, ListConfig


class MaisiAutoencoder():

    def __init__(self, path_autoencoder, device):

        config_dict = {
            "_target_": "autoencoderkl_maisi.AutoencoderKlMaisi", 
            "ckpt_path": path_autoencoder,
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

        self.device = device

        config = OmegaConf.create(config_dict)
        self.model = self.instantiate_from_config(config)
        self.model = self.model.to(self.device)
        self.model.eval()

    def get_obj_from_str(self, string: str, reload: bool = False):
        module, cls = string.rsplit(".", 1)
        module_imp = importlib.import_module(module)
        if reload:
            importlib.reload(module_imp)
        return getattr(module_imp, cls)

    def instantiate_from_config(self, config, *args, **kwargs):
        if isinstance(config, (DictConfig, ListConfig)):
            config = OmegaConf.to_container(config, resolve=True)

        if config is None:
            return None

        if isinstance(config, list):
            return [self.instantiate_from_config(item) for item in config]

        if isinstance(config, dict):
            if "_target_" in config:
                cls = self.get_obj_from_str(config["_target_"])
                init_args = {
                    k: self.instantiate_from_config(v) for k, v in config.items() if k != "_target_"
                }
                return cls(*args, **init_args, **kwargs)
            else:
                return {k: self.instantiate_from_config(v) for k, v in config.items()}

        return config

    def encode(self, data):
        with torch.no_grad(), torch.autocast(device_type=self.device.type, dtype=torch.float16):
            latent = self.model.sampling(*self.model.encode(data)) * 1.0 # latents variance needs to be scaled, but already is 1.0 for maisi
            return latent
        
    def decode(self, data):
        with torch.no_grad(), torch.autocast(device_type=self.device.type):
            reconstruction = self.model.decode(data)
            return reconstruction
    


# for path_t1 in tqdm(paths_t1):
#     folder = path_t1.parent.name

#     original = nib.load(path_t1)
#     original = original.get_fdata()

#     transform_intensity = transforms.ScaleIntensity(minv=0.0, maxv=1.0)
#     # transform_intensity = transforms.ScaleIntensityRangePercentiles(lower=0.001, upper=99.999, b_min=0, b_max=1)
#     transform_original = transforms.SpatialPad(spatial_size=(240, 240, 160))
#     transform_reconstruction = transforms.CenterSpatialCrop(roi_size=(240, 240, 155))

#     original = transform_intensity(original)
#     # original = torch.clamp(original, 0.0, 1.0)
#     assert original.min() == 0.0 and original.max() == 1.0, "Intensity values should be in the range [0, 1]"
    
#     original_processed = transform_original(original.unsqueeze(0))
#     original_processed = torch.as_tensor(original_processed).unsqueeze(0).to("cuda")
#     assert original_processed.shape == (1, 1, 240, 240, 160), f"Expected shape (1, 1, 240, 240, 160), got {original_processed.shape}"
#     assert original_processed.min() == 0.0 and original_processed.max() == 1.0, "Intensity values should be in the range [0, 1]"

#     with torch.no_grad(), torch.autocast("cuda"):
#         latent = model.sampling(*model.encode(original_processed)) * 1.0 # latents variance needs to be scaled, but already is 1.0 for maisi
#         reconstruction_processed = model.decode(latent)

#         # reconstruction_processed = transform_intensity(reconstruction_processed)
#         reconstruction_processed = torch.clamp(reconstruction_processed, 0.0, 1.0)
#         # assert reconstruction_processed.min() == 0.0 and reconstruction_processed.max() == 1.0, "Intensity values should be in the range [0, 1]"
        
#         reconstruction = transform_reconstruction(reconstruction_processed.squeeze(0))
#         reconstruction = reconstruction.squeeze(0).cpu()
#         # reconstruction = transform_intensity(reconstruction.squeeze(0)).cpu()

#         # print(f'Original_Processed shape: {original_processed.shape}')
#         # print(f'Reconstruction_Processed shape: {reconstruction_processed.shape}')
#         # print(f'Original shape: {original.shape}')
#         # print(f'Reconstruction shape: {reconstruction.shape}')

#         print('PSNR Padded not clipped:', psnr(reconstruction_processed, original_processed))
#         print('PSNR Padded clipped:', psnr(torch.clamp(reconstruction_processed, 0.0, 1.0), torch.clamp(original_processed, 0.0, 1.0)))
#         print('PSNR Original not clipped:', psnr(reconstruction, original))
#         print('PSNR Original clipped:', psnr(torch.clamp(reconstruction, 0.0, 1.0), torch.clamp(original, 0.0, 1.0)))
        
#         path_temp = f'/vol/miltank/users/bilv/ldm/maisi/output/{folder}'
#         os.makedirs(path_temp, exist_ok=True)
#         nib.save(nib.Nifti1Image(original.float().numpy(), np.eye(4)), f'{path_temp}/t1_original.nii.gz')
#         nib.save(nib.Nifti1Image(reconstruction.float().numpy(), np.eye(4)), f'{path_temp}/t1_reconstruction.nii.gz')
