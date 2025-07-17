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