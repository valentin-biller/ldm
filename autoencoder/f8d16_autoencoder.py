import sys
from pathlib import Path
dir_current = Path(__file__).resolve().parent
dir_autoencoder_networks = dir_current/ 'networks'
sys.path.append(str(dir_autoencoder_networks))

import torch
import nibabel as nib
from vaekl_f8d16 import AutoencoderKL_f8


class F8D16Autoencoder:

    def __init__(self, path_autoencoder, device):
        self.device = device

        self.model = AutoencoderKL_f8(ckpt_path=path_autoencoder, dims=3, in_channels=1, out_ch=1, z_channels=16)
        self.model = self.model.to(self.device)
        self.model.eval()

    def encode(self, data):
        with torch.no_grad():
            latent = self.model.encode(data)
            return latent
        
    def decode(self, data):
        with torch.no_grad():
            reconstruction = self.model.decode_sliding(data, roi_size=(16, 16, 16))
            return reconstruction