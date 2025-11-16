from models.vaekl_f8d16 import AutoencoderKL_f8
import nibabel as nib
import torch

path = "/vol/miltank/users/bilv/data/BraTS2021_00658/t1.nii.gz"
x = nib.load(path).get_fdata()
x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).float().to("cuda")
x = x / x.max()  #[0, 1]
x = x * 2 - 1   #[-1, 1]


class F8D16Autoencoder:

    def __init__(self, path_autoencoder, device):
        self.device = device

        self.model = AutoencoderKL_f8(ckpt_path=path_autoencoder, dims=3, in_channels=1, out_ch=1, z_channels=16)
        self.model = self.model.to(self.device)
        self.model.eval()

    # b,c,d,h,w -> b,c,d,h,w !!! # TODO
    def encode(self, data):
        with torch.no_grad(), torch.autocast(device_type=self.device.type, dtype=torch.float16):
            # latent = self.model.encode_sliding(data)
            latent = self.model.encode(data)
            print('Encoded latent shape:', latent.shape) # TODO
            return latent
        
    # b,c,d,h,w -> b,c,d,h,w !!! # TODO
    def decode(self, data):
        with torch.no_grad(), torch.autocast(device_type=self.device.type):
            # reconstruction = self.model.decode_sliding(data, roi_size=(16, 16, 16))
            reconstruction = self.model.decode(data)
            print('Decoded reconstruction shape:', reconstruction.shape) # TODO
            return reconstruction