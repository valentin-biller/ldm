import torch.nn as nn
from monai.networks.nets import ControlNet, DiffusionModelUNet

class UNet(nn.Module):
    def __init__(self, mask_conditioning, modality_conditioning, scheduler_, latent_shape):
        super().__init__()
        self.mask_conditioning = mask_conditioning
        self.modality_conditioning = modality_conditioning

        config_unet = {
            "spatial_dims": 3,
            "in_channels": 4,
            "out_channels": 8 if scheduler_ == 'iddpm' else 4,  # output 8 channels to learn sigma
            "channels": (64, 128, 256, 512) if latent_shape == (4, 64, 64, 40) else (128, 256, 512),
            "attention_levels": [False, False, True, True] if latent_shape == (4, 64, 64, 40) else [False, True, True],
            "num_head_channels": (0, 0, 32, 32) if latent_shape == (4, 64, 64, 40) else (0, 32, 32),
            "num_class_embeds": 4 if self.modality_conditioning else None,
            "num_res_blocks": 2,
            "use_flash_attention": True,
            # 'with_conditioning' only needed for 'context'
        }
        config_controlnet = config_unet.copy()
        config_controlnet.pop('out_channels')
        conditioning_embedding_in_channels = latent_shape[0]*2  # original: 8
        conditioning_embedding_num_channels = [latent_shape[0]*2,]  # original 8, 32, 64

        self.unet = DiffusionModelUNet(**config_unet)
        if self.mask_conditioning is not None:
            self.controlnet = ControlNet(
                **config_controlnet,
                conditioning_embedding_in_channels=conditioning_embedding_in_channels,
                conditioning_embedding_num_channels=conditioning_embedding_num_channels,
            )
            self.controlnet.load_state_dict(self.unet.state_dict(), strict=False)
            self.parameters_unet_controlnet = list(self.unet.parameters()) + list(self.controlnet.parameters())
        else:
            self.parameters_unet_controlnet = list(self.unet.parameters())

    def forward(self, sample, timesteps, modality, conditioning):
        if self.mask_conditioning is not None:
            down_block_res_samples, mid_block_res_sample = self.controlnet(
                x=sample,
                timesteps=timesteps,
                class_labels=modality if self.modality_conditioning else None,
                controlnet_cond=conditioning,
            )
            noise_pred = self.unet(
                x=sample,
                timesteps=timesteps,
                class_labels=modality if self.modality_conditioning else None,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample
            )
        else:
            noise_pred = self.unet(
                x=sample,
                timesteps=timesteps,
                class_labels=modality if self.modality_conditioning else None,
            )
        return noise_pred