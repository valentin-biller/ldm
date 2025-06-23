import torch 
from src.utils import instantiate_from_config
from .unet import UNetModel as ADM
from .unet import EncoderUNetModel as ADM_classifier

"""
Defining the available diffusion models for ADM with the configs
given in the paper

"""

__all__ = [
    "ADM",
    "ADM_classifier",
    "ADM_diffusion_64_conditioned",
    "ADM_diffusion_64_unconditioned", 
    "ADM_diffusion_128_conditioned",
    "ADM_diffusion_128_unconditioned",
    "ADM_diffusion_256_conditioned",
    "ADM_diffusion_256_unconditioned",
    "ADM_diffusion_512_conditioned", 
    "ADM_diffusion_512_unconditioned",
    "ADM_classifier_64",
    "ADM_classifier_128",
    "ADM_classifier_256", 
    "ADM_classifier_512",
    "ADM_U",
    "ADM_G",
]

diffusion_cfg = {
    "target": "src.diffusion.create_gaussian_diffusion",
    "params": {
        "steps": 1000,
        "learn_sigma": True,
        "sigma_small": False,
        "noise_schedule": "linear",
        "use_kl": False,
        "predict_xstart": False,
        "rescale_timesteps": False,
        "rescale_learned_sigmas": False,
        "timestep_respacing": ""
    }
}

class ADM_diffusion_64_conditioned(ADM):
    def __init__(self, *args, **kwargs):
        super().__init__(
            image_size=64,
            in_channels=3,
            model_channels=192,
            num_classes=1000,
            out_channels=6,  # 6 for learn_sigma=True
            num_res_blocks=3,
            attention_resolutions=[32, 16, 8],
            dropout=0.1,
            channel_mult=(1, 2, 3, 4),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=True,
            num_heads=-1,  # Not used when num_head_channels specified
            num_head_channels=64,
            num_heads_upsample=-1,
            use_scale_shift_norm=True,
            resblock_updown=True,
            use_new_attention_order=True,
            **kwargs
        )
        self.diffusion_cfg = diffusion_cfg

class ADM_diffusion_64_unconditioned(ADM):
    def __init__(self, *args, **kwargs):
        super().__init__(
            image_size=64,
            in_channels=3,
            model_channels=192,
            num_classes=None,
            out_channels=6,  # 6 for learn_sigma=True
            num_res_blocks=3,
            attention_resolutions=[32, 16, 8],
            dropout=0.1,
            channel_mult=(1, 2, 3, 4),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,    #actually true but has to be defined from accelerator
            num_heads=-1,  # Not used when num_head_channels specified
            num_head_channels=64,
            num_heads_upsample=-1,
            use_scale_shift_norm=True,
            resblock_updown=True,
            use_new_attention_order=True,
            **kwargs
        )
        self.diffusion_cfg = diffusion_cfg

class ADM_diffusion_128_conditioned(ADM):
    def __init__(self, *args, **kwargs):
        super().__init__(
            image_size=128,
            in_channels=3,
            num_classes=1000,
            model_channels=256,
            out_channels=6,
            num_res_blocks=2,
            attention_resolutions=[32, 16, 8],
            dropout=0.0,
            channel_mult=(1, 1, 2, 3, 4),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=4,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=True,
            resblock_updown=True,
            use_new_attention_order=False,
            **kwargs
        )
        self.diffusion_cfg = diffusion_cfg

class ADM_diffusion_128_unconditioned(ADM):
    def __init__(self, *args, **kwargs):
        super().__init__(
            image_size=128,
            in_channels=3,
            num_classes=None,
            model_channels=256,
            out_channels=6,
            num_res_blocks=2,
            attention_resolutions=[32, 16, 8],
            dropout=0.0,
            channel_mult=(1, 1, 2, 3, 4),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=4,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=True,
            resblock_updown=True,
            use_new_attention_order=False,
            **kwargs
        )
        self.diffusion_cfg = diffusion_cfg
        
        
class ADM_diffusion_256_conditioned(ADM):
    def __init__(self, *args, **kwargs):
        super().__init__(
            image_size=256,
            in_channels=3,
            num_classes=1000,
            model_channels=256,
            out_channels=6,
            num_res_blocks=2,
            attention_resolutions=[32, 16, 8],
            dropout=0.0,
            channel_mult=(1, 1, 2, 2, 4, 4),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=-1,  # Not used when num_head_channels specified
            num_head_channels=64,
            num_heads_upsample=-1,
            use_scale_shift_norm=True,
            resblock_updown=True,
            use_new_attention_order=False,
            **kwargs
        )
        self.diffusion_cfg = diffusion_cfg

class ADM_diffusion_256_unconditioned(ADM):
    def __init__(self, *args, **kwargs):
        super().__init__(
            image_size=256,
            in_channels=3,
            num_classes=None,
            model_channels=256,
            out_channels=6,
            num_res_blocks=2,
            attention_resolutions=[32, 16, 8],
            dropout=0.0,
            channel_mult=(1, 1, 2, 2, 4, 4),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=-1,  # Not used when num_head_channels specified
            num_head_channels=64,
            num_heads_upsample=-1,
            use_scale_shift_norm=True,
            resblock_updown=True,
            use_new_attention_order=False,
            **kwargs
        )
        self.diffusion_cfg = diffusion_cfg


class ADM_diffusion_512_conditioned(ADM):
    def __init__(self, *args, **kwargs):
        super().__init__(
            image_size=512,
            in_channels=3,
            num_classes=1000,
            model_channels=256,
            out_channels=6,  # learn_sigma=True -> 6 channels
            num_res_blocks=2,
            attention_resolutions=[32, 16, 8],
            dropout=0.0,
            channel_mult=(0.5, 1, 1, 2, 2, 4, 4),  # For 512x512 images
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,  # use_fp16=False from flags
            num_heads=-1,  # Not used when num_head_channels specified
            num_head_channels=64,
            num_heads_upsample=-1,
            use_scale_shift_norm=True,
            resblock_updown=True,
            use_new_attention_order=False,
            **kwargs
        )
        self.diffusion_cfg = diffusion_cfg

class ADM_diffusion_512_unconditioned(ADM):
    def __init__(self, *args, **kwargs):
        super().__init__(
            image_size=512,
            in_channels=3,
            num_classes=None,
            model_channels=256,
            out_channels=6,  # learn_sigma=True -> 6 channels
            num_res_blocks=2,
            attention_resolutions=[32, 16, 8],
            dropout=0.0,
            channel_mult=(0.5, 1, 1, 2, 2, 4, 4),  # For 512x512 images
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,  # use_fp16=False from flags
            num_heads=-1,  # Not used when num_head_channels specified
            num_head_channels=64,
            num_heads_upsample=-1,
            use_scale_shift_norm=True,
            resblock_updown=True,
            use_new_attention_order=False,
            **kwargs
        )
        self.diffusion_cfg = diffusion_cfg


class ADM_classifier_64(ADM_classifier):
    def __init__(self, *args, **kwargs):
        super().__init__(
            image_size=64,
            in_channels=3,
            model_channels=192,
            out_channels=1000,  # num_classes for classifier
            num_res_blocks=3,
            attention_resolutions=[32, 16, 8],
            dropout=0.1,
            channel_mult=(1, 2, 3, 4),  # For 64x64 images
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=-1,  # Not used when num_head_channels specified
            num_head_channels=64,
            num_heads_upsample=-1,
            use_scale_shift_norm=True,
            resblock_updown=True,
            use_new_attention_order=True,
            pool="adaptive",
            **kwargs
        )
        self.diffusion_cfg = diffusion_cfg
        
class ADM_classifier_128(ADM_classifier):
    def __init__(self, *args, **kwargs):
        super().__init__(
            image_size=128,
            in_channels=3,
            model_channels=256,  # num_channels=256
            out_channels=1000,  # num_classes for classifier
            num_res_blocks=2,
            attention_resolutions=[32, 16, 8],
            dropout=0.0,
            channel_mult=(1, 1, 2, 3, 4),  # For 128x128 images
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=4,
            num_head_channels=-1,  # Not used when num_heads specified
            num_heads_upsample=-1,
            use_scale_shift_norm=True,
            resblock_updown=True,
            use_new_attention_order=False,
            pool="adaptive",
            **kwargs
        )
        self.diffusion_cfg = diffusion_cfg

class ADM_classifier_256(ADM_classifier):
    def __init__(self, *args, **kwargs):
        super().__init__(
            image_size=256,
            in_channels=3,
            model_channels=256,  # num_channels=256
            out_channels=1000,  # num_classes for classifier
            num_res_blocks=2,
            attention_resolutions=[32, 16, 8],
            dropout=0.0,
            channel_mult=(1, 1, 2, 2, 4, 4),  # For 256x256 images
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=-1,  # Not used when num_head_channels specified
            num_head_channels=64,
            num_heads_upsample=-1,
            use_scale_shift_norm=True,
            resblock_updown=True,
            use_new_attention_order=False,
            pool="adaptive",
            **kwargs
        )
        self.diffusion_cfg = diffusion_cfg
        
class ADM_classifier_512(ADM_classifier):
    def __init__(self, *args, **kwargs):
        super().__init__(
            image_size=512,
            in_channels=3,
            model_channels=256,  # num_channels=256
            out_channels=1000,  # num_classes for classifier
            num_res_blocks=2,
            attention_resolutions=[32, 16, 8],
            dropout=0.0,
            channel_mult=(0.5, 1, 1, 2, 2, 4, 4),  # For 512x512 images
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,  # Changed based on MODEL_FLAGS
            num_heads=-1,  # Not used when num_head_channels specified
            num_head_channels=64,
            num_heads_upsample=-1,
            use_scale_shift_norm=True,
            resblock_updown=True,
            use_new_attention_order=False,
            pool="adaptive",
            **kwargs
        )
        self.diffusion_cfg = diffusion_cfg

class ADM_U(ADM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class ADM_G(torch.nn.Module):
    """
    This model is two stage without possibiliy of E2E training! It's only usable for inference and testing! 
    """
    def __init__(self, diffmodel_cfg, classmodel_cfg):
        self.diffusion = instantiate_from_config(diffmodel_cfg).eval()
        self.classifier = instantiate_from_config(classmodel_cfg).eval()
        
    def cond_fn(self, x, t, y=None, classifier_scale=1.):
        assert y is not None
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = self.classifier(x_in, t)
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            return torch.autograd.grad(selected.sum(), x_in)[0] * classifier_scale
        
    def forward(self, *args, **kwargs): 
        return self.diffusion.forward(*args, **kwargs)