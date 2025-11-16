
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
from einops import rearrange
from einops.layers.torch import Rearrange
from timm.models.vision_transformer import Attention, Mlp
# from src.models.utils import PatchEmbed, ToPixel
# from src.models.utils.pos_embed import get_2d_sincos_pos_embed, get_3d_sincos_pos_embed

__all__ = ["DiT", "DiT_XL_1", "DiT_XL_2", "DiT_XL_4", "DiT_XL_8", "DiT_L_1", "DiT_L_2", "DiT_L_4", "DiT_L_8", "DiT_B_1", "DiT_B_2", "DiT_B_4", "DiT_B_8", "DiT_S_1", "DiT_S_2", "DiT_S_4", "DiT_S_8"]

####################################################################



class SineLayer(nn.Module):
    """
    Paper: Implicit Neural Representation with Periodic Activ ation Function (SIREN)
    """

    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))


class PatchEmbed(nn.Module):
    """Image/Volume to Patch Embedding - Supports both 2D and 3D"""
    def __init__(self, to_embed='linear', img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        
        # Handle img_size: can be int (2D), tuple of 2 (2D), or tuple of 3 (3D)
        if isinstance(img_size, int):
            self.img_size = (img_size, img_size)
            self.dims = 2
        elif len(img_size) == 2:
            self.img_size = img_size
            self.dims = 2
        elif len(img_size) == 3:
            self.img_size = img_size
            self.dims = 3
        else:
            raise ValueError(f"img_size must be int or tuple of length 2 or 3, got {img_size}")
        
        # Handle patch_size: can be int (2D), tuple of 2 (2D), or tuple of 3 (3D)
        if isinstance(patch_size, int):
            if self.dims == 2:
                self.patch_size = (patch_size, patch_size)
            else:  # 3D
                self.patch_size = (patch_size, patch_size, patch_size)
        elif len(patch_size) == 2:
            if self.dims == 2:
                self.patch_size = patch_size
            else:
                raise ValueError(f"patch_size tuple of length 2 not compatible with 3D img_size {img_size}")
        elif len(patch_size) == 3:
            if self.dims == 3:
                self.patch_size = patch_size
            else:
                raise ValueError(f"patch_size tuple of length 3 not compatible with 2D img_size {img_size}")
        else:
            raise ValueError(f"patch_size must be int or tuple of length 2 or 3, got {patch_size}")
        
        # Calculate grid size and number of patches
        if self.dims == 2:
            self.grid_size = (self.img_size[0] // self.patch_size[0], self.img_size[1] // self.patch_size[1])
            self.num_patches = self.grid_size[0] * self.grid_size[1]
        else:  # 3D
            self.grid_size = (self.img_size[0] // self.patch_size[0], 
                             self.img_size[1] // self.patch_size[1], 
                             self.img_size[2] // self.patch_size[2])
            self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        
        self.flatten = flatten
        
        if to_embed == 'conv':
            # Create appropriate convolution layer
            if self.dims == 2:
                self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
            else:  # 3D
                self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        
        elif to_embed == 'siren':
            self.proj = nn.Sequential(
                    SineLayer(in_chans, embed_dim, is_first=True, omega_0=30.),
                    SineLayer(embed_dim, embed_dim * 2, is_first=False, omega_0=30)
                )
        
        elif to_embed == 'linear':
            self.proj = nn.Linear(in_chans, embed_dim)
        elif to_embed == 'identity':
            self.proj = nn.Identity()
        else:
            raise NotImplementedError

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()


    def forward(self, x):
        if self.dims == 2:
            B, C, H, W = x.shape
            assert H == self.img_size[0] and W == self.img_size[1], \
                f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        else:  # 3D
            B, C, D, H, W = x.shape
            assert D == self.img_size[0] and H == self.img_size[1] and W == self.img_size[2], \
                f"Input volume size ({D}*{H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]}*{self.img_size[2]})."
        
        x = self.proj(x)
        
        if self.flatten:
            if self.dims == 2:
                x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
            else:  # 3D
                x = x.flatten(2).transpose(1, 2)  # BCDHW -> BNC
        
        x = self.norm(x)
        return x


class ToPixel(nn.Module):
    def __init__(self, to_pixel='linear', img_size=256, in_channels=3, in_dim=512, patch_size=16) -> None:
        super().__init__()
        self.to_pixel_name = to_pixel
        
        # Handle img_size: can be int (2D), tuple of 2 (2D), or tuple of 3 (3D)
        if isinstance(img_size, int):
            self.img_size = (img_size, img_size)
            self.dims = 2
        elif len(img_size) == 2:
            self.img_size = img_size
            self.dims = 2
        elif len(img_size) == 3:
            self.img_size = img_size
            self.dims = 3
        else:
            raise ValueError(f"img_size must be int or tuple of length 2 or 3, got {img_size}")
        
        # Handle patch_size: can be int (2D), tuple of 2 (2D), or tuple of 3 (3D)
        if isinstance(patch_size, int):
            if self.dims == 2:
                self.patch_size = (patch_size, patch_size)
            else:  # 3D
                self.patch_size = (patch_size, patch_size, patch_size)
        elif len(patch_size) == 2:
            if self.dims == 2:
                self.patch_size = patch_size
            else:
                raise ValueError(f"patch_size tuple of length 2 not compatible with 3D img_size {img_size}")
        elif len(patch_size) == 3:
            if self.dims == 3:
                self.patch_size = patch_size
            else:
                raise ValueError(f"patch_size tuple of length 3 not compatible with 2D img_size {img_size}")
        else:
            raise ValueError(f"patch_size must be int or tuple of length 2 or 3, got {patch_size}")
        
        # Calculate grid size and number of patches
        if self.dims == 2:
            self.grid_size = (self.img_size[0] // self.patch_size[0], self.img_size[1] // self.patch_size[1])
            self.num_patches = self.grid_size[0] * self.grid_size[1]
        else:  # 3D
            self.grid_size = (self.img_size[0] // self.patch_size[0], 
                             self.img_size[1] // self.patch_size[1], 
                             self.img_size[2] // self.patch_size[2])
            self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]

        self.in_channels = in_channels
        if to_pixel == 'linear':
            if self.dims == 2:
                self.model = nn.Linear(in_dim, in_channels * self.patch_size[0] * self.patch_size[1])
            else:  # 3D
                self.model = nn.Linear(in_dim, in_channels * self.patch_size[0] * self.patch_size[1] * self.patch_size[2])
        elif to_pixel == 'conv':
            if self.dims == 2:
                self.model = nn.Sequential(
                    Rearrange("b (h w) c -> b c h w", h=self.grid_size[0], w=self.grid_size[1]),
                    nn.Conv2d(in_dim, self.patch_size[0] * self.patch_size[1] * in_channels, 1, padding=0),
                    Rearrange("b (p1 p2 c) h w -> b c (h p1) (w p2)", p1=self.patch_size[0], p2=self.patch_size[1]),
                    nn.Conv2d(in_channels, in_channels, 3, padding=1)
                    )
            else:  # 3D
                self.model = nn.Sequential(
                    Rearrange("b (d h w) c -> b c d h w", d=self.grid_size[0], h=self.grid_size[1], w=self.grid_size[2]),
                    nn.Conv3d(in_dim, self.patch_size[0] * self.patch_size[1] * self.patch_size[2] * in_channels, 1, padding=0),
                    Rearrange("b (p1 p2 p3 c) d h w -> b c (d p1) (h p2) (w p3)", p1=self.patch_size[0], p2=self.patch_size[1], p3=self.patch_size[2]),
                    nn.Conv3d(in_channels, in_channels, 3, padding=1)
                    )
        elif to_pixel == 'siren':
            if self.dims == 2:
                self.model = nn.Sequential(
                    SineLayer(in_dim, in_dim * 2, is_first=True, omega_0=30.),
                    SineLayer(in_dim * 2, self.img_size[0] // self.patch_size[0] * self.patch_size[0] * in_channels, is_first=False, omega_0=30)
                )
            else:  # 3D
                self.model = nn.Sequential(
                    SineLayer(in_dim, in_dim * 2, is_first=True, omega_0=30.),
                    SineLayer(in_dim * 2, self.img_size[0] // self.patch_size[0] * self.patch_size[0] * in_channels, is_first=False, omega_0=30)
                )
        elif to_pixel == 'identity':
            self.model = nn.Identity()
        else:
            raise NotImplementedError

    def get_last_layer(self):
        if self.to_pixel_name == 'linear':
            return self.model.weight
        elif self.to_pixel_name == 'siren':
            return self.model[1].linear.weight
        elif self.to_pixel_name == 'conv':
            return self.model[-1].weight
        else:
            return None

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 * in_channels) for 2D 
        (N, L, patch_size**3 * in_channels) for 3D
        imgs: (N, in_channels, H, W) for 2D 
            (N, in_channels, D, H, W) for 3D
        """
        B = x.shape[0]
        C = self.in_channels

        if self.dims == 2:
            p_h, p_w = self.patch_size
            H, W = self.img_size
            h, w = H // p_h, W // p_w
            assert h * w == x.shape[1], f"Expected {h*w} patches, got {x.shape[1]}"
            imgs = rearrange(
                x, "b (h w) (ph pw c) -> b c (h ph) (w pw)",
                h=h, w=w, ph=p_h, pw=p_w, c=C
            )

        else:  # 3D
            p_d, p_h, p_w = self.patch_size
            D, H, W = self.img_size
            d, h, w = D // p_d, H // p_h, W // p_w
            assert d * h * w == x.shape[1], f"Expected {d*h*w} patches, got {x.shape[1]}"
            imgs = rearrange(
                x, "b (d h w) (pd ph pw c) -> b c (d pd) (h ph) (w pw)",
                d=d, h=h, w=w, pd=p_d, ph=p_h, pw=p_w, c=C
            )
        return imgs


    def forward(self, x):
        if self.to_pixel_name == 'linear':
            x = self.model(x)
            x = self.unpatchify(x)
        elif self.to_pixel_name == 'siren':
            x = self.model(x)
            if self.dims == 2:
                x = x.view(x.shape[0], self.in_channels, self.patch_size[0] * int(self.num_patches ** 0.5),
                           self.patch_size[0] * int(self.num_patches ** 0.5))
            else:  # 3D
                d = self.img_size[0] // self.patch_size[0]
                h = self.img_size[1] // self.patch_size[1]
                w = self.img_size[2] // self.patch_size[2]
                x = x.view(x.shape[0], self.in_channels, self.patch_size[0] * d,
                           self.patch_size[1] * h, self.patch_size[2] * w)
        elif self.to_pixel_name == 'conv':
            x = self.model(x)
        elif self.to_pixel_name == 'identity':
            x = self.model(x)
            x = self.unpatchify(x)
        return x


def get_sincos_pos_embed(embed_dim, grid_size, dims):
    """
    grid_size: int or tuple of the grid dimensions (depth, height, width) for 3D or (height, width) for 2D
    dims: int, 2 for 2D or 3 for 3D
    return:
    pos_embed: [grid_size*grid_size*grid_size, embed_dim] or [1+grid_size*grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    if isinstance(grid_size, int):
        grid_size = (grid_size,) * dims
    if dims == 2:
        return get_2d_sincos_pos_embed(embed_dim, grid_size)
    elif dims == 3:
        return get_3d_sincos_pos_embed(embed_dim, grid_size)
    else:
        raise ValueError("dims must be 2 or 3.")

def get_2d_sincos_pos_embed(embed_dim, grid_size):
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size[0], grid_size[1]])
    return get_2d_sincos_pos_embed_from_grid(embed_dim, grid)

def get_3d_sincos_pos_embed(embed_dim, grid_size):
    grid_d = np.arange(grid_size[0], dtype=np.float32)
    grid_h = np.arange(grid_size[1], dtype=np.float32)
    grid_w = np.arange(grid_size[2], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h, grid_d)
    grid = np.stack(grid, axis=0).reshape([3, 1, grid_size[0], grid_size[1], grid_size[2]])
    return get_3d_sincos_pos_embed_from_grid(embed_dim, grid)

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)

def get_3d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 3 == 0
    emb_d = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[0])
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[1])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[2])
    return np.concatenate([emb_d, emb_h, emb_w], axis=1)

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)






#####################################################################

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, math.prod(patch_size) * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size: list[int] = [32, 32],
        patch_size: list[int] = [2, 2],
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
        mask_in_channels=8,  # TODO for mask_conditioning
    ):
        super().__init__()
        assert len(input_size) == len(patch_size), f"input_size and patch_size must have the same length, got {input_size} and {patch_size}"
        self.dims = len(input_size)
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.z_embedder = PatchEmbed(to_embed='conv', img_size=input_size, patch_size=patch_size, in_chans=mask_in_channels, embed_dim=hidden_size)  # TODO for mask_conditioning

        self.x_embedder = PatchEmbed(to_embed='conv', img_size=input_size, patch_size=patch_size, in_chans=in_channels, embed_dim=hidden_size)
        self.to_pixel = ToPixel(to_pixel='identity', img_size=input_size, in_channels=self.out_channels, in_dim=hidden_size, patch_size=patch_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        if self.dims == 2:
            pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.x_embedder.grid_size)
        elif self.dims == 3:
            pos_embed = get_3d_sincos_pos_embed(self.pos_embed.shape[-1], self.x_embedder.grid_size)
        else:
            raise ValueError(f"dims must be 2 or 3, got {self.dims}")
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # TODO for mask_conditioning
        w = self.z_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.z_embedder.proj.bias, 0)
        # TODO for mask_conditioning

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, y, z):  # TODO z is for mask_conditioning
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, self.training)    # (N, D)

        z = self.z_embedder(z)  # TODO for mask_conditioning z.B. 16, 2560, 1152
        x = x + z  # TODO for mask_conditioning
        
        c = t + y                                # (N, D)
        for block in self.blocks:
            x = block(x, c)                      # (N, T, D)
        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.to_pixel(x)                   # (N, out_channels, H, W)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


#################################################################################
#                                   DiT Configs                                  #
#################################################################################
def DiT_XL_1(dims=2,**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=[1]*dims, num_heads=16, **kwargs)

def DiT_XL_2(dims=2,**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=[2]*dims, num_heads=16, **kwargs)

def DiT_XL_4(dims=2,**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=[4]*dims, num_heads=16, **kwargs)

def DiT_XL_8(dims=2,**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=[8]*dims, num_heads=16, **kwargs)

def DiT_L_1(dims=2,**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=[1]*dims, num_heads=16, **kwargs)

def DiT_L_2(dims=2,**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=[2]*dims, num_heads=16, **kwargs)

def DiT_L_4(dims=2,**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=[4]*dims, num_heads=16, **kwargs)

def DiT_L_8(dims=2,**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=[8]*dims, num_heads=16, **kwargs)

def DiT_B_1(dims=2,**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=[1]*dims, num_heads=12, **kwargs)

def DiT_B_2(dims=2,**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=[2]*dims, num_heads=12, **kwargs)

def DiT_B_4(dims=2,**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=[4]*dims, num_heads=12, **kwargs)

def DiT_B_8(dims=2,**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=[8]*dims, num_heads=12, **kwargs)

def DiT_S_1(dims=2, **kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=[1]*dims, num_heads=6, **kwargs)

def DiT_S_2(dims=2, **kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=[2]*dims, num_heads=6, **kwargs)

def DiT_S_4(dims=2, **kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=[4]*dims, num_heads=6, **kwargs)

def DiT_S_8(dims=2, **kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=[8]*dims, num_heads=6, **kwargs)

def DiT_Custom(scheduler_, latent_shape):
    return DiT(
        input_size=[latent_shape[1], latent_shape[2], latent_shape[3]], 
        patch_size=[2, 2, 2],  # DiT_B_2: [2]*dims
        in_channels=latent_shape[0],
        hidden_size=768,  # DiT_B_2: 768, Default: 1152
        depth=12,  # DiT_B_2: 12, Default: 28
        num_heads=12,  # DiT_B_2: 12, Default: 16
        num_classes=4,
        learn_sigma=True if scheduler_ == 'iddpm' else False,
        mask_in_channels=latent_shape[0]*2,  # TODO for mask_conditioning
    )

if __name__ == "__main__":
    model = DiT_S_1(input_size=[32, 32, 32], patch_size=[2, 2, 2])
    x = torch.randn(1, 4, 32, 32, 32) # (N, C, H, W, D)
    t = torch.randint(0, 1000, (1,))
    y = torch.randint(0)
    out = model(x, t, y)
    print(out.shape)