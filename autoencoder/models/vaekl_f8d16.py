# Adopted from LDM's KL-VAE: https://github.com/CompVis/latent-diffusion
import torch
import torch.nn as nn
import numpy as np
from typing import Union
import logging
logger = logging.getLogger(__name__)
import gc
from monai.inferers import sliding_window_inference

__all__ = ["AutoencoderKL", "AutoencoderKL_f4", "AutoencoderKL_f8", "AutoencoderKL_f16", "AutoencoderKL_f32"]

############### modules ####################

def get_conv_layer(dims):
    if dims == 2:
        return nn.Conv2d
    elif dims == 3:
        return nn.Conv3d
    else:
        raise ValueError(f"Unsupported number of dimensions: {dims}")


def get_interpolate_mode(dims):
    if dims == 2:
        return "nearest"
    elif dims == 3:
        return "nearest"
    else:
        raise ValueError(f"Unsupported number of dimensions: {dims}")


def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(
        num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True
    )


def _empty_cuda_cache(save_mem: bool) -> None:
    if torch.cuda.is_available() and save_mem:
        torch.cuda.empty_cache()
    return

class SplitConvWrapper(nn.Module):
    """
    Convolutional layer with optional print_info output and custom splitting mechanism.

    Args:
        spatial_dims: Number of spatial dimensions (1D, 2D, 3D).
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        num_splits: Number of splits for the input tensor.
        dim_split: Dimension of splitting for the input tensor.
        print_info: Whether to print information.
        save_mem: Whether to clean CUDA cache in order to save GPU memory, default to `True`.
        Additional arguments for the convolution operation.
        https://docs.monai.io/en/stable/networks.html#convolution
    """

    def __init__(
        self,
        conv: Union[nn.Conv2d, nn.Conv3d],
        num_splits: int,
        dim_split: int,
        print_info: bool = False,
        save_mem: bool = True,
    ) -> None:
        super().__init__()

        self.conv = conv
        strides = self.conv.stride
        self.dim_split = dim_split
        self.stride = strides[self.dim_split] if isinstance(strides, (tuple, list)) else strides
        self.num_splits = num_splits
        self.print_info = print_info
        self.save_mem = save_mem

    def _split_tensor(self, x: torch.Tensor, split_size: int, padding: int) -> list[torch.Tensor]:
        overlaps = [0] + [padding] * (self.num_splits - 1)
        last_padding = x.size(self.dim_split + 2) % split_size

        slices = [slice(None)] * x.dim()
        splits: list[torch.Tensor] = []
        for i in range(self.num_splits):
            slices[self.dim_split + 2] = slice(
                i * split_size - overlaps[i],
                (i + 1) * split_size + (padding if i != self.num_splits - 1 else last_padding),
            )
            splits.append(x[tuple(slices)])

        if self.print_info:
            for j in range(len(splits)):
                logger.info(f"Split {j + 1}/{len(splits)} size: {splits[j].size()}")

        return splits

    def _concatenate_tensors(self, outputs: list[torch.Tensor], split_size: int, padding: int) -> torch.Tensor:
        slices = [slice(None)] * outputs[0].dim()
        for i in range(self.num_splits):
            slices[self.dim_split + 2] = slice(None, split_size) if i == 0 else slice(padding, padding + split_size)
            outputs[i] = outputs[i][tuple(slices)]

        if self.print_info:
            for i in range(self.num_splits):
                logger.info(f"Output {i + 1}/{len(outputs)} size after: {outputs[i].size()}")

        if max(outputs[0].size()) < 500:
            x = torch.cat(outputs, dim=self.dim_split + 2)
        else:
            x = outputs[0].clone().to("cpu", non_blocking=True)
            outputs[0] = torch.tensor(0)
            _empty_cuda_cache(self.save_mem)
            for k in range(len(outputs) - 1):
                x = torch.cat((x, outputs[k + 1].cpu()), dim=self.dim_split + 2)
                outputs[k + 1] = torch.tensor(0)
                _empty_cuda_cache(self.save_mem)
                gc.collect()
                if self.print_info:
                    logger.info(f"MaisiConvolution concat progress: {k + 1}/{len(outputs) - 1}.")

            x = x.to("cuda", non_blocking=True)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.print_info:
            logger.info(f"Number of splits: {self.num_splits}")

        # compute size of splits
        l = x.size(self.dim_split + 2)
        split_size = l // self.num_splits

        # update padding length if necessary
        padding = 3
        if padding % self.stride > 0:
            padding = (padding // self.stride + 1) * self.stride
        if self.print_info:
            logger.info(f"Padding size: {padding}")

        # split tensor into a list of tensors
        splits = self._split_tensor(x, split_size, padding)

        del x
        _empty_cuda_cache(self.save_mem)

        # convolution
        outputs = [self.conv(split) for split in splits]
        if self.print_info:
            for j in range(len(outputs)):
                logger.info(f"Output {j + 1}/{len(outputs)} size before: {outputs[j].size()}")

        # update size of splits and padding length for output
        split_size_out = split_size
        padding_s = padding
        non_dim_split = self.dim_split + 1 if self.dim_split < 2 else 0
        if outputs[0].size(non_dim_split + 2) // splits[0].size(non_dim_split + 2) == 2:
            split_size_out *= 2
            padding_s *= 2
        elif splits[0].size(non_dim_split + 2) // outputs[0].size(non_dim_split + 2) == 2:
            split_size_out //= 2
            padding_s //= 2

        # concatenate list of tensors
        x = self._concatenate_tensors(outputs, split_size_out, padding_s)

        del outputs
        _empty_cuda_cache(self.save_mem)

        return x



class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv=True, dims=2, num_splits=1, dim_split=0):
        super().__init__()
        self.with_conv = with_conv
        self.dims = dims
        if self.with_conv:
            conv_layer = get_conv_layer(dims)
            kernel_size = 3
            self.conv = conv_layer(
                in_channels, in_channels, kernel_size=kernel_size, stride=1, padding=1
            )
            if num_splits > 1:
                self.conv = SplitConvWrapper(conv=self.conv, num_splits=num_splits, dim_split=dim_split)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode=get_interpolate_mode(self.dims))
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv, dims=2, num_splits=1, dim_split=0):
        super().__init__()
        self.with_conv = with_conv
        self.dims = dims
        if self.with_conv:
            conv_layer = get_conv_layer(dims)
            kernel_size = 3
            self.conv = conv_layer(
                in_channels, in_channels, kernel_size=kernel_size, stride=2, padding=0
            )
            if num_splits > 1:
                self.conv = SplitConvWrapper(conv=self.conv, num_splits=num_splits, dim_split=dim_split)

    def forward(self, x):
        if self.with_conv:
            pad_size = (0, 1) * self.dims  # Creates appropriate padding tuple for 2D/3D
            x = torch.nn.functional.pad(x, pad_size, mode="constant", value=0)
            x = self.conv(x)
        else:
            if self.dims == 2:
                x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
            else:
                x = torch.nn.functional.avg_pool3d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout,
        temb_channels=512,
        dims=2,
        num_splits=1,
        dim_split=0
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        conv_layer = get_conv_layer(dims)

        self.norm1 = Normalize(in_channels)
        self.conv1 = conv_layer(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )

        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = conv_layer(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )

        # Split the convolution layers if num_splits > 1
        if num_splits > 1:
            self.conv1 = SplitConvWrapper(conv=self.conv1, num_splits=num_splits, dim_split=dim_split)
            self.conv2 = SplitConvWrapper(conv=self.conv2, num_splits=num_splits, dim_split=dim_split)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = conv_layer(
                    in_channels, out_channels, kernel_size=3, stride=1, padding=1
                )
                if num_splits > 1:
                    self.conv_shortcut = SplitConvWrapper(conv=self.conv_shortcut, num_splits=num_splits, dim_split=dim_split)
            else:
                self.nin_shortcut = conv_layer(
                    in_channels, out_channels, kernel_size=1, stride=1, padding=0
                )
                if num_splits > 1:
                    self.nin_shortcut = SplitConvWrapper(conv=self.nin_shortcut, num_splits=num_splits, dim_split=dim_split)



    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels, dims=2):
        super().__init__()
        self.in_channels = in_channels
        self.dims = dims
        conv_layer = get_conv_layer(dims)

        self.norm = Normalize(in_channels)
        self.q = conv_layer(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k = conv_layer(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v = conv_layer(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = conv_layer(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        if self.dims == 2:
            b, c, h, w = q.shape
            q = q.reshape(b, c, h * w)
            k = k.reshape(b, c, h * w)
            v = v.reshape(b, c, h * w)
        else:  # dims == 3
            b, c, d, h, w = q.shape
            q = q.reshape(b, c, d * h * w)
            k = k.reshape(b, c, d * h * w)
            v = v.reshape(b, c, d * h * w)

        q = q.permute(0, 2, 1)  # b,hw,c
        w_ = torch.bmm(q, k)  # b,hw,hw
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        w_ = w_.permute(0, 2, 1)
        h_ = torch.bmm(v, w_)

        if self.dims == 2:
            h_ = h_.reshape(b, c, h, w)
        else:  # dims == 3
            h_ = h_.reshape(b, c, d, h, w)

        h_ = self.proj_out(h_)

        return x + h_


class Encoder(nn.Module):
    def __init__(
        self,
        *,
        ch=128,
        out_ch=3,
        ch_mult=(1, 1, 2, 2, 4),
        num_res_blocks=2,
        attn_resolutions=(16,),
        dropout=0.0,
        resamp_with_conv=True,
        in_channels=3,
        resolution=256,
        z_channels=16,
        double_z=True,
        dims=2,
        num_splits=1,
        dim_split=0,
        ignore_mid_attn=False,
        **ignore_kwargs,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.dims = dims
        self.ignore_mid_attn = ignore_mid_attn

        conv_layer = get_conv_layer(dims)

        # downsampling
        self.conv_in = conv_layer(
            in_channels, self.ch, kernel_size=3, stride=1, padding=1
        )
        if num_splits > 1:
            self.conv_in = SplitConvWrapper(conv=self.conv_in, num_splits=num_splits, dim_split=dim_split)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                        dims=dims,
                        num_splits=num_splits,
                        dim_split=dim_split
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in, dims=dims))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv, dims=dims, num_splits=num_splits, dim_split=dim_split)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
            dims=dims,
            num_splits=num_splits,
            dim_split=dim_split
        )
        if not ignore_mid_attn:
            self.mid.attn_1 = AttnBlock(block_in, dims=dims)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
            dims=dims,
            num_splits=num_splits,
            dim_split=dim_split
        )

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = conv_layer(
            block_in,
            2 * z_channels if double_z else z_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        if num_splits > 1:
            self.conv_out = SplitConvWrapper(conv=self.conv_out, num_splits=num_splits, dim_split=dim_split)

    def forward(self, x):
        # assert x.shape[2] == x.shape[3] == self.resolution, "{}, {}, {}".format(x.shape[2], x.shape[3], self.resolution)

        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        if not self.ignore_mid_attn:
            h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        *,
        ch=128,
        out_ch=3,
        ch_mult=(1, 1, 2, 2, 4),
        num_res_blocks=2,
        attn_resolutions=(),
        dropout=0.0,
        resamp_with_conv=True,
        in_channels=3,
        resolution=256,
        z_channels=16,
        give_pre_end=False,
        dims=2,
        num_splits=1,
        dim_split=0,
        ignore_mid_attn=False,
        **ignore_kwargs,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.dims = dims
        self.ignore_mid_attn = ignore_mid_attn
        conv_layer = get_conv_layer(dims)

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        
        # Adjust z_shape based on dimensions
        if dims == 2:
            self.z_shape = (1, z_channels, curr_res, curr_res)
        else:  # dims == 3
            self.z_shape = (1, z_channels, curr_res, curr_res, curr_res)
            
        print(
            "Working with z of shape {} = {} dimensions.".format(
                self.z_shape, np.prod(self.z_shape)
            )
        )

        # z to block_in
        self.conv_in = conv_layer(
            z_channels, block_in, kernel_size=3, stride=1, padding=1
        )
        if num_splits > 1:
            self.conv_in = SplitConvWrapper(conv=self.conv_in, num_splits=num_splits, dim_split=dim_split)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
            dims=dims,
            num_splits=num_splits,
            dim_split=dim_split
        )
        if not ignore_mid_attn:
            self.mid.attn_1 = AttnBlock(block_in, dims=dims)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
            dims=dims,
            num_splits=num_splits,
            dim_split=dim_split
        )

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                        dims=dims,
                        num_splits=num_splits,
                        dim_split=dim_split
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in, dims=dims))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv, dims=dims, num_splits=num_splits, dim_split=dim_split)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = conv_layer(
            block_in, out_ch, kernel_size=3, stride=1, padding=1
        )
        if num_splits > 1:
            self.conv_out = SplitConvWrapper(conv=self.conv_out, num_splits=num_splits, dim_split=dim_split)

    def forward(self, z):
        # assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        if not self.ignore_mid_attn:
            h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

    def get_last_layer(self) -> nn.Parameter:
        return self.conv_out.weight

### FOR AEKL ###

class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False, channel_dim=1):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=channel_dim)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(
                device=self.parameters.device
            )

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(
            device=self.parameters.device
        )
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.tensor([0.0], device=self.mean.device)
        else:
            reduce_dims = list(range(1, self.mean.ndim))  # all dims except batch
            if other is None:
                return 0.5 * torch.sum(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                    dim=reduce_dims
                )
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=reduce_dims
                )

    def nll(self, sample):
        if self.deterministic:
            return torch.tensor([0.0], device=self.mean.device)
        logtwopi = np.log(2.0 * np.pi)
        reduce_dims = list(range(1, self.mean.ndim))  # sum over everything but batch
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=reduce_dims
        )

    def mode(self):
        return self.mean
    













################ helpers #################
def init_from_ckpt(self, path):
    # Load the checkpoint
    try:
        sd = torch.load(path, map_location="cpu", weights_only=False)['state_dict']
    except:
        try: 
            sd = torch.load(path, map_location="cpu", weights_only=False)['model']
        except:
            sd = torch.load(path, map_location="cpu", weights_only=False)

    # Remove '.module' prefix from keys if present
    new_sd = {}
    for k, v in sd.items():
        if "loss" in k: continue
        new_key = k.replace("module.", "") if k.startswith("module.") else k
        new_sd[new_key] = v
    
    # Try to load the state dictionary, but allow for dimension mismatches
    try:
        msg = self.load_state_dict(new_sd, strict=True)
    except RuntimeError as e:
        # If error is due to dimension mismatch (2D vs 3D), print warning and skip loading affected layers
        print(f"Warning: {str(e)}")
        print("This might be due to loading 2D weights into 3D model or vice versa.")
        print("Loading weights with strict=False to skip incompatible layers.")
        msg = self.load_state_dict(new_sd, strict=False)

    # Clean up memory
    torch.cuda.empty_cache()

    # Print loading information
    print(f"Loading pre-trained {self.__class__.__name__}")
    print("Missing keys:")
    print(msg.missing_keys)
    print("Unexpected keys:")
    print(msg.unexpected_keys)
    print(f"Restored from {path}")


def get_conv_layer(dims):
    if dims == 2:
        return nn.Conv2d
    elif dims == 3:
        return nn.Conv3d
    else:
        raise ValueError(f"Unsupported number of dimensions: {dims}")

def merge_ddconfig(default_ddconfig, **kwargs):
    """
    Merge default ddconfig with overrides from kwargs.

    Priority:
    1. kwargs["ddconfig"] (if provided)
    2. Direct kwargs that match ddconfig keys
    """
    ddconfig = default_ddconfig.copy()

    # Pop optional nested ddconfig overrides
    user_ddconfig = kwargs.pop("ddconfig", {})
    ddconfig.update(user_ddconfig)

    # Override with direct kwargs
    for key in list(kwargs.keys()):
        if key in ddconfig:
            ddconfig[key] = kwargs.pop(key)

    return ddconfig, kwargs


############################## AUTOENCODER KL ##############################

class AutoencoderKL(nn.Module):
    def __init__(self,
                 ddconfig,
                 embed_dim,
                 ckpt_path=None,
                 ):
        super().__init__()
        self.dims = ddconfig.get("dims", 2)  # Default to 2D for backward compatibility
        conv_layer = get_conv_layer(self.dims)

        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        assert ddconfig["double_z"]
        self.quant_conv = conv_layer(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = conv_layer(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim

        if ckpt_path is not None:
            init_from_ckpt(self, ckpt_path)


    def encode(self, x, return_posterior=False):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        if return_posterior:
            return posterior
        else:
            z = posterior.sample()
        return z

    def encode_sliding(self, x, roi_size=(64, 64, 64), sw_batch_size=1, overlap=0.0):
        z = sliding_window_inference(
            inputs=x,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            predictor=self.encode,
            mode="gaussian",
            overlap=overlap,
        )
        return z

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def decode_sliding(self, z, roi_size=(8, 8, 8), sw_batch_size=1, overlap=0.0):
        dec = sliding_window_inference(
            inputs=z,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            predictor=self.decode,
            mode="gaussian",
            overlap=overlap,
        )
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

def AutoencoderKL_f4(**kwargs):
    """
    AutoencoderKL with compression factor 4
    Args:
        dims (int): Number of dimensions (2 for 2D, 3 for 3D)
    """
    ddconfig = {
        "double_z": True,
        "z_channels": 3,
        "resolution": 256,
        "in_channels": 3,
        "out_ch": 3,
        "ch": 128,
        "ch_mult": [1, 2, 4],
        "num_res_blocks": 2,
        "attn_resolutions": [],
        "dropout": 0.0,
        "dims": 2,
        "num_splits": 1,
        "dim_split": 0,
    }

    ddconfig, kwargs = merge_ddconfig(ddconfig, **kwargs)
    return AutoencoderKL(ddconfig, embed_dim=3, **kwargs)

def AutoencoderKL_f8(**kwargs):
    """
    AutoencoderKL with compression factor 8.
    Args:
        dims (int): Number of dimensions (2 for 2D, 3 for 3D)
        kwargs: Can include 'ddconfig' dict or any ddconfig key directly
    """
    # Default ddconfig
    ddconfig = {
        "double_z": True,
        "z_channels": 4,
        "resolution": 256,
        "in_channels": 3,
        "out_ch": 3,
        "ch": 128,
        "ch_mult": [1, 2, 4, 4],
        "num_res_blocks": 2,
        "attn_resolutions": [],
        "dropout": 0.0,
        "dims": 2,
        "num_splits": 1,
        "dim_split": 0,
    }

    ddconfig, kwargs = merge_ddconfig(ddconfig, **kwargs)
    return AutoencoderKL(ddconfig, embed_dim=4, **kwargs)

def AutoencoderKL_f16(**kwargs):
    """
    AutoencoderKL with compression factor 16
    Args:
        dims (int): Number of dimensions (2 for 2D, 3 for 3D)
    """
    ddconfig = {
        "double_z": True,
        "z_channels": 16,
        "resolution": 256,
        "in_channels": 3,
        "out_ch": 3,
        "ch": 128,
        "ch_mult": [1, 1, 2, 2, 4],
        "num_res_blocks": 2,
        "attn_resolutions": [16],
        "dropout": 0.0,
        "dims": 2,
        "num_splits": 1,
        "dim_split": 0,
    }
    ddconfig, kwargs = merge_ddconfig(ddconfig, **kwargs)
    return AutoencoderKL(ddconfig, embed_dim=16, **kwargs)   

def AutoencoderKL_f32(**kwargs):
    """
    AutoencoderKL with compression factor 32
    Args:
        dims (int): Number of dimensions (2 for 2D, 3 for 3D)
    """
    ddconfig = {
        "double_z": True,
        "z_channels": 64,
        "resolution": 256,
        "in_channels": 3,
        "out_ch": 3,
        "ch": 128,
        "ch_mult": [1, 1, 2, 2, 4, 4],
        "num_res_blocks": 2,
        "attn_resolutions": [16, 8],
        "dropout": 0.0,
        "dims": 2,
        "num_splits": 1,
        "dim_split": 0,
    }
    ddconfig, kwargs = merge_ddconfig(ddconfig, **kwargs)
    return AutoencoderKL(ddconfig, embed_dim=64, **kwargs)


if __name__ == "__main__":
    import nibabel as nib
    import torch
    path = "/vol/miltank/users/bilv/data/BraTS2021_00658/t1.nii.gz"
    x = nib.load(path).get_fdata()
    x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).float().to("cuda")
    x = x / x.max()  #[0, 1]
    x = x * 2 - 1   #[-1, 1]
    nib.save(nib.Nifti1Image(x[0,0,...].cpu().numpy(), affine=np.eye(4)), "input.nii.gz")
    model = AutoencoderKL_f8(ckpt_path="/vol/miltank/users/bubeckn/autoMED/outputs/vaekl/3d_vaekl_f8d16_BraTS/train/models/ckpt.1.pt", dims=3, in_channels=1, out_ch=1, z_channels=16)
    model.eval()
    model.to("cuda")
    with torch.no_grad():
        print(x.shape)
        x = model.encode_sliding(x)
        print(x.shape)
        x = model.decode_sliding(x, roi_size=(16, 16, 16))
        print(x.shape)

    nib.save(nib.Nifti1Image(x[0,0,...].cpu().numpy(), affine=np.eye(4)), "output.nii.gz")
