import torch
import numpy as np
from tqdm import tqdm

# --------- Gradients ---------  (doesn't slow down the training too much)
def _gradients_compute_norm(parameters):
    sqsum = 0.0
    for p in parameters:
        if p.grad is None:
            continue
        sqsum += (p.grad ** 2).sum().item()
    return np.sqrt(sqsum)

# --------- Distribution Shift ---------
def _distribution_shift(z, ae_latent_mean, ae_latent_std):
    batch_mean = z.mean()
    batch_std = z.std(unbiased=False)
    z = (z - batch_mean) / (batch_std + 1e-6)
    z = z * ae_latent_std + ae_latent_mean
    return z

# --------- Debugging ---------
@torch.no_grad()
def _debugging(self, tensor, tag, print_=False, logging_=False, distribution_=False, target_mean=0.0, target_std=1.0, target_tolerance=0.1):  # print_: False, 'scientific', 'float'
    if not self.debugging:
        return
    
    tensor = tensor.detach().float()
    mean = tensor.mean().detach()
    std = tensor.std(unbiased=False).detach()
    var = (std * std)
    min = tensor.min().detach()
    max = tensor.max().detach()

    if print_ == 'scientific':
        msg = f'[DEBUGGING] {tag}: mean={mean:.4e}, std={std:.4e}, var={var:.4e}, min={min:.4e}, max={max:.4e}'
    elif print_ == 'float':
        msg = f'[DEBUGGING] {tag}: mean={mean:.2f}, std={std:.2f}, var={var:.2f}, min={min:.2f}, max={max:.2f}'
    if print_ is not False:    
        tqdm.write(msg)

    if logging_:
        self.log(f'{tag}_mean', mean, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)
        self.log(f'{tag}_std', std, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)
        self.log(f'{tag}_var', var, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)
        self.log(f'{tag}_min', min, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)
        self.log(f'{tag}_max', max, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)

    if distribution_:
        if not (abs(mean - target_mean) < target_tolerance and abs(std - target_std) < target_tolerance):
            tqdm.write(f'[DEBUGGING] {tag}: Distribution Warning (mean≈{mean:.4f}, std≈{std:.4f})')

    return  # {'mean': mean, 'std': std, 'var': var, 'min': min, 'max': max}