import torch
import numpy as np
from collections import defaultdict
from scipy.ndimage import binary_dilation


def _repaint_init(self):
    self.betas = _repaint_get_named_beta_schedule(self, 'linear', self.num_inference_steps, use_scale=True)
    self.betas = np.array(self.betas, dtype=np.float64)
    self.alphas = 1.0 - self.betas
    self.alphas_cumprod = np.cumprod(self.alphas, axis=0)

def _repaint_start(self, patients, modality, conditioning, affines, latent_modality=None, latent_mask=None, spatio_temporal=None):
    self.scheduler_inference.set_timesteps(self.num_inference_steps)

    final = None
    for sample in _repaint_p_sample_loop_progressive(self, modality, conditioning, latent_modality=latent_modality, latent_mask=latent_mask, spatio_temporal=spatio_temporal):
        final = sample
    return final["sample"]

def _repaint_p_sample_loop_progressive(self, modality, conditioning, latent_modality=None, latent_mask=None, spatio_temporal=None):
    """
    Generate samples from the model and yield intermediate samples from
    each timestep of diffusion.

    Arguments are the same as p_sample_loop().
    Returns a generator over dicts, where each dict is the return value of
    p_sample().
    """
    shape = latent_modality.shape

    image_after_step = torch.randn(
        shape,
        device=self.device
    )
    
    self.gt_noises = None  # reset for next image

    pred_xstart = None

    idx_wall = -1
    
    sample_idxs = defaultdict(lambda: 0)

    # schedule_jump_params
    t_T = 250
    n_sample = 1
    jump_length = 10
    jump_n_sample = 10

    times = _repaint_get_schedule_jump(self, t_T=t_T, n_sample=n_sample, jump_length=jump_length, jump_n_sample=jump_n_sample)

    time_pairs = list(zip(times[:-1], times[1:]))

    for t_last, t_cur in time_pairs:
        idx_wall += 1
        t_last_t = torch.tensor([t_last] * shape[0],  # pylint: disable=not-callable
                                device=self.device)

        if t_cur < t_last:  # reverse
            with torch.no_grad():
                image_before_step = image_after_step.clone()
                out = _repaint_p_sample(
                    self,
                    image_after_step,
                    t_last_t,
                    modality,
                    conditioning,
                    latent_modality=latent_modality,
                    latent_mask=latent_mask,
                    spatio_temporal=spatio_temporal
                )
                image_after_step = out["sample"]
                pred_xstart = out["pred_xstart"]

                sample_idxs[t_cur] += 1

                yield out

        else:
            t_shift = 1

            image_before_step = image_after_step.clone()
            image_after_step = _repaint_undo(self, image_before_step, image_after_step, est_x_0=out['pred_xstart'], t=t_last_t+t_shift, debug=False)
            pred_xstart = out["pred_xstart"]
    
def _repaint_p_sample(self, x, t, modality, conditioning, latent_modality=None, latent_mask=None, spatio_temporal=None):
    """
    Sample x_{t-1} from the model at the given timestep.

    :param model: the model to sample from.
    :param x: the current tensor at x_{t-1}.
    :param t: the value of t, starting at 0 for the first diffusion step.
    :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
    :param denoised_fn: if not None, a function which applies to the
        x_start prediction before it is used to sample.
    :param cond_fn: if not None, this is a gradient function that acts
                    similarly to the model.
    :param model_kwargs: if not None, a dict of extra keyword arguments to
        pass to the model. This can be used for conditioning.
    :return: a dict containing the following keys:
                - 'sample': a random sample from the model.
                - 'pred_xstart': a prediction of x_0.
    """

    mask_np = latent_mask.cpu().numpy()
    dilated_mask_np = binary_dilation(mask_np, iterations=1)
    gt_keep_mask = 1 - torch.from_numpy(dilated_mask_np).to(latent_mask.device).float().clamp(0, 1)
    gt = latent_modality  # model_kwargs['gt']

    alpha_cumprod = _repaint_extract_into_tensor(self, self.alphas_cumprod, t, x.shape)

    gt_weight = torch.sqrt(alpha_cumprod)
    gt_part = gt_weight * gt

    noise_weight = torch.sqrt((1 - alpha_cumprod))
    noise_part = noise_weight * torch.randn_like(x)

    weighed_gt = gt_part + noise_part

    x = gt_keep_mask * weighed_gt + (1 - gt_keep_mask) * x

    noise_pred = self._predict_noise(
        x,  # x is the current sample
        t,  # t is the current timestep
        modality=modality,
        conditioning=conditioning
    )
    step_result = self.scheduler_inference.step(noise_pred, t[0], x)

    sample = step_result[0]
    pred_xstart = step_result[1]

    result = {
        "sample": sample,
        "pred_xstart": pred_xstart,
        "gt": latent_modality
    }
    
    return result

def _repaint_get_schedule_jump(self, t_T, n_sample, jump_length, jump_n_sample,
                    jump2_length=1, jump2_n_sample=1,
                    jump3_length=1, jump3_n_sample=1,
                    start_resampling=100000000):

    jumps = {}
    for j in range(0, t_T - jump_length, jump_length):
        jumps[j] = jump_n_sample - 1

    jumps2 = {}
    for j in range(0, t_T - jump2_length, jump2_length):
        jumps2[j] = jump2_n_sample - 1

    jumps3 = {}
    for j in range(0, t_T - jump3_length, jump3_length):
        jumps3[j] = jump3_n_sample - 1

    t = t_T
    ts = []

    while t >= 1:
        t = t-1
        ts.append(t)

        if (
            t + 1 < t_T - 1 and
            t <= start_resampling
        ):
            for _ in range(n_sample - 1):
                t = t + 1
                ts.append(t)

                if t >= 0:
                    t = t - 1
                    ts.append(t)

        if (
            jumps3.get(t, 0) > 0 and
            t <= start_resampling - jump3_length
        ):
            jumps3[t] = jumps3[t] - 1
            for _ in range(jump3_length):
                t = t + 1
                ts.append(t)

        if (
            jumps2.get(t, 0) > 0 and
            t <= start_resampling - jump2_length
        ):
            jumps2[t] = jumps2[t] - 1
            for _ in range(jump2_length):
                t = t + 1
                ts.append(t)
            jumps3 = {}
            for j in range(0, t_T - jump3_length, jump3_length):
                jumps3[j] = jump3_n_sample - 1

        if (
            jumps.get(t, 0) > 0 and
            t <= start_resampling - jump_length
        ):
            jumps[t] = jumps[t] - 1
            for _ in range(jump_length):
                t = t + 1
                ts.append(t)
            jumps2 = {}
            for j in range(0, t_T - jump2_length, jump2_length):
                jumps2[j] = jump2_n_sample - 1

            jumps3 = {}
            for j in range(0, t_T - jump3_length, jump3_length):
                jumps3[j] = jump3_n_sample - 1

    ts.append(-1)

    _repaint_check_times(self, ts, -1, t_T)

    return ts

def _repaint_check_times(self, times, t_0, t_T):
    # Check end
    assert times[0] > times[1], (times[0], times[1])

    # Check beginning
    assert times[-1] == -1, times[-1]

    # Steplength = 1
    for t_last, t_cur in zip(times[:-1], times[1:]):
        assert abs(t_last - t_cur) == 1, (t_last, t_cur)

    # Value range
    for t in times:
        assert t >= t_0, (t, t_0)
        assert t <= t_T, (t, t_T)

def _repaint_extract_into_tensor(self, arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

def _repaint_get_named_beta_schedule(self, schedule_name, num_diffusion_timesteps, use_scale):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.

        if use_scale:
            scale = 1000 / num_diffusion_timesteps
        else:
            scale = 1

        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )

def _repaint_undo(self, image_before_step, img_after_model, est_x_0, t, debug=False):
    return _repaint__undo(self, img_after_model, t)

def _repaint__undo(self, img_out, t):
    beta = _repaint_extract_into_tensor(self, self.betas, t, img_out.shape)

    img_in_est = torch.sqrt(1 - beta) * img_out + \
        torch.sqrt(beta) * torch.randn_like(img_out)

    return img_in_est