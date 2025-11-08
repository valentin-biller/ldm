import torch
import schedulers_helper
from monai.networks.schedulers import DDPMScheduler, DDIMScheduler

class Scheduler():
    def __init__(self, scheduler_, num_train_timesteps, num_inference_steps, device):
        self.scheduler_ = scheduler_
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_steps = num_inference_steps
        self.device = device

        if self.scheduler_ in ['ddpm', 'ddim']:
            self.diffusion = None
            self.schedulers = {
                "ddpm": DDPMScheduler,
                "ddim": DDIMScheduler,
            }
            self.scheduler = self.schedulers[self.scheduler_](
                num_train_timesteps=self.num_train_timesteps,
                beta_start=0.0015,
                beta_end=0.0195,
                schedule="scaled_linear_beta",  # or "linear_beta" or "scaled_linear_beta" ?
                prediction_type="epsilon",
                clip_sample=False,
            )
        elif self.scheduler_ == 'iddpm':
            steps=4000  # TODO diffusion steps 4000
            learn_sigma=True
            sigma_small=False
            noise_schedule="cosine"  # TODO noise schedule cosine
            use_kl=False
            predict_xstart=False
            rescale_timesteps=False  # TODO rescale_learned_sigmas False
            rescale_learned_sigmas=False  # TODO rescale_timesteps False
            timestep_respacing=""


            betas = schedulers_helper.get_named_beta_schedule(noise_schedule, steps)
            if use_kl:
                loss_type = schedulers_helper.LossType.RESCALED_KL
            elif rescale_learned_sigmas:
                loss_type = schedulers_helper.LossType.RESCALED_MSE
            else:
                print('Here 1')
                loss_type = schedulers_helper.LossType.MSE
            if not timestep_respacing:
                print('Here 2')
                timestep_respacing = [steps]
            self.diffusion = schedulers_helper.SpacedDiffusion(
                use_timesteps=schedulers_helper.space_timesteps(steps, timestep_respacing),
                betas=betas,
                model_mean_type=(
                    schedulers_helper.ModelMeanType.EPSILON if not predict_xstart else schedulers_helper.ModelMeanType.START_X
                ),
                model_var_type=(
                    (
                        schedulers_helper.ModelVarType.FIXED_LARGE
                        if not sigma_small
                        else schedulers_helper.ModelVarType.FIXED_SMALL
                    )
                    if not learn_sigma
                    else schedulers_helper.ModelVarType.LEARNED_RANGE  # this one
                ),
                loss_type=loss_type,
                rescale_timesteps=rescale_timesteps,
            )
            self.scheduler = schedulers_helper.LossSecondMomentResampler(self.diffusion)  # TODO schedule_sampler loss-second-moment

        for key, value in self.scheduler.__dict__.items():
            if isinstance(value, torch.Tensor):
                self.scheduler.__dict__[key] = value.to(self.device)
    
# def run(self, batch, cond):
#     t, weights = self.scheduler.sample(batch_size)

#     compute_losses = functools.partial(
#         self.diffusion.training_losses,
#         self.ddp_model,
#         micro,
#         t,
#         model_kwargs=micro_cond,
#     )

#     losses = compute_losses()
#     if isinstance(self.scheduler, LossAwareSampler):
#         self.scheduler.update_with_local_losses(
#             t, losses["loss"].detach()
#         )

#     loss = (losses["loss"] * weights).mean()