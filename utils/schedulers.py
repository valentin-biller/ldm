import torch
import schedulers_helper
from monai.networks.schedulers import DDPMScheduler, DDIMScheduler

class Scheduler():
    def __init__(self, scheduler_, num_train_timesteps, num_inference_steps):
        self.scheduler_ = scheduler_
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_steps = num_inference_steps

        if self.scheduler_ == 'ddpm':
            self.diffusion = None
            self.schedulers = {
                "ddpm": DDPMScheduler,
                "ddim": DDIMScheduler,
            }
            self.scheduler_training = self.schedulers['ddpm'](
                num_train_timesteps=self.num_train_timesteps,
                beta_start=0.0015,
                beta_end=0.0195,
                schedule="scaled_linear_beta",  # or "linear_beta" or "scaled_linear_beta" ?
                prediction_type="epsilon",
                clip_sample=False,
            )
            self.scheduler_inference = self.schedulers['ddim'](
                num_train_timesteps=self.num_inference_steps,
                beta_start=0.0015,
                beta_end=0.0195,
                schedule="scaled_linear_beta",  # or "linear_beta" or "scaled_linear_beta" ?
                prediction_type="epsilon",
                clip_sample=False,
            )
        elif self.scheduler_ == 'iddpm':
            steps=self.num_train_timesteps  # diffusions steps 4000
            learn_sigma=True
            sigma_small=False
            noise_schedule="cosine"  # noise schedule cosine
            use_kl=False
            predict_xstart=False
            rescale_timesteps=False  # rescale_timesteps False
            rescale_learned_sigmas=False  # rescale_learned_sigmas False
            timestep_respacing=""

            betas = schedulers_helper.get_named_beta_schedule(noise_schedule, steps)
            if use_kl:
                loss_type = schedulers_helper.LossType.RESCALED_KL
            elif rescale_learned_sigmas:
                loss_type = schedulers_helper.LossType.RESCALED_MSE
            else:
                loss_type = schedulers_helper.LossType.MSE  # this condition is used
            if not timestep_respacing:
                timestep_respacing = [steps]  # this condition is used
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
                    else schedulers_helper.ModelVarType.LEARNED_RANGE  # this condition is used
                ),
                loss_type=loss_type,
                rescale_timesteps=rescale_timesteps,
            )

            self.scheduler = schedulers_helper.LossSecondMomentResampler(self.diffusion)  # schedule_sampler loss-second-moment

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