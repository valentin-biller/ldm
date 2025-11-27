import sys
from pathlib import Path
dir_current = Path(__file__).resolve().parent
dir_autoencoder = dir_current.parent
dir_utils = dir_current.parent.parent / 'utils'
dir_models = dir_current.parent.parent.parent / 'master-thesis' / 'models'
sys.path.append(str(dir_autoencoder))
sys.path.append(str(dir_utils))
sys.path.append(str(dir_models))

from data import DataModule
from mlflow_continue import MLFlowContinue
from maisi_f8_autoencoder import MaisiF8Autoencoder

import math
import mlflow
import nibabel as nib
from tqdm import tqdm
from monai import transforms
from torchmetrics.image import PeakSignalNoiseRatio
autoencoder_crop = transforms.CenterSpatialCrop(roi_size=(240, 240, 155))


import argparse
import os
from pathlib import Path

import torch
from monai.networks.nets import PatchDiscriminator
from monai.inferers.inferer import SimpleInferer, SlidingWindowInferer
from monai.losses.adversarial_loss import PatchAdversarialLoss
from monai.losses.perceptual import PerceptualLoss
from monai.utils import set_determinism
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from torch.nn import L1Loss, MSELoss
from torch.optim import lr_scheduler


# Configuration
args = argparse.Namespace()

args.batch_size = 2  # TODO
args.patch_size = [64, 64, 64]
args.val_batch_size = 1
args.val_patch_size = None
args.val_sliding_window_patch_size = [96, 96, 64]
args.lr = 1e-4
args.perceptual_weight = 0.3
args.kl_weight = 1e-7
args.adv_weight = 0.1
args.recon_loss = "l1"
args.val_interval = 10  # TODO
args.cache = 0.5
args.amp = True
args.n_epochs = 1000  # TODO

args.spatial_dims = 3


dir_data = '/vol/miltank/users/bilv/data'


####################################################################################################
# Initialize mlflow and networks
####################################################################################################

# initialize mlflow
mlflow_params = vars(args)
mlflow_continue = MLFlowContinue(
    identifier='ldm-autoencoder',
    mlflow_params=mlflow_params
)
resume, dir_output_model, mlf_logger = mlflow_continue.mlflow_continue()

# initialize networks
device = torch.device("cuda")

discriminator_norm = "INSTANCE"
discriminator = PatchDiscriminator(
    spatial_dims=args.spatial_dims,
    num_layers_d=3,
    channels=32,
    in_channels=1,
    out_channels=1,
    norm=discriminator_norm,
).to(device)

if resume[0]:
    autoencoder = MaisiF8Autoencoder(path_autoencoder=str(resume[1]).replace('discriminator', 'autoencoder'), device=device).model
    discriminator.load_state_dict(torch.load(resume[1], map_location=device))
else:
    autoencoder = MaisiF8Autoencoder(path_autoencoder=None, device=device).model

trained_g_path = os.path.join(dir_output_model, "autoencoder.pt")
trained_d_path = os.path.join(dir_output_model, "discriminator.pt")

####################################################################################################
# Set deterministic training for reproducibility
####################################################################################################

set_determinism(seed=42)

####################################################################################################
# Build training dataset and data loader
####################################################################################################

# Create data module
datamodule = DataModule(
    dir_data=dir_data,
    use_latents=None,
    mask_conditioning=None,
    modality_conditioning=None,
    latent_shape=None,
    mode='autoencoder',
    oversampling=True,
    undersampling=True,
    batch_size=args.batch_size,
    num_workers=4,
)
datamodule.setup(stage='fit')

dataloader_train = datamodule.dataloader_train
dataloader_val = datamodule.dataloader_val

####################################################################################################
# Training config
####################################################################################################

# config loss and loss weight
if args.recon_loss == "l2":
    intensity_loss = MSELoss()
    print("Use l2 loss")
else:
    intensity_loss = L1Loss(reduction="mean")
    print("Use l1 loss")
adv_loss = PatchAdversarialLoss(criterion="least_squares")

loss_perceptual = (
    PerceptualLoss(spatial_dims=3, network_type="squeeze", is_fake_3d=True, fake_3d_ratio=0.2).eval().to(device)
)

def KL_loss(z_mu, z_sigma):
    eps = 1e-10
    kl_loss = 0.5 * torch.sum(
        z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2) + eps) - 1,
        dim=list(range(1, len(z_sigma.shape))),
    )
    return torch.sum(kl_loss) / kl_loss.shape[0]

def loss_weighted_sum(losses):
    return losses["recons_loss"] + args.kl_weight * losses["kl_loss"] + args.perceptual_weight * losses["p_loss"]


# config optimizer and lr scheduler
optimizer_g = torch.optim.Adam(params=autoencoder.parameters(), lr=args.lr, eps=1e-06 if args.amp else 1e-08)
optimizer_d = torch.optim.Adam(params=discriminator.parameters(), lr=args.lr, eps=1e-06 if args.amp else 1e-08)


# please adjust the learning rate warmup rule based on your dataset and n_epochs
def warmup_rule(epoch):
    # learning rate warmup rule
    if epoch < 10:
        return 0.01
    elif epoch < 20:
        return 0.1
    else:
        return 1.0


scheduler_g = lr_scheduler.LambdaLR(optimizer_g, lr_lambda=warmup_rule)
scheduler_d = lr_scheduler.LambdaLR(optimizer_d, lr_lambda=warmup_rule)


# set AMP scaler
if args.amp:
    # test use mean reduction for everything
    # scaler_g = GradScaler("cuda", init_scale=2.0**8, growth_factor=1.5)
    # scaler_d = GradScaler("cuda", init_scale=2.0**8, growth_factor=1.5)
    scaler_g = GradScaler(init_scale=2.0**8, growth_factor=1.5)
    scaler_d = GradScaler(init_scale=2.0**8, growth_factor=1.5)

####################################################################################################
# Training
####################################################################################################

# Initialize variables
val_interval = args.val_interval
best_val_recon_epoch_loss = 10000000.0
total_step = 0
start_epoch = 0
max_epochs = args.n_epochs

# Setup validation inferer
val_inferer = (
    SlidingWindowInferer(
        roi_size=args.val_sliding_window_patch_size,
        sw_batch_size=1,
        progress=False,
        overlap=0.0,
        device=torch.device("cpu"),
        sw_device=device,
    )
    if args.val_sliding_window_patch_size
    else SimpleInferer()
)

def dynamic_infer(inferer, model, images):
    if torch.numel(images[0:1, 0:1, ...]) <= math.prod(inferer.roi_size):
        return model(images)
    else:
        # Extract the spatial dimensions from the images tensor (H, W, D)
        spatial_dims = images.shape[2:]
        orig_roi = inferer.roi_size

        # Check that roi has the same number of dimensions as spatial_dims
        if len(orig_roi) != len(spatial_dims):
            raise ValueError(f"ROI length ({len(orig_roi)}) does not match spatial dimensions ({len(spatial_dims)}).")

        # Iterate and adjust each ROI dimension
        adjusted_roi = [min(roi_dim, img_dim) for roi_dim, img_dim in zip(orig_roi, spatial_dims)]
        inferer.roi_size = adjusted_roi
        output = inferer(network=model, inputs=images)
        inferer.roi_size = orig_roi
        return output


# Training and validation loops
for epoch in range(start_epoch, max_epochs):
    print("lr:", scheduler_g.get_lr())
    autoencoder.train()
    discriminator.train()
    train_epoch_losses = {"recons_loss": 0, "kl_loss": 0, "p_loss": 0}

    for batch in tqdm(dataloader_train):
        # images = batch["image"].to(device).contiguous()
        images = batch["padded_modality"].to(device).contiguous()
        optimizer_g.zero_grad(set_to_none=True)
        optimizer_d.zero_grad(set_to_none=True)
        with autocast("cuda", enabled=args.amp):
            # Train Generator
            reconstruction, z_mu, z_sigma = autoencoder(images)
            losses = {
                "recons_loss": intensity_loss(reconstruction, images),
                "kl_loss": KL_loss(z_mu, z_sigma),
                "p_loss": loss_perceptual(reconstruction.float(), images.float()),
            }
            logits_fake = discriminator(reconstruction.contiguous().float())[-1]
            generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
            loss_g = loss_weighted_sum(losses) + args.adv_weight * generator_loss

            if args.amp:
                scaler_g.scale(loss_g).backward()
                scaler_g.unscale_(optimizer_g)
                scaler_g.step(optimizer_g)
                scaler_g.update()
            else:
                loss_g.backward()
                optimizer_g.step()

            # Train Discriminator
            logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
            loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
            logits_real = discriminator(images.contiguous().detach())[-1]
            loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
            loss_d = (loss_d_fake + loss_d_real) * 0.5

            if args.amp:
                scaler_d.scale(loss_d).backward()
                scaler_d.step(optimizer_d)
                scaler_d.update()
            else:
                loss_d.backward()
                optimizer_d.step()

        # Log training loss
        total_step += 1
        for loss_name, loss_value in losses.items():
            # tensorboard_writer.add_scalar(f"train_{loss_name}_iter", loss_value.item(), total_step)
            mlflow.log_metric(f"train_{loss_name}_iter", loss_value.item(), step=total_step)
            train_epoch_losses[loss_name] += loss_value.item()
        # tensorboard_writer.add_scalar("train_adv_loss_iter", generator_loss, total_step)
        # tensorboard_writer.add_scalar("train_fake_loss_iter", loss_d_fake, total_step)
        # tensorboard_writer.add_scalar("train_real_loss_iter", loss_d_real, total_step)
        mlflow.log_metric("train_adv_loss_iter", generator_loss.item(), step=total_step)
        mlflow.log_metric("train_fake_loss_iter", loss_d_fake.item(), step=total_step)
        mlflow.log_metric("train_real_loss_iter", loss_d_real.item(), step=total_step)

    scheduler_g.step()
    scheduler_d.step()
    for key in train_epoch_losses:
        train_epoch_losses[key] /= len(dataloader_train)
    print(f"Epoch {epoch} train_vae_loss {loss_weighted_sum(train_epoch_losses)}: {train_epoch_losses}.")
    for loss_name, loss_value in train_epoch_losses.items():
        # tensorboard_writer.add_scalar(f"train_{loss_name}_epoch", loss_value, epoch)
        mlflow.log_metric(f"train_{loss_name}_epoch", loss_value, step=epoch)
    torch.save(autoencoder.state_dict(), trained_g_path)
    torch.save(discriminator.state_dict(), trained_d_path)
    print("Save trained autoencoder to", trained_g_path)
    print("Save trained discriminator to", trained_d_path)

    # Validation
    if epoch % val_interval == 0:
        autoencoder.eval()
        val_epoch_losses = {"recons_loss": 0, "kl_loss": 0, "p_loss": 0}
        val_loader_iter = iter(dataloader_val)
        for batch in tqdm(dataloader_val):
            with torch.no_grad():
                with autocast("cuda", enabled=args.amp):
                    # images = batch["image"]
                    images = batch["padded_modality"]
                    reconstruction, z_mu, z_sigma = dynamic_infer(val_inferer, autoencoder, images)
                    reconstruction = reconstruction.to(device)

                    ### saving & psnr ###
                    dir_output = dir_output_model.parent / "images" / f"epoch_{epoch}"

                    patients = batch["patient"]
                    affines = batch["affine"].cpu().float().numpy()
                    modality = batch["modality"]
                    normalized = batch["normalized_modality"].cpu().float().numpy()

                    for i, (patient, modality_) in enumerate(zip(patients, modality)):
                        dir_patient = dir_output / patient
                        dir_patient.mkdir(exist_ok=True, parents=True)

                        affine = affines[i]
                        
                        normalized_modality = normalized[i][0]  # (240, 240, 155)
                        reconstructed_modality = autoencoder_crop(reconstruction[i])[0].detach().cpu().float().numpy()  # (240, 240, 155)

                        normalized_modality_nii = nib.Nifti1Image(normalized_modality, affine)
                        reconstructed_modality_nii = nib.Nifti1Image(reconstructed_modality, affine)

                        nib.save(normalized_modality_nii, dir_patient / f"normalized_{modality_}.nii.gz")
                        nib.save(reconstructed_modality_nii, dir_patient / f"reconstructed_{modality_}.nii.gz")
                    
                    psnr = PeakSignalNoiseRatio()
                    print(f"PSNR: {psnr(preds=reconstruction.to('cpu'), target=images.to('cpu'))}")
                    ### saving & psnr ###

                    val_epoch_losses["recons_loss"] += intensity_loss(reconstruction, images.to(device)).item()
                    val_epoch_losses["kl_loss"] += KL_loss(z_mu, z_sigma).item()
                    val_epoch_losses["p_loss"] += loss_perceptual(reconstruction, images.to(device)).item()

        for key in val_epoch_losses:
            val_epoch_losses[key] /= len(dataloader_val)

        val_loss_g = loss_weighted_sum(val_epoch_losses)
        print(f"Epoch {epoch} val_vae_loss {val_loss_g}: {val_epoch_losses}.")

        if val_loss_g < best_val_recon_epoch_loss:
            best_val_recon_epoch_loss = val_loss_g
            trained_g_path_epoch = f"{trained_g_path[:-3]}_epoch{epoch}.pt"
            torch.save(autoencoder.state_dict(), trained_g_path_epoch)
            print("Got best val vae loss.")
            print("Save trained autoencoder to", trained_g_path_epoch)

        for loss_name, loss_value in val_epoch_losses.items():
            # tensorboard_writer.add_scalar(loss_name, loss_value, epoch)
            mlflow.log_metric(loss_name, loss_value, step=epoch)

        # Monitor scale_factor
        # We'd like to tune kl_weights in order to make scale_factor close to 1.
        scale_factor_sample = 1.0 / z_mu.flatten().std()
        # tensorboard_writer.add_scalar("val_one_sample_scale_factor", scale_factor_sample, epoch)
        mlflow.log_metric("val_one_sample_scale_factor", scale_factor_sample.item(), step=epoch)
