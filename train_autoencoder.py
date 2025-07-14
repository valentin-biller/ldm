import sys
sys.path.append('/vol/miltank/users/bilv/ldm/maisi')
sys.path.append('/vol/miltank/users/bilv/master-thesis/models')

import os
import math
from tqdm import tqdm

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.nn import L1Loss, MSELoss
from torch.optim import lr_scheduler
from torch.amp import GradScaler, autocast
import pytorch_lightning as pl

import mlflow
from mlflow_continue import MLFlowContinue
from utils.data import DataModule
from utils.maisi_autoencoder import MaisiAutoencoder

from monai.networks.nets import PatchDiscriminator
from monai.losses.perceptual import PerceptualLoss
from monai.losses.adversarial_loss import PatchAdversarialLoss
from monai.inferers.inferer import SimpleInferer, SlidingWindowInferer


def setup_distributed():
    """Initialize distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank


def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


# Initialize distributed training
rank, world_size, local_rank = setup_distributed()
is_main_process = rank == 0

# Configuration
path_data = "/vol/miltank/users/bilv/data"
path_data_challenge = "/vol/miltank/datasets/glioma/brats_inpainting/ASNR-MICCAI-BraTS2023-Local-Synthesis-Challenge-Validation"
path_autoencoder = "/vol/miltank/users/bilv/ldm/maisi/maisi_vae.pt"
latent_shape = (60, 60, 40)

pl.seed_everything(42)
device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

amp = True
batch_size = 1
num_workers = 4
learning_rate = 1e-4

val_interval = 1
best_val_recon_epoch_loss = 10000000.0

total_step = 0
start_epoch = 0
max_epochs = 10000

perceptual_weight = 0.3
kl_weight = 1e-07
adv_weight = 0.1

val_batch_size = 1
val_patch_size = None
val_sliding_window_patch_size = [60, 60, 40]

# Setup MLflow continuation
mlflow_params = {
    "batch_size": batch_size,
    "learning_rate": learning_rate,
    "max_epochs": max_epochs,
    "num_workers": num_workers,
    # "precision": amp,
    # "conditioning_channels": 4,
    # "image_size": "240x240x155",
    # "latent_size": "60x60x40",
}
mlflow_continue = MLFlowContinue(
    identifier='ldm-autoencoder',
    mlflow_params=mlflow_params
)
resume, dir_output_model, mlf_logger = mlflow_continue.mlflow_continue()

# Initialize models
discriminator_norm = "INSTANCE"
discriminator = PatchDiscriminator(
    spatial_dims=3,
    num_layers_d=3,
    channels=32,
    in_channels=1,
    out_channels=1,
    norm=discriminator_norm,
).to(device)

if resume[0]:
    autoencoder = MaisiAutoencoder(path_autoencoder=dir_output_model + "/autoencoder.pt", device=device).model
    discriminator.load_state_dict(torch.load(dir_output_model + "/discriminator.pt", map_location=device))
else:
    autoencoder = MaisiAutoencoder(path_autoencoder=path_autoencoder, device=device).model

# Wrap models with DDP for multi-GPU
if world_size > 1:
    autoencoder = DDP(autoencoder, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    discriminator = DDP(discriminator, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

# Create data module
datamodule = DataModule(
    mode='training',
    path_data=path_data,
    path_data_challenge=path_data_challenge,
    latent_shape=latent_shape,
    batch_size=batch_size,
    num_workers=num_workers,
)
datamodule.setup(stage='fit')

# Add distributed samplers
train_sampler = DistributedSampler(
    datamodule.dataset_train, 
    num_replicas=world_size, 
    rank=rank
) if world_size > 1 else None

val_sampler = DistributedSampler(
    datamodule.dataset_val, 
    num_replicas=world_size, 
    rank=rank,
    shuffle=False
) if world_size > 1 else None

# Update dataloaders
dataloader_train = DataLoader(
    datamodule.dataset_train,
    batch_size=batch_size,
    shuffle=(train_sampler is None),
    num_workers=num_workers,
    sampler=train_sampler,
    pin_memory=True
)

dataloader_val = DataLoader(
    datamodule.dataset_val,
    batch_size=val_batch_size,
    shuffle=False,
    num_workers=num_workers,
    sampler=val_sampler,
    pin_memory=True
)


def psnr(input, target, max_val=1.0):
    mse = torch.mean((input - target) ** 2)
    return 20 * torch.log10(max_val / torch.sqrt(mse))

def KL_loss(z_mu, z_sigma):
    eps = 1e-10
    kl_loss = 0.5 * torch.sum(
        z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2) + eps) - 1,
        dim=list(range(1, len(z_sigma.shape))),
    )
    return torch.sum(kl_loss) / kl_loss.shape[0]

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

def loss_weighted_sum(losses):
    return losses["recons_loss"] + kl_weight * losses["kl_loss"] + perceptual_weight * losses["p_loss"]

def warmup_rule(epoch):
    # learning rate warmup rule
    if epoch < 10:
        return 0.01
    elif epoch < 20:
        return 0.1
    else:
        return 1.0

def reduce_tensor(tensor, world_size):
    """Reduce tensor across all processes"""
    if world_size > 1:
        tensor = tensor.clone()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor = tensor / world_size
    return tensor


# Initialize loss functions
intensity_loss = L1Loss(reduction="mean")
adv_loss = PatchAdversarialLoss(criterion="least_squares")
loss_perceptual = (
    PerceptualLoss(spatial_dims=3, network_type="squeeze", is_fake_3d=True, fake_3d_ratio=0.2).eval().to(device)
)

# Initialize optimizers
optimizer_g = torch.optim.Adam(
    params=autoencoder.parameters(), 
    lr=learning_rate, 
    eps=1e-06 if amp else 1e-08
)
optimizer_d = torch.optim.Adam(
    params=discriminator.parameters(), 
    lr=learning_rate, 
    eps=1e-06 if amp else 1e-08
)
    
scheduler_g = lr_scheduler.LambdaLR(optimizer_g, lr_lambda=warmup_rule)
scheduler_d = lr_scheduler.LambdaLR(optimizer_d, lr_lambda=warmup_rule)

scaler_g = GradScaler("cuda", init_scale=2.0**8, growth_factor=1.5)
scaler_d = GradScaler("cuda", init_scale=2.0**8, growth_factor=1.5)

# Setup validation inferer
val_inferer = (
    SlidingWindowInferer(
        roi_size=val_sliding_window_patch_size,
        sw_batch_size=1,
        progress=False,
        overlap=0.0,
        device=torch.device("cpu"),
        sw_device=device,
    )
    if val_sliding_window_patch_size
    else SimpleInferer()
)

# Training and validation loops
for epoch in range(start_epoch, max_epochs):
    # Set epoch for distributed sampler
    if world_size > 1:
        train_sampler.set_epoch(epoch)
    
    if is_main_process:
        print("lr:", scheduler_g.get_lr())
    
    autoencoder.train()
    discriminator.train()
    train_epoch_losses = {"recons_loss": 0, "kl_loss": 0, "p_loss": 0}

    progress_bar = tqdm(dataloader_train, desc=f"Epoch {epoch} Training", ncols=100) if is_main_process else dataloader_train
    for batch in progress_bar:
        images = batch["original_t1_autoencoder"].to(device, non_blocking=True).contiguous()
        optimizer_g.zero_grad(set_to_none=True)
        optimizer_d.zero_grad(set_to_none=True)
        
        with autocast("cuda", enabled=amp):
            # Train Generator
            reconstruction, z_mu, z_sigma = autoencoder(images)
            losses = {
                "recons_loss": intensity_loss(reconstruction, images),
                "kl_loss": KL_loss(z_mu, z_sigma),
                "p_loss": loss_perceptual(reconstruction.float(), images.float()),
            }
            logits_fake = discriminator(reconstruction.contiguous().float())[-1]
            generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
            loss_g = loss_weighted_sum(losses) + adv_weight * generator_loss

            if amp:
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

            if amp:
                scaler_d.scale(loss_d).backward()
                scaler_d.step(optimizer_d)
                scaler_d.update()
            else:
                loss_d.backward()
                optimizer_d.step()

        # Update progress bar only on main process
        if is_main_process and hasattr(progress_bar, 'set_postfix'):
            progress_bar.set_postfix({
                "loss_g": f"{loss_g.item():.4f}",
                "loss_d": f"{loss_d.item():.4f}"
            })

        # Log training loss
        total_step += 1
        for loss_name, loss_value in losses.items():
            if is_main_process:
                mlflow.log_metric(f"train_{loss_name}_iter", loss_value.item(), step=total_step)
            train_epoch_losses[loss_name] += loss_value.item()
        if is_main_process:
            mlflow.log_metric("train_adv_loss_iter", generator_loss.item(), step=total_step)
            mlflow.log_metric("train_fake_loss_iter", loss_d_fake.item(), step=total_step)
            mlflow.log_metric("train_real_loss_iter", loss_d_real.item(), step=total_step)

    # Synchronize losses across GPUs
    if world_size > 1:
        for key in train_epoch_losses:
            train_epoch_losses[key] = reduce_tensor(
                torch.tensor(train_epoch_losses[key], device=device), world_size
            ).item()

    scheduler_g.step()
    scheduler_d.step()
    
    # Calculate average losses
    for key in train_epoch_losses:
        train_epoch_losses[key] /= len(dataloader_train)
    if is_main_process:
        print(f"Epoch {epoch} train_vae_loss {loss_weighted_sum(train_epoch_losses)}: {train_epoch_losses}.")
        for loss_name, loss_value in train_epoch_losses.items():
            mlflow.log_metric(f"train_{loss_name}_epoch", loss_value, step=epoch)
        
        # Save models (only on main process)
        model_to_save = autoencoder.module if hasattr(autoencoder, 'module') else autoencoder
        discriminator_to_save = discriminator.module if hasattr(discriminator, 'module') else discriminator
        
        torch.save(model_to_save.state_dict(), dir_output_model + "/autoencoder.pt")
        torch.save(discriminator_to_save.state_dict(), dir_output_model + "/discriminator.pt")
        print("Save trained autoencoder to", dir_output_model)
        print("Save trained discriminator to", dir_output_model)

    # Validation
    if epoch % val_interval == 0:
        autoencoder.eval()
        val_epoch_losses = {"recons_loss": 0, "kl_loss": 0, "p_loss": 0}
        total_psnr = 0.0
        
        progress_bar = tqdm(dataloader_val, desc=f"Epoch {epoch} Validation", ncols=100) if is_main_process else dataloader_val
        for batch in progress_bar:
            with torch.no_grad():
                with autocast("cuda", enabled=amp):
                    images = batch["original_t1_autoencoder"]
                    reconstruction, z_mu, z_sigma = dynamic_infer(val_inferer, autoencoder, images)
                    reconstruction = reconstruction.to(device)

                    total_psnr += psnr(reconstruction, images.to(device)).item()

                    val_epoch_losses["recons_loss"] += intensity_loss(reconstruction, images.to(device)).item()
                    val_epoch_losses["kl_loss"] += KL_loss(z_mu, z_sigma).item()
                    val_epoch_losses["p_loss"] += loss_perceptual(reconstruction, images.to(device)).item()

        # Synchronize validation losses across GPUs
        if world_size > 1:
            for key in val_epoch_losses:
                val_epoch_losses[key] = reduce_tensor(
                    torch.tensor(val_epoch_losses[key], device=device), world_size
                ).item()
            total_psnr = reduce_tensor(torch.tensor(total_psnr, device=device), world_size).item()

        # Calculate average validation losses
        for key in val_epoch_losses:
            val_epoch_losses[key] /= len(dataloader_val)

        val_loss_g = loss_weighted_sum(val_epoch_losses)

        val_loss_g = reduce_tensor(torch.tensor(val_loss_g, device=device), world_size).item()  # multi-GPU support
        
        if is_main_process:
            print(f"Epoch {epoch} val_vae_loss {val_loss_g}: {val_epoch_losses}.")
            print(f"Epoch {epoch} val_psnr: {total_psnr / len(dataloader_val):.4f}")

            # Save best model only on main process
            if val_loss_g < best_val_recon_epoch_loss:
                best_val_recon_epoch_loss = val_loss_g
                trained_g_path_epoch = f"{dir_output_model}/autoencoder_epoch{epoch}.pt"
                model_to_save = autoencoder.module if hasattr(autoencoder, 'module') else autoencoder
                torch.save(model_to_save.state_dict(), trained_g_path_epoch)
                print("Got best val vae loss.")
                print("Save trained autoencoder to", trained_g_path_epoch)

            for loss_name, loss_value in val_epoch_losses.items():
                mlflow.log_metric(loss_name, loss_value, step=epoch)

        # Monitor scale_factor
        # We'd like to tune kl_weights in order to make scale_factor close to 1.
        # scale_factor_sample = 1.0 / z_mu.flatten().std()
        # mlflow.log_metric("val_one_sample_scale_factor", scale_factor_sample, step=epoch)

        # # Monitor reconstruction result
        # center_loc_axis = find_label_center_loc(images[0, 0, ...])
        # vis_image = get_xyz_plot(images[0, ...], center_loc_axis, mask_bool=False)
        # vis_recon_image = get_xyz_plot(reconstruction[0, ...], center_loc_axis, mask_bool=False)

        # tensorboard_writer.add_image(
        #     "val_orig_img",
        #     vis_image.transpose([2, 0, 1]),
        #     epoch,
        # )
        # tensorboard_writer.add_image(
        #     "val_recon_img",
        #     vis_recon_image.transpose([2, 0, 1]),
        #     epoch,
        # )

        # show_image(vis_image, title="val image")
        # show_image(vis_recon_image, title="val recon result")

# Clean up distributed training
cleanup_distributed()