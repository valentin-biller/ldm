# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -

# # 3D AutoencoderKL
#
# This demo is a toy example of how to use MONAI's AutoencoderKL. In particular, it uses the Autoencoder with a Kullback-Leibler regularisation as implemented by Rombach et. al [1].
#
# [1] Rombach et. al "High-Resolution Image Synthesis with Latent Diffusion Models" https://arxiv.org/pdf/2112.10752.pdf
#
# This tutorial was based on:
#
# [Brain tumor 3D segmentation with MONAI](https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/brats_segmentation_3d.ipynb)


# !python -c "import monai" || pip install -q "monai-weekly[tqdm, nibabel]"
# !python -c "import matplotlib" || pip install -q matplotlib
# %matplotlib inline

# ## Setup imports

# +
import os
import shutil
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from monai import transforms
from monai.apps import DecathlonDataset
from monai.config import print_config
from monai.data import DataLoader
from monai.networks.layers import Act
from monai.utils import first, set_determinism
from torch.cuda.amp import autocast
from tqdm import tqdm

from generative.losses import PatchAdversarialLoss, PerceptualLoss
from generative.networks.nets import AutoencoderKL, PatchDiscriminator

from pathlib import Path  # valentin
from monai.data import Dataset  # valentin
from monai.transforms import CenterSpatialCropd  # valentin

print_config()
# -

# for reproducibility purposes set a seed
set_determinism(42)

# ## Setup a data directory and download dataset
#
# Specify a `MONAI_DATA_DIRECTORY` variable, where the data will be downloaded. If not specified a temporary directory will be used.

# directory = os.environ.get("MONAI_DATA_DIRECTORY")
# root_dir = tempfile.mkdtemp() if directory is None else directory
# print(root_dir)

# ## Download the training set

# Note: The DecatholonDataset has 7GB. So make sure that you have enought space when running the next line

# valentin
batch_size = 2
lr_generator = 1e-4
lr_discriminator = 1e-6 # 5e-4
# valentin

# valentin
root_dir = Path('/vol/miltank/users/bilv/2025_challenge/data')
t1n_files = list(root_dir.rglob("**/BraTS2021_*-t1n.nii.gz"))  # valentin
print(len(t1n_files))

data_dicts = [
    {"image": file} for file in t1n_files
]
train_files, val_files = torch.utils.data.random_split(data_dicts, [0.8, 0.2])
# train_files = [train_files.dataset[i] for i in train_files.indices][:1]
# val_files = [val_files.dataset[i] for i in val_files.indices]
# valentin

# +
channel = 0  # 0 = Flair
assert channel in [0, 1, 2, 3], "Choose a valid channel"

train_transforms = transforms.Compose(
    [
        transforms.LoadImaged(keys=["image"]),
        transforms.EnsureChannelFirstd(keys=["image"]),
        transforms.Lambdad(keys="image", func=lambda x: x[channel, :, :, :]),
        transforms.EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
        transforms.EnsureTyped(keys=["image"]),
        transforms.Orientationd(keys=["image"], axcodes="RAS"),
        transforms.SpatialPadd(keys=["image"], spatial_size=(240, 240, 160)), # valentin
        # transforms.Spacingd(keys=["image"], pixdim=(2.4, 2.4, 2.2), mode=("bilinear")),
        # transforms.CenterSpatialCropd(keys=["image"], roi_size=(96, 96, 64)),
        transforms.ScaleIntensityRangePercentilesd(keys="image", lower=0, upper=99.5, b_min=0, b_max=1),
    ]
)
# train_ds = DecathlonDataset(
#     root_dir=root_dir,
#     task="Task01_BrainTumour",
#     section="training",
#     cache_rate=1.0,  # you may need a few Gb of RAM... Set to 0 otherwise
#     num_workers=4,
#     download=True,
#     seed=0,
#     transform=train_transforms,
# )
train_ds = Dataset(data=train_files, transform=train_transforms)  # valentin
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True)
print(f'Image shape {train_ds[0]["image"].shape}')
# -

# ## Visualise examples from the training set

# +
check_data = first(train_loader)

# Select the first image from the batch
img = check_data["image"][0]
fig, axs = plt.subplots(nrows=1, ncols=3)
for ax in axs:
    ax.axis("off")
ax = axs[0]
ax.imshow(img[0, ..., img.shape[3] // 2].rot90(), cmap="gray")
ax = axs[1]
ax.imshow(img[0, :, img.shape[2] // 2, ...].rot90(), cmap="gray")
ax = axs[2]
ax.imshow(img[0, img.shape[1] // 2, ...].rot90(), cmap="gray")
# -

# ## Download the validation set

val_transforms = transforms.Compose(
    [
        transforms.LoadImaged(keys=["image"]),
        transforms.EnsureChannelFirstd(keys=["image"]),
        transforms.Lambdad(keys="image", func=lambda x: x[channel, :, :, :]),
        transforms.EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
        transforms.EnsureTyped(keys=["image"]),
        transforms.Orientationd(keys=["image"], axcodes="RAS"),
        transforms.SpatialPadd(keys=["image"], spatial_size=(240, 240, 160)), # valentin
        # transforms.Spacingd(keys=["image"], pixdim=(2.4, 2.4, 2.2), mode=("bilinear")),
        # transforms.CenterSpatialCropd(keys=["image"], roi_size=(96, 96, 64)),
        transforms.ScaleIntensityRangePercentilesd(keys="image", lower=0, upper=99.5, b_min=0, b_max=1),
    ]
)
# val_ds = DecathlonDataset(
#     root_dir=root_dir,
#     task="Task01_BrainTumour",
#     section="validation",
#     cache_rate=1.0,  # you may need a few Gb of RAM... Set to 0 otherwise
#     num_workers=4,
#     download=True,
#     seed=0,
#     transform=val_transforms,
# )
val_ds = Dataset(data=val_files, transform=val_transforms)  # valentin
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, persistent_workers=True)
print(f'Image shape {val_ds[0]["image"].shape}')

# +
check_data = first(val_loader)

img = check_data["image"][0]
fig, axs = plt.subplots(nrows=1, ncols=3)
for ax in axs:
    ax.axis("off")
ax = axs[0]
ax.imshow(img[0, ..., img.shape[3] // 2].rot90(), cmap="gray")
ax = axs[1]
ax.imshow(img[0, :, img.shape[2] // 2, ...].rot90(), cmap="gray")
ax = axs[2]
ax.imshow(img[0, img.shape[1] // 2, ...].rot90(), cmap="gray")
# -

# ## Define the network

# +
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

model = AutoencoderKL(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    num_channels=(32, 64, 64),
    # num_channels=(32, 64, 128, 128),  # valentin -> ich muss einmal mehr downsamplen (und dort attention anwenden) weil er selbst ohne attention_levels irgendein attention module instantiiert, was zu oom führt
    latent_channels=3,
    # latent_channels=6,  # valentin
    num_res_blocks=1,
    norm_num_groups=32,
    # norm_num_groups=32,  # valentin
    # attention_levels=(False, False, True),
    attention_levels=(False, False, False),
    # attention_levels=(False, False, False, True),  # valentin
    with_encoder_nonlocal_attn=False,
    with_decoder_nonlocal_attn=False,
)
model.to(device)

discriminator = PatchDiscriminator(
    spatial_dims=3,
    num_layers_d=3,
    num_channels=32,
    in_channels=1,
    out_channels=1,
    kernel_size=4,
    activation=(Act.LEAKYRELU, {"negative_slope": 0.2}),
    norm="BATCH",
    bias=False,
    padding=1,
)

# valentin
from torch.nn.utils import spectral_norm
for name, module in discriminator.named_modules():
    if isinstance(module, (torch.nn.Conv3d, torch.nn.Linear)):
        spectral_norm(module)
        print(f"Applied spectral norm to: {name}")
# valentin

discriminator.to(device)

# valentin
from torchsummary import summary
print("=== AUTOENCODER MODEL SUMMARY ===")
summary(model, input_size=train_ds[0]["image"].shape)
print("\n=== DISCRIMINATOR MODEL SUMMARY ===")
summary(discriminator, input_size=train_ds[0]["image"].shape)
# valentin

# +
perceptual_loss = PerceptualLoss(spatial_dims=3, network_type="squeeze", fake_3d_ratio=0.25)
perceptual_loss.to(device)

adv_loss = PatchAdversarialLoss(criterion="least_squares")
adv_weight = 0.01  # 0.01
perceptual_weight = 0.001

optimizer_g = torch.optim.Adam(model.parameters(), lr_generator)
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr_discriminator)
# -

# valentin
# from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
# scheduler_g = ReduceLROnPlateau(optimizer_g, mode='min', factor=0.5, patience=10)
# scheduler_d = ReduceLROnPlateau(optimizer_d, mode='min', factor=0.5, patience=5)  # More aggressive for discriminator
# valentin

scaler_g = torch.cuda.amp.GradScaler()
scaler_d = torch.cuda.amp.GradScaler()

# ## Model training

# +
kl_weight = 1e-6
# n_epochs = 100
n_epochs = 1000  # valentin
val_interval = 10
epoch_recon_loss_list = []
epoch_gen_loss_list = []
epoch_disc_loss_list = []
val_recon_epoch_loss_list = []
intermediary_images = []
n_example_images = 4

# valentin
import os
import json
import shutil
import datetime
import mlflow


mlflow_params = {
    'dataset': root_dir,
    'kl_weight': kl_weight,
    'n_epochs': n_epochs,
    'val_interval': val_interval,
    'adv_weight': adv_weight,
    'perceptual_weight': perceptual_weight,
    'batch_size': batch_size,
    'lr_generator': lr_generator,
    'lr_discriminator': lr_discriminator,
}

identifier = 'ldm-autoencoder'
dir_master_thesis = '/vol/miltank/users/bilv/master-thesis'
dir_model = os.path.join(dir_master_thesis, 'models', identifier)
dir_output = os.path.join(dir_model, 'output')

mlflow_overview_json = os.path.join(dir_master_thesis, 'models', 'mlflow_overview.json')
mlflow_info_json = os.path.join(dir_model, 'mlflow_info.json')

mlflow_tracking_uri = os.path.join('file://' + dir_master_thesis, 'mlruns')
mlflow_experiment_name = identifier
mlflow.set_tracking_uri(mlflow_tracking_uri)
mlflow.set_experiment(mlflow_experiment_name)
mlflow_experiment_id = mlflow.get_experiment_by_name(mlflow_experiment_name).experiment_id

# Read name and counter of the last run
with open(mlflow_info_json, 'r') as f:
    mlflow_info = json.load(f)
mlflow_run_name = mlflow_info['run_name']
mlflow_run_counter_last = mlflow_info['run_counter']

# Set directory paths for the two latest runs
if mlflow_run_name is None:
    mlflow_run_counter_last_last = None
else:
    mlflow_run_counter_last_last = int(mlflow_run_counter_last) - 1
dir_output_run_last = os.path.join(dir_output, str(mlflow_run_name) + '_' + str(mlflow_run_counter_last))
dir_output_run_last_last = os.path.join(dir_output, str(mlflow_run_name) + '_' + str(mlflow_run_counter_last_last))

if identifier == 'med-ddpm':
    dir_output_model_folder = 'model'
elif identifier == 'MOTFM':
    dir_output_model_folder = 'latest'
elif identifier == 'pix2pix':
    dir_output_model_folder = 'checkpoints'
elif identifier == 'spade':
    dir_output_model_folder = 'checkpoints'
elif identifier == 'ldm-autoencoder':
    dir_output_model_folder = 'checkpoints'

dir_output_model_last = os.path.join(dir_output_run_last, dir_output_model_folder)
dir_output_model_last_last = os.path.join(dir_output_run_last_last, dir_output_model_folder)

def get_checkpoint_model(dir_output_model):
    if identifier == 'med-ddpm':
        checkpoint_model = sorted([f for f in os.listdir(dir_output_model)], key=lambda x: int(x.split('-')[1].split('.')[0]))[-1]
    else:
        checkpoint_model = sorted([f for f in os.listdir(dir_output_model)])[-1]
    return checkpoint_model

# Check if a new run should be started (if it's supposed to be the first run or if a first run was started but got interrupted (and therefore no checkpint exists))
if mlflow_run_name is None:
    resume = (False, None)
    mlflow_run_name = datetime.datetime.now().strftime("%d.%m.%Y-%H:%M:%S")
    mlflow_run_counter = '1'
# Check if a run should be continued
else:
    # Check if the last run has a checkpoint, if not, delete the last run and continue with the second last run
    if os.listdir(dir_output_model_last) == []:
        if os.path.exists(dir_output_run_last):
            shutil.rmtree(dir_output_run_last)
        mlflow_temp_runs = mlflow.search_runs(
            experiment_ids=[mlflow_experiment_id],
            filter_string=f"tags.mlflow.runName = '{mlflow_run_name + '_' + mlflow_run_counter_last}'"
        )
        if not mlflow_temp_runs.empty:    
            mlflow.delete_run(mlflow_temp_runs.iloc[0]['run_id'])
            shutil.rmtree(os.path.join(dir_master_thesis, 'mlruns', mlflow_experiment_id, mlflow_temp_runs.iloc[0]['run_id']))

        if mlflow_run_counter_last_last == 0:
            resume = (False, None)
            mlflow_run_name = datetime.datetime.now().strftime("%d.%m.%Y-%H:%M:%S")
            mlflow_run_counter = '1'
        else:
            checkpoint_model = get_checkpoint_model(dir_output_model_last_last)
            checkpoint_path = os.path.join(dir_output_model_last_last, checkpoint_model)
            mlflow_run_counter = mlflow_run_counter_last
            resume = (True, checkpoint_path)
    # If the last run has a checkpoint, continue with it
    else:
        checkpoint_model = get_checkpoint_model(dir_output_model_last)
        checkpoint_path = os.path.join(dir_output_model_last, checkpoint_model)
        mlflow_run_counter = str(int(mlflow_run_counter_last) + 1)
        resume = (True, checkpoint_path)
mlflow_run_name_counter = mlflow_run_name + '_' + mlflow_run_counter
dir_output_run = os.path.join(dir_output, mlflow_run_name_counter)
dir_output_model = os.path.join(dir_output_run, dir_output_model_folder)
os.makedirs(dir_output_run, exist_ok=True)
os.makedirs(dir_output_model, exist_ok=True)
with open(mlflow_info_json, 'w') as f:
    mlflow_info['run_name'] = mlflow_run_name
    mlflow_info['run_counter'] = mlflow_run_counter
    json.dump(mlflow_info, f)
if identifier == 'pix2pix':
    # # Setup MLflow logger for PyTorch Lightning
    # mlf_logger = MLFlowLogger(
    #     tracking_uri=mlflow_tracking_uri,
    #     experiment_name=mlflow_experiment_name,
    #     run_name=mlflow_run_name_counter
    # )
    # mlf_logger.log_hyperparams(mlflow_params)
    pass
else:
    mlflow.start_run(run_name=mlflow_run_name_counter)
    mlflow.log_params(mlflow_params)
if os.path.exists(mlflow_overview_json):
    with open(mlflow_overview_json, 'r') as f:
        mlflow_overview = json.load(f)
else:
    mlflow_overview = {}
with open(mlflow_overview_json, 'w') as f:
    mlflow_overview[identifier] = mlflow_experiment_id
    json.dump(mlflow_overview, f)
print('====================================================================================================')
print('=========================')
print(f'========================= {mlflow_run_name}_{mlflow_run_counter}')
print(f'========================= Resume: {resume}')
print('=========================')
print('====================================================================================================')
# valentin

# valentin
from utils import save_checkpoint, load_checkpoint
save_every_n_epochs = 1
if resume[0]:
    start_epoch, last_recon_loss, last_gen_loss, last_disc_loss = load_checkpoint(
        resume[1], model, discriminator, optimizer_g, optimizer_d, scaler_g, scaler_d
    )
    start_epoch += 1
else:
    start_epoch = 0
# valentin

# for epoch in range(n_epochs):
for epoch in range(start_epoch, n_epochs):  # valentin
    model.train()
    discriminator.train()
    epoch_loss = 0
    gen_epoch_loss = 0
    disc_epoch_loss = 0

    # valentin
    epoch_gen_grad_norm = 0
    epoch_disc_grad_norm = 0
    d_train_freq = 3  # Train discriminator every N iterations
    g_train_freq = 1  # Train generator every N iterations
    # valentin

    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=70)
    progress_bar.set_description(f"Epoch {epoch}")
    for step, batch in progress_bar:
        images = batch["image"].to(device)

        if step % g_train_freq == 0: # valentin

            optimizer_g.zero_grad(set_to_none=True)

            # Generator part
            with autocast(enabled=True):
                reconstruction, z_mu, z_sigma = model(images)
                logits_fake = discriminator(reconstruction.contiguous().float())[-1]

                recons_loss = F.l1_loss(reconstruction.float(), images.float())
                p_loss = perceptual_loss(reconstruction.float(), images.float())
                generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)

                kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3, 4])
                kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

                loss_g = recons_loss + (kl_weight * kl_loss) + (perceptual_weight * p_loss) + (adv_weight * generator_loss)

            scaler_g.scale(loss_g).backward()

            # valentin
            # gen_grad_norm, _ = compute_gradient_norm(model)
            # epoch_gen_grad_norm += gen_grad_norm
            # valentin
            # valentin
            scaler_g.unscale_(optimizer_g)
            gen_grad_norm, _ = compute_gradient_norm(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # valentin

            scaler_g.step(optimizer_g)
            scaler_g.update()

        if step % d_train_freq == 0:  # valentin

            # Discriminator part
            optimizer_d.zero_grad(set_to_none=True)

            with autocast(enabled=True):
                logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                logits_real = discriminator(images.contiguous().detach())[-1]
                loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

                loss_d = adv_weight * discriminator_loss

            scaler_d.scale(loss_d).backward()

            # valentin
            scaler_d.unscale_(optimizer_d)
            disc_grad_norm, _ = compute_gradient_norm(discriminator)
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
            # disc_grad_norm, _ = compute_gradient_norm(discriminator)
            # epoch_disc_grad_norm += disc_grad_norm
            # valentin

            scaler_d.step(optimizer_d)
            scaler_d.update()

        epoch_loss += recons_loss.item()
        gen_epoch_loss += generator_loss.item()
        disc_epoch_loss += discriminator_loss.item()

        progress_bar.set_postfix(
            {
                "recons_loss": epoch_loss / (step + 1),
                "gen_loss": gen_epoch_loss / (step + 1),
                "disc_loss": disc_epoch_loss / (step + 1),
            }
        )

        # valentin
        epoch_gen_grad_norm += gen_grad_norm
        epoch_disc_grad_norm += disc_grad_norm
        # valentin

        # valentin
        # log_gradient_stats(model, "generator")
        # log_gradient_stats(discriminator, "discriminator")
        # gen_healthy = check_gradient_health(model, "Generator", step, epoch)
        # disc_healthy = check_gradient_health(discriminator, "Discriminator", step, epoch)
        # if not (gen_healthy and disc_healthy):
        #     print("Consider adjusting learning rates or loss weights!")
        # valentin

    # valentin
    # avg_gen_grad_norm = epoch_gen_grad_norm / (step + 1)
    # avg_disc_grad_norm = epoch_disc_grad_norm / (step + 1)
    # gen_grad_stats = log_gradient_stats(model, "generator")
    # disc_grad_stats = log_gradient_stats(discriminator, "discriminator")
    # valentin

    # valentin
    # scheduler_g.step(gen_epoch_loss / (step + 1))
    # scheduler_d.step(disc_epoch_loss / (step + 1))
    # valentin

    epoch_recon_loss_list.append(epoch_loss / (step + 1))
    epoch_gen_loss_list.append(gen_epoch_loss / (step + 1))
    epoch_disc_loss_list.append(disc_epoch_loss / (step + 1))
    
    # valentin
    mlflow.log_metrics({
        'train_recon_loss': epoch_loss / (step + 1),
        'train_gen_loss': gen_epoch_loss / (step + 1),
        'train_disc_loss': disc_epoch_loss / (step + 1),
        'train_gen_grad_norm_avg': epoch_gen_grad_norm / (step + 1),
        'train_disc_grad_norm_avg': epoch_disc_grad_norm / (step + 1),
        'epoch': epoch
    }, step=epoch)
    # valentin

    if (epoch + 1) % val_interval == 0:
        print(f"Validation at epoch {epoch + 1}")  # valentin
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for val_step, batch in enumerate(val_loader, start=1):
                images = batch["image"].to(device)
                optimizer_g.zero_grad(set_to_none=True)

                reconstruction, z_mu, z_sigma = model(images)

                # valentin
                cropper = CenterSpatialCropd(keys=["image"], roi_size=(240, 240, 155))
                original_data = {"image": images[0]}
                reconstruction_data = {"image": reconstruction[0]}
                original_cropped = cropper(original_data)["image"]
                reconstruction_cropped = cropper(reconstruction_data)["image"]

                import nibabel as nib
                ref = nib.load("/vol/miltank/users/bilv/data/BraTS2021_00000/t1.nii.gz")
                original_nii = nib.Nifti1Image(original_cropped[0].cpu().numpy(), affine=ref.affine)
                reconstruction_nii = nib.Nifti1Image(reconstruction_cropped[0].cpu().numpy(), affine=ref.affine)

                nib.save(original_nii, f'{'/'.join(dir_output_model.split('/')[:-1])}/{epoch+1:04d}_original.nii.gz')
                nib.save(reconstruction_nii, f'{'/'.join(dir_output_model.split('/')[:-1])}/{epoch+1:04d}_reconstruction.nii.gz')
                # valentin

                # get the first sammple from the first validation batch for visualisation
                # purposes
                if val_step == 1:
                    intermediary_images.append(reconstruction[:n_example_images, 0])

                recons_loss = F.l1_loss(reconstruction.float(), images.float())

                val_loss += recons_loss.item()

        val_loss /= val_step
        val_recon_epoch_loss_list.append(val_loss)

        # valentin
        mlflow.log_metrics({
            'val_loss': val_loss,
        }, step=epoch)
        # valentin

    # valentin
    if (epoch + 1) % save_every_n_epochs == 0:
        save_checkpoint(
            model, discriminator, optimizer_g, optimizer_d, scaler_g, scaler_d,
            epoch, epoch_loss / (step + 1), gen_epoch_loss / (step + 1), disc_epoch_loss / (step + 1),
            dir_output_model
        )
    # valentin

# progress_bar.close()
# -
# ## Evaluate the trainig

plt.figure()
val_samples = np.linspace(val_interval, n_epochs, int(n_epochs / val_interval))
plt.plot(np.linspace(1, n_epochs, n_epochs), epoch_recon_loss_list, label="Train")
plt.plot(val_samples, val_recon_epoch_loss_list, label="Validation")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.show()
plt.close()

plt.title("Adversarial Training Curves", fontsize=20)
plt.plot(np.linspace(1, n_epochs, n_epochs), epoch_gen_loss_list, color="C0", linewidth=2.0, label="Generator")
plt.plot(np.linspace(1, n_epochs, n_epochs), epoch_disc_loss_list, color="C1", linewidth=2.0, label="Discriminator")
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.xlabel("Epochs", fontsize=16)
plt.ylabel("Loss", fontsize=16)
plt.legend(prop={"size": 14})
plt.show()

# ### Visualise some reconstruction images

# +
# get the first 5 examples to plot
n_evaluations = 5

fig, axs = plt.subplots(nrows=n_evaluations, ncols=3, constrained_layout=True, figsize=(8, 6))


# Remove ticks
for ax in axs.flatten():
    ax.set_xticks([])
    ax.set_yticks([])


for image_n in range(n_evaluations):
    axs[image_n, 0].imshow(
        intermediary_images[image_n][0, ..., intermediary_images[image_n].shape[3] // 2].cpu(), cmap="gray"
    )
    axs[image_n, 1].imshow(
        intermediary_images[image_n][0, :, intermediary_images[image_n].shape[2] // 2, ...].cpu().rot90(), cmap="gray"
    )
    axs[image_n, 2].imshow(
        intermediary_images[image_n][0, intermediary_images[image_n].shape[1] // 2, ...].cpu().rot90(), cmap="gray"
    )
    axs[image_n, 0].set_ylabel(f"Epoch {val_samples[image_n]:.0f}")
# -

fig, ax = plt.subplots(nrows=1, ncols=2)
ax[0].imshow(images[0, channel, ..., images.shape[2] // 2].cpu(), vmin=0, vmax=1, cmap="gray")
ax[0].axis("off")
ax[0].title.set_text("Inputted Image")
ax[1].imshow(reconstruction[0, channel, ..., reconstruction.shape[2] // 2].detach().cpu(), vmin=0, vmax=1, cmap="gray")
ax[1].axis("off")
ax[1].title.set_text("Reconstruction")
plt.show()

# ## Clean up data directory
#
# Remove directory if a temporary storage was used

# if directory is None:
#     shutil.rmtree(root_dir)