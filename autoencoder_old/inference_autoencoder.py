#!/usr/bin/env python3
"""
Simple script to test the trained AutoencoderKL model.
Loads a sample image, runs it through the autoencoder, and saves the reconstruction.
"""

import torch
import numpy as np
import nibabel as nib
from pathlib import Path
from monai import transforms
from generative.networks.nets import AutoencoderKL
import matplotlib.pyplot as plt

def load_model_only(checkpoint_path, model):
    """Load only model weights from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    return model

def main():
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Paths
    root_dir = Path('/vol/miltank/users/bilv/2025_challenge/data')
    checkpoint_path = Path('/vol/miltank/users/bilv/master-thesis/models/ldm-autoencoder/output/25.06.2025-11:56:31_4/checkpoints/0100-checkpoint.pth')
    
    # Initialize model with same architecture as training
    model = AutoencoderKL(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        num_channels=(32, 64, 128, 128),
        latent_channels=6,
        num_res_blocks=1,
        norm_num_groups=32,
        attention_levels=(False, False, False, True),
    )
    
    # Load trained weights
    model = load_model_only(checkpoint_path, model)
    model.to(device)
    model.eval()
    
    # Find a sample image
    t1n_files = list(root_dir.rglob("**/BraTS2021_*-t1n.nii.gz"))
    print(len(t1n_files))
    
    sample_file = t1n_files[0]
    print(f"Using sample image: {sample_file}")
    
    # Define the same transforms as used in training
    channel = 0  # 0 = Flair
    test_transforms = transforms.Compose([
        transforms.LoadImaged(keys=["image"]),
        transforms.EnsureChannelFirstd(keys=["image"]),
        transforms.Lambdad(keys="image", func=lambda x: x[channel, :, :, :]),
        transforms.EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
        transforms.EnsureTyped(keys=["image"]),
        transforms.Orientationd(keys=["image"], axcodes="RAS"),
        transforms.SpatialPadd(keys=["image"], spatial_size=(240, 240, 160)),
        transforms.ScaleIntensityRangePercentilesd(keys="image", lower=0, upper=99.5, b_min=0, b_max=1),
    ])
    
    # Load and preprocess the image
    data_dict = {"image": sample_file}
    transformed_data = test_transforms(data_dict)
    image = transformed_data["image"]
    
    print(f"Image shape: {image.shape}")
    
    # Add batch dimension and move to device
    image_batch = image.unsqueeze(0).to(device)
    
    # Run through autoencoder
    with torch.no_grad():
        reconstruction, z_mu, z_sigma = model(image_batch)
    
    # Move back to CPU and remove batch dimension
    original = image_batch[0, 0].cpu().numpy()
    reconstructed = reconstruction[0, 0].cpu().numpy()
    
    print(f"Original image range: [{original.min():.3f}, {original.max():.3f}]")
    print(f"Reconstructed image range: [{reconstructed.min():.3f}, {reconstructed.max():.3f}]")
    
    # Save the reconstructed image as NIfTI
    output_dir = Path('/vol/miltank/users/bilv/ldm/autoencoder/output')
    output_dir.mkdir(exist_ok=True)
    
    # Create NIfTI image with same affine as original
    template = nib.load(sample_file)
    
    original_nii = nib.Nifti1Image(
        original, 
        affine=template.affine,
    )
    reconstructed_nii = nib.Nifti1Image(
        reconstructed,
        affine=template.affine,
    )
    
    output_file_original = output_dir / f"original_{sample_file.stem}.gz"
    output_file_reconstructed = output_dir / f"reconstructed_{sample_file.stem}.gz"
    nib.save(original_nii, output_file_original)
    nib.save(reconstructed_nii, output_file_reconstructed)
    
    # Calculate and print reconstruction metrics
    mse = np.mean((original - reconstructed) ** 2)
    mae = np.mean(np.abs(original - reconstructed))
    
    print("\nReconstruction metrics:")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    
    print("\nLatent space statistics:")
    print(f"Latent mean shape: {z_mu.shape}")
    print(f"Latent std shape: {z_sigma.shape}")
    print(f"Mean of latent means: {z_mu.mean().item():.6f}")
    print(f"Mean of latent stds: {z_sigma.mean().item():.6f}")

if __name__ == "__main__":
    main()
