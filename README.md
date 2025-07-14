====================================================================================================
====================================================================================================
====================================================================================================

GENERATED NOT FINAL !!!

====================================================================================================
====================================================================================================
====================================================================================================



# Latent Diffusion Model for Brain MRI Inpainting

This project implements a 3D latent diffusion model for brain MRI inpainting using PyTorch Lightning and MLflow.

## Overview

The model performs inpainting of brain MRI images in latent space using:
- **AutoencoderKL**: Encodes images to 6-channel latent space (6, 30, 30, 20)
- **3D ADM**: Adapted Ablated Diffusion Model for 3D latent diffusion
- **4-channel conditioning**: Growth model + tissue segmentation
- **Inpainting masks**: Binary masks indicating regions to inpaint

## Architecture

### Data Flow
1. **Input**: Voided T1n image (1, 240, 240, 156) → pad → (1, 240, 240, 160)
2. **Encoding**: AutoencoderKL → latent space (6, 30, 30, 20)
3. **Conditioning**: 
   - Growth model (thresholded at 0.2) → Channel 0
   - Tissue segmentation (one-hot) → Channels 1-3
   - Combined: (4, 30, 30, 20)
4. **Diffusion**: 3D ADM with conditioning
5. **Decoding**: AutoencoderKL → (1, 240, 240, 160) → crop → (1, 240, 240, 156)

### Model Components
- **3D UNet**: Modified ADM with 3 downsampling levels, 2x channel multipliers
- **Frozen Autoencoder**: Pre-trained weights are frozen during training
- **DDPM Scheduler**: 1000 training timesteps, linear noise schedule

## Data Structure

### Input Data
```
data/
├── BraTS2021_00000/
│   ├── BraTS2021_00000-mask-0000.nii.gz      # Inpainting mask
│   ├── BraTS2021_00000-mask-0001.nii.gz      # Inpainting mask  
│   ├── BraTS2021_00000-mask-0002.nii.gz      # Inpainting mask
│   ├── BraTS2021_00000-t1n-voided-0000.nii.gz  # Voided input
│   ├── BraTS2021_00000-t1n-voided-0001.nii.gz  # Voided input
│   ├── BraTS2021_00000-t1n-voided-0002.nii.gz  # Voided input
│   └── BraTS2021_00000-t1n.nii.gz            # Ground truth
└── ...
```

### Conditioning Data
```
conditioning_data/
├── growth_model/
│   └── BraTS2021_00000.nii.gz
└── tissue_segmentation/
    └── BraTS2021_00000.nii.gz
```

## Training Strategy

### Data Preparation
- **3 samples per patient**: Using mask-0000, mask-0001, mask-0002
- **80/20 split**: By patient IDs to prevent data leakage
- **Conditioning**: Growth model (threshold 0.2) + tissue segmentation (one-hot)

### Training Process
1. Load voided input and encode to latent space
2. Add noise to clean latent representation
3. Apply inpainting mask (original in non-masked, noisy in masked regions)
4. Predict noise using 3D ADM with conditioning
5. Compute loss only on masked regions
6. Decode result and compare to ground truth in image space

## Usage

### Training
```bash
# Basic training
python main.py

# Custom parameters
python main.py \
    --batch_size 4 \
    --learning_rate 2e-4 \
    --max_epochs 200 \
    --gpus 2 \
    --accumulate_grad_batches 2
```

### Configuration
Edit `config.py` to modify:
- Data paths
- Model architecture
- Training hyperparameters
- Augmentation settings

### Key Parameters
- `batch_size`: Batch size (default: 2)
- `learning_rate`: Learning rate (default: 1e-4)
- `max_epochs`: Maximum training epochs (default: 100)
- `num_inference_steps`: DDPM sampling steps during validation (default: 50)
- `save_samples_every`: Save sample images every N epochs (default: 10)

## Features

### MLflow Integration
- Automatic experiment tracking
- Hyperparameter logging
- Metric monitoring
- Artifact storage (checkpoints, samples)

### Checkpointing
- Best model saving (based on validation loss)
- Early stopping (patience: 20 epochs)
- Automatic resume from checkpoints

### Validation & Sampling
- DDPM sampling during validation
- Sample image generation (NIfTI format)
- Image-space loss computation

### Augmentations
- Latent space augmentations (flips)
- Image space augmentations (rotations, intensity)
- Noise augmentation in latent space

## Output

### Checkpoints
- Saved in `./checkpoints/`
- Best model: `brain_mri_inpainting-{epoch}-{val_loss}.ckpt`
- Last checkpoint for resuming

### Sample Images
- Saved every N epochs in `./samples/epoch_{N}/`
- Format: `{patient_id}_{mask_id}_inpainted.nii.gz`
- Includes both inpainted and target images

### MLflow Tracking
- Experiments logged to `./mlruns/`
- View with: `mlflow ui`

## Requirements

```bash
pip install torch torchvision pytorch-lightning
pip install monai[all] monai-generative
pip install mlflow nibabel
pip install numpy matplotlib tqdm
```

## Model Details

### Architecture Changes from 2D ADM
- **3D Convolutions**: All conv2d → conv3d
- **Reduced Downsampling**: 3 levels instead of 4 for small latent size
- **Channel Multipliers**: 2x base channels (128 instead of 64)
- **Attention Resolutions**: Adjusted for 30x30x20 latent space

### Loss Function
- **Training**: MSE loss on noise prediction (masked regions only)
- **Validation**: MSE loss on decoded images (full image)

### Memory Optimization
- Mixed precision training (16-bit)
- Gradient accumulation
- Gradient clipping
- Frozen autoencoder weights

## Monitoring

### Training Metrics
- `train_loss`: Training loss per step/epoch
- `val_loss`: Validation loss per epoch
- `learning_rate`: Current learning rate

### Sample Quality
- Visual inspection of generated samples
- Comparison with ground truth images
- Inpainting quality in masked regions

## Troubleshooting

### Common Issues
1. **CUDA OOM**: Reduce batch_size or increase accumulate_grad_batches
2. **Slow training**: Increase num_workers or reduce num_inference_steps
3. **Poor quality**: Check autoencoder loading, adjust learning_rate
4. **Data loading errors**: Verify file paths and data structure

### Debug Mode
```bash
# Single batch overfitting test
python main.py --max_epochs 10 --batch_size 1 --overfit_batches 1
```
