# Latent Diffusion Model

This repository contains a framework for training and evaluating a Latent Diffusion Model (LDM) on 3d brain imaging data.

## Core Components

- **Autoencoder Finetuning**: [`train_autoencoder.py`](/ldm/train_autoencoder.py)
- **LDM Training**: [`train_diffusion.py`](/ldm/train_diffusion.py)
- **LDM Inference**: [`inference_diffusion.py`](/ldm/inference_diffusion.py)

## Getting Started

### Installation

This project uses `uv` for dependency management. Install dependencies with:

```sh
uv sync
```

### Usage

The diffusion model can be trained from scratch or from a checkpoint. The evaluation script currently only supports the inpainting task.