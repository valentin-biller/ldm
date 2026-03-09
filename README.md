# Latent Generative Model for 3D Brain MRI Synthesis

A framework for training and evaluating **Latent Flow Matching** and **Latent Diffusion Models** on 3D brain MRI data, with support for conditional synthesis, tumor inpainting and longitudinal generation.

> Paper: [arxiv.org/abs/2603.04058](https://arxiv.org/abs/2603.04058)

---

## Overview

The model operates in two stages:

1. **Autoencoder** — compresses 240×240×155 MRI volumes into compact latents (e.g. 4×64×64×40)
2. **Latent Generative Model** — a UNet or DiT trained on latents, conditioned on tumour growth model and tissue segmentation

Supported schedulers: `ddpm` · `flow_matching`

---

## Installation

```sh
uv sync
```

---

## Usage

```sh
uv run train.py
  --model unet
  --scheduler flow_matching
```

---

## Project Structure

```
lgm/
├── train.py               # Main Training Script
├── utils/
│   ├── data.py            # DataModule & Dataset
│   └── trainer.py         # LightningModule
└── autoencoder/           # Autoencoder
```