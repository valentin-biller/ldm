# GBMbench

Glioblastoma Model Benchmark (working title) is a work in progress with the goal of assessing the possible benefit of Glioblastoma models for better radiotherapy planning.


## Features

- Standardized processing pipeline for Brain MRIs
- Extensible benchmark framework for dockered Glioblastoma models
- Easy-to-use minimal API

## Setting up

### Prerequisites

- **Docker**: Required for running BRATS and custom Glioblastoma models. See [Docker and NVIDIA Container Toolkit Setup](#docker-and-nvidia-container-toolkit-setup).
- **dicom2niix**: Required if you plan to process raw DICOM data.

### Installation

```bash
git clone https://github.com/LMZimmer/gbm_bench.git
cd gbm_bench
pip install .
```

### Docker and NVIDIA Container Toolkit Setup

- **Docker**: Installation instructions on the official [website](https://docs.docker.com/get-docker/)
- **NVIDIA Container Toolkit**: Refer to the [NVIDIA install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) and the official [GitHub page](https://github.com/NVIDIA/nvidia-container-toolkit)


## Adding new growth models

This repository can be used to perform inference or benchmark with your own tumor growth model. To this end, you need to create a docker image of your growth model. The following sections serve as guideline on how the image should be created. 

### Directory structure

Input and output data are passed to/from the container using mounted directories:

**Input:**

```bash
/mlcube_io0
   ┗ Patient-00000
      ┣ 00000-t1c.nii.gz
      ┣ 00000-gm.nii.gz
      ┣ 00000-wm.nii.gz
      ┣ 00000-csf.nii.gz
      ┣ 00000-tumorseg.nii.gz
      ┗ 00000-pet.nii.gz
```

**Output:**

```bash
/mlcube_io1
   ┗ 00000.nii.gz
```

### Dockerfile Example

Ensure the container adheres to the above I/O structure. An example Dockerfile could be:

```dockerfile
# Image and environment variables
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install python
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev git && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --no-cache-dir --upgrade pip

WORKDIR /app

# Install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your code to workdir
COPY . .
ENTRYPOINT ["python3", "inference.py"]
```

## Adding new datasets

Datasets are handled by the `LongitudinalDataset` class in `utils.parsing`. This class supports:

- Parsing from a specific directory structure
- Reading/writing dataset paths via JSON

The recommended approach is to provide a compatible `.json` file. Example: [`data/datasets/rhuh.json`](data/datasets/rhuh.json). Automatic parsing is also supported, though naming consistency (e.g., for `preop`, `postop`, `followup`) can be a challenge.
