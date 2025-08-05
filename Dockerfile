#Dockerfile

FROM --platform=linux/amd64 pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime
LABEL authors="JonasWeidner"

RUN apt-get update -y && \
    apt-get install -y curl

# Install uv
RUN curl -Ls https://astral.sh/uv/install.sh | sh

# Copy Python version and dependency files
COPY .python-version .
COPY pyproject.toml .
COPY uv.lock .

# Install dependencies with uv
RUN ~/.cargo/bin/uv sync

# Copy source code and other necessary files
COPY requirements.txt .
COPY tools tools/
COPY checkpoints checkpoints/
COPY main.py .
COPY gbm_bench gbm_bench/
COPY maisi maisi/
COPY models/diffusion_big_ddpm_v2.ckpt models/
COPY utils utils/
COPY inference_diffusion.py .

CMD ["uv", "run", "inference_diffusion.py", "--mode", "inference_challenge", "--dir_data_challenge", "/input", "--dir_output_model", "/output"]