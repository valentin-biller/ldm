"""
Main training script for diffusion model. 

Autoencoder:
    - Input: 240x240x160 (Padded From: 240x240x155)
    - Latent: 60x60x40
    - Output: 240x240x160 (Cropped To: 240x240x155)
Diffusion:
    - Input: 60x60x40
    - Latent: 15x15x10
    - Output: 60x60x40
    - Conditioning: 4 channels
"""
import os
import sys
import argparse
from pathlib import Path
dir_current = Path(__file__).resolve().parent
dir_models = dir_current.parent / 'master-thesis' / 'models'
sys.path.append(str(dir_models))

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

from mlflow.system_metrics.system_metrics_monitor import SystemMetricsMonitor

from mlflow_continue import MLFlowContinue
from utils.data import DataModule
from utils.trainer import LatentDiffusion


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train latent diffusion inpainting model")
    
    parser.add_argument("--debug", action="store_true", help="Run in debug mode with limited data.")

    parser.add_argument("--model", type=str, default="big", choices=["small", "big", "big_old"], help="Model Type")
    parser.add_argument("--scheduler", type=str, default="ddpm", choices=["ddpm", "ddim"], help="Scheduler Type")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning Rate")

    # Mode options:
    #   training                          - Standard training (also used for standard inference)
    #   inpainting_inference              - [Inpainting] Save reconstructions for own test dataset
    #   inpainting_inference_conditioning - [Inpainting] Save reconstructions for own test dataset (non-zero tumor concentration)
    #   inpainting_inference_challenge    - [Inpainting] Save reconstructions for challenge dataset
    #   baseline                          - Dataloader for baseline dataset (cropped volumes)
    parser.add_argument("--mode", type=str, default="training", choices=["training"], help="Mode of Operation")
    parser.add_argument("--oversampling", action="store_false", help="Disable Oversampling")

    parser.add_argument("--batch_size", type=int, default=4, help="Batch Size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of Workers")

    parser.add_argument("--max_epochs", type=int, default=100000, help="Maximum Epochs")
    parser.add_argument("--precision", type=str, default="16", help="Training Precision")
    parser.add_argument("--save_samples_every", type=int, default=10, help="Save samples every N epochs.")
    
    args = parser.parse_args()

    pl.seed_everything(42)
    torch.set_float32_matmul_precision('medium')
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    dir_data = "/vol/miltank/users/bilv/data"
    dir_data_challenge = "/vol/miltank/datasets/glioma/brats_inpainting/ASNR-MICCAI-BraTS2023-Local-Synthesis-Challenge-Validation"
    
    path_autoencoder = dir_current / 'maisi' / 'maisi_vae.pt'

    print('Model', args.model)
    print('Scheduler', args.scheduler)
    print('Learning Rate', args.learning_rate)
    print('Mode', args.mode)
    print('Oversampling', args.oversampling)
    print('Batch Size', args.batch_size)
    print('Num Workers', args.num_workers)
    if not args.debug:
        # Setup MLflow continuation
        mlflow_params = {
            "model": args.model,
            "scheduler": args.scheduler,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "max_epochs": args.max_epochs,
            "num_workers": args.num_workers,
            "precision": args.precision,
            "conditioning_channels": 4,
            "image_size": "240x240x155",
            "latent_size": "60x60x40",
        }
        if local_rank == 0:
            mlflow_continue = MLFlowContinue(
                identifier='ldm-diffusion',
                mlflow_params=mlflow_params
            )
            resume, dir_output_model, mlf_logger = mlflow_continue.mlflow_continue()
        else:
            resume = (False, None)
            dir_output_model = None
            mlf_logger = None
        # Setup MLflow continuation
    else:
        resume = (False, None)
        dir_output_model = dir_current / 'output/output'
    
    if resume[0]:
        model = LatentDiffusion.load_from_checkpoint(
            resume[1],
            path_autoencoder=path_autoencoder,
            dir_output_model=dir_output_model,
            model=args.model,
            scheduler=args.scheduler,
            learning_rate=args.learning_rate,
        )
    else:
        model = LatentDiffusion(
            path_autoencoder=path_autoencoder,
            dir_output_model=dir_output_model,
            model=args.model,
            scheduler=args.scheduler,
            learning_rate=args.learning_rate,
        )

    # Create data module
    datamodule = DataModule(
        debug=args.debug,
        mode=args.mode,
        oversampling=args.oversampling,
        dir_data=dir_data,
        dir_data_challenge=dir_data_challenge,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=dir_output_model,
        filename="checkpoint-{epoch:06d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
        verbose=True
    )
    
    early_stopping = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=20,
        verbose=True
    )
    
    lr_monitor = LearningRateMonitor(
        logging_interval="epoch"
    )
    
    class MLFlowSystemMonitorCallback(pl.Callback):
        def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
            # Only run on rank 0 and only if MLFlowLogger is available
            if trainer.global_rank == 0:
                print("Starting system metrics monitoring...", flush=True)
                self.system_monitor = SystemMetricsMonitor(
                    run_id=trainer.logger.run_id,
                )
                self.system_monitor.start()
            else:
                print("Not Starting system metrics monitoring...", flush=True)
                self.system_monitor = None

        def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
            if hasattr(self, 'system_monitor') and self.system_monitor is not None:
                self.system_monitor.finish()

    callbacks = [checkpoint_callback]
    if not args.debug and local_rank == 0 and mlf_logger is not None:
        system_monitor = MLFlowSystemMonitorCallback()
        callbacks.append(system_monitor)
        callbacks.append(lr_monitor) 

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=torch.cuda.device_count(),
        strategy="ddp_find_unused_parameters_true",
        precision=args.precision,
        gradient_clip_val=1.0,
        logger=mlf_logger if not args.debug else False,
        callbacks=callbacks,
        log_every_n_steps=10,
        check_val_every_n_epoch=args.save_samples_every,
        enable_model_summary=True,
        enable_progress_bar=True,
        deterministic=False,  # had it originally set to True, but caused issues with avg_pool3d_backward_cuda
    )
    
    trainer.fit(model, datamodule)
    print(f"Training completed! Best model saved at: {trainer.checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()