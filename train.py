"""
Main training script for diffusion model. 

Autoencoder:
    - Input: 256x256x160 (Padded From: 240x240x155)
    - Latent: 4x64x64x40
    - Output: 256x256x160 (Cropped To: 240x240x155)
Diffusion:
    - UNet
        - Input: 4x64x64x40
        - Latent: 4x8x8x5
        - Output: 4x64x64x40
    - ControlNet: 4x256x256x160 (Padded From: 240x240x155)
"""
import os
import sys
import argparse
from pathlib import Path
dir_current = Path(__file__).resolve().parent
dir_models = dir_current.parent / 'master-thesis' / 'models'
sys.path.append(str(dir_models))

import torch
import lightning.pytorch as L
from lightning.pytorch.callbacks import ModelCheckpoint

from mlflow.system_metrics.system_metrics_monitor import SystemMetricsMonitor

from mlflow_continue import MLFlowContinue
from utils.data import DataModule
from utils.trainer import LatentDiffusion


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train latent diffusion inpainting model")
    
    parser.add_argument("--mlflow_use", action="store_false", help="Disable MLFlow logging.")
    parser.add_argument("--mlflow_info_slot", type=int, default=None, help="Slot number for MLFlow continuation.")

    parser.add_argument("--use_latents", action="store_false", help="Disable using precomputed latents.")
    parser.add_argument("--use_distribution_shift", action="store_true", help="Activate using distribution shift.")

    parser.add_argument("--model", type=str, default="unet", choices=["unet", "dit"], help="Model Type")
    parser.add_argument("--mask_conditioning", type=str, default="64", choices=["64", "32", "none"], help="Mask Conditioning")
    parser.add_argument("--modality_conditioning", action="store_false", help="Disable Modality Conditioning")
    parser.add_argument("--denoising", type=str, default="own", choices=["own", "repaint"], help="Denoising Type")
    parser.add_argument("--scheduler", type=str, default="ddpm", choices=["ddpm", "iddpm", "flow_matching"], help="Scheduler Type")
    parser.add_argument("--latent_shape", type=str, default="4,64,64,40", choices=["4,64,64,40", "4,32,32,20"], help="Latent Shape")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning Rate")

    # Mode options:
    #   training                          - Standard training (also used for standard inference)
    #   inpainting_inference_healthy      - [Inpainting] Save reconstructions for own test dataset (healthy: zero tumor concentration)
    #   inpainting_inference_tumor        - [Inpainting] Save reconstructions for own test dataset (tumor: non-zero tumor concentration)
    #   baseline                          - Dataloader for baseline dataset (cropped volumes)
    parser.add_argument("--mode", type=str, default="training", choices=["training"], help="Mode of Operation")
    parser.add_argument("--oversampling", action="store_false", help="Disable Oversampling (Training)")
    parser.add_argument("--undersampling", action="store_false", help="Disable Undersampling (Validation)")

    parser.add_argument("--batch_size", type=int, default=16, help="Batch Size")  # batch size of 16 and big model fits into 40 GB GPU if precision set to bf16-mixed / for controlnet 8
    parser.add_argument("--num_workers", type=int, default=16, help="Number of Workers")  # for 16 workers around 128G mem is needed (no speed improvements / degradations) / for controlnet 4
    parser.add_argument("--accumulate_grad_batches", type=int, default=2, help="Accumulate Grad Batches")  # no speed improvements / degradations / for controlnet 4

    parser.add_argument("--precision", type=str, default="bf16-mixed", help="Training Precision")  # 32 extremly slows down training and needs more memory, use bf16-mixed
    parser.add_argument("--save_samples_every", type=int, default=10, help="Save samples every N epochs.")
    
    args = parser.parse_args()

    L.seed_everything(42)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    # torch.set_float32_matmul_precision('medium')  # no speed improvements / degradations

    dir_data = "/vol/miltank/users/bilv/data"
    dir_autoencoder = dir_current / 'autoencoder' / 'checkpoints'
    # dir_data_challenge = "/vol/miltank/datasets/glioma/brats_inpainting/ASNR-MICCAI-BraTS2023-Local-Synthesis-Challenge-Validation"
    

    # processing argparser arguments (type conversions)
    if args.mask_conditioning == "none":
        args.mask_conditioning = None
    else:
        args.mask_conditioning = int(args.mask_conditioning)
    args.latent_shape = tuple(int(s) for s in args.latent_shape.split(','))
    assert args.latent_shape in [(4, 64, 64, 40), (4, 32, 32, 20)], "Latent shape must be (4,64,64,40) or (4,32,32,20)."


    # automatic checks and changes
    if args.mask_conditioning == 64:
        args.batch_size = 8 if args.model == "unet" else 4
        args.num_workers = 4
        args.accumulate_grad_batches = 4
    if args.model == "dit":
        assert args.mask_conditioning is not None, "DiT requires mask conditioning."
        assert args.modality_conditioning, "DiT requires modality conditioning."


    # automatic path autoencoder
    if args.latent_shape == (4, 64, 64, 40):
        path_autoencoder = dir_autoencoder / 'maisi_vae.pt'
    elif args.latent_shape == (4, 32, 32, 20):
        path_autoencoder = dir_autoencoder / 'f8d16_vae.pt'


    print('MLFlow Use:', args.mlflow_use)
    print('MLFlow Info Slot:', args.mlflow_info_slot)
    print('Use Latents:', args.use_latents)
    print('Use Distribution Shift:', args.use_distribution_shift)
    print('Model:', args.model)
    print('Mask Conditioning:', args.mask_conditioning)
    print('Modality Conditioning:', args.modality_conditioning)
    print('Denoising:', args.denoising)
    print('Scheduler:', args.scheduler)
    print('Latent Shape:', args.latent_shape)
    print('Learning Rate:', args.learning_rate)
    print('Mode:', args.mode)
    print('Oversampling:', args.oversampling)
    print('Undersampling:', args.undersampling)
    print('Batch Size:', args.batch_size)
    print('Num Workers:', args.num_workers)
    print('Accumulate Grad Batches:', args.accumulate_grad_batches)
    if args.mlflow_use:
        # Setup MLFlow continuation
        mlflow_params = {
            "mlflow_use": args.mlflow_use,
            "mlflow_info_slot": args.mlflow_info_slot,
            "use_latents": args.use_latents,
            "use_distribution_shift": args.use_distribution_shift,
            "model": args.model,
            "mask_conditioning": args.mask_conditioning ,
            "modality_conditioning": args.modality_conditioning,
            "denoising": args.denoising,
            "scheduler": args.scheduler,
            "latent_shape": args.latent_shape,
            "learning_rate": args.learning_rate,
            "mode": args.mode,
            "oversampling": args.oversampling,
            "undersampling": args.undersampling,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
        }
        if local_rank == 0:
            mlflow_continue = MLFlowContinue(
                identifier='ldm-diffusion',
                mlflow_params=mlflow_params,
                mlflow_info_slot=args.mlflow_info_slot, 
            )
            resume, dir_output_model, mlf_logger = mlflow_continue.mlflow_continue()
        else:
            resume = (False, None)
            dir_output_model = None
            mlf_logger = None
    else:
        resume = (False, None)
        dir_output_model = dir_current / 'output/output'
    
    if resume[0]:
        model = LatentDiffusion.load_from_checkpoint(
            resume[1],
            path_autoencoder=path_autoencoder,
            dir_output_model=dir_output_model,
            dir_ema=resume[1].parent,
            use_latents=args.use_latents,
            use_distribution_shift=args.use_distribution_shift,
            model_=args.model,
            mask_conditioning=args.mask_conditioning,
            modality_conditioning=args.modality_conditioning,
            denoising=args.denoising,
            scheduler_=args.scheduler,
            latent_shape=args.latent_shape,
            learning_rate=args.learning_rate,
        )
    else:
        model = LatentDiffusion(
            path_autoencoder=path_autoencoder,
            dir_output_model=dir_output_model,
            dir_ema=None,
            use_latents=args.use_latents,
            use_distribution_shift=args.use_distribution_shift,
            model_=args.model,
            mask_conditioning=args.mask_conditioning,
            modality_conditioning=args.modality_conditioning,
            denoising=args.denoising,
            scheduler_=args.scheduler,
            latent_shape=args.latent_shape,
            learning_rate=args.learning_rate,
        )

    # Create data module
    datamodule = DataModule(
        dir_data=dir_data,
        use_latents=args.use_latents,
        mask_conditioning=args.mask_conditioning,
        modality_conditioning=args.modality_conditioning,
        latent_shape=args.latent_shape,
        mode=args.mode,
        oversampling=args.oversampling,
        undersampling=args.undersampling,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    
    # Setup callbacks
    best_checkpoint_callback = ModelCheckpoint(
        dirpath=dir_output_model,
        filename="checkpoint-{epoch:06d}-{val/loss_l1:.4f}",
        monitor="val/loss_l1",
        mode="min",
        save_top_k=3,
        save_last=False,
        verbose=True
    )
    every_checkpoint_callback = ModelCheckpoint(
        dirpath=dir_output_model,
        filename="last",
        every_n_epochs=1,
        save_top_k=0,
        save_last=True,
        verbose=True,
        save_on_train_epoch_end=True,
    )
    
    class ProfilerLoggerCallback(L.Callback):
        def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
            if trainer.is_global_zero and trainer.profiler is not None:
                summary = trainer.profiler.summary()
                print(summary, flush=True)

    class MLFlowSystemMonitorCallback(L.Callback):
        def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
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

        def on_fit_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
            if hasattr(self, 'system_monitor') and self.system_monitor is not None:
                self.system_monitor.finish()

    callbacks = [ProfilerLoggerCallback(), every_checkpoint_callback]
    if args.save_samples_every > 0:
        callbacks.append(best_checkpoint_callback)
    if args.mlflow_use and local_rank == 0 and mlf_logger is not None:
        callbacks.append(MLFlowSystemMonitorCallback())

    # Create trainer
    trainer = L.Trainer(
        max_steps=1_000_000,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=torch.cuda.device_count(),
        strategy="ddp",  # no speed differences between ddp and ddp_find_unused_parameters_true
        precision=args.precision,
        gradient_clip_val=1.0,  # should be the best clipping value as most of the gradient norms are at around 0.5 and lower ?
        accumulate_grad_batches=args.accumulate_grad_batches,
        logger=mlf_logger if args.mlflow_use else False,
        callbacks=callbacks,
        log_every_n_steps=1,
        limit_val_batches=1.0 if args.save_samples_every > 0 else 0,
        check_val_every_n_epoch=args.save_samples_every if args.save_samples_every > 0 else 1,
        enable_model_summary=True,
        enable_progress_bar=True,
        deterministic=False,  # had it originally set to True, but caused issues with avg_pool3d_backward_cuda
        profiler="simple",
    )
    
    trainer.fit(model, datamodule)
    print(f"Training completed! Best model saved at: {trainer.checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()