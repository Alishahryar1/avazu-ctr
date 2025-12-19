"""Training script for CTR prediction model."""

import pyperclip
from typing import Any, cast
from torch.optim import optimizer
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from config import CONFIG, seed_everything
from src.processing.data_processor import (
    load_metadata,
    get_parquet_path,
    get_parquet_row_count,
)
from src.processing.dataset import ParquetFullDataset
from src.models.architectures import create_model
from src.training.schedulers import LRSchedulerWithWarmup
from src.training.evaluator import evaluate


def train():
    # Cache paths early for type checker (avoids unbound-name errors in finally blocks)
    models_path = CONFIG["models_path"]
    seed_everything(CONFIG["seed"])

    # 1. Load Metadata (data stays in parquet)
    print("=" * 80)
    print("AVAZU CTR PREDICTION - MEMORY-EFFICIENT TRAINING")
    print("=" * 80)
    print("\nStep 1: Loading Metadata...")
    try:
        vocab_sizes, feature_names = load_metadata()
    except FileNotFoundError:
        print(
            "Processed data not found. Please run 'python data_processor.py' to generate it."
        )
        return

    train_parquet = get_parquet_path("train")
    total_samples = get_parquet_row_count("train")

    print(f"Train parquet: {train_parquet}")
    print(f"Total samples: {total_samples:,}")
    print(f"Number of features: {len(feature_names)}")

    # 2. Create Train/Validation Split (batch-level, not row-level)
    print("\nStep 2: Creating Train/Validation Split...")

    # Create dataset - Loads entire file into memory
    full_dataset = ParquetFullDataset(
        parquet_path=train_parquet, feature_cols=feature_names, label_col="click"
    )

    # Check if validation is enabled
    use_validation = CONFIG["validation_split"] > 0

    if use_validation:
        # Random split
        indices = np.arange(len(full_dataset))
        np.random.seed(CONFIG["seed"])  # Reproducible split
        np.random.shuffle(indices)

        split_idx = int(len(full_dataset) * (1 - CONFIG["validation_split"]))
        train_indices = indices[:split_idx].tolist()
        val_indices = indices[split_idx:].tolist()

        print(f"Random split: {CONFIG['validation_split'] * 100:.1f}% as validation")

        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)

        print(f"Training samples: {len(train_dataset):,}")
        print(f"Validation samples: {len(val_dataset):,}")
    else:
        print("Validation disabled (validation_split=0)")
        train_dataset = full_dataset
        val_dataset = None
        print(f"Training samples: {len(train_dataset):,}")

    # Standard DataLoaders
    # Pin memory for faster transfer to GPU
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=CONFIG["shuffle_train"],  # Can disable for time-sorted datasets
        num_workers=CONFIG["num_workers"],
        pin_memory=True,
        persistent_workers=True,
    )

    val_loader = None
    if use_validation and val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=CONFIG["batch_size"],
            shuffle=False,
            num_workers=CONFIG["num_workers"],
            pin_memory=True,
            persistent_workers=True,
        )

    # 3. Model Initialization
    print("\nStep 3: Initializing Model...")
    model = create_model(CONFIG, vocab_sizes, feature_names)
    print(f"Using {model.model_name()} model")
    model.to(CONFIG["device"])
    if CONFIG["compile_model"]:
        model = torch.compile(model, mode="reduce-overhead")
        print("Model compiled with torch.compile (mode='reduce-overhead')")
    else:
        print("Model compilation disabled (compile_model=False)")
    assert isinstance(model, torch.nn.Module)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # 4. Optimizer, Scheduler (loss is now handled by model internally)
    print("\nStep 4: Setting up Training Components...")

    # Read optimizer configs
    dense_opt_cfg = CONFIG["dense_optimizer"]
    embed_opt_cfg = CONFIG["embedding_optimizer"]
    dense_opt_type = str(dense_opt_cfg.get("type", "adamw"))
    embed_opt_type = str(embed_opt_cfg.get("type", "adagrad"))

    # Helper to create optimizer from config
    def create_optimizer(params, opt_cfg):  # pyrefly: ignore
        """Create optimizer based on config type."""
        opt_type = opt_cfg.get("type", "adamw")
        if opt_type == "ftrl":
            from src.training.optimizers import FTRLProximal

            return FTRLProximal(
                params,
                alpha=opt_cfg.get("alpha", 0.1),
                beta=opt_cfg.get("beta", 1.0),
                l1=opt_cfg.get("l1", 0.0),
                l2=opt_cfg.get("l2", 0.0),
            )
        elif opt_type == "adagrad":
            return optim.Adagrad(
                params,
                lr=opt_cfg.get("lr", 1e-2),
                weight_decay=opt_cfg.get("weight_decay", 0.0),
            )
        else:  # adamw
            return optim.AdamW(
                params,
                lr=opt_cfg.get("lr", 1e-4),
                weight_decay=opt_cfg.get("weight_decay", 1e-4),
            )

    # Check if both optimizers are FTRL (single optimizer mode)
    use_single_ftrl = dense_opt_cfg == embed_opt_cfg

    if use_single_ftrl:
        # FTRL Proximal for ALL parameters (use dense config)
        optimizer = create_optimizer(model.parameters(), dense_opt_cfg)
        embedding_optimizer = None
        other_optimizer = None
        print(
            f"Optimizer: FTRL Proximal (alpha={dense_opt_cfg.get('alpha', 0.1)}, "
            f"beta={dense_opt_cfg.get('beta', 1.0)}, l1={dense_opt_cfg.get('l1', 0.0)}, "
            f"l2={dense_opt_cfg.get('l2', 0.0)})"
        )
    else:
        # Separate parameters for embeddings vs other layers
        embedding_params = []
        other_params = []
        for name, param in model.named_parameters():
            if "embeddings" in name:
                embedding_params.append(param)
            else:
                other_params.append(param)

        print(f"Embedding parameters: {sum(p.numel() for p in embedding_params):,}")
        print(f"Dense parameters: {sum(p.numel() for p in other_params):,}")

        # Create embedding optimizer
        embedding_optimizer = create_optimizer(embedding_params, embed_opt_cfg)
        print(
            f"Embedding optimizer: {embed_opt_type.upper()} (lr={embed_opt_cfg.get('lr', 'N/A')}, "
            f"weight_decay={embed_opt_cfg.get('weight_decay', 'N/A')})"
        )

        # Create dense optimizer
        other_optimizer = create_optimizer(other_params, dense_opt_cfg)
        print(
            f"Dense optimizer: {dense_opt_type.upper()} (lr={dense_opt_cfg.get('lr', 'N/A')}, "
            f"weight_decay={dense_opt_cfg.get('weight_decay', 'N/A')})"
        )
        optimizer = None

    # Learning rate scheduler (only for non-FTRL optimizers)
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * CONFIG["epochs"]
    dense_warmup_ratio = (
        float(cast(dict[str, Any], dense_opt_cfg).get("warmup_epoch_ratio", 0.0))
        if dense_opt_type != "ftrl"
        else 0.0
    )
    warmup_steps = int(steps_per_epoch * dense_warmup_ratio)

    if not use_single_ftrl and other_optimizer is not None and dense_opt_type != "ftrl":
        scheduler = LRSchedulerWithWarmup(
            other_optimizer, warmup_steps=warmup_steps, total_steps=total_steps
        )
        print(
            f"LR warmup steps: {warmup_steps} ({dense_warmup_ratio * 100:.0f}% of {steps_per_epoch} steps)"
        )
    else:
        scheduler = None
        print("LR scheduler disabled (FTRL mode uses per-coordinate learning rates)")

    # Setup Automatic Mixed Precision (AMP)
    use_amp = CONFIG["auto_amp"] and CONFIG["device"] == "cuda"
    amp_dtype_str = CONFIG.get("amp_dtype", "float16")
    amp_dtype = torch.bfloat16 if amp_dtype_str == "bfloat16" else torch.float16

    if use_amp:
        scaler = torch.amp.GradScaler("cuda")
        print(f"Automatic Mixed Precision (AMP) ENABLED with dtype={amp_dtype_str}")
    else:
        scaler = None
        if CONFIG["auto_amp"] and CONFIG["device"] != "cuda":
            print("AMP disabled (requires CUDA device)")
        else:
            print("Automatic Mixed Precision (AMP) disabled")

    # 5. Training Loop
    # Ensure models directory exists
    os.makedirs(CONFIG["models_path"], exist_ok=True)

    print("\nStep 5: Starting Training...")
    print("=" * 80)

    best_val_loss = float("inf")
    best_val_auc = 0.0
    patience_counter = 0
    epoch = 0  # Initialize for graceful interrupt handling
    val_loss = 0
    val_auc = 0
    val_logloss = 0

    # Setup TensorBoard writer with timestamped run directory
    writer = None
    run_dir = None
    if CONFIG["use_tensorboard"]:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(CONFIG["tensorboard_logdir"], f"run_{timestamp}")
        writer = SummaryWriter(log_dir=run_dir)
        command = f"python -m tensorboard.main --logdir={run_dir} --reload_interval=30"
        pyperclip.copy(command)
        print(f"TensorBoard logging to: {run_dir}")
        print(f"Run '{command}' to view training progress")

    try:
        for epoch in range(CONFIG["epochs"]):
            model.train()
            total_loss = 0
            start_time = time.time()

            # Reset dataset shuffle order at start of each epoch
            # train_dataset.reset()

            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{CONFIG['epochs']}")
            for batch_idx, batch_data in enumerate(pbar):  # pyrefly: ignore
                X_batch, y_batch = batch_data
                X_batch = X_batch.to(CONFIG["device"])
                y_batch = y_batch.to(CONFIG["device"]).unsqueeze(1)

                # Zero gradients based on optimizer mode
                if use_single_ftrl and optimizer is not None:
                    optimizer.zero_grad()
                else:
                    if embedding_optimizer is not None:
                        embedding_optimizer.zero_grad()
                    if other_optimizer is not None:
                        other_optimizer.zero_grad()

                # Forward pass with optional AMP
                with torch.amp.autocast(
                    device_type="cuda", dtype=amp_dtype, enabled=use_amp
                ):
                    # Unified interface: all models handle their own loss internally
                    output = model(X_batch)
                    loss = model.compute_loss(output, y_batch)  # type: ignore[operator]

                # Backward pass with gradient scaling for AMP
                if use_amp and scaler is not None:
                    scaler.scale(loss).backward()

                    # Gradient clipping (unscale first for proper clipping)
                    if CONFIG["grad_clip"] > 0:
                        if use_single_ftrl and optimizer is not None:
                            scaler.unscale_(optimizer)
                        else:
                            if embedding_optimizer is not None:
                                scaler.unscale_(embedding_optimizer)
                            if other_optimizer is not None:
                                scaler.unscale_(other_optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), CONFIG["grad_clip"]
                        )

                    # Step optimizers
                    if use_single_ftrl and optimizer is not None:
                        scaler.step(optimizer)
                    else:
                        if embedding_optimizer is not None:
                            scaler.step(embedding_optimizer)
                        if other_optimizer is not None:
                            scaler.step(other_optimizer)
                    scaler.update()
                else:
                    loss.backward()

                    # Gradient clipping
                    if CONFIG["grad_clip"] > 0:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), CONFIG["grad_clip"]
                        )

                    # Step optimizers
                    if use_single_ftrl and optimizer is not None:
                        optimizer.step()
                    else:
                        if embedding_optimizer is not None:
                            embedding_optimizer.step()
                        if other_optimizer is not None:
                            other_optimizer.step()

                # Step scheduler (only in adamw_adagrad mode)
                if scheduler is not None:
                    scheduler.step()

                total_loss += loss.item()

                # Log to TensorBoard (at configured interval to reduce I/O overhead)
                if (
                    writer is not None
                    and batch_idx % CONFIG["tensorboard_log_interval"] == 0
                ):
                    global_step = epoch * len(train_loader) + batch_idx
                    writer.add_scalar("Loss/train_batch", loss.item(), global_step)
                    if scheduler is not None:
                        writer.add_scalar(
                            "LR/learning_rate", scheduler.get_lr(), global_step
                        )

                # Update progress bar
                current_lr = (
                    scheduler.get_lr()
                    if scheduler is not None
                    else dense_opt_cfg.get("alpha", dense_opt_cfg.get("lr", 0.0))
                )
                pbar.set_postfix(
                    {"loss": f"{loss.item():.4f}", "lr": f"{current_lr:.2e}"}
                )

            # Epoch statistics
            avg_train_loss = total_loss / len(train_loader)
            epoch_time = time.time() - start_time

            # Validation (only if enabled)
            if use_validation and val_loader is not None:
                val_loss, val_auc, val_logloss = evaluate(
                    model,
                    val_loader,
                    CONFIG["device"],
                    use_amp=use_amp,
                    amp_dtype=amp_dtype,
                )

                # Log epoch metrics to TensorBoard
                if writer is not None:
                    writer.add_scalar("Loss/train_epoch", avg_train_loss, epoch)
                    writer.add_scalar("Loss/val", val_loss, epoch)
                    writer.add_scalar("Metrics/val_auc", val_auc, epoch)
                    writer.add_scalar("Metrics/val_logloss", val_logloss, epoch)

                print(f"\n{'=' * 80}")
                print(f"Epoch {epoch + 1}/{CONFIG['epochs']} Summary:")
                print(f"  Train Loss: {avg_train_loss:.5f}")
                print(f"  Val Loss:   {val_loss:.5f}")
                print(f"  Val AUC:    {val_auc:.5f}")
                print(f"  Val LogLoss: {val_logloss:.5f}")
                current_lr = (
                    scheduler.get_lr()
                    if scheduler is not None
                    else dense_opt_cfg.get("alpha", dense_opt_cfg.get("lr", 0.0))
                )
                print(f"  LR:         {current_lr:.2e}")
                print(f"  Time:       {epoch_time:.0f}s")
                print(f"{'=' * 80}\n")

                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_auc = val_auc
                    patience_counter = 0

                    # Save best model - checkpoint format depends on optimizer mode
                    checkpoint = {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "val_loss": val_loss,
                        "val_auc": val_auc,
                        "val_logloss": val_logloss,
                        "dense_optimizer_type": dense_opt_type,
                        "embedding_optimizer_type": embed_opt_type,
                    }
                    if use_single_ftrl and optimizer is not None:
                        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
                    else:
                        if embedding_optimizer is not None:
                            checkpoint["embedding_optimizer_state_dict"] = (
                                embedding_optimizer.state_dict()
                            )
                        if other_optimizer is not None:
                            checkpoint["other_optimizer_state_dict"] = (
                                other_optimizer.state_dict()
                            )
                    torch.save(
                        checkpoint,
                        os.path.join(CONFIG["models_path"], "best_model.pth"),
                    )
                    print(f"✓ New best model saved! (Val AUC: {val_auc:.5f})")
                else:
                    patience_counter += 1
                    print(f"No improvement for {patience_counter} epoch(s)")

                    if patience_counter >= CONFIG["early_stopping_patience"]:
                        print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                        break
            else:
                # No validation - just log training metrics
                if writer is not None:
                    writer.add_scalar("Loss/train_epoch", avg_train_loss, epoch)

                # Save best model - checkpoint format depends on optimizer mode
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "val_auc": val_auc,
                    "val_logloss": val_logloss,
                    "dense_optimizer_type": dense_opt_type,
                    "embedding_optimizer_type": embed_opt_type,
                }
                if use_single_ftrl and optimizer is not None:
                    checkpoint["optimizer_state_dict"] = optimizer.state_dict()
                else:
                    if embedding_optimizer is not None:
                        checkpoint["embedding_optimizer_state_dict"] = (
                            embedding_optimizer.state_dict()
                        )
                    if other_optimizer is not None:
                        checkpoint["other_optimizer_state_dict"] = (
                            other_optimizer.state_dict()
                        )
                torch.save(checkpoint, os.path.join(models_path, "best_model.pth"))
                print(f"✓ New best model saved! (Val AUC: {val_auc:.5f})")

                current_lr = (
                    scheduler.get_lr()
                    if scheduler is not None
                    else dense_opt_cfg.get("alpha", dense_opt_cfg.get("lr", 0.0))
                )
                print(f"\n{'=' * 80}")
                print(f"Epoch {epoch + 1}/{CONFIG['epochs']} Summary:")
                print(f"  Train Loss: {avg_train_loss:.5f}")
                print(f"  LR:         {current_lr:.2e}")
                print(f"  Time:       {epoch_time:.0f}s")
                print(f"{'=' * 80}\n")

    except KeyboardInterrupt:
        print("\n" + "=" * 80)
        print("TRAINING INTERRUPTED BY USER (Ctrl+C)")
        print("=" * 80)
        if writer is not None:
            writer.close()
        return

    # 6. Final Results
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    if use_validation:
        print(f"Best Validation Loss: {best_val_loss:.5f}")
        print(f"Best Validation AUC:  {best_val_auc:.5f}")
        print(f"\nBest model saved to: {models_path}/best_model.pth")
    print(f"Latest model saved to: {models_path}/model.pth")
    print("=" * 80)

    # Save final model - checkpoint format depends on optimizer mode
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "val_loss": val_loss,
        "val_auc": val_auc,
        "val_logloss": val_logloss,
        "dense_optimizer_type": dense_opt_type,
        "embedding_optimizer_type": embed_opt_type,
    }
    if use_single_ftrl and optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    else:
        if embedding_optimizer is not None:
            checkpoint["embedding_optimizer_state_dict"] = (
                embedding_optimizer.state_dict()
            )
        if other_optimizer is not None:
            checkpoint["other_optimizer_state_dict"] = other_optimizer.state_dict()
    torch.save(checkpoint, os.path.join(models_path, "model.pth"))

    # Cleanup TensorBoard writer
    if writer is not None:
        writer.close()
        print(f"TensorBoard logs saved to: {run_dir}")


if __name__ == "__main__":
    train()
