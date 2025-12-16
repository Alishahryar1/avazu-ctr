"""Training script for CTR prediction model."""
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

from src.config.config import CONFIG, seed_everything
from src.processing.data_processor import load_metadata, get_parquet_path, get_parquet_row_count
from src.processing.dataset import ParquetFullDataset
from src.models.architectures.base_model import GatedDCNModel
from src.models.architectures.ensemble import EnsembleModel
from src.models.architectures.fcnv2 import FCNv2Model
from src.training.losses import FocalLoss, KBCELoss
from src.training.schedulers import LRSchedulerWithWarmup
from src.training.evaluator import evaluate


def train():
    # Cache paths early for type checker (avoids unbound-name errors in finally blocks)
    models_path = CONFIG['models_path']
    seed_everything(CONFIG['seed'])

    # 1. Load Metadata (data stays in parquet)
    print("=" * 80)
    print("AVAZU CTR PREDICTION - MEMORY-EFFICIENT TRAINING")
    print("=" * 80)
    print("\nStep 1: Loading Metadata...")
    try:
        vocab_sizes, feature_names = load_metadata()
    except FileNotFoundError:
        print("Processed data not found. Please run 'python data_processor.py' to generate it.")
        return

    train_parquet = get_parquet_path('train')
    total_samples = get_parquet_row_count('train')

    print(f"Train parquet: {train_parquet}")
    print(f"Total samples: {total_samples:,}")
    print(f"Number of features: {len(feature_names)}")

    # 2. Create Train/Validation Split (batch-level, not row-level)
    print("\nStep 2: Creating Train/Validation Split...")

    # Create dataset - Loads entire file into memory
    full_dataset = ParquetFullDataset(
        parquet_path=train_parquet,
        feature_cols=feature_names,
        label_col='click'
    )

    # Check if validation is enabled
    use_validation = CONFIG['validation_split'] > 0

    if use_validation:
        # Random split
        indices = np.arange(len(full_dataset))
        np.random.seed(CONFIG['seed'])  # Reproducible split
        np.random.shuffle(indices)

        split_idx = int(len(full_dataset) * (1 - CONFIG['validation_split']))
        train_indices = indices[:split_idx].tolist()
        val_indices = indices[split_idx:].tolist()

        print(f"Random split: {CONFIG['validation_split']*100:.1f}% as validation")

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
        batch_size=CONFIG['batch_size'],
        shuffle=True,  # Shuffle row-by-row now possible in memory
        num_workers=CONFIG['num_workers'],
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = None
    if use_validation and val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=CONFIG['batch_size'],
            shuffle=False,
            num_workers=CONFIG['num_workers'],
            pin_memory=True,
            persistent_workers=True
        )

    # 3. Model Initialization
    print("\nStep 3: Initializing Model...")
    use_ensemble = CONFIG['use_ensemble']
    use_fcnv2 = CONFIG.get('use_fcnv2', False)
    
    if use_fcnv2:
        model = FCNv2Model(vocab_sizes, feature_names, CONFIG)
        print("Using FCNv2 model (dual-path cross network)")
    elif use_ensemble:
        model = EnsembleModel(vocab_sizes, feature_names, CONFIG)
        print(f"Using ensemble of {CONFIG['ensemble_k']} models (aggregation={CONFIG['ensemble_aggregation']})")
    else:
        model = GatedDCNModel(vocab_sizes, feature_names, CONFIG)
        print("Using GatedDCNModel")
    model.to(CONFIG['device'])
    if CONFIG['compile_model']:
        model = torch.compile(model, mode="reduce-overhead")
        print("Model compiled with torch.compile (mode='reduce-overhead')")
    else:
        print("Model compilation disabled (compile_model=False)")
    assert(isinstance(model, torch.nn.Module))

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # 4. Loss, Optimizer, Scheduler
    print("\nStep 4: Setting up Training Components...")

    # Use appropriate loss function
    if use_fcnv2 or use_ensemble:
        criterion = KBCELoss()
        print("Using KBCELoss for multi-branch architecture")
    elif CONFIG['focal_loss_gamma'] > 0:
        criterion = FocalLoss(gamma=CONFIG['focal_loss_gamma'])
        print(f"Using Focal Loss (gamma={CONFIG['focal_loss_gamma']})")
    else:
        criterion = nn.BCEWithLogitsLoss()
        print("Using BCEWithLogits Loss")

    # Separate parameters for embeddings vs other layers
    embedding_params = []
    other_params = []
    for name, param in model.named_parameters():
        if 'embeddings' in name:
            embedding_params.append(param)
        else:
            other_params.append(param)

    print(f"Embedding parameters: {sum(p.numel() for p in embedding_params):,}")
    print(f"Other parameters: {sum(p.numel() for p in other_params):,}")

    # Adagrad for embeddings (commonly used for sparse/categorical features)
    # Separate weight decay for embeddings (usually lower or zero)
    embedding_optimizer = optim.Adagrad(
        embedding_params,
        lr=CONFIG['embedding_lr'],
        weight_decay=CONFIG['embedding_weight_decay']
    )
    print(f"Embedding optimizer: Adagrad (lr={CONFIG['embedding_lr']}, weight_decay={CONFIG['embedding_weight_decay']})")

    # AdamW for other parameters (MLP, DCN, etc.)
    other_optimizer = optim.AdamW(
        other_params,
        lr=CONFIG['lr'],
        weight_decay=CONFIG['weight_decay']
    )
    print(f"Other optimizer: AdamW (lr={CONFIG['lr']}, weight_decay={CONFIG['weight_decay']})")

    # Learning rate scheduler (only for non-embedding params)
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * CONFIG['epochs']
    warmup_steps = int(steps_per_epoch * CONFIG['lr_warmup_epoch_ratio'])
    scheduler = LRSchedulerWithWarmup(
        other_optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps
    )
    print(f"LR warmup steps: {warmup_steps} ({CONFIG['lr_warmup_epoch_ratio']*100:.0f}% of {steps_per_epoch} steps)")

    # Setup Automatic Mixed Precision (AMP)
    use_amp = CONFIG['auto_amp'] and CONFIG['device'] == 'cuda'
    amp_dtype_str = CONFIG.get('amp_dtype', 'float16')
    amp_dtype = torch.bfloat16 if amp_dtype_str == 'bfloat16' else torch.float16

    if use_amp:
        scaler = torch.amp.GradScaler('cuda')
        print(f"Automatic Mixed Precision (AMP) ENABLED with dtype={amp_dtype_str}")
    else:
        scaler = None
        if CONFIG['auto_amp'] and CONFIG['device'] != 'cuda':
            print("AMP disabled (requires CUDA device)")
        else:
            print("Automatic Mixed Precision (AMP) disabled")

    # 5. Training Loop
    # Ensure models directory exists
    os.makedirs(CONFIG['models_path'], exist_ok=True)

    print("\nStep 5: Starting Training...")
    print("=" * 80)

    best_val_loss = float('inf')
    best_val_auc = 0.0
    patience_counter = 0
    epoch = 0  # Initialize for graceful interrupt handling
    val_loss = 0
    val_auc = 0
    val_logloss = 0

    # Setup TensorBoard writer with timestamped run directory
    writer = None
    run_dir = None
    if CONFIG['use_tensorboard']:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(CONFIG['tensorboard_logdir'], f"run_{timestamp}")
        writer = SummaryWriter(log_dir=run_dir)
        print(f"TensorBoard logging to: {run_dir}")
        print("Run 'tensorboard --logdir=runs' to view training progress")

    try:
        for epoch in range(CONFIG['epochs']):
            model.train()
            total_loss = 0
            start_time = time.time()

            # Reset dataset shuffle order at start of each epoch
            # train_dataset.reset()

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
            for batch_idx, batch_data in enumerate(pbar):
                X_batch, y_batch = batch_data
                X_batch = X_batch.to(CONFIG['device'])
                y_batch = y_batch.to(CONFIG['device']).unsqueeze(1)

                embedding_optimizer.zero_grad()
                other_optimizer.zero_grad()

                # Forward pass with optional AMP
                with torch.amp.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                    if isinstance(model, EnsembleModel):
                        # Get logits from all models: [K, Batch, 1]
                        stacked_logits = model(X_batch, return_all_logits=True)

                        # Aggregate logits for combined prediction
                        if model.ensemble_aggregation == 'mean':
                            ensemble_logits = stacked_logits.mean(dim=0)
                        elif model.ensemble_aggregation == 'median':
                            ensemble_logits = stacked_logits.median(dim=0).values
                        else:
                            raise ValueError(f"Unknown aggregation: {model.ensemble_aggregation}")

                        # Convert stacked logits to list of branches for KBCELoss
                        # stacked_logits shape: [K, Batch, 1] -> list of K tensors [Batch, 1]
                        branch_logits = [stacked_logits[i] for i in range(model.k)]
                        
                        # Use same interface as FCNv2: combined output + list of branches
                        loss = criterion(ensemble_logits, branch_logits, y_batch)
                    elif isinstance(model, FCNv2Model):
                        # FCNv2 returns dict with y_pred, y_d, y_s
                        output = model(X_batch)
                        loss = criterion(
                            output['y_pred'],
                            [output['y_d'], output['y_s']],  # k=2 branches for FCN
                            y_batch
                        )
                    else:
                        # Standard single model (GatedDCNModel)
                        logits = model(X_batch)
                        loss = criterion(logits, y_batch)

                # Backward pass with gradient scaling for AMP
                if use_amp and scaler is not None:
                    scaler.scale(loss).backward()

                    # Gradient clipping (unscale first for proper clipping)
                    if CONFIG['grad_clip'] > 0:
                        scaler.unscale_(embedding_optimizer)
                        scaler.unscale_(other_optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip'])

                    scaler.step(embedding_optimizer)
                    scaler.step(other_optimizer)
                    scaler.update()
                else:
                    loss.backward()

                    # Gradient clipping
                    if CONFIG['grad_clip'] > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip'])

                    embedding_optimizer.step()
                    other_optimizer.step()

                scheduler.step()

                total_loss += loss.item()

                # Log to TensorBoard (at configured interval to reduce I/O overhead)
                if writer is not None and batch_idx % CONFIG['tensorboard_log_interval'] == 0:
                    global_step = epoch * len(train_loader) + batch_idx
                    writer.add_scalar('Loss/train_batch', loss.item(), global_step)
                    writer.add_scalar('LR/learning_rate', scheduler.get_lr(), global_step)

                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'lr': f"{scheduler.get_lr():.2e}"
                })

            # Epoch statistics
            avg_train_loss = total_loss / len(train_loader)
            epoch_time = time.time() - start_time

            # Validation (only if enabled)
            if use_validation and val_loader is not None:
                val_loss, val_auc, val_logloss = evaluate(model, val_loader, criterion, CONFIG['device'], use_amp=use_amp, amp_dtype=amp_dtype)

                # Log epoch metrics to TensorBoard
                if writer is not None:
                    writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch)
                    writer.add_scalar('Loss/val', val_loss, epoch)
                    writer.add_scalar('Metrics/val_auc', val_auc, epoch)
                    writer.add_scalar('Metrics/val_logloss', val_logloss, epoch)

                print(f"\n{'='*80}")
                print(f"Epoch {epoch+1}/{CONFIG['epochs']} Summary:")
                print(f"  Train Loss: {avg_train_loss:.5f}")
                print(f"  Val Loss:   {val_loss:.5f}")
                print(f"  Val AUC:    {val_auc:.5f}")
                print(f"  Val LogLoss: {val_logloss:.5f}")
                print(f"  LR:         {scheduler.get_lr():.2e}")
                print(f"  Time:       {epoch_time:.0f}s")
                print(f"{'='*80}\n")

                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_auc = val_auc
                    patience_counter = 0

                    # Save best model
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'embedding_optimizer_state_dict': embedding_optimizer.state_dict(),
                        'other_optimizer_state_dict': other_optimizer.state_dict(),
                        'val_loss': val_loss,
                        'val_auc': val_auc,
                        'val_logloss': val_logloss,
                    }, os.path.join(CONFIG['models_path'], "best_model.pth"))
                    print(f"✓ New best model saved! (Val AUC: {val_auc:.5f})")
                else:
                    patience_counter += 1
                    print(f"No improvement for {patience_counter} epoch(s)")

                    if patience_counter >= CONFIG['early_stopping_patience']:
                        print(f"\nEarly stopping triggered after {epoch+1} epochs")
                        break
            else:
                # No validation - just log training metrics
                if writer is not None:
                    writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch)

                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'embedding_optimizer_state_dict': embedding_optimizer.state_dict(),
                    'other_optimizer_state_dict': other_optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_auc': val_auc,
                    'val_logloss': val_logloss,
                }, os.path.join(models_path, "best_model.pth"))
                print(f"✓ New best model saved! (Val AUC: {val_auc:.5f})")

                print(f"\n{'='*80}")
                print(f"Epoch {epoch+1}/{CONFIG['epochs']} Summary:")
                print(f"  Train Loss: {avg_train_loss:.5f}")
                print(f"  LR:         {scheduler.get_lr():.2e}")
                print(f"  Time:       {epoch_time:.0f}s")
                print(f"{'='*80}\n")

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

    # Save final model
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'other_optimizer_state_dict': other_optimizer.state_dict(),
            'val_loss': val_loss,
            'val_auc': val_auc,
            'val_logloss': val_logloss,
        }, os.path.join(models_path, "model.pth")
    )

    # Cleanup TensorBoard writer
    if writer is not None:
        writer.close()
        print(f"TensorBoard logs saved to: {run_dir}")


if __name__ == "__main__":
    train()
