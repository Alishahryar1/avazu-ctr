import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, log_loss
from tqdm import tqdm
import time
import gc
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from config import CONFIG, seed_everything
from data_processor import load_processed_data
from dataset import AvazuDataset
from model import GatedDCNModel


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    Focuses on hard examples by down-weighting easy ones.
    """
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # Optional class weights

    def forward(self, logits, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        probs = torch.sigmoid(logits)
        pt = targets * probs + (1 - targets) * (1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        loss = focal_weight * bce_loss

        if self.alpha is not None:
            alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
            loss = alpha_t * loss

        return loss.mean()


class LRSchedulerWithWarmup:
    """
    Learning rate scheduler with linear warmup and cosine decay.
    """
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_step = 0

    def step(self):
        self.current_step += 1
        if self.current_step < self.warmup_steps:
            # Linear warmup
            lr = self.base_lr * self.current_step / self.warmup_steps
        else:
            # Cosine decay
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']


def evaluate(model, data_loader, criterion, device):
    """
    Evaluate model on validation set.
    Returns loss, AUC, and LogLoss.
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).unsqueeze(1)

            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item()

            # Collect predictions and targets for metrics
            preds = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(preds.flatten())
            all_targets.extend(y_batch.cpu().numpy().flatten())

    avg_loss = total_loss / len(data_loader)

    # Calculate metrics
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Manual clipping since eps argument is not supported in recent sklearn versions
    all_preds = np.clip(all_preds, 1e-7, 1 - 1e-7)

    auc = roc_auc_score(all_targets, all_preds)
    logloss = log_loss(all_targets, all_preds)

    return avg_loss, auc, logloss


def train():
    seed_everything(CONFIG['seed'])

    # 1. Load and Process Data
    print("=" * 80)
    print("AVAZU CTR PREDICTION - IMPROVED TRAINING")
    print("=" * 80)
    print("\nStep 1: Loading Processed Data...")
    try:
        X_train, y_train, train_hours, X_test, test_ids, vocab_sizes, feature_names = load_processed_data(mode='train')
    except FileNotFoundError:
        print("Processed data not found. Please run 'python data_processor.py' to generate it.")
        return

    # Type assertions for train mode - these are always non-None
    assert X_train is not None and y_train is not None and train_hours is not None

    print(f"Train samples: {len(X_train):,}")
    print(f"Positive rate: {y_train.mean():.4f}")

    # 2. Temporal Train/Validation Split (prevents data leakage from future to past)
    print("\nStep 2: Creating Temporal Train/Validation Split...")
    
    # Sort indices by hour to enable temporal split
    # Hour format is YYMMDDHH (e.g., "14102100" for 2014-10-21 00:00)
    sorted_indices = np.argsort(train_hours)
    
    # Calculate split point: use last N% of data (chronologically) for validation
    val_size = int(len(sorted_indices) * CONFIG['validation_split'])
    train_size = len(sorted_indices) - val_size
    
    train_indices = sorted_indices[:train_size]
    val_indices = sorted_indices[train_size:]
    
    # Get the hour boundary for logging
    train_max_hour = train_hours[train_indices[-1]]
    val_min_hour = train_hours[val_indices[0]]
    print(f"Temporal split: Train ends at hour {train_max_hour}, Val starts at hour {val_min_hour}")
    
    # Create datasets using the temporal indices
    train_dataset = AvazuDataset(X_train[train_indices], y_train[train_indices])
    val_dataset = AvazuDataset(X_train[val_indices], y_train[val_indices])

    print(f"Training samples: {len(train_dataset):,}")
    print(f"Validation samples: {len(val_dataset):,}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,  # Shuffle within training set is fine
        num_workers=CONFIG['num_workers'],
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'] * 2,  # Larger batch for validation
        shuffle=False,
        num_workers=CONFIG['num_workers'],
        pin_memory=True
    )

    # Free memory
    del X_train, y_train, train_hours, X_test, train_indices, val_indices
    gc.collect()

    # 3. Model Initialization
    print("\nStep 3: Initializing Model...")
    model = GatedDCNModel(vocab_sizes, feature_names, CONFIG)
    model.to(CONFIG['device'])

    # # Compile model for faster training
    # model = torch.compile(model, mode="reduce-overhead")
    # print("Model compiled with torch.compile (mode='reduce-overhead')")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # 4. Loss, Optimizer, Scheduler
    print("\nStep 4: Setting up Training Components...")

    # Use Focal Loss to handle class imbalance
    if CONFIG['focal_loss_gamma'] > 0:
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

    # 5. Training Loop
    # Ensure models directory exists
    os.makedirs(CONFIG['models_path'], exist_ok=True)

    print("\nStep 5: Starting Training...")
    print("=" * 80)

    best_val_loss = float('inf')
    best_val_auc = 0.0
    patience_counter = 0
    epoch = 0  # Initialize for graceful interrupt handling

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

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
            for batch_idx, (X_batch, y_batch) in enumerate(pbar):
                X_batch = X_batch.to(CONFIG['device'])
                y_batch = y_batch.to(CONFIG['device']).unsqueeze(1)

                embedding_optimizer.zero_grad()
                other_optimizer.zero_grad()

                logits = model(X_batch)
                loss = criterion(logits, y_batch)

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

            # Validation
            val_loss, val_auc, val_logloss = evaluate(model, val_loader, criterion, CONFIG['device'])

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
    print(f"Best Validation Loss: {best_val_loss:.5f}")
    print(f"Best Validation AUC:  {best_val_auc:.5f}")
    print(f"\nBest model saved to: {CONFIG['models_path']}/best_model.pth")
    print(f"Latest model saved to: {CONFIG['models_path']}/model.pth")
    print("=" * 80)

    # Save final model
    torch.save(model.state_dict(), os.path.join(CONFIG['models_path'], "model.pth"))

    # Cleanup TensorBoard writer
    if writer is not None:
        writer.close()
        print(f"TensorBoard logs saved to: {run_dir}")


if __name__ == "__main__":
    train()
