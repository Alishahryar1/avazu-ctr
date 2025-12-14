import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import roc_auc_score, log_loss
from tqdm import tqdm
import time
import gc
import numpy as np

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
        X_train, y_train, X_test, test_ids, vocab_sizes, feature_names = load_processed_data(mode='train')
    except FileNotFoundError:
        print("Processed data not found. Please run 'python data_processor.py' to generate it.")
        return

    print(f"Train samples: {len(X_train):,}")
    print(f"Positive rate: {y_train.mean():.4f}")

    # 2. Train/Validation Split
    print("\nStep 2: Creating Train/Validation Split...")
    full_dataset = AvazuDataset(X_train, y_train)

    val_size = int(len(full_dataset) * CONFIG['validation_split'])
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(CONFIG['seed'])
    )

    print(f"Training samples: {len(train_dataset):,}")
    print(f"Validation samples: {len(val_dataset):,}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
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
    del X_train, y_train, X_test, full_dataset
    gc.collect()

    # 3. Model Initialization
    print("\nStep 3: Initializing Model...")
    model = GatedDCNModel(
        vocab_sizes,
        CONFIG['embedding_dim'],
        feature_names,
        dcn_num_layers=CONFIG['dcn_num_layers'],
        mlp_hidden_dims=CONFIG['mlp_hidden_dims'],
        mlp_dropout=CONFIG['mlp_dropout'],
        use_batch_norm=CONFIG['use_batch_norm']
    )
    model.to(CONFIG['device'])

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

    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG['lr'],
        weight_decay=CONFIG['weight_decay']
    )

    # Learning rate scheduler
    total_steps = len(train_loader) * CONFIG['epochs']
    scheduler = LRSchedulerWithWarmup(
        optimizer,
        warmup_steps=CONFIG['lr_warmup_steps'],
        total_steps=total_steps
    )

    # 5. Training Loop
    print("\nStep 5: Starting Training...")
    print("=" * 80)

    best_val_loss = float('inf')
    best_val_auc = 0.0
    patience_counter = 0

    for epoch in range(CONFIG['epochs']):
        model.train()
        total_loss = 0
        start_time = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
        for batch_idx, (X_batch, y_batch) in enumerate(pbar):
            X_batch = X_batch.to(CONFIG['device'])
            y_batch = y_batch.to(CONFIG['device']).unsqueeze(1)

            optimizer.zero_grad()

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            loss.backward()

            # Gradient clipping
            if CONFIG['grad_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip'])

            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

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
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_auc': val_auc,
                'val_logloss': val_logloss,
            }, "best_model.pth")
            print(f"✓ New best model saved! (Val AUC: {val_auc:.5f})")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epoch(s)")

            if patience_counter >= CONFIG['early_stopping_patience']:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break

    # 6. Final Results
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best Validation Loss: {best_val_loss:.5f}")
    print(f"Best Validation AUC:  {best_val_auc:.5f}")
    print(f"\nBest model saved to: best_model.pth")
    print(f"Latest model saved to: model.pth")
    print("=" * 80)

    # Save final model
    torch.save(model.state_dict(), "model.pth")


if __name__ == "__main__":
    train()
