"""Evaluation utilities for model performance."""

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss


def evaluate(model, data_loader, device, use_amp=False, amp_dtype=torch.float16):
    """
    Evaluate model on validation set.
    Returns loss, AUC, and LogLoss.

    Args:
        model: The model to evaluate (must have compute_loss and get_predictions methods)
        data_loader: DataLoader for validation data
        device: Device to run on
        use_amp: Whether to use automatic mixed precision
        amp_dtype: Data type for AMP (torch.float16 or torch.bfloat16)
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_data in data_loader:
            X_batch, y_batch = batch_data
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).unsqueeze(1)

            # Use autocast for mixed precision inference
            with torch.amp.autocast(
                device_type="cuda", dtype=amp_dtype, enabled=use_amp
            ):
                # Unified interface: all models use same methods
                output = model(X_batch)
                loss = model.compute_loss(output, y_batch)

            total_loss += loss.item()

            # Use get_predictions for probabilities
            preds = model.get_predictions(X_batch).cpu().numpy()
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
