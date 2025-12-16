import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import polars as pl

from config import CONFIG, seed_everything
from data_processor import load_metadata, get_parquet_path
from dataset import ParquetFullDataset
from model import GatedDCNModel, EnsembleModel


def inference():
    seed_everything(CONFIG['seed'])

    # 1. Load Metadata (data stays in parquet)
    print("Step 1: Loading Metadata...")
    try:
        vocab_sizes, feature_names = load_metadata()
    except FileNotFoundError:
        print("Processed data not found. Please run 'python data_processor.py' to generate it.")
        return

    test_parquet = get_parquet_path('test')
    print(f"Test parquet: {test_parquet}")

    # 2. Dataset and DataLoader
    print("Step 2: Preparing Test DataLoader...")
    test_dataset = ParquetFullDataset(
        parquet_path=test_parquet,
        feature_cols=feature_names,
        label_col=None  # No labels for test data
    )

    print(f"Test samples: {len(test_dataset):,}")

    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=CONFIG['num_workers'],
        pin_memory=True
    )

    # 3. Model Initialization and Loading
    print("Step 3: Loading Model...")
    use_ensemble = CONFIG['use_ensemble']
    if use_ensemble:
        model = EnsembleModel(vocab_sizes, feature_names, CONFIG)
        print(f"Using ensemble of {CONFIG['ensemble_k']} models (aggregation={CONFIG['ensemble_aggregation']})")
    else:
        model = GatedDCNModel(vocab_sizes, feature_names, CONFIG)

    # Try to load best model first, fall back to model.pth
    model_path = os.path.join(CONFIG['models_path'], "best_model.pth")
    try:
        checkpoint = torch.load(model_path, map_location=CONFIG['device'], weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(CONFIG['device'])
        print(f"Best model loaded successfully (epoch {checkpoint['epoch']+1})")
        if 'val_auc' in checkpoint:
            print(f"  Val AUC: {checkpoint['val_auc']:.5f}, Val LogLoss: {checkpoint['val_logloss']:.5f}")
    except FileNotFoundError:
        try:
            model_path = os.path.join(CONFIG['models_path'], "model.pth")
            checkpoint = torch.load(model_path, map_location=CONFIG['device'], weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(CONFIG['device'])
            print(f"Model loaded from {model_path}")
        except FileNotFoundError:
            print(f"Error: No model found in {CONFIG['models_path']}. Please run train.py first.")
            return

    # 4. Inference
    print("Step 4: Starting Inference...")
    model.eval()
    predictions = []

    with torch.no_grad():
        for X_batch in tqdm(test_loader, desc="Predicting"):
            X_batch = X_batch.to(CONFIG['device'])

            logits = model(X_batch)
            # Apply sigmoid to convert logits to probabilities
            preds = torch.sigmoid(logits)

            predictions.append(preds.cpu().numpy())

    # Concatenate predictions
    predictions = np.concatenate(predictions).flatten()
    print(f"Prediction stats - Min: {predictions.min():.6f}, Max: {predictions.max():.6f}, Mean: {predictions.mean():.6f}")

    # 5. Read test IDs from parquet and save submission
    print("Step 5: Creating submission file...")

    # Read IDs from parquet (memory efficient - only read ID column)
    test_ids = pl.scan_parquet(test_parquet).select('id').collect()['id'].to_numpy()

    submission = pl.DataFrame({
        "id": test_ids,
        "click": predictions
    })

    submission.write_csv(CONFIG['sub_path'])
    print(f"Submission saved to {CONFIG['sub_path']}")
    print(submission.head())


if __name__ == "__main__":
    inference()
