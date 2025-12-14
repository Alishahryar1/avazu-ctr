import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import polars as pl
import gc

from config import CONFIG, seed_everything
from data_processor import load_processed_data
from dataset import AvazuDataset
from model import GatedDCNModel

def inference():
    seed_everything(CONFIG['seed'])
    
    # 1. Load and Process Data (Need to get vocab_sizes and feature_names)
    print("Step 1: Loading Processed Data (for Inference)...")
    try:
        _, _, X_test, test_ids, vocab_sizes, feature_names = load_processed_data(mode='inference')
    except FileNotFoundError:
        print("Processed data not found. Please run 'python data_processor.py' to generate it.")
        return
    
    # 2. Dataset and DataLoader
    print("Step 2: Preparing Test DataLoader...")
    test_dataset = AvazuDataset(X_test)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=False, 
        num_workers=CONFIG['num_workers'], 
        pin_memory=True
    )
    
    # 3. Model Initialization and Loading
    print("Step 3: Loading Model...")
    model = GatedDCNModel(
        vocab_sizes,
        CONFIG['embedding_dim'],
        feature_names,
        dcn_num_layers=CONFIG['dcn_num_layers'],
        mlp_hidden_dims=CONFIG['mlp_hidden_dims'],
        mlp_dropout=CONFIG['mlp_dropout'],
        use_batch_norm=CONFIG['use_batch_norm']
    )

    # Try to load best model first, fall back to model.pth
    model_path = "best_model.pth"
    try:
        checkpoint = torch.load(model_path, map_location=CONFIG['device'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(CONFIG['device'])
        print(f"Best model loaded successfully (epoch {checkpoint['epoch']+1})")
        print(f"  Val AUC: {checkpoint['val_auc']:.5f}, Val LogLoss: {checkpoint['val_logloss']:.5f}")
    except FileNotFoundError:
        try:
            model_path = "model.pth"
            model.load_state_dict(torch.load(model_path, map_location=CONFIG['device']))
            model.to(CONFIG['device'])
            print("Model loaded from model.pth")
        except FileNotFoundError:
            print("Error: No model found. Please run train.py first.")
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
    
    # 5. Save Submission
    print("Step 5: Creating submission file...")
    submission = pl.DataFrame({
        "id": test_ids,
        "click": predictions
    })
    
    submission.write_csv(CONFIG['sub_path'])
    print(f"Submission saved to {CONFIG['sub_path']}")
    print(submission.head())

if __name__ == "__main__":
    inference()
