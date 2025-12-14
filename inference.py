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
        mlp_dropout=CONFIG['mlp_dropout']
    )
    try:
        model.load_state_dict(torch.load("model.pth"))
        model.to(CONFIG['device'])
        print("Model state loaded successfully.")
    except FileNotFoundError:
        print("Error: 'model.pth' not found. Please run train.py first.")
        return

    # 4. Inference
    print("Step 4: Starting Inference...")
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for X_batch in tqdm(test_loader, desc="Predicting"):
            X_batch = X_batch.to(CONFIG['device'])
            
            preds = model(X_batch)
                
            predictions.append(preds.cpu().numpy())
    
    # Concatenate predictions
    predictions = np.concatenate(predictions).flatten()
    
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
