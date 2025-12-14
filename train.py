import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import gc

from config import CONFIG, seed_everything
from data_processor import process_data_polars
from dataset import AvazuDataset
from model import GatedDCNModel

def train():
    seed_everything(CONFIG['seed'])
    
    # 1. Load and Process Data
    # Note: For this verification structure, we process everything.
    # In a real production pipeline, you would separate feature map generation.
    print("Step 1: Processing Data...")
    X_train, y_train, X_test, test_ids, vocab_sizes, feature_names = process_data_polars()
    
    # 2. Dataset and DataLoader
    print("Step 2: Preparing DataLoaders...")
    train_dataset = AvazuDataset(X_train, y_train)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=True, 
        num_workers=CONFIG['num_workers'], 
        pin_memory=True
    )
    
    # Free memory
    del X_train, y_train, X_test
    gc.collect()
    
    # 3. Model Initialization
    print("Step 3: Initializing Model...")
    model = GatedDCNModel(vocab_sizes, CONFIG['embedding_dim'], feature_names)
    model.to(CONFIG['device'])
    print(model)
    
    # 4. Optimizer and Loss
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=1e-5)
    
    # 5. Training Loop
    print("Step 4: Starting Training...")
    
    for epoch in range(CONFIG['epochs']):
        model.train()
        total_loss = 0
        start_time = time.time()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for X_batch, y_batch in pbar:
            X_batch = X_batch.to(CONFIG['device'])
            y_batch = y_batch.to(CONFIG['device']).unsqueeze(1)
            
            optimizer.zero_grad()
            
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} Done. Avg Loss: {avg_loss:.5f}. Time: {time.time() - start_time:.0f}s")
        
    # 6. Save Model
    print("Step 5: Saving Model...")
    torch.save(model.state_dict(), "model.pth")
    print("Model saved to model.pth")

if __name__ == "__main__":
    train()
