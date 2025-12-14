import os
import numpy as np
import torch

# --- CONFIGURATION ---
CONFIG = {
    "seed": 42,
    "batch_size": 256,
    "embedding_dim": 64, # Keep small for memory efficiency
    "lr": 1e-3,
    "epochs": 1,         # 1 Epoch is usually enough for Avazu on this scale
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "min_freq": 10,      # Map categories appearing < 10 times to <UNK>
    "num_workers": 2,
    "train_path": "./data/train.gz",
    "test_path": "./data/test.gz",
    "sub_path": "submission.csv"
}

def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    print(f"Using device: {CONFIG['device']}")
