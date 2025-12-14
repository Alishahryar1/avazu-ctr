import os
import numpy as np
import torch

# --- CONFIGURATION ---
CONFIG = {
    "seed": 42,
    "batch_size": 512,
    "embedding_dim": 64,
    "lr": 1e-3,
    "epochs": 2,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "min_freq": 10,
    "num_workers": 2,
    "train_path": "./data/train.gz",
    "test_path": "./data/test.gz",
    "sub_path": "submission.csv",
    "dcn_num_layers": 3,
    "mlp_hidden_dims": [1024, 512],
    "mlp_dropout": 0.3,
    "processed_path": "./data"
}

def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    print(f"Using device: {CONFIG['device']}")
