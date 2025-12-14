import os
import numpy as np
import torch
from typing import TypedDict

class ConfigType(TypedDict):
    seed: int
    batch_size: int
    embedding_dim: int
    lr: float
    epochs: int
    device: str
    min_freq: int
    num_workers: int
    train_path: str
    test_path: str
    sub_path: str
    dcn_num_layers: int
    mlp_hidden_dims: list[int]
    mlp_dropout: float
    processed_path: str
    validation_split: float
    early_stopping_patience: int
    lr_warmup_steps: int
    grad_clip: float
    weight_decay: float
    use_batch_norm: bool
    focal_loss_gamma: float
    label_smoothing: float
    models_path: str

# --- CONFIGURATION ---
CONFIG: ConfigType = {
    "seed": 42,
    "batch_size": 2048,  # Increased for faster training
    "embedding_dim": 64,
    "lr": 5e-3,  # Lower initial LR for better convergence
    "epochs": 15,  # Increased from 2 (critical!)
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "min_freq": 10,
    "num_workers": 4,  # Increased for faster data loading
    "train_path": "./data/train.gz",
    "test_path": "./data/test.gz",
    "sub_path": "submission.csv",
    "dcn_num_layers": 4,  # Increased for more feature interactions
    "mlp_hidden_dims": [512, 256, 128],  # Deeper network
    "mlp_dropout": 0.2,  # Reduced dropout
    "processed_path": "./data",
    "validation_split": 0.01,  # NEW: Hold out 10% for validation
    "early_stopping_patience": 3,  # NEW: Early stopping
    "lr_warmup_steps": 1000,  # NEW: LR warmup
    "grad_clip": 1.0,  # NEW: Gradient clipping
    "weight_decay": 1e-5,  # L2 regularization
    "use_batch_norm": True,  # NEW: Batch normalization
    "focal_loss_gamma": 2.0,  # NEW: Focal loss for imbalance
    "label_smoothing": 0.0,  # NEW: Optional label smoothing
    "models_path": "./models",  # Directory for saving model checkpoints
}

def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    print(f"Using device: {CONFIG['device']}")
