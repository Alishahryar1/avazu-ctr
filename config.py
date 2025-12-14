import os
import numpy as np
import torch
from typing import TypedDict

class ConfigType(TypedDict):
    # General
    seed: int
    device: str
    
    # Data Loading
    batch_size: int
    num_workers: int
    min_freq: int
    validation_split: float
    
    # Model Architecture
    embedding_dim: int
    dcn_num_layers: int
    mlp_hidden_dims: list[int]
    mlp_activation: str
    gating_activation: str
    use_batch_norm: bool
    
    # Training
    lr: float
    embedding_lr: float
    embedding_optimizer: str
    epochs: int
    lr_warmup_steps: int
    early_stopping_patience: int
    show_live_plot: bool
    
    # Regularization
    mlp_dropout: float
    grad_clip: float
    weight_decay: float
    focal_loss_gamma: float
    label_smoothing: float
    
    # Paths
    train_path: str
    test_path: str
    sub_path: str
    processed_path: str
    models_path: str

# --- CONFIGURATION ---
CONFIG: ConfigType = {
    # === General ===
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    
    # === Data Loading ===
    "batch_size": 2048,  # Increased for faster training
    "num_workers": 4,  # Increased for faster data loading
    "min_freq": 5,
    "validation_split": 0.01,  # Hold out 1% for validation
    
    # === Model Architecture ===
    "embedding_dim": 64,
    "dcn_num_layers": 4,  # Increased for more feature interactions
    "mlp_hidden_dims": [512, 256, 128],  # Deeper network
    "mlp_activation": "gelu",  # Options: relu, gelu, silu, leaky_relu, tanh
    "gating_activation": "tanh",  # Options: sigmoid, tanh, relu, softmax
    "use_batch_norm": True,
    
    # === Training ===
    "lr": 1e-4,  # Lower initial LR for better convergence
    "embedding_lr": 1.0,  # Higher LR for embeddings (Adagrad style)
    "embedding_optimizer": "adagrad",  # Separate optimizer for embeddings
    "epochs": 15,
    "lr_warmup_steps": 1000,
    "early_stopping_patience": 3,
    "show_live_plot": False,
    
    # === Regularization ===
    "mlp_dropout": 0.1,
    "grad_clip": 10.0,
    "weight_decay": 1e-5,  # L2 regularization
    "focal_loss_gamma": 2.0,  # Focal loss for imbalance
    "label_smoothing": 0.0,  # Optional label smoothing
    
    # === Paths ===
    "train_path": "./data/train.gz",
    "test_path": "./data/test.gz",
    "sub_path": "submission.csv",
    "processed_path": "./data",
    "models_path": "./models",
}

def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    print(f"Using device: {CONFIG['device']}")
