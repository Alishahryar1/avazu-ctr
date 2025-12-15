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
    use_dcn: bool
    dcn_num_layers: int
    dcn_use_layernorm: bool
    dcn_low_rank: int | None  # None = full-rank, int = low-rank dimension
    use_senet: bool
    senet_squeeze_funcs: list[str]  # Options: 'mean', 'max' - can combine multiple
    senet_reduction_ratio: int
    senet_activation: str
    mlp_hidden_dims: list[int]
    mlp_activation: str
    use_layer_norm: bool
    
    # Training
    lr: float
    embedding_lr: float
    embedding_optimizer: str
    epochs: int
    lr_warmup_epoch_ratio: float
    early_stopping_patience: int
    use_tensorboard: bool
    tensorboard_logdir: str
    tensorboard_log_interval: int  # Log every N batches
    
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
    
    "use_dcn": True,  # Enable/disable DCNv2 cross network
    "dcn_num_layers": 4,  # Increased for more feature interactions
    "dcn_use_layernorm": False,  # LayerNorm for cross layer stability
    "dcn_low_rank": 32,  # None = full-rank, int (e.g. 32) = low-rank decomposition
    
    "use_senet": True,  # Enable/disable SENET (Squeeze-and-Excitation) layer
    "senet_squeeze_funcs": ["mean", "max", "min"],  # Squeeze functions to combine
    "senet_reduction_ratio": 3,  # Reduction ratio for excitation bottleneck
    "senet_activation": "tanh",  # Options: sigmoid, tanh, relu, softmax
    
    "mlp_hidden_dims": [1024, 512, 512],  # Deeper network
    "mlp_activation": "gelu",  # Options: relu, gelu, silu, leaky_relu, tanh
    "use_layer_norm": True,
    
    # === Training ===
    "lr": 1e-3,  # Lower initial LR for better convergence
    "embedding_lr": 1.0,  # Higher LR for embeddings (Adagrad style)
    "embedding_optimizer": "adagrad",  # Separate optimizer for embeddings
    "epochs": 50,
    "lr_warmup_epoch_ratio": 0.25,
    "early_stopping_patience": 50,
    "use_tensorboard": False,
    "tensorboard_logdir": "./runs",
    "tensorboard_log_interval": 1000,  # Log every N batches (reduces I/O overhead)
    
    # === Regularization ===
    "mlp_dropout": 0.1,
    "grad_clip": 10.0,
    "weight_decay": 1e-5,  # L2 regularization
    "focal_loss_gamma": 0.0,  # Focal loss for imbalance
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
