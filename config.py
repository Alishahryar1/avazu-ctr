import os
import numpy as np
import torch
from typing import TypedDict


class FeatureEmbeddingConfig(TypedDict, total=False):
    """Per-feature embedding configuration with optional overrides."""
    embedding_dim: int  # Override embedding dimension for this feature


class ConfigType(TypedDict):
    # General
    seed: int
    device: str
    
    # Data Loading
    batch_size: int
    num_workers: int
    min_freq: int
    validation_split: float
    
    # Model Architecture - Embeddings
    embedding_dim: int  # Default/uniform embedding dimension
    use_variable_embeddings: bool  # Enable cardinality-based embedding dimensions
    embedding_dim_rules: list[tuple[int, int]]  # (max_cardinality, embed_dim) sorted ascending
    embedding_projection_dim: int | None  # None = sum of all, int = project to this dim
    feature_embedding_overrides: dict[str, FeatureEmbeddingConfig]  # Per-feature overrides
    
    # Model Architecture - DCN/Attention
    use_dcn: bool
    dcn_num_layers: int
    dcn_use_layernorm: bool
    dcn_low_rank: int | None  # None = full-rank, int = low-rank dimension
    use_senet: bool
    senet_squeeze_funcs: list[str]  # Options: 'mean', 'max' - can combine multiple
    senet_reduction_ratio: int
    senet_activation: str
    use_feature_gating: bool  # Alternative to SENET (mutually exclusive)
    feature_gating_activation: str  # Options: sigmoid, tanh, relu, etc.
    feature_gating_low_rank: int | None  # None = full-rank, int = low-rank dimension
    mlp_hidden_dims: list[int]
    mlp_activation: str
    mlp_use_skip_connections: bool  # Add residual/skip connections to MLP layers
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
    
    # Automatic Mixed Precision (AMP)
    auto_amp: bool  # Enable automatic mixed precision for faster training
    amp_dtype: str  # Options: 'float16' or 'bfloat16'
    
    # Regularization
    mlp_dropout: float
    grad_clip: float
    weight_decay: float
    embedding_weight_decay: float
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
    "batch_size": 4096,  # Increased for faster training
    "num_workers": 4,  # Increased for faster data loading
    "min_freq": 10,
    "validation_split": 0.001,  # Hold out 1% for validation
    
    # === Model Architecture - Embeddings ===
    "embedding_dim": 64,  # Default/fallback embedding dimension
    "use_variable_embeddings": True,  # Enable cardinality-based embedding dimensions
    # Cardinality rules: (max_vocab_size, embedding_dim) - sorted ascending
    # Based on EDA analysis of actual feature cardinalities
    "embedding_dim_rules": [
        (10, 4),       # 10 features: device_type, C18, C1, banner_pos, day_of_week, C15, C16, month, day_of_month, device_conn_type
        (100, 16),     # 5 features: site_category, hour_of_day, app_category, C21, C19
        (500, 24),     # 3 features: C20, app_domain, C17
        (5000, 32),    # 3 features: C14, site_id, site_domain
        (20000, 48),   # 2 features: app_id, device_model
        # Anything above 20000 uses embedding_dim (128): device_id, device_id_x_app_id, device_ip_x_C14, device_ip, user_proxy
    ],
    "embedding_projection_dim": None,  # None = no projection, int = project to uniform dim
    "feature_embedding_overrides": {},  # Per-feature overrides, e.g., {"device_id": {"embedding_dim": 128}}
    
    "use_dcn": True,  # Enable/disable DCNv2 cross network
    "dcn_num_layers": 6,  # Increased for more feature interactions
    "dcn_use_layernorm": False,  # LayerNorm for cross layer stability
    "dcn_low_rank": 32,  # None = full-rank, int (e.g. 32) = low-rank decomposition
    
    "use_senet": False,  # Enable/disable SENET (Squeeze-and-Excitation) layer
    "senet_squeeze_funcs": ["mean", "max", "min"],  # Squeeze functions to combine
    "senet_reduction_ratio": 4,  # Reduction ratio for excitation bottleneck
    "senet_activation": "tanh",  # Options: sigmoid, tanh, relu, softmax
    
    "use_feature_gating": True,  # Alternative to SENET (mutually exclusive)
    "feature_gating_activation": "sigmoid",  # Options: sigmoid, tanh, relu, etc.
    "feature_gating_low_rank": 32,  # None = full-rank, int (e.g. 32) = low-rank decomposition
    
    "mlp_hidden_dims": [1024, 512],  # Deeper network
    "mlp_activation": "gelu",  # Options: relu, gelu, silu, leaky_relu, tanh
    "mlp_use_skip_connections": True,  # Add residual/skip connections to MLP
    "use_layer_norm": True,
    
    # === Training ===
    "lr": 1e-3,  # Lower initial LR for better convergence
    "embedding_lr": 1.0,  # Higher LR for embeddings (Adagrad style)
    "embedding_optimizer": "adagrad",  # Separate optimizer for embeddings
    "epochs": 50,
    "early_stopping_patience": 50,
    "use_tensorboard": False,
    "tensorboard_logdir": "./runs",
    "tensorboard_log_interval": 1000,  # Log every N batches (reduces I/O overhead)
    
    # === Automatic Mixed Precision (AMP) ===
    "auto_amp": True,  # Enable AMP for faster training on CUDA (uses float16/bfloat16)
    "amp_dtype": "bfloat16",  # Options: 'float16' (more compatible), 'bfloat16' (better numerics)
    
    # === Regularization ===
    "lr_warmup_epoch_ratio": 0.1,
    "mlp_dropout": 0.1,
    "grad_clip": 1.0,
    "weight_decay": 1e-5,  # L2 regularization for MLP/DCN params
    "embedding_weight_decay": 0.0,  # L2 regularization for embeddings (usually 0)
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
    torch.backends.cudnn.benchmark = True
    print(f"Using device: {CONFIG['device']}")
