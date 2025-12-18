import os
import numpy as np
import torch
from typing import TypedDict, Literal


class FeatureEmbeddingConfig(TypedDict, total=False):
    """Per-feature embedding configuration.
    
    Attributes:
        type: Embedding type - 'standard' (nn.Embedding) or 'hash' (HashEmbedding)
        dim: Embedding dimension (output size)
        num_buckets: For hash embeddings: size of shared embedding pool
        num_hashes: For hash embeddings: number of hash functions (default: 2)
        aggregation_mode: For hash embeddings: 'sum', 'concatenate', or 'median'
    """
    type: Literal['standard', 'hash']
    dim: int
    num_buckets: int  # Only for type='hash'
    num_hashes: int  # Only for type='hash', default: 2
    aggregation_mode: Literal['sum', 'concatenate', 'median']  # Only for type='hash', default: 'sum'


class ConfigType(TypedDict):
    # General
    seed: int
    device: str
    
    # Data Loading
    batch_size: int
    num_workers: int
    min_freq: int
    validation_split: float
    shuffle_train: bool  # Shuffle training data (set False for time-sorted datasets)
    
    # Model Architecture - Embeddings
    embedding_dim: int  # Default embedding dimension (fallback if feature not in feature_embeddings)
    feature_embeddings: dict[str, FeatureEmbeddingConfig]  # Per-feature embedding config
    embedding_projection_dim: int | None  # None = sum of all, int = project to this dim
    
    # Model Architecture - DCN/Attention
    use_dcn: bool
    dcn_num_layers: int
    dcn_use_layernorm: bool
    dcn_low_rank: int | None  # None = full-rank, int = low-rank dimension
    use_senet: bool
    senet_squeeze_funcs: list[str]  # Options: 'mean', 'max' - can combine multiple
    senet_reduction_ratio: int
    senet_hidden_activation: str  # Bottleneck hidden layer activation
    senet_excitation_activation: str  # Final excitation output activation
    senet_num_groups: int  # SENet+: Number of groups for grouped squeeze (1 = no grouping)
    senet_reweight_mode: str  # SENet+: 'feature' (one weight per field) or 'element' (weight per element)
    senet_use_fuse: bool  # SENet+: Add original to reweighted (residual)
    senet_use_layer_norm: bool  # SENet+: Apply LayerNorm after fuse
    use_feature_gating: bool  # Alternative to SENET (mutually exclusive)
    feature_gating_activation: str  # Options: sigmoid, tanh, relu, etc.
    feature_gating_low_rank: int | None  # None = full-rank, int = low-rank dimension

    # === Model Architecture - STEC ===
    use_stec: bool
    stec_num_layers: int
    stec_num_heads: int
    stec_hidden_dim: int | None
    stec_dropout: float
    stec_use_ffn: bool
    stec_mlp_hidden_dims: list[int]
    

    mlp_hidden_dims: list[int]
    mlp_activation: str
    mlp_use_skip_connections: bool  # Add residual/skip connections to MLP layers
    use_layer_norm: bool
    
    # Training
    lr: float
    embedding_lr: float
    optimizer_mode: str  # Options: 'adamw_adagrad' or 'ftrl'
    ftrl_alpha: float  # FTRL learning rate proportionality constant
    ftrl_beta: float  # FTRL learning rate smoothing parameter
    ftrl_l1: float  # FTRL L1 regularization (sparsity)
    ftrl_l2: float  # FTRL L2 regularization
    epochs: int
    lr_warmup_epoch_ratio: float
    early_stopping_patience: int
    use_tensorboard: bool
    tensorboard_logdir: str
    tensorboard_log_interval: int  # Log every N batches
    
    # Automatic Mixed Precision (AMP)
    auto_amp: bool  # Enable automatic mixed precision for faster training
    amp_dtype: str  # Options: 'float16' or 'bfloat16'
    
    # Model Compilation
    compile_model: bool  # Enable torch.compile for faster training
    
    # Ensemble
    use_ensemble: bool  # Enable ensemble training
    ensemble_k: int  # Number of models in ensemble
    ensemble_aggregation: str  # Aggregation method: 'mean' or 'median'
    
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
    "min_freq": 0,
    "validation_split": 0.0,  # Hold out 1% for validation
    "shuffle_train": False,  # Set False for time-sorted datasets to preserve temporal order
    
    # === Model Architecture - Embeddings ===
    "embedding_dim": 16,  # Default/fallback embedding dimension
    # Per-feature embedding configuration
    # - type: 'standard' (nn.Embedding) or 'hash' (HashEmbedding)
    # - dim: embedding dimension
    # - num_buckets: for hash only, size of shared pool
    # - num_hashes: for hash only, number of hash functions (default: 2)
    # - aggregation_mode: for hash only, 'sum'/'concatenate'/'median' (default: 'sum')
    "feature_embeddings": {
        # --- Standard embeddings (low-medium cardinality) ---
        # Very low cardinality (dim 8)
        "month": {"type": "standard", "dim": 8},
        "C21_count_bin": {"type": "standard", "dim": 8},
        "device_conn_type": {"type": "standard", "dim": 8},
        "C18": {"type": "standard", "dim": 8},
        "user_hourly_impressions_bin": {"type": "standard", "dim": 8},
        "device_type": {"type": "standard", "dim": 8},
        "hours_since_last_click_bin": {"type": "standard", "dim": 8},
        "user_proxy_prev_clicks_bin": {"type": "standard", "dim": 8},
        "device_ip_cumcount_bin": {"type": "standard", "dim": 8},
        "device_id_cumcount_bin": {"type": "standard", "dim": 8},
        "C1": {"type": "standard", "dim": 8},
        "banner_pos": {"type": "standard", "dim": 8},
        "day_of_week": {"type": "standard", "dim": 8},
        "C15": {"type": "standard", "dim": 8},
        "device_ip_count_bin": {"type": "standard", "dim": 8},
        "device_id_count_bin": {"type": "standard", "dim": 8},
        "C14_count_bin": {"type": "standard", "dim": 8},
        "C17_count_bin": {"type": "standard", "dim": 8},
        "user_proxy_count_bin": {"type": "standard", "dim": 8},
        # Low cardinality (dim 16)
        "C16": {"type": "standard", "dim": 16},
        "day_of_month": {"type": "standard", "dim": 16},
        "hour_of_day": {"type": "standard", "dim": 16},
        "site_category": {"type": "standard", "dim": 16},
        "app_category": {"type": "standard", "dim": 16},
        "C21": {"type": "standard", "dim": 16},
        "C19": {"type": "standard", "dim": 16},
        # Medium cardinality (dim 24)
        "C20": {"type": "standard", "dim": 24},
        "C17": {"type": "standard", "dim": 24},
        "app_domain": {"type": "standard", "dim": 24},
        # High cardinality (dim 32)
        "C14": {"type": "standard", "dim": 32},
        "site_id": {"type": "standard", "dim": 32},
        "site_domain": {"type": "standard", "dim": 32},
        "device_model": {"type": "standard", "dim": 32},
        "app_id": {"type": "standard", "dim": 32},
        # --- Hash embeddings (very high cardinality) ---
        "device_id": {"type": "hash", "dim": 32, "num_buckets": 3500, "num_hashes": 2},
        "device_id_x_app_id": {"type": "hash", "dim": 32, "num_buckets": 3500, "num_hashes": 2},
        "device_ip": {"type": "hash", "dim": 32, "num_buckets": 5000, "num_hashes": 2},
        "user_proxy": {"type": "hash", "dim": 32, "num_buckets": 7000, "num_hashes": 2},
        "device_ip_x_C14": {"type": "hash", "dim": 32, "num_buckets": 8000, "num_hashes": 2},
    },
    "embedding_projection_dim": None,  # None = no projection, int = project to uniform dim
    
    # === Model Architecture - DCN ===
    "use_dcn": True,  # Enable/disable DCNv2 cross network
    "dcn_num_layers": 6,  # Increased for more feature interactions
    "dcn_use_layernorm": False,  # LayerNorm for cross layer stability
    "dcn_low_rank": 64,  # None = full-rank, int (e.g. 32) = low-rank decomposition
    
    # === Model Architecture - SENET ===
    "use_senet": True,  # Enable/disable SENET (Squeeze-and-Excitation) layer
    "senet_squeeze_funcs": ["mean", "max", "min", "std", "norm"],  # Squeeze functions to combine
    "senet_reduction_ratio": 4,  # Reduction ratio for excitation bottleneck
    "senet_hidden_activation": "gelu",  # Bottleneck hidden layer activation
    "senet_excitation_activation": "gelu",  # Final excitation output: sigmoid, tanh, softmax
    "senet_num_groups": 2,  # SENet+: 1 = no grouping (backward compatible)
    "senet_reweight_mode": "element",  # SENet+: 'feature' or 'element'
    "senet_use_fuse": True,  # SENet+: residual connection
    "senet_use_layer_norm": False,  # SENet+: layer norm after fuse
    
    # === Model Architecture - Feature Gating ===   
    "use_feature_gating": False,  # Alternative to SENET (mutually exclusive)
    "feature_gating_activation": "sigmoid",  # Options: sigmoid, tanh, relu, etc.
    "feature_gating_low_rank": 64,  # None = full-rank, int (e.g. 32) = low-rank decomposition
    
    # === Model Architecture - STEC ===
    "use_stec": False,
    "stec_num_layers": 4,
    "stec_num_heads": 4,
    "stec_hidden_dim": None,  # Defaults to 4 * embed_dim
    "stec_dropout": 0.0,
    "stec_use_ffn": True,
    "stec_mlp_hidden_dims": [2048],
    
    # === Model Architecture - MLP ===
    "mlp_hidden_dims": [2048, 1024, 512],  # Deeper network
    "mlp_activation": "gelu",  # Options: relu, gelu, silu, leaky_relu, tanh
    "mlp_use_skip_connections": True,  # Add residual/skip connections to MLP
    "use_layer_norm": True,
    
    # === Training ===
    "lr": 1e-4,  # Lower initial LR for better convergence
    "embedding_lr": 1e-1,  # Higher LR for embeddings (Adagrad style)
    "optimizer_mode": "adamw_adagrad",  # Options: 'adamw_adagrad' or 'ftrl'
    "ftrl_alpha": 0.1,  # FTRL learning rate proportionality constant
    "ftrl_beta": 1.0,  # FTRL learning rate smoothing parameter
    "ftrl_l1": 2.0,  # FTRL L1 regularization (enables sparsity)
    "ftrl_l2": 1.0,  # FTRL L2 regularization
    "epochs": 1,
    "early_stopping_patience": 50,
    "use_tensorboard": True,
    "tensorboard_logdir": "./runs",
    "tensorboard_log_interval": 50,  # Log every N batches (reduces I/O overhead)
    
    # === Automatic Mixed Precision (AMP) ===
    "auto_amp": True,  # Enable AMP for faster training on CUDA (uses float16/bfloat16)
    "amp_dtype": "float16",  # Options: 'float16' (more compatible), 'bfloat16' (better numerics)
    
    # === Model Compilation ===
    "compile_model": False,  # Enable torch.compile for faster training (requires PyTorch 2.0+)
    
    # === Ensemble ===
    "use_ensemble": True,  # Enable ensemble of k identical models
    "ensemble_k": 3,  # Number of models in ensemble
    "ensemble_aggregation": "mean",  # Aggregation method: 'mean' or 'median'
    
    # === Regularization ===
    "lr_warmup_epoch_ratio": 0.0,
    "mlp_dropout": 0.25,
    "grad_clip": 1.0,
    "weight_decay": 1e-4,  # L2 regularization for MLP/DCN params
    "embedding_weight_decay": 0.0,  # L2 regularization for embeddings (usually 0)
    "focal_loss_gamma": 0.0,  # Focal loss for imbalance
    "label_smoothing": 0.0,  # Optional label smoothing
    
    # === Paths ===
    "train_path": "data/raw/train.gz",
    "test_path": "data/raw/test.gz",
    "sub_path": "submission.csv",
    "processed_path": "data/processed",
    "models_path": "./checkpoints",
}

def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    print(f"Using device: {CONFIG['device']}")
