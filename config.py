import os
import numpy as np
import torch
from src.config_types import (
    FeatureEmbeddingConfig,
    GatedDCNConfig,
    STECConfig,
    EnsembleConfig,
    ModelConfig,
    ConfigType,
    MultiHeadDiversityConfig
)


# --- CONFIGURATION ---
CONFIG: ConfigType = {
    # === General ===
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    
    # === Data Loading ===
    "batch_size": 4096,  # Increased for faster training
    "num_workers": 4,  # Increased for faster data loading
    "validation_split": 0.0,  # Hold out 1% for validation
    "shuffle_train": False,  # Set False for time-sorted datasets to preserve temporal order
    
    # === Data Processing ===
    # Sort keys for data processor (applied before feature engineering)
    # Default: app_id, site_id, banner_pos, C1, day_of_month, hour_of_day
    "data_processor_sort_keys": ["app_id", "site_id", "banner_pos", "C1", "day_of_month", "hour"],
    "min_freq": 0,
    
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
    
    # === MULTI-HEAD DIVERSITY MODEL ===
    "model": {
        "backbone_type": "gated_dcn",
        "diversity_weight": 0.1,
        "feature_bagging_ratio": 0.8,
        
        # Backbone Configuration (GatedDCN w/ neutralized MLP)
        "backbone_config": {
            # Interaction Layers
            "use_dcn": True,
            "dcn_num_layers": 3,
            "dcn_use_layernorm": True,
            "dcn_low_rank": 64,
            
            # Feature Gating (used instead of SENET)
            "use_feature_gating": False,
            "feature_gating_activation": "sigmoid",
            "feature_gating_low_rank": None,
            
            # SENET (Disabled)
            "use_senet": True,
            "senet_squeeze_funcs": ["mean", "max", "min", "std"],
            "senet_reduction_ratio": 3,
            "senet_hidden_activation": "gelu",
            "senet_excitation_activation": "tanh",
            "senet_num_groups": 2,
            "senet_reweight_mode": "element",
            "senet_use_fuse": True,
            "senet_use_layer_norm": True,
            
            # MLP Props (Neutralized/Ignored by MultiHeadDiversityModel, but required by type)
            "mlp_hidden_dims": [0],  # Dummy value
            "mlp_activation": "gelu",
            "mlp_use_skip_connections": True,
            "mlp_dropout": 0.0,
            "use_layer_norm": True,
            "focal_loss_gamma": 0.0,
            "label_smoothing": 0.0,
        },
        
        # Multiple Independent Heads
        "heads": [
            # Head 1
            {
                "hidden_dims": [256],
                "activation": "gelu",
                "dropout": 0.15,
                "use_layer_norm": True,
                "use_skip_connections": True
            },
            # Head 2
            {
                "hidden_dims": [512],
                "activation": "relu",
                "dropout": 0.2,
                "use_layer_norm": True,
                "use_skip_connections": True
            },
            # Head 3
            {
                "hidden_dims": [128],
                "activation": "silu",
                "dropout": 0.1,
                "use_layer_norm": True,
                "use_skip_connections": True
            },
            # Head 4
            {
                "hidden_dims": [1024],
                "activation": "mish",
                "dropout": 0.25,
                "use_layer_norm": True,
                "use_skip_connections": True
            }
        ]
    },
    
    # === Training ===
    "epochs": 1,
    "early_stopping_patience": 50,
    "grad_clip": 1.0,
    "use_tensorboard": True,
    "tensorboard_logdir": "./runs",
    "tensorboard_log_interval": 50,  # Log every N batches (reduces I/O overhead)
    
    # === Optimizer Configuration ===
    # Dense optimizer: for MLP, DCN, and other dense parameters
    # Options: "adamw", "adagrad", "ftrl"
    "dense_optimizer": {
        "type": "adamw",
        "lr": 1e-4,
        "warmup_epoch_ratio": 0.2,
        "weight_decay": 1e-4
    },
    # Embedding optimizer: for embedding layers (typically benefits from adaptive LR)
    # Options: "adamw", "adagrad", "ftrl"
    "embedding_optimizer": {
        "type": "adagrad",
        "lr": 1e-2,
        "warmup_epoch_ratio": 0.0,
        "weight_decay": 0.0
    },
    # FTRL config example (uncomment to use):
    # "embedding_optimizer": {
    #     "type": "ftrl",
    #     "alpha": 0.1,  # Learning rate proportionality constant
    #     "beta": 1.0,   # Learning rate smoothing parameter
    #     "l1": 2.0,     # L1 regularization (enables sparsity)
    #     "l2": 1.0,     # L2 regularization
    # },
    
    # === Automatic Mixed Precision (AMP) ===
    "auto_amp": True,  # Enable AMP for faster training on CUDA (uses float16/bfloat16)
    "amp_dtype": "float16",  # Options: 'float16' (more compatible), 'bfloat16' (better numerics)
    
    # === Model Compilation ===
    "compile_model": False,  # Enable torch.compile for faster training (requires PyTorch 2.0+)
    
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
