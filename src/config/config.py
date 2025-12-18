import os
import numpy as np
import torch
from .types import (
    FeatureEmbeddingConfig,
    GatedDCNConfig,
    STECConfig,
    EnsembleConfig,
    ModelConfig,
    ConfigType
)


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
    
    # === ENSEMBLE MODEL ===
    # Contains: GatedDCN, STEC, and inner ensemble (MLP-only + SENET+DCN)
    "model": {
        "models": [
            # --- Model 1: Small GatedDCN (DCN + Feature Gating) ---
            {
                "use_dcn": True,
                "dcn_num_layers": 3,
                "dcn_use_layernorm": True,
                "dcn_low_rank": 32,
                "use_senet": False,
                "senet_squeeze_funcs": ["mean"],
                "senet_reduction_ratio": 3,
                "senet_hidden_activation": "relu",
                "senet_excitation_activation": "sigmoid",
                "senet_num_groups": 1,
                "senet_reweight_mode": "feature",
                "senet_use_fuse": False,
                "senet_use_layer_norm": False,
                "use_feature_gating": True,
                "feature_gating_activation": "sigmoid",
                "feature_gating_low_rank": None,
                "mlp_hidden_dims": [512, 256],
                "mlp_activation": "gelu",
                "mlp_use_skip_connections": True,
                "mlp_dropout": 0.1,
                "use_layer_norm": True,
                "focal_loss_gamma": 0.0,
                "label_smoothing": 0.0,
            },
            # --- Model 2: Small STEC ---
            {
                "stec_num_layers": 2,
                "stec_num_heads": 4,
                "stec_hidden_dim": None,
                "stec_dropout": 0.0,
                "stec_use_ffn": True,
                "stec_mlp_hidden_dims": [256, 128],
            },
            # --- Model 3: Inner Ensemble (MLP-only + SENET+DCN) ---
            {
                "models": [
                    # Inner Model A: Small MLP-only (no DCN, no SENET)
                    {
                        "use_dcn": False,
                        "dcn_num_layers": 0,
                        "dcn_use_layernorm": False,
                        "dcn_low_rank": None,
                        "use_senet": False,
                        "senet_squeeze_funcs": ["mean"],
                        "senet_reduction_ratio": 3,
                        "senet_hidden_activation": "relu",
                        "senet_excitation_activation": "sigmoid",
                        "senet_num_groups": 1,
                        "senet_reweight_mode": "feature",
                        "senet_use_fuse": False,
                        "senet_use_layer_norm": False,
                        "use_feature_gating": False,
                        "feature_gating_activation": "sigmoid",
                        "feature_gating_low_rank": None,
                        "mlp_hidden_dims": [512, 256, 128],
                        "mlp_activation": "gelu",
                        "mlp_use_skip_connections": True,
                        "mlp_dropout": 0.1,
                        "use_layer_norm": True,
                        "focal_loss_gamma": 0.0,
                        "label_smoothing": 0.0,
                    },
                    # Inner Model B: Small SENET + DCN
                    {
                        "use_dcn": True,
                        "dcn_num_layers": 3,
                        "dcn_use_layernorm": True,
                        "dcn_low_rank": 32,
                        "use_senet": True,
                        "senet_squeeze_funcs": ["mean", "max", "min", "std", "norm"],
                        "senet_reduction_ratio": 3,
                        "senet_hidden_activation": "gelu",
                        "senet_excitation_activation": "tanh",
                        "senet_num_groups": 1,
                        "senet_reweight_mode": "feature",
                        "senet_use_fuse": True,
                        "senet_use_layer_norm": True,
                        "use_feature_gating": False,
                        "feature_gating_activation": "sigmoid",
                        "feature_gating_low_rank": None,
                        "mlp_hidden_dims": [512, 256],
                        "mlp_activation": "gelu",
                        "mlp_use_skip_connections": True,
                        "mlp_dropout": 0.25,
                        "use_layer_norm": True,
                        "focal_loss_gamma": 0.0,
                        "label_smoothing": 0.0,
                    },
                ],
                "ensemble_aggregation": "mean",
            },
        ],
        "ensemble_aggregation": "mean",
    },
    
    # === Training ===
    "lr": 1e-4,  # Lower initial LR for better convergence
    "embedding_lr": 1e-2,  # Higher LR for embeddings (Adagrad style)
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
    
    # === Regularization ===
    "lr_warmup_epoch_ratio": 0.2,
    "grad_clip": 1.0,
    "weight_decay": 1e-4,  # L2 regularization for MLP/DCN params
    "embedding_weight_decay": 0.0,  # L2 regularization for embeddings (usually 0)
    
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
