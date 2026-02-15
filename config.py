import os
from typing import cast

import numpy as np
import torch
from src.config_types import (
    ConfigType,
)


# --- CONFIGURATION ---
CONFIG: ConfigType = cast(
    ConfigType,
    {
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
    "data_processor_sort_keys": [
        "app_id",
        "site_id",
        "banner_pos",
        "C1",
        "day_of_month",
        "hour",
    ],
    "min_freq": 0,
    # === Automatic Mixed Precision (AMP) ===
    "auto_amp": True,  # Enable AMP for faster training on CUDA (uses float16/bfloat16)
    "amp_dtype": "float16",  # Options: 'float16' (more compatible), 'bfloat16' (better numerics)
    # === Model Compilation ===
    "compile_model": True,  # Enable torch.compile for faster training (requires PyTorch 2.0+)
    # === Paths ===
    "train_path": "data/raw/train.gz",
    "test_path": "data/raw/test.gz",
    "sub_path": "submission.csv",
    "processed_path": "data/processed",
    "models_path": "./checkpoints",
    # === Model Architecture - Embeddings ===
    "embedding_dim": 16,  # Default/fallback embedding dimension
    "feature_embeddings": {
        # --- Binned features (categorical versions of numerical features) ---
        # Count features (binned: 0, 1, 2-5, 6-10, 11-50, 51-100, 101-500, 501-1000, 1000+)
        "device_ip_count_bin": {"type": "standard", "dim": 8 * 2},
        "device_id_count_bin": {"type": "standard", "dim": 8 * 2},
        "C14_count_bin": {"type": "standard", "dim": 8 * 2},
        "C17_count_bin": {"type": "standard", "dim": 8 * 2},
        "C21_count_bin": {"type": "standard", "dim": 8 * 2},
        "user_proxy_count_bin": {"type": "standard", "dim": 8 * 2},
        # Cumcount features (binned: first, 2-3, 4-10, 11-50, 51-100, 100+)
        "device_ip_cumcount_bin": {"type": "standard", "dim": 8 * 2},
        "device_id_cumcount_bin": {"type": "standard", "dim": 8 * 2},
        # Nunique features (binned: 1, 2, 3-5, 6-10, 11-50, 50+)
        "device_ip_nunique_apps_bin": {"type": "standard", "dim": 8 * 2},
        "device_ip_nunique_sites_bin": {"type": "standard", "dim": 8 * 2},
        "user_proxy_nunique_apps_bin": {"type": "standard", "dim": 8 * 2},
        "user_proxy_nunique_sites_bin": {"type": "standard", "dim": 8 * 2},
        # Likelihood features (binned: very_low, low, medium, high, very_high)
        "app_id_likelihood_bin": {"type": "standard", "dim": 8 * 2},
        "site_id_likelihood_bin": {"type": "standard", "dim": 8 * 2},
        "site_domain_likelihood_bin": {"type": "standard", "dim": 8 * 2},
        "app_domain_likelihood_bin": {"type": "standard", "dim": 8 * 2},
        "C14_likelihood_bin": {"type": "standard", "dim": 8 * 2},
        "C17_likelihood_bin": {"type": "standard", "dim": 8 * 2},
        # Time/sequence features (binned)
        "hours_since_last_click_bin": {"type": "standard", "dim": 8 * 2},
        "user_proxy_prev_clicks_bin": {"type": "standard", "dim": 8 * 2},
        "user_hourly_impressions_bin": {"type": "standard", "dim": 8 * 2},
        # --- Raw numerical features (continuous values with log transform) ---
        # Count features (raw)
        "device_ip_count": {"type": "numerical", "use_log_transform": True},
        "device_id_count": {"type": "numerical", "use_log_transform": True},
        "C14_count": {"type": "numerical", "use_log_transform": True},
        "C17_count": {"type": "numerical", "use_log_transform": True},
        "C21_count": {"type": "numerical", "use_log_transform": True},
        "user_proxy_count": {"type": "numerical", "use_log_transform": True},
        # Cumcount features (raw)
        "device_ip_cumcount": {"type": "numerical", "use_log_transform": True},
        "device_id_cumcount": {"type": "numerical", "use_log_transform": True},
        # Nunique features (raw)
        "device_ip_nunique_apps": {"type": "numerical", "use_log_transform": True},
        "device_ip_nunique_sites": {"type": "numerical", "use_log_transform": True},
        "user_proxy_nunique_apps": {"type": "numerical", "use_log_transform": True},
        "user_proxy_nunique_sites": {"type": "numerical", "use_log_transform": True},
        # Likelihood features (raw probabilities, no log transform)
        "app_id_likelihood": {"type": "numerical", "use_log_transform": False},
        "site_id_likelihood": {"type": "numerical", "use_log_transform": False},
        "site_domain_likelihood": {"type": "numerical", "use_log_transform": False},
        "app_domain_likelihood": {"type": "numerical", "use_log_transform": False},
        "C14_likelihood": {"type": "numerical", "use_log_transform": False},
        "C17_likelihood": {"type": "numerical", "use_log_transform": False},
        # Time/sequence features (raw)
        "hours_since_last_click": {"type": "numerical", "use_log_transform": True},
        "user_proxy_prev_clicks": {"type": "numerical", "use_log_transform": True},
        "user_hourly_impressions": {"type": "numerical", "use_log_transform": True},
        # --- Standard embeddings (true categorical features) ---
        # Very low cardinality (dim 8)
        "month": {"type": "standard", "dim": 8 * 2},
        "device_conn_type": {"type": "standard", "dim": 8 * 2},
        "C18": {"type": "standard", "dim": 8 * 2},
        "device_type": {"type": "standard", "dim": 8 * 2},
        "C1": {"type": "standard", "dim": 8 * 2},
        "banner_pos": {"type": "standard", "dim": 8 * 2},
        "day_of_week": {"type": "standard", "dim": 8 * 2},
        "C15": {"type": "standard", "dim": 8 * 2},
        # Low cardinality (dim 16)
        "C16": {"type": "standard", "dim": 16 * 2},
        "day_of_month": {"type": "standard", "dim": 16 * 2},
        "hour_of_day": {"type": "standard", "dim": 16 * 2},
        "site_category": {"type": "standard", "dim": 16 * 2},
        "app_category": {"type": "standard", "dim": 16 * 2},
        "C21": {"type": "standard", "dim": 16 * 2},
        "C19": {"type": "standard", "dim": 16 * 2},
        # Medium cardinality (dim 24)
        "C20": {"type": "standard", "dim": 24 * 2},
        "C17": {"type": "standard", "dim": 24 * 2},
        "app_domain": {"type": "standard", "dim": 24 * 2},
        # High cardinality (dim 32)
        "C14": {"type": "standard", "dim": 32 * 2},
        "site_id": {"type": "standard", "dim": 32 * 2},
        "site_domain": {"type": "standard", "dim": 32 * 2},
        "device_model": {"type": "standard", "dim": 32 * 2},
        "app_id": {"type": "standard", "dim": 32 * 2},
        # --- Hash embeddings (very high cardinality) ---
        "device_id": {
            "type": "hash",
            "dim": 32 * 2,
            "num_buckets": 500_000,
            "num_hashes": 2,
        },
        "device_id_x_app_id": {
            "type": "hash",
            "dim": 32 * 2,
            "num_buckets": 500_000,
            "num_hashes": 2,
        },
        "device_ip": {
            "type": "hash",
            "dim": 32 * 2,
            "num_buckets": 500_000,
            "num_hashes": 2,
        },
        "user_proxy": {
            "type": "hash",
            "dim": 32 * 2,
            "num_buckets": 500_000,
            "num_hashes": 2,
        },
        "device_ip_x_C14": {
            "type": "hash",
            "dim": 32 * 2,
            "num_buckets": 500_000,
            "num_hashes": 2,
        },
    },
    # === NORMALIZED MULTIHEAD DIVERSITY MODEL (nGPT-style) ===
    # Reference: "nGPT: Normalized Transformer with Representation Learning
    #            on the Hypersphere" (Loshchilov et al., ICLR 2025)
    "model": {
        "model_type": "normalized_multi_head_diversity",
        "backbone_type": "gated_dcn",
        "diversity_weight": 0.001,
        "feature_bagging_ratio": 0.8,
        "aggregation_method": "mean",  # 'mean' | 'gated'
        "gating_hidden_dim": None,  # Optional hidden dim for gated aggregation
        # === nGPT-specific parameters ===
        "use_normalized_embeddings": True,  # Normalize embeddings to unit norm (hypersphere)
        "use_normalized_weights": True,  # Normalize weight matrices
        "alpha_init": 0.05,  # Initial eigen learning rate (~1/num_layers)
        "alpha_scale": None,  # Scale for effective LR in Adam (None = 1/sqrt(dim))
        "su_init": 1.0,  # Initial scaling for MLP u gate
        "sv_init": 1.0,  # Initial scaling for MLP v gate (multiplied by sqrt(dim))
        "use_lerp_updates": True,  # Use LERP: h ← Norm(h + α(h_block - h))
        "normalize_before_head": True,  # Normalize hidden state before each head
        # === Backbone config ===
        "backbone_config": {
            # Feature Gating
            "use_feature_gating": True,
            "feature_gating_activation": "gelu",
            "feature_gating_low_rank": None,
            # SENET
            "use_senet": False,
            "senet_squeeze_funcs": ["mean", "max", "min", "std"],
            "senet_reduction_ratio": 3,
            "senet_hidden_activation": "gelu",
            "senet_excitation_activation": "tanh",
            "senet_num_groups": 2,
            "senet_reweight_mode": "element",
            "senet_use_fuse": True,
            "senet_use_layer_norm": False,  # Disabled for nGPT (uses unit norm instead)
            # DCN
            "use_dcn": True,
            "dcn_num_layers": 8,  # Reduced for nGPT (faster convergence expected)
            "dcn_use_layernorm": False,  # Disabled for nGPT
            "dcn_low_rank": 32,
            # MLP
            "mlp_hidden_dims": [
                512
            ],  # Single layer (nGPT uses 4x expansion internally)
            "mlp_activation": "silu",  # SwiGLU uses SiLU internally
            "mlp_use_skip_connections": False,  # LERP handles residuals
            "mlp_dropout": 0.1,
            "use_layer_norm": False,  # Disabled for nGPT
        },
        # === Prediction heads ===
        "heads": [
            {
                "hidden_dims": [128],
                "activation": "silu",
                "dropout": 0.3,
                "use_layer_norm": False,
                "use_skip_connections": False,
            },
            {
                "hidden_dims": [64],
                "activation": "silu",
                "dropout": 0.3,
                "use_layer_norm": False,
                "use_skip_connections": False,
            },
            {
                "hidden_dims": [32],
                "activation": "silu",
                "dropout": 0.2,
                "use_layer_norm": False,
                "use_skip_connections": False,
            },
        ],
    },
    # === Training ===
    "epochs": 1,
    "early_stopping_patience": 50,
    "grad_clip": 4.968085896356788,
    "use_tensorboard": True,
    "tensorboard_logdir": "./runs",
    "tensorboard_log_interval": 50,  # Log every N batches (reduces I/O overhead)
    # === Optimizer Configuration (nGPT-style: no weight decay, no warmup) ===
    # Note: nGPT normalizes weights, making weight decay unnecessary
    "dense_optimizer": {
        "type": "adamw",
        "lr": 0.001,  # nGPT can use higher LR due to normalized optimization
        "weight_decay": 0.0,  # No weight decay for nGPT (normalization handles it)
        "scheduler": {
            "warmup_epoch_ratio": 0.0,  # No warmup for nGPT
            "min_lr": 1e-6,
            "decay_type": "cosine",  # Cosine annealing as in nGPT paper
        },
    },
    "embedding_optimizer": {
        "type": "adamw",  # Use Adam for embeddings too (nGPT style)
        "lr": 0.001,  # Same LR for all parameters in nGPT
        "weight_decay": 0.0,  # No weight decay
        "scheduler": {
            "warmup_epoch_ratio": 0.0,  # No warmup
            "min_lr": 1e-6,
            "decay_type": "cosine",
        },
    },
    # FTRL config example (uncomment to use):
    # "dense_optimizer": {
    #     "type": "ftrl",
    #     "alpha": 0.1,
    #     "beta": 1.0,
    #     "l1": 2.0,
    #     "l2": 1.0,
    # },
    # "embedding_optimizer": {
    #     "type": "ftrl",
    #     "alpha": 0.001,  # Learning rate proportionality constant
    #     "beta": 1.0,   # Learning rate smoothing parameter
    #     "l1": 0.0,     # L1 regularization (enables sparsity)
    #     "l2": 0.0,     # L2 regularization
    # }
},
)


def seed_everything(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    print(f"Using device: {CONFIG['device']}")
