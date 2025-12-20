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
    MultiHeadDiversityConfig,
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
    "compile_model": False,  # Enable torch.compile for faster training (requires PyTorch 2.0+)
    # === Paths ===
    "train_path": "data/raw/train.gz",
    "test_path": "data/raw/test.gz",
    "sub_path": "submission.csv",
    "processed_path": "data/processed",
    "models_path": "./checkpoints",
    # === Model Architecture - Embeddings ===
    "embedding_dim": 16,  # Default/fallback embedding dimension
    "feature_embeddings": {
        # --- Standard embeddings (low-medium cardinality) ---
        # Very low cardinality (dim 8)
        "month": {"type": "standard", "dim": 8 * 2},
        "C21_count_bin": {"type": "standard", "dim": 8 * 2},
        "device_conn_type": {"type": "standard", "dim": 8 * 2},
        "C18": {"type": "standard", "dim": 8 * 2},
        "user_hourly_impressions_bin": {"type": "standard", "dim": 8 * 2},
        "device_type": {"type": "standard", "dim": 8 * 2},
        "hours_since_last_click_bin": {"type": "standard", "dim": 8 * 2},
        "user_proxy_prev_clicks_bin": {"type": "standard", "dim": 8 * 2},
        "device_ip_cumcount_bin": {"type": "standard", "dim": 8 * 2},
        "device_id_cumcount_bin": {"type": "standard", "dim": 8 * 2},
        # Reverse cumcount bins (session length proxy)
        "device_ip_reverse_cumcount_bin": {"type": "standard", "dim": 8 * 2},
        "device_id_reverse_cumcount_bin": {"type": "standard", "dim": 8 * 2},
        # Target encoding CTR bins
        "app_id_ctr_bin": {"type": "standard", "dim": 8 * 2},
        "site_id_ctr_bin": {"type": "standard", "dim": 8 * 2},
        "device_ip_ctr_bin": {"type": "standard", "dim": 8 * 2},
        "user_proxy_ctr_bin": {"type": "standard", "dim": 8 * 2},
        # Unique counts bin (bot detection)
        "device_ip_unique_app_ids_bin": {"type": "standard", "dim": 8 * 2},
        "C1": {"type": "standard", "dim": 8 * 2},
        "banner_pos": {"type": "standard", "dim": 8 * 2},
        "day_of_week": {"type": "standard", "dim": 8 * 2},
        "C15": {"type": "standard", "dim": 8 * 2},
        "device_ip_count_bin": {"type": "standard", "dim": 8 * 2},
        "device_id_count_bin": {"type": "standard", "dim": 8 * 2},
        "C14_count_bin": {"type": "standard", "dim": 8 * 2},
        "C17_count_bin": {"type": "standard", "dim": 8 * 2},
        "user_proxy_count_bin": {"type": "standard", "dim": 8 * 2},
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
    # === MULTIHEAD DIVERSITY MODEL ===
    "model": {
        "backbone_type": "gated_dcn",
        "diversity_weight": 0.1,
        "feature_bagging_ratio": 0.9,
        "aggregation_method": "mean",  # 'mean' | 'gated'
        "gating_hidden_dim": None,  # Optional hidden dim for gated aggregation
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
            "senet_use_layer_norm": True,
            # DCN
            "use_dcn": True,
            "dcn_num_layers": 6,
            "dcn_use_layernorm": True,
            "dcn_low_rank": None,
            # MLP
            "mlp_hidden_dims": [2048, 2048, 1024, 512],
            "mlp_activation": "gelu",
            "mlp_use_skip_connections": True,
            "mlp_dropout": 0.1,
            "use_layer_norm": True,
        },
        "heads": [
            {
                "hidden_dims": [128],
                "activation": "relu",
                "dropout": 0.2,
                "use_layer_norm": False,
                "use_skip_connections": True,
            },
            {
                "hidden_dims": [64],
                "activation": "gelu",
                "dropout": 0.15,
                "use_layer_norm": True,
                "use_skip_connections": True,
            },
            {
                "hidden_dims": [32],
                "activation": "silu",
                "dropout": 0.1,
                "use_layer_norm": True,
                "use_skip_connections": True,
            },
            {
                "hidden_dims": [256],
                "activation": "mish",
                "dropout": 0.25,
                "use_layer_norm": True,
                "use_skip_connections": True,
            },
        ],
    },
    # === Training ===
    "epochs": 1,
    "early_stopping_patience": 50,
    "grad_clip": 1.0,
    "use_tensorboard": True,
    "tensorboard_logdir": "./runs",
    "tensorboard_log_interval": 50,  # Log every N batches (reduces I/O overhead)
    # === Optimizer Configuration ===
    "dense_optimizer": {
        "type": "adamw",
        "lr": 1e-5,
        "warmup_epoch_ratio": 0.0,
        "weight_decay": 1e-5,
    },
    "embedding_optimizer": {
        "type": "adagrad",
        "lr": 1e-2,
        "warmup_epoch_ratio": 0.0,
        "weight_decay": 0.0,
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
}


def seed_everything(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    print(f"Using device: {CONFIG['device']}")
