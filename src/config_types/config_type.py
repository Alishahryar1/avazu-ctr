from typing import TypedDict
from .feature_embedding_config import FeatureEmbeddingConfig
from .model_config import ModelConfig

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
    
    # Data Processing
    data_processor_sort_keys: list[str]  # Sort keys for data processor

    # Embeddings
    embedding_dim: int
    feature_embeddings: dict[str, FeatureEmbeddingConfig]
    embedding_projection_dim: int | None  # None = no projection
    
    # Model
    model: ModelConfig
    
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

    # Regularization
    lr_warmup_epoch_ratio: float 
    grad_clip: float
    weight_decay: float
    embedding_weight_decay: float
    
    # Paths
    train_path: str
    test_path: str
    sub_path: str
    processed_path: str
    models_path: str
