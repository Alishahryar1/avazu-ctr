from typing import TypedDict
from .feature_embedding_config import FeatureEmbeddingConfig
from .model_config import ModelConfig
from .optimizer_config import OptimizerConfig


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

    # Model
    model: ModelConfig

    # Training
    epochs: int
    early_stopping_patience: int
    grad_clip: float
    use_tensorboard: bool
    tensorboard_logdir: str
    tensorboard_log_interval: int  # Log every N batches

    # Optimizer Configuration
    dense_optimizer: OptimizerConfig  # For MLP, DCN, and other dense parameters
    embedding_optimizer: OptimizerConfig  # For embedding layers

    # Automatic Mixed Precision (AMP)
    auto_amp: bool  # Enable automatic mixed precision for faster training
    amp_dtype: str  # Options: 'float16' or 'bfloat16'

    # Model Compilation
    compile_model: bool  # Enable torch.compile for faster training

    # Paths
    train_path: str
    test_path: str
    sub_path: str
    processed_path: str
    models_path: str
