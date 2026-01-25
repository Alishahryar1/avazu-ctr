"""Model architectures for CTR prediction."""

from config import ConfigType
from src.models.architectures.base import BaseCTRModel, ModelOutput
from src.models.architectures.gated_dcn import GatedDCNModel
from src.models.architectures.stec import STECModel
from src.models.architectures.ensemble import EnsembleModel
from src.models.architectures.multi_head_diversity import MultiHeadDiversityModel
from src.models.architectures.normalized_multi_head_diversity import (
    NormalizedMultiHeadDiversityModel,
)
from src.config_types import (
    GatedDCNConfig,
    STECConfig,
    EnsembleConfig,
    MultiHeadDiversityConfig,
)
from src.config_types.normalized_multi_head_diversity_config import (
    NormalizedMultiHeadDiversityConfig,
)


def create_model(
    config: ConfigType, vocab_sizes: dict[str, int], feature_names: list[str]
) -> BaseCTRModel:
    """
    Factory function to create the appropriate model based on config.

    Config flags checked (in order of priority):
    - use_ensemble: Creates EnsembleModel
    - use_stec: Creates STECModel
    - MultiHeadDiversityConfig detected: Creates MultiHeadDiversityModel
    - Otherwise: Creates GatedDCNModel (default)

    Args:
        config: Configuration dictionary
        vocab_sizes: Feature vocabulary sizes
        feature_names: List of feature names

    Returns:
        Instantiated model
    """
    model_config = config.get("model", {})

    # Check for ensemble config (has 'models' key)
    if "models" in model_config:
        return EnsembleModel(vocab_sizes, feature_names, config)
    # Check for STEC config (has 'stec_num_layers' key)
    elif "stec_num_layers" in model_config:
        return STECModel(vocab_sizes, feature_names, config)
    # Check for NormalizedMultiHeadDiversity config (has 'use_normalized_embeddings' key)
    elif "heads" in model_config and "use_normalized_embeddings" in model_config:
        return NormalizedMultiHeadDiversityModel(vocab_sizes, feature_names, config)
    # Check for MultiHeadDiversity config (has 'heads' key)
    elif "heads" in model_config and "backbone_type" in model_config:
        return MultiHeadDiversityModel(vocab_sizes, feature_names, config)
    # Default to GatedDCN (has 'use_dcn' key or fallback)
    elif "use_dcn" in model_config:
        return GatedDCNModel(vocab_sizes, feature_names, config)
    else:
        raise ValueError(f"Unsupported model config keys: {model_config.keys()}")


__all__ = [
    "BaseCTRModel",
    "ModelOutput",
    "GatedDCNModel",
    "STECModel",
    "EnsembleModel",
    "MultiHeadDiversityModel",
    "NormalizedMultiHeadDiversityModel",
    "create_model",
]
