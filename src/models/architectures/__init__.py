"""Model architectures for CTR prediction."""

from src.config_types import ConfigType
from src.models.architectures.base import BaseCTRModel
from src.models.types import ModelOutput
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

    Uses model_type if present, otherwise falls back to key-based detection.

    Args:
        config: Configuration dictionary
        vocab_sizes: Feature vocabulary sizes
        feature_names: List of feature names

    Returns:
        Instantiated model
    """
    model_config = config.get("model", {})
    model_type = model_config.get("model_type")

    if model_type == "ensemble":
        return EnsembleModel(vocab_sizes, feature_names, config)
    if model_type == "stec":
        return STECModel(vocab_sizes, feature_names, config)
    if model_type == "normalized_multi_head_diversity":
        return NormalizedMultiHeadDiversityModel(vocab_sizes, feature_names, config)
    if model_type == "multi_head_diversity":
        return MultiHeadDiversityModel(vocab_sizes, feature_names, config)
    if model_type == "gated_dcn":
        return GatedDCNModel(vocab_sizes, feature_names, config)

    # Fallback: key-based detection for backward compatibility
    if "models" in model_config:
        return EnsembleModel(vocab_sizes, feature_names, config)
    if "stec_num_layers" in model_config:
        return STECModel(vocab_sizes, feature_names, config)
    if "heads" in model_config and "use_normalized_embeddings" in model_config:
        return NormalizedMultiHeadDiversityModel(vocab_sizes, feature_names, config)
    if "heads" in model_config and "backbone_type" in model_config:
        return MultiHeadDiversityModel(vocab_sizes, feature_names, config)
    if "use_dcn" in model_config:
        return GatedDCNModel(vocab_sizes, feature_names, config)

    raise ValueError(f"Unsupported model config. Set model_type or use known keys. Keys: {list(model_config.keys())}")


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
