"""Model architectures for CTR prediction."""
from src.config.config import ConfigType
from src.models.architectures.base import BaseCTRModel, ModelOutput
from src.models.architectures.gated_dcn import GatedDCNModel
from src.models.architectures.stec import STECModel
from src.models.architectures.ensemble import EnsembleModel


# Model registry - maps config flags to model classes
MODEL_REGISTRY: dict[str, type[BaseCTRModel]] = {
    "gated_dcn": GatedDCNModel,
    "stec": STECModel,
    "ensemble": EnsembleModel,
}


def create_model(
    config: ConfigType,
    vocab_sizes: dict[str, int],
    feature_names: list[str]
) -> BaseCTRModel:
    """
    Factory function to create the appropriate model based on config.
    
    Config flags checked (in order of priority):
    - use_ensemble: Creates EnsembleModel  
    - use_stec: Creates STECModel
    - Otherwise: Creates GatedDCNModel (default)
    
    Args:
        config: Configuration dictionary
        vocab_sizes: Feature vocabulary sizes
        feature_names: List of feature names
        
    Returns:
        Instantiated model
    """
    if config.get('use_ensemble', False):
        return EnsembleModel(vocab_sizes, feature_names, config)
    elif config.get('use_stec', False):
        return STECModel(vocab_sizes, feature_names, config)
    else:
        return GatedDCNModel(vocab_sizes, feature_names, config)


__all__ = [
    'BaseCTRModel',
    'ModelOutput',
    'GatedDCNModel',
    'STECModel',
    'EnsembleModel',
    'MODEL_REGISTRY',
    'create_model',
]
