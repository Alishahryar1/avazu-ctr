"""Model architectures for CTR prediction."""
from src.config.config import ConfigType
from src.models.architectures.base import BaseCTRModel, ModelOutput
from src.models.architectures.base_model import GatedDCNModel
from src.models.architectures.ensemble import EnsembleModel
from src.models.architectures.fcnv2 import FCNv2Model


# Model registry - maps config flags to model classes
MODEL_REGISTRY: dict[str, type[BaseCTRModel]] = {
    "gated_dcn": GatedDCNModel,
    "fcnv2": FCNv2Model,
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
    - use_fcnv2: Creates FCNv2Model
    - use_ensemble: Creates EnsembleModel  
    - Otherwise: Creates GatedDCNModel (default)
    
    Args:
        config: Configuration dictionary
        vocab_sizes: Feature vocabulary sizes
        feature_names: List of feature names
        
    Returns:
        Instantiated model
    """
    if config.get('use_fcnv2', False):
        return FCNv2Model(vocab_sizes, feature_names, config)
    elif config.get('use_ensemble', False):
        return EnsembleModel(vocab_sizes, feature_names, config)
    else:
        return GatedDCNModel(vocab_sizes, feature_names, config)


__all__ = [
    'BaseCTRModel',
    'ModelOutput',
    'GatedDCNModel',
    'EnsembleModel',
    'FCNv2Model',
    'MODEL_REGISTRY',
    'create_model',
]
