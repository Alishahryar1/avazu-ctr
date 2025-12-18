"""Model architectures for CTR prediction."""
from src.config.config import ConfigType
from src.models.architectures.base import BaseCTRModel, ModelOutput
from src.models.architectures.gated_dcn import GatedDCNModel
from src.models.architectures.stec import STECModel
from src.models.architectures.ensemble import EnsembleModel
from src.config.config import GatedDCNConfig, STECConfig, EnsembleConfig



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
    model_config = config.get('model', {})
    
    # Check for ensemble config (has 'models' key)
    if 'models' in model_config:
        return EnsembleModel(vocab_sizes, feature_names, config)
    # Check for STEC config (has 'stec_num_layers' key)
    elif 'stec_num_layers' in model_config:
        return STECModel(vocab_sizes, feature_names, config)
    # Default to GatedDCN (has 'use_dcn' key or fallback)
    elif 'use_dcn' in model_config:
        return GatedDCNModel(vocab_sizes, feature_names, config)
    else:
        raise ValueError(f"Unsupported model config: {model_config.keys()}")


__all__ = [
    'BaseCTRModel',
    'ModelOutput',
    'GatedDCNModel',
    'STECModel',
    'EnsembleModel',
    'create_model',
]
