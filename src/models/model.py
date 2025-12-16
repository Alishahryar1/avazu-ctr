"""
DEPRECATED: This module is kept for backward compatibility.

All components have been moved to a modular structure:
- Utils: src.models.utils
- Layers: src.models.layers
- Architectures: src.models.architectures

Please update your imports to use the new module structure.
"""

# Backward compatibility imports
from src.models.utils import get_activation, compute_embedding_dim
from src.models.layers.senet import SENetLayer
from src.models.layers.gating import FeatureGatingLayer
from src.models.layers.cross_network import DCNv2
from src.models.layers.mlp import ResidualMLP
from src.models.architectures.gated_dcn import GatedDCNModel
from src.models.architectures.ensemble import EnsembleModel

__all__ = [
    'get_activation',
    'compute_embedding_dim',
    'SENetLayer',
    'FeatureGatingLayer',
    'DCNv2',
    'ResidualMLP',
    'GatedDCNModel',
    'EnsembleModel',
]
