"""
DEPRECATED: This module is kept for backward compatibility.

All components have been moved to a modular structure:
- Utils: src.models.utils
- Layers: src.models.layers
- Architectures: src.models.architectures

Please update your imports to use the new module structure.
"""

# Backward compatibility imports
from src.models.utils import get_activation, get_embedding
from src.models.layers.senet import SENetLayer
from src.models.layers.gating import FeatureGatingLayer
from src.models.layers.cross_network import DCNv2
from src.models.layers.mlp import ResidualMLP
from src.models.layers.hash_embedding import HashEmbedding
from src.models.architectures.gated_dcn import GatedDCNModel
from src.models.architectures.ensemble import EnsembleModel

__all__ = [
    "get_activation",
    "get_embedding",
    "SENetLayer",
    "FeatureGatingLayer",
    "DCNv2",
    "ResidualMLP",
    "HashEmbedding",
    "GatedDCNModel",
    "EnsembleModel",
]
