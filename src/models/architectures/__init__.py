"""Model architectures for CTR prediction."""
from src.models.architectures.base_model import GatedDCNModel
from src.models.architectures.ensemble import EnsembleModel

__all__ = [
    'GatedDCNModel',
    'EnsembleModel',
]
