"""Model architectures for CTR prediction."""
from src.models.architectures.base_model import GatedDCNModel
from src.models.architectures.ensemble import EnsembleModel
from src.models.architectures.fcnv2 import FCNv2Model

__all__ = [
    'GatedDCNModel',
    'EnsembleModel',
    'FCNv2Model',
]
