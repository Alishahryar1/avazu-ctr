"""Loss functions for CTR prediction.

This module re-exports losses from src.models.losses for backward compatibility.
The actual implementations are in src/models/losses.py to avoid circular imports.
"""

from src.models.losses import FocalLoss, KBCELoss

__all__ = ["FocalLoss", "KBCELoss"]
