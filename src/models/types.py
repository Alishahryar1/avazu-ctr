from typing import TypedDict
import torch

class ModelOutput(TypedDict):
    """Standardized output from all CTR models."""
    logits: torch.Tensor  # Primary prediction logits [B, 1]
    aux_logits: list[torch.Tensor] | None  # Optional branch logits for multi-branch models
