"""Base model contract."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn

from avazu_ctr.contracts import FeatureBatch, ModelOutput


class CTRModel(nn.Module, ABC):
    @abstractmethod
    def forward(self, batch: FeatureBatch) -> ModelOutput:
        """Return aggregate logits and optional auxiliary structure."""

    def predict_proba(self, batch: FeatureBatch) -> torch.Tensor:
        return self(batch).probabilities()

    def post_step(self) -> None:
        """Apply model constraints after a successful optimizer step."""

    def embedding_table_parameters(self) -> list[nn.Parameter]:
        encoder = getattr(self, "encoder", None)
        if not isinstance(encoder, nn.Module):
            return []
        return [module.weight for module in encoder.modules() if isinstance(module, nn.Embedding)]
