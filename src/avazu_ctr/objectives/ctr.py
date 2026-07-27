"""Aggregate-supervised multihead and ensemble objectives."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from avazu_ctr.config.schema import ObjectiveConfig
from avazu_ctr.contracts import ModelOutput


@dataclass(slots=True)
class LossBreakdown:
    total: torch.Tensor
    aggregate: torch.Tensor
    auxiliary: torch.Tensor
    diversity: torch.Tensor
    children: torch.Tensor

    def scalars(self) -> dict[str, float]:
        return {
            "loss": float(self.total.detach()),
            "aggregate_loss": float(self.aggregate.detach()),
            "auxiliary_loss": float(self.auxiliary.detach()),
            "diversity_loss": float(self.diversity.detach()),
            "child_loss": float(self.children.detach()),
        }


class CTRObjective(nn.Module):
    def __init__(self, config: ObjectiveConfig) -> None:
        super().__init__()
        self.config = config
        self.binary = nn.BCEWithLogitsLoss()

    def _residual_correlation(
        self, auxiliary_logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        predictions = torch.sigmoid(auxiliary_logits.squeeze(-1))
        residuals = predictions - labels
        residuals = residuals - residuals.mean(dim=0, keepdim=True)
        norms = torch.sqrt(residuals.square().sum(dim=0).clamp_min(self.config.correlation_epsilon))
        normalized = residuals / norms
        correlation = normalized.transpose(0, 1) @ normalized
        upper = torch.triu(correlation, diagonal=1)
        pairs = auxiliary_logits.shape[1] * (auxiliary_logits.shape[1] - 1) // 2
        if pairs == 0:
            return auxiliary_logits.new_zeros(())
        return upper.clamp_min(0).square().sum() / pairs

    def forward(self, output: ModelOutput, labels: torch.Tensor) -> LossBreakdown:
        aggregate = self.binary(output.aggregate_logits, labels)
        zero = aggregate.new_zeros(())
        auxiliary = zero
        diversity = zero
        if output.auxiliary_logits is not None:
            expanded = labels.unsqueeze(1).expand_as(output.auxiliary_logits)
            auxiliary = self.binary(output.auxiliary_logits, expanded)
            if output.auxiliary_logits.shape[1] > 1:
                diversity = self._residual_correlation(output.auxiliary_logits, labels)
        child_loss = zero
        if output.children:
            child_loss = torch.stack(
                [self.forward(child, labels).total for child in output.children]
            ).mean()
        total = (
            self.config.aggregate_weight * aggregate
            + self.config.auxiliary_weight * auxiliary
            + self.config.diversity_weight * diversity
            + self.config.child_weight * child_loss
        )
        return LossBreakdown(
            total=total,
            aggregate=aggregate,
            auxiliary=auxiliary,
            diversity=diversity,
            children=child_loss,
        )
