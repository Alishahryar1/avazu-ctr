"""Faithful See-Through Transformer-based Encoder for CTR prediction."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional

from avazu_ctr.config.schema import ModelConfig, STECModelConfig
from avazu_ctr.contracts import FeatureBatch, ModelOutput
from avazu_ctr.models.base import CTRModel
from avazu_ctr.models.layers import FeatureEncoder, PredictionHead


def _split_heads(values: torch.Tensor, heads: int) -> torch.Tensor:
    batch, fields, dimension = values.shape
    return values.view(batch, fields, heads, dimension // heads).transpose(1, 2)


class STECBlock(nn.Module):
    """Return self-attention and its unpooled grouped bilinear interaction."""

    def __init__(self, dimension: int, heads: int, dropout: float) -> None:
        super().__init__()
        if dimension % heads:
            raise ValueError("STEC dimension must be divisible by attention heads")
        self.heads = heads
        self.head_dimension = dimension // heads
        self.query = nn.Linear(dimension, dimension, bias=False)
        self.key = nn.Linear(dimension, dimension, bias=False)
        self.value = nn.Linear(dimension, dimension, bias=False)
        self.output = nn.Linear(dimension, dimension, bias=False)
        self.attention_dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in (self.query, self.key, self.value, self.output):
            nn.init.xavier_uniform_(module.weight)

    def projected_interactions(
        self,
        values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        query = _split_heads(self.query(values), self.heads)
        key = _split_heads(self.key(values), self.heads)
        value = _split_heads(self.value(values), self.heads)
        bilinear = key.unsqueeze(3) * query.unsqueeze(2)
        return query, key, value, bilinear

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, _, value, bilinear = self.projected_interactions(values)
        logits = bilinear.mean(dim=-1) * math.sqrt(self.head_dimension)
        weights = self.attention_dropout(functional.softmax(logits, dim=-1))
        attended = weights @ value
        batch, _, fields, _ = attended.shape
        merged = attended.transpose(1, 2).contiguous().view(batch, fields, -1)
        return self.output(merged), bilinear.flatten(start_dim=1)


class STECEncoderLayer(nn.Module):
    """STEC attention followed by the paper's Add-and-Norm and position-wise FFN."""

    def __init__(self, config: STECModelConfig) -> None:
        super().__init__()
        self.stec = STECBlock(config.dimension, config.heads, config.dropout)
        hidden = config.dimension * config.ffn_multiplier
        self.attention_dropout = nn.Dropout(config.dropout)
        self.attention_norm = nn.LayerNorm(config.dimension)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.dimension, hidden),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, config.dimension),
            nn.Dropout(config.dropout),
        )
        self.feed_forward_norm = nn.LayerNorm(config.dimension)

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attended, bilinear = self.stec(values)
        values = self.attention_norm(values + self.attention_dropout(attended))
        values = self.feed_forward_norm(values + self.feed_forward(values))
        return values, bilinear


class GroupBilinearInteraction(nn.Module):
    """Produce the final grouped bilinear interaction from the last hidden state."""

    def __init__(self, dimension: int, heads: int) -> None:
        super().__init__()
        if dimension % heads:
            raise ValueError("STEC dimension must be divisible by attention heads")
        self.heads = heads
        self.head_dimension = dimension // heads
        self.projection = nn.Linear(dimension, dimension, bias=False)
        nn.init.xavier_uniform_(self.projection.weight)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        left = _split_heads(self.projection(values), self.heads)
        right = _split_heads(values, self.heads)
        interactions = left.unsqueeze(3) * right.unsqueeze(2)
        return interactions.flatten(start_dim=1)


class STECModel(CTRModel):
    """Fuse grouped bilinear interactions from every STEC level and the final state."""

    def __init__(
        self,
        categorical_columns: tuple[str, ...],
        numerical_columns: tuple[str, ...],
        cardinalities: dict[str, int],
        config: ModelConfig,
        *,
        seed: int,
    ) -> None:
        super().__init__()
        if config.stec is None:
            raise ValueError("STEC model requires stec configuration")
        architecture = config.stec
        self.encoder = FeatureEncoder(
            categorical_columns,
            numerical_columns,
            cardinalities,
            config,
            seed=seed,
            numerical_bias=False,
        )
        self.input_projection: nn.Module = (
            nn.Identity()
            if self.encoder.dimension == architecture.dimension
            else nn.Linear(self.encoder.dimension, architecture.dimension, bias=False)
        )
        self.layers = nn.ModuleList(
            [STECEncoderLayer(architecture) for _ in range(architecture.layers)]
        )
        self.final_interaction = GroupBilinearInteraction(
            architecture.dimension,
            architecture.heads,
        )
        interaction_width = self.encoder.fields * self.encoder.fields * architecture.dimension
        self.interaction_norms = nn.ModuleList(
            [
                nn.BatchNorm1d(
                    interaction_width,
                    eps=architecture.batch_norm_epsilon,
                    momentum=architecture.batch_norm_momentum,
                )
                for _ in range(architecture.layers + 1)
            ]
        )
        self.prediction = PredictionHead(
            interaction_width * (architecture.layers + 1),
            architecture.prediction_hidden,
            architecture.prediction_dropout,
        )

    def forward(self, batch: FeatureBatch) -> ModelOutput:
        values = self.input_projection(self.encoder(batch))
        interactions: list[torch.Tensor] = []
        for layer, normalization in zip(
            self.layers,
            self.interaction_norms[:-1],
            strict=True,
        ):
            values, bilinear = layer(values)
            interactions.append(normalization(bilinear))
        interactions.append(self.interaction_norms[-1](self.final_interaction(values)))
        fused = torch.cat(interactions, dim=1)
        return ModelOutput(aggregate_logits=self.prediction(fused))
