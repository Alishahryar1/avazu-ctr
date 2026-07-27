"""CTR architecture implementations sharing one deployed-logit contract."""

from __future__ import annotations

from typing import cast

import torch
from torch import nn
from torch.nn import functional

from avazu_ctr.config.schema import Aggregation, ModelConfig
from avazu_ctr.contracts import FeatureBatch, ModelOutput
from avazu_ctr.models.base import CTRModel
from avazu_ctr.models.layers import (
    Backbone,
    FeatureEncoder,
    LogitGate,
    NormalizedLinear,
    PredictionHead,
    StableHashEmbedding,
)


class GatedDCNModel(CTRModel):
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
        self.encoder = FeatureEncoder(
            categorical_columns,
            numerical_columns,
            cardinalities,
            config,
            seed=seed,
        )
        self.backbone = Backbone(self.encoder.fields, self.encoder.dimension, config.backbone)
        self.output = nn.Linear(self.backbone.output_dimension, 1)

    def forward(self, batch: FeatureBatch) -> ModelOutput:
        shared = self.backbone(self.encoder(batch))
        return ModelOutput(aggregate_logits=self.output(shared))


class MultiHeadModel(CTRModel):
    def __init__(
        self,
        categorical_columns: tuple[str, ...],
        numerical_columns: tuple[str, ...],
        cardinalities: dict[str, int],
        config: ModelConfig,
        *,
        seed: int,
        normalized: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = FeatureEncoder(
            categorical_columns,
            numerical_columns,
            cardinalities,
            config,
            seed=seed,
            normalized=normalized,
        )
        self.backbone = Backbone(self.encoder.fields, self.encoder.dimension, config.backbone)
        head_type: type[nn.Linear] = NormalizedLinear if normalized else nn.Linear
        self.heads = nn.ModuleList()
        for head_config in config.heads:
            if normalized and not head_config.hidden:
                self.heads.append(head_type(self.backbone.output_dimension, 1))
            else:
                self.heads.append(
                    PredictionHead(
                        self.backbone.output_dimension,
                        head_config.hidden,
                        head_config.dropout,
                    )
                )
        self.aggregation = config.aggregation
        self.gate = LogitGate(len(self.heads)) if config.aggregation is Aggregation.GATED else None
        self.normalized = normalized
        self.feature_masks: torch.Tensor
        self.register_buffer(
            "feature_masks",
            self._make_masks(
                len(self.heads),
                self.encoder.fields,
                config.feature_bagging,
                seed,
            ),
        )
        if normalized:
            self.post_step()

    @staticmethod
    def _make_masks(heads: int, fields: int, ratio: float, seed: int) -> torch.Tensor:
        generator = torch.Generator().manual_seed(seed)
        masks = torch.rand((heads, fields), generator=generator) < ratio
        for mask in masks:
            if not mask.any():
                mask[torch.randint(fields, (), generator=generator)] = True
        return masks.to(torch.float32)

    def forward(self, batch: FeatureBatch) -> ModelOutput:
        embeddings = self.encoder(batch)
        logits = []
        for index, head in enumerate(self.heads):
            masked = embeddings * self.feature_masks[index].view(1, -1, 1)
            shared = self.backbone(masked)
            if self.normalized:
                shared = functional.normalize(shared, dim=1)
            logits.append(head(shared))
        auxiliary = torch.stack(logits, dim=1)
        diagnostics: dict[str, torch.Tensor] = {}
        if self.gate is not None:
            aggregate, weights = self.gate(auxiliary.squeeze(-1))
            diagnostics["aggregation_weights"] = weights
        else:
            aggregate = auxiliary.mean(dim=1)
        return ModelOutput(
            aggregate_logits=aggregate,
            auxiliary_logits=auxiliary,
            diagnostics=diagnostics,
        )

    @torch.no_grad()
    def post_step(self) -> None:
        if not self.normalized:
            return
        for module in self.modules():
            if isinstance(module, NormalizedLinear):
                module.normalize_()
            elif isinstance(module, nn.Linear):
                module.weight.copy_(functional.normalize(module.weight, dim=1))
            elif isinstance(module, nn.Embedding):
                module.weight.copy_(functional.normalize(module.weight, dim=1))
                if module.padding_idx is not None:
                    module.weight[module.padding_idx].zero_()
            elif isinstance(module, StableHashEmbedding):
                for child_module in module.tables:
                    table = cast(nn.Embedding, child_module)
                    table.weight.copy_(functional.normalize(table.weight, dim=1))
        self.encoder.numerical.weight.copy_(
            functional.normalize(self.encoder.numerical.weight, dim=1)
        )


class STECModel(CTRModel):
    """See-through transformer with direct interaction-level supervision path."""

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
        self.encoder = FeatureEncoder(
            categorical_columns,
            numerical_columns,
            cardinalities,
            config,
            seed=seed,
        )
        dimension = self.encoder.dimension
        if dimension % config.stec_heads:
            raise ValueError("STEC embedding dimension must be divisible by attention heads")
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=dimension,
                    nhead=config.stec_heads,
                    dim_feedforward=max(dimension * 4, 32),
                    dropout=config.backbone.dropout,
                    activation=config.backbone.activation,
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(config.stec_layers)
            ]
        )
        interactions = self.encoder.fields * self.encoder.fields
        self.level_projections = nn.ModuleList(
            [nn.Linear(interactions, dimension) for _ in range(config.stec_layers + 1)]
        )
        self.head = PredictionHead(
            dimension * (config.stec_layers + 1),
            config.heads[0].hidden,
            config.heads[0].dropout,
        )

    @staticmethod
    def _interactions(values: torch.Tensor) -> torch.Tensor:
        scaled = values / math_sqrt(values.shape[-1])
        return torch.bmm(scaled, scaled.transpose(1, 2)).flatten(start_dim=1)

    def forward(self, batch: FeatureBatch) -> ModelOutput:
        values = self.encoder(batch)
        levels = [self.level_projections[0](self._interactions(values))]
        for index, layer in enumerate(self.layers, start=1):
            values = layer(values)
            levels.append(self.level_projections[index](self._interactions(values)))
        fused = torch.cat(levels, dim=1)
        return ModelOutput(
            aggregate_logits=self.head(fused),
            diagnostics={"interaction_levels": torch.stack(levels, dim=1)},
        )


def math_sqrt(value: int) -> float:
    return float(value) ** 0.5


class EnsembleModel(CTRModel):
    def __init__(self, children: list[CTRModel], aggregation: Aggregation) -> None:
        super().__init__()
        if not children:
            raise ValueError("ensemble requires children")
        self.members = nn.ModuleList(children)
        self.aggregation = aggregation
        self.gate = LogitGate(len(children)) if aggregation is Aggregation.GATED else None

    def forward(self, batch: FeatureBatch) -> ModelOutput:
        outputs = tuple(child(batch) for child in self.members)
        child_logits = torch.cat([output.aggregate_logits for output in outputs], dim=1)
        diagnostics: dict[str, torch.Tensor] = {"child_logits": child_logits}
        if self.gate is not None:
            aggregate, weights = self.gate(child_logits)
            diagnostics["aggregation_weights"] = weights
        else:
            aggregate = child_logits.mean(dim=1, keepdim=True)
        return ModelOutput(
            aggregate_logits=aggregate,
            auxiliary_logits=child_logits.unsqueeze(-1),
            diagnostics=diagnostics,
            children=outputs,
        )

    def post_step(self) -> None:
        for module in self.members:
            child = cast(CTRModel, module)
            child.post_step()

    def embedding_parameters(self) -> list[nn.Parameter]:
        parameters: list[nn.Parameter] = []
        for module in self.members:
            parameters.extend(cast(CTRModel, module).embedding_parameters())
        return parameters
