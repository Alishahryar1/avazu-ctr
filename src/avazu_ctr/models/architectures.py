"""DCN and ensemble architectures sharing the deployed-logit contract."""

from __future__ import annotations

from typing import cast

import torch
from torch import nn

from avazu_ctr.config.schema import Aggregation, ModelConfig
from avazu_ctr.contracts import FeatureBatch, ModelOutput
from avazu_ctr.models.base import CTRModel
from avazu_ctr.models.layers import Backbone, FeatureEncoder, LogitGate, PredictionHead


class DCNModel(CTRModel):
    """One or more prediction heads over a shared SENet/DCNv2 backbone."""

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
        if config.dcn is None:
            raise ValueError("DCN model requires dcn configuration")
        architecture = config.dcn
        self.encoder = FeatureEncoder(
            categorical_columns,
            numerical_columns,
            cardinalities,
            config,
            seed=seed,
        )
        self.backbone = Backbone(
            self.encoder.fields,
            self.encoder.dimension,
            architecture.backbone,
        )
        self.heads = nn.ModuleList(
            [
                PredictionHead(
                    self.backbone.output_dimension,
                    head.hidden,
                    head.dropout,
                )
                for head in architecture.heads
            ]
        )
        self.aggregation = architecture.aggregation
        self.gate = (
            LogitGate(len(self.heads))
            if len(self.heads) > 1 and architecture.aggregation is Aggregation.GATED
            else None
        )
        self.feature_masks: torch.Tensor
        self.register_buffer(
            "feature_masks",
            self._make_masks(
                len(self.heads),
                self.encoder.fields,
                architecture.feature_bagging,
                seed,
            ),
        )

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
            logits.append(head(self.backbone(masked)))
        auxiliary = torch.stack(logits, dim=1)
        diagnostics: dict[str, torch.Tensor] = {}
        if self.gate is not None:
            aggregate, weights = self.gate(auxiliary.squeeze(-1))
            diagnostics["aggregation_weights"] = weights
        elif len(self.heads) == 1:
            aggregate = auxiliary[:, 0]
        else:
            aggregate = auxiliary.mean(dim=1)
        return ModelOutput(
            aggregate_logits=aggregate,
            auxiliary_logits=auxiliary,
            diagnostics=diagnostics,
        )


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
