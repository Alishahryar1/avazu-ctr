"""nGPT field transformer adapted from hyperspherical language modeling to CTR."""

from __future__ import annotations

import math
from typing import cast

import torch
from torch import nn
from torch.nn import functional

from avazu_ctr.config.schema import ModelConfig, NGPTModelConfig
from avazu_ctr.contracts import FeatureBatch, ModelOutput
from avazu_ctr.models.base import CTRModel
from avazu_ctr.models.layers import FeatureEncoder, StableHashEmbedding


def unit_norm(values: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Normalize in float32 and restore the input dtype."""

    dtype = values.dtype
    normalized = functional.normalize(values.float(), dim=dim)
    return normalized.to(dtype=dtype)


@torch.no_grad()
def normalize_parameter_(parameter: torch.Tensor, dim: int) -> None:
    parameter.copy_(unit_norm(parameter, dim=dim))


class NGPTAttention(nn.Module):
    """Normalized self-attention with learned q/k scales and eigen update rates."""

    def __init__(self, config: NGPTModelConfig) -> None:
        super().__init__()
        self.dimension = config.dimension
        self.heads = config.heads
        self.head_dimension = config.dimension // config.heads
        self.dropout = config.dropout
        self.base_scale = 1.0 / math.sqrt(config.dimension)
        self.alpha_init = config.alpha_init
        self.query = nn.Linear(config.dimension, config.dimension, bias=False)
        self.key = nn.Linear(config.dimension, config.dimension, bias=False)
        self.value = nn.Linear(config.dimension, config.dimension, bias=False)
        self.output = nn.Linear(config.dimension, config.dimension, bias=False)
        self.qk_scale = nn.Parameter(torch.full((config.dimension,), self.base_scale))
        self.alpha = nn.Parameter(torch.full((config.dimension,), self.base_scale))

    def _heads(self, values: torch.Tensor) -> torch.Tensor:
        batch, fields, _ = values.shape
        return values.view(batch, fields, self.heads, self.head_dimension).transpose(1, 2)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        batch, fields, _ = hidden.shape
        scale = (self.qk_scale / self.base_scale).view(
            1,
            self.heads,
            1,
            self.head_dimension,
        )
        query = unit_norm(self._heads(self.query(hidden)))
        key = unit_norm(self._heads(self.key(hidden)))
        scale = scale.to(dtype=query.dtype)
        query = query * scale
        key = key * scale
        value = self._heads(self.value(hidden))
        attended = functional.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
            scale=math.sqrt(self.head_dimension),
        )
        attended = attended.transpose(1, 2).contiguous().view(batch, fields, self.dimension)
        proposal = unit_norm(self.output(attended))
        alpha = (self.alpha.abs() * (self.alpha_init / self.base_scale)).view(1, 1, -1)
        return unit_norm(hidden + alpha * (proposal - hidden))

    @torch.no_grad()
    def normalize_parameters(self) -> None:
        for projection in (self.query, self.key, self.value):
            normalize_parameter_(projection.weight, dim=1)
        normalize_parameter_(self.output.weight, dim=0)


class NGPTFeedForward(nn.Module):
    """Normalized SwiGLU MLP with paper-specific intermediate scaling."""

    def __init__(self, config: NGPTModelConfig) -> None:
        super().__init__()
        hidden = config.dimension * config.mlp_multiplier
        self.dimension = config.dimension
        self.base_scale = 1.0 / math.sqrt(config.dimension)
        self.alpha_init = config.alpha_init
        self.dropout = nn.Dropout(config.dropout)
        self.up = nn.Linear(config.dimension, hidden, bias=False)
        self.gate = nn.Linear(config.dimension, hidden, bias=False)
        self.down = nn.Linear(hidden, config.dimension, bias=False)
        self.up_scale = nn.Parameter(torch.ones(hidden))
        self.gate_scale = nn.Parameter(torch.ones(hidden))
        self.alpha = nn.Parameter(torch.full((config.dimension,), self.base_scale))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        up = self.up(hidden) * self.up_scale
        gate = self.gate(hidden) * self.gate_scale * math.sqrt(self.dimension)
        proposal = self.down(up * functional.silu(gate))
        proposal = unit_norm(self.dropout(proposal))
        alpha = (self.alpha.abs() * (self.alpha_init / self.base_scale)).view(1, 1, -1)
        return unit_norm(hidden + alpha * (proposal - hidden))

    @torch.no_grad()
    def normalize_parameters(self) -> None:
        normalize_parameter_(self.up.weight, dim=1)
        normalize_parameter_(self.gate.weight, dim=1)
        normalize_parameter_(self.down.weight, dim=0)


class NGPTBlock(nn.Module):
    def __init__(self, config: NGPTModelConfig) -> None:
        super().__init__()
        self.attention = NGPTAttention(config)
        self.feed_forward = NGPTFeedForward(config)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.feed_forward(self.attention(hidden))

    @torch.no_grad()
    def normalize_parameters(self) -> None:
        self.attention.normalize_parameters()
        self.feed_forward.normalize_parameters()


class NGPTModel(CTRModel):
    """Use field embeddings as a non-causal nGPT sequence and classify a unit CLS token."""

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
        if config.ngpt is None:
            raise ValueError("nGPT model requires ngpt configuration")
        architecture = config.ngpt
        self.base_scale = 1.0 / math.sqrt(architecture.dimension)
        self.encoder = FeatureEncoder(
            categorical_columns,
            numerical_columns,
            cardinalities,
            config,
            seed=seed,
            normalized=True,
        )
        self.input_projection: nn.Module = (
            nn.Identity()
            if self.encoder.dimension == architecture.dimension
            else nn.Linear(self.encoder.dimension, architecture.dimension, bias=False)
        )
        generator = torch.Generator().manual_seed(seed)
        self.cls_token = nn.Parameter(
            torch.randn((1, 1, architecture.dimension), generator=generator)
        )
        self.blocks = nn.ModuleList([NGPTBlock(architecture) for _ in range(architecture.layers)])
        self.output = nn.Linear(architecture.dimension, 2, bias=False)
        self.logit_scale = nn.Parameter(torch.full((2,), self.base_scale))
        self._initialize_matrices()
        self.post_step()

    def _initialize_matrices(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=self.base_scale)

    def forward(self, batch: FeatureBatch) -> ModelOutput:
        encoded = self.input_projection(self.encoder(batch))
        encoded = unit_norm(encoded)
        cls = unit_norm(self.cls_token).expand(encoded.shape[0], -1, -1)
        hidden = torch.cat((cls, encoded), dim=1)
        for block in self.blocks:
            hidden = block(hidden)
        class_logits = self.output(hidden[:, 0]) * (self.logit_scale / self.base_scale)
        logits = class_logits[:, 1:2] - class_logits[:, 0:1]
        return ModelOutput(aggregate_logits=logits)

    @torch.no_grad()
    def post_step(self) -> None:
        for module in self.encoder.embeddings.values():
            if isinstance(module, StableHashEmbedding):
                for table in module.tables:
                    embedding = cast(nn.Embedding, table)
                    normalize_parameter_(embedding.weight, dim=1)
            elif isinstance(module, nn.Embedding):
                normalize_parameter_(module.weight, dim=1)
        for projection in self.encoder.projections.values():
            normalize_parameter_(cast(nn.Linear, projection).weight, dim=1)
        normalize_parameter_(self.encoder.numerical.weight, dim=1)
        if self.encoder.numerical.bias is not None:
            normalize_parameter_(self.encoder.numerical.bias, dim=1)
        if isinstance(self.input_projection, nn.Linear):
            normalize_parameter_(self.input_projection.weight, dim=1)
        normalize_parameter_(self.cls_token, dim=-1)
        for block in self.blocks:
            cast(NGPTBlock, block).normalize_parameters()
        normalize_parameter_(self.output.weight, dim=1)
