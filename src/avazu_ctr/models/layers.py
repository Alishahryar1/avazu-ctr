"""Correct, bounded model building blocks."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import cast

import torch
from torch import nn
from torch.nn import functional

from avazu_ctr.config.schema import BackboneConfig, EmbeddingKind, ModelConfig
from avazu_ctr.contracts import FeatureBatch


def activation(name: str) -> nn.Module:
    choices: dict[str, type[nn.Module]] = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
    }
    return choices[name]()


class StableHashEmbedding(nn.Module):
    """Routes full-width feature hashes through independent bounded tables."""

    _PRIME = 2_147_483_647
    _CHUNK_BITS = 21
    _CHUNK_MASK = (1 << _CHUNK_BITS) - 1

    def __init__(
        self,
        buckets: int,
        dimension: int,
        hashes: int,
        *,
        seed: int,
    ) -> None:
        super().__init__()
        if buckets <= 1 or dimension <= 0 or hashes <= 0:
            raise ValueError("invalid hash embedding dimensions")
        generator = torch.Generator().manual_seed(seed)
        coefficients = torch.randint(1, self._PRIME, (hashes, 4), generator=generator)
        offsets = torch.randint(0, self._PRIME, (hashes,), generator=generator)
        self.coefficients: torch.Tensor
        self.offsets: torch.Tensor
        self.register_buffer("coefficients", coefficients.to(torch.int64))
        self.register_buffer("offsets", offsets.to(torch.int64))
        self.tables = nn.ModuleList([nn.Embedding(buckets, dimension) for _ in range(hashes)])
        self.mixing_logits = nn.Parameter(torch.zeros(hashes))
        self.buckets = buckets
        self.dimension = dimension
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.tables:
            table = cast(nn.Embedding, module)
            nn.init.normal_(table.weight, mean=0.0, std=1.0 / math.sqrt(self.dimension))
        nn.init.zeros_(self.mixing_logits)

    def hash_locations(self, values: torch.Tensor) -> torch.Tensor:
        """Return one independently mixed bucket location per hash table."""

        values = values.to(torch.int64)
        chunks = torch.stack(
            (
                values & self._CHUNK_MASK,
                (values >> self._CHUNK_BITS) & self._CHUNK_MASK,
                (values >> (2 * self._CHUNK_BITS)) & self._CHUNK_MASK,
                (values >> 63) & 1,
            ),
            dim=-1,
        )
        mixed = (chunks.unsqueeze(1) * self.coefficients.unsqueeze(0)).sum(
            dim=-1
        ) + self.offsets.unsqueeze(0)
        return mixed.remainder(self._PRIME).remainder(self.buckets)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        locations = self.hash_locations(values)
        outputs = []
        for index, module in enumerate(self.tables):
            table = cast(nn.Embedding, module)
            outputs.append(table(locations[:, index]))
        stacked = torch.stack(outputs, dim=1)
        weights = torch.softmax(self.mixing_logits, dim=0).view(1, -1, 1)
        return (stacked * weights).sum(dim=1)


class NumericalProjection(nn.Module):
    """Projects each scalar into a vector before optional normalization."""

    def __init__(self, count: int, dimension: int, *, normalized: bool) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(count, dimension))
        self.bias = nn.Parameter(torch.empty(count, dimension))
        self.normalized = normalized
        nn.init.normal_(self.weight, std=1.0 / math.sqrt(dimension))
        nn.init.normal_(self.bias, std=1.0 / math.sqrt(dimension))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        projected = values.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
        if self.normalized:
            projected = functional.normalize(projected, dim=-1)
        return projected


class FeatureEncoder(nn.Module):
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
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns
        self.dimension = config.default_embedding.dim
        self.embeddings = nn.ModuleDict()
        self.projections = nn.ModuleDict()
        for index, feature in enumerate(categorical_columns):
            embedding = config.feature_embeddings.get(feature, config.default_embedding)
            if embedding.kind is EmbeddingKind.HASH:
                module: nn.Module = StableHashEmbedding(
                    embedding.buckets,
                    embedding.dim,
                    embedding.hashes,
                    seed=seed + index * 101,
                )
            else:
                cardinality = cardinalities.get(feature)
                if cardinality is None:
                    raise ValueError(f"manifest has no cardinality for {feature}")
                module = nn.Embedding(cardinality, embedding.dim, padding_idx=0)
            self.embeddings[feature] = module
            if embedding.dim != self.dimension:
                self.projections[feature] = nn.Linear(embedding.dim, self.dimension, bias=False)
        self.numerical = NumericalProjection(
            len(numerical_columns), self.dimension, normalized=normalized
        )
        self.normalized = normalized

    @property
    def fields(self) -> int:
        return len(self.categorical_columns) + len(self.numerical_columns)

    def forward(self, batch: FeatureBatch) -> torch.Tensor:
        categorical: list[torch.Tensor] = []
        for index, feature in enumerate(self.categorical_columns):
            value = self.embeddings[feature](batch.categorical[:, index])
            if feature in self.projections:
                value = self.projections[feature](value)
            if self.normalized:
                value = functional.normalize(value, dim=-1)
            categorical.append(value)
        numerical = self.numerical(batch.numerical)
        if not categorical and not numerical.shape[1]:
            raise ValueError("model requires at least one feature")
        categorical_tensor = (
            torch.stack(categorical, dim=1)
            if categorical
            else numerical.new_empty((numerical.shape[0], 0, self.dimension))
        )
        return torch.cat((categorical_tensor, numerical), dim=1)


class SENet(nn.Module):
    def __init__(self, fields: int, reduction_ratio: int, activation_name: str) -> None:
        super().__init__()
        hidden = max(1, fields // reduction_ratio)
        self.excitation = nn.Sequential(
            nn.Linear(fields, hidden),
            activation(activation_name),
            nn.Linear(hidden, fields),
            nn.Sigmoid(),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        weights = self.excitation(embeddings.mean(dim=-1)).unsqueeze(-1)
        return embeddings * weights


class DCNv2(nn.Module):
    def __init__(self, dimension: int, layers: int, rank: int | None) -> None:
        super().__init__()
        self.rank = rank
        if rank is None:
            self.weights = nn.ParameterList(
                [nn.Parameter(torch.empty(dimension, dimension)) for _ in range(layers)]
            )
            self.left = self.right = nn.ParameterList()
        else:
            self.left = nn.ParameterList(
                [nn.Parameter(torch.empty(dimension, rank)) for _ in range(layers)]
            )
            self.right = nn.ParameterList(
                [nn.Parameter(torch.empty(rank, dimension)) for _ in range(layers)]
            )
            self.weights = nn.ParameterList()
        self.biases = nn.ParameterList(
            [nn.Parameter(torch.zeros(dimension)) for _ in range(layers)]
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for weight in self.weights:
            nn.init.xavier_uniform_(weight)
        for left, right in zip(self.left, self.right, strict=True):
            nn.init.xavier_uniform_(left)
            nn.init.xavier_uniform_(right)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        current = inputs
        for index, bias in enumerate(self.biases):
            if self.rank is None:
                crossed = current @ self.weights[index]
            else:
                crossed = (current @ self.left[index]) @ self.right[index]
            current = inputs * (crossed + bias) + current
        return current


class MLP(nn.Module):
    def __init__(
        self,
        input_dimension: int,
        hidden: Iterable[int],
        *,
        activation_name: str,
        dropout: float,
        layer_norm: bool,
    ) -> None:
        super().__init__()
        modules: list[nn.Module] = []
        current = input_dimension
        for output in hidden:
            modules.append(nn.Linear(current, output))
            if layer_norm:
                modules.append(nn.LayerNorm(output))
            modules.extend((activation(activation_name), nn.Dropout(dropout)))
            current = output
        self.network = nn.Sequential(*modules)
        self.output_dimension = current

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class Backbone(nn.Module):
    def __init__(self, fields: int, dimension: int, config: BackboneConfig) -> None:
        super().__init__()
        flattened = fields * dimension
        self.senet = (
            SENet(
                fields,
                config.senet.reduction_ratio,
                config.senet.activation,
            )
            if config.senet.enabled
            else nn.Identity()
        )
        self.cross = DCNv2(flattened, config.dcn_layers, config.dcn_rank)
        self.mlp = MLP(
            flattened,
            config.mlp_hidden,
            activation_name=config.activation,
            dropout=config.dropout,
            layer_norm=config.layer_norm,
        )
        self.output_dimension = flattened + self.mlp.output_dimension

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        embeddings = self.senet(embeddings)
        flattened = embeddings.flatten(start_dim=1)
        crossed = self.cross(flattened)
        deep = self.mlp(flattened)
        return torch.cat((crossed, deep), dim=1)


class PredictionHead(nn.Module):
    def __init__(self, input_dimension: int, hidden: tuple[int, ...], dropout: float) -> None:
        super().__init__()
        self.mlp = MLP(
            input_dimension,
            hidden,
            activation_name="silu",
            dropout=dropout,
            layer_norm=True,
        )
        self.output = nn.Linear(self.mlp.output_dimension, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.output(self.mlp(inputs))


class LogitGate(nn.Module):
    def __init__(self, heads: int) -> None:
        super().__init__()
        hidden = max(4, heads * 2)
        self.network = nn.Sequential(
            nn.Linear(heads, hidden),
            nn.SiLU(),
            nn.Linear(hidden, heads),
        )

    def forward(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = torch.softmax(self.network(logits), dim=1)
        return (weights * logits).sum(dim=1, keepdim=True), weights


class NormalizedLinear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return functional.linear(input, functional.normalize(self.weight, dim=1), self.bias)

    @torch.no_grad()
    def normalize_(self) -> None:
        self.weight.copy_(functional.normalize(self.weight, dim=1))
