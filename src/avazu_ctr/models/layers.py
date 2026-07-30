"""Correct, bounded model building blocks."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import cast

import torch
from torch import nn
from torch.nn import functional

from avazu_ctr.config.schema import (
    BackboneConfig,
    DCNv2CrossConfig,
    DeltaRoutedDCNv2CrossConfig,
    EmbeddingKind,
    ModelConfig,
)
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

    def __init__(
        self,
        count: int,
        dimension: int,
        *,
        normalized: bool,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(count, dimension))
        self.bias = nn.Parameter(torch.empty(count, dimension)) if bias else None
        self.normalized = normalized
        nn.init.normal_(self.weight, std=1.0 / math.sqrt(dimension))
        if self.bias is not None:
            nn.init.normal_(self.bias, std=1.0 / math.sqrt(dimension))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        projected = values.unsqueeze(-1) * self.weight.unsqueeze(0)
        if self.bias is not None:
            projected = projected + self.bias.unsqueeze(0)
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
        numerical_bias: bool = True,
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
            len(numerical_columns),
            self.dimension,
            normalized=normalized,
            bias=numerical_bias,
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
        self.dimension = dimension
        self.layers = layers
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

    def _delta(
        self,
        inputs: torch.Tensor,
        context: torch.Tensor,
        index: int,
    ) -> torch.Tensor:
        if self.rank is None:
            crossed = context @ self.weights[index]
        else:
            crossed = (context @ self.left[index]) @ self.right[index]
        return inputs * (crossed + self.biases[index])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        current = inputs
        for index in range(self.layers):
            current = current + self._delta(inputs, current, index)
        return current


class DeltaRoutedDCNv2(DCNv2):
    """Redistribute prior cross deltas without changing their total residual mass."""

    _ROUTING_EPSILON = 1e-6

    def __init__(self, dimension: int, layers: int, rank: int | None) -> None:
        if layers < 3:
            raise ValueError("delta-routed DCNv2 requires at least three layers")
        super().__init__(dimension, layers, rank)
        self.routing_queries = nn.Parameter(torch.zeros(layers - 2, dimension))
        self.routing_scale = 1.0 / math.sqrt(dimension)

    def _routing_context(
        self,
        current: torch.Tensor,
        deltas: list[torch.Tensor],
        query: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sources = torch.stack(deltas, dim=1)
        stable_sources = sources.to(torch.float32)
        normalized = functional.rms_norm(
            stable_sources,
            (self.dimension,),
            eps=self._ROUTING_EPSILON,
        )
        scores = torch.einsum("bsd,d->bs", normalized, query.to(torch.float32))
        weights = torch.softmax(scores * self.routing_scale, dim=1)

        # Standard DCNv2 assigns coefficient one to every prior delta. Centering
        # n * softmax(scores) around one preserves that total coefficient mass
        # and makes a zero query exactly equivalent to the original recurrence.
        coefficients = weights * len(deltas) - 1.0
        correction = torch.bmm(coefficients.unsqueeze(1), stable_sources).squeeze(1)
        return current + correction.to(current.dtype), weights

    def _forward(
        self,
        inputs: torch.Tensor,
        *,
        collect_routing: bool,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        current = inputs
        deltas: list[torch.Tensor] = []
        routing: list[torch.Tensor] = []
        for index in range(self.layers):
            context = current
            if len(deltas) >= 2:
                context, weights = self._routing_context(
                    current,
                    deltas,
                    self.routing_queries[index - 2],
                )
                if collect_routing:
                    routing.append(weights)
            delta = self._delta(inputs, context, index)
            current = current + delta
            deltas.append(delta)
        return current, tuple(routing)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output, _ = self._forward(inputs, collect_routing=False)
        return output

    def forward_with_routing(
        self,
        inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Return per-example routing weights for offline diagnostics."""

        return self._forward(inputs, collect_routing=True)


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
        cross = config.cross
        if isinstance(cross, DeltaRoutedDCNv2CrossConfig):
            self.cross: nn.Module = DeltaRoutedDCNv2(
                flattened,
                cross.layers,
                cross.rank,
            )
        elif isinstance(cross, DCNv2CrossConfig):
            self.cross = DCNv2(
                flattened,
                cross.layers,
                cross.rank,
            )
        else:
            raise TypeError(f"unsupported cross-network config: {type(cross).__name__}")
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
