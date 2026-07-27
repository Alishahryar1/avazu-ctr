"""Validated model construction and artifact-size accounting."""

from __future__ import annotations

from torch import nn

from avazu_ctr.config.schema import Aggregation, ModelConfig, ModelKind
from avazu_ctr.data.manifest import DatasetManifest
from avazu_ctr.models.architectures import (
    EnsembleModel,
    GatedDCNModel,
    MultiHeadModel,
    STECModel,
)
from avazu_ctr.models.base import CTRModel


def create_model(
    config: ModelConfig,
    manifest: DatasetManifest,
    *,
    seed: int,
) -> CTRModel:
    common = (
        manifest.categorical_columns,
        manifest.numerical_columns,
        manifest.cardinalities,
        config,
    )
    if config.kind is ModelKind.GATED_DCN:
        return GatedDCNModel(*common, seed=seed)
    if config.kind is ModelKind.MULTIHEAD:
        return MultiHeadModel(*common, seed=seed)
    if config.kind is ModelKind.NORMALIZED_MULTIHEAD:
        return MultiHeadModel(*common, seed=seed, normalized=True)
    if config.kind is ModelKind.STEC:
        return STECModel(*common, seed=seed)
    if config.kind is ModelKind.ENSEMBLE:
        children = [
            create_model(child, manifest, seed=seed + index + 1)
            for index, child in enumerate(config.children)
        ]
        return EnsembleModel(children, config.aggregation)
    raise ValueError(f"unsupported model kind: {config.kind}")


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def serialized_weight_bytes(model: nn.Module) -> int:
    return sum(value.numel() * value.element_size() for value in model.state_dict().values())


def enforce_weight_budget(model: nn.Module, maximum_bytes: int) -> None:
    size = serialized_weight_bytes(model)
    if size > maximum_bytes:
        raise ValueError(
            f"model weights require {size / 1024**2:.1f} MiB, "
            f"exceeding the {maximum_bytes / 1024**2:.1f} MiB budget"
        )


def _linear_values(input_dimension: int, output_dimension: int) -> int:
    return input_dimension * output_dimension + output_dimension


def _encoder_bytes(config: ModelConfig, manifest: DatasetManifest) -> tuple[int, int, int]:
    dimension = config.default_embedding.dim
    fields = len(manifest.categorical_columns) + len(manifest.numerical_columns)
    total = 0
    for feature in manifest.categorical_columns:
        embedding = config.feature_embeddings.get(feature, config.default_embedding)
        if embedding.kind.value == "hash":
            total += embedding.hashes * embedding.buckets * embedding.dim * 4
            total += embedding.hashes * 4
            total += embedding.hashes * 5 * 8
        else:
            total += manifest.cardinalities[feature] * embedding.dim * 4
        if embedding.dim != dimension:
            total += embedding.dim * dimension * 4
    total += len(manifest.numerical_columns) * dimension * 2 * 4
    return total, fields, dimension


def _mlp_values(
    input_dimension: int,
    hidden: tuple[int, ...],
    *,
    layer_norm: bool,
) -> tuple[int, int]:
    values = 0
    current = input_dimension
    for output in hidden:
        values += _linear_values(current, output)
        if layer_norm:
            values += 2 * output
        current = output
    return values, current


def _backbone_values(config: ModelConfig, fields: int, dimension: int) -> tuple[int, int]:
    backbone = config.backbone
    flattened = fields * dimension
    values = 0
    if backbone.senet.enabled:
        hidden = max(1, fields // backbone.senet.reduction_ratio)
        values += _linear_values(fields, hidden) + _linear_values(hidden, fields)
    if backbone.dcn_rank is None:
        values += backbone.dcn_layers * (flattened * flattened + flattened)
    else:
        values += backbone.dcn_layers * (2 * flattened * backbone.dcn_rank + flattened)
    mlp_values, mlp_output = _mlp_values(
        flattened,
        backbone.mlp_hidden,
        layer_norm=backbone.layer_norm,
    )
    values += mlp_values
    return values, flattened + mlp_output


def _head_values(input_dimension: int, hidden: tuple[int, ...]) -> int:
    values, output = _mlp_values(input_dimension, hidden, layer_norm=True)
    return values + _linear_values(output, 1)


def estimate_model_weight_bytes(
    config: ModelConfig,
    manifest: DatasetManifest,
) -> int:
    """Conservative pre-allocation estimate for model parameters and buffers."""

    if config.kind is ModelKind.ENSEMBLE:
        total = sum(estimate_model_weight_bytes(child, manifest) for child in config.children)
        if config.aggregation is Aggregation.GATED:
            heads = len(config.children)
            hidden = max(4, heads * 2)
            total += (_linear_values(heads, hidden) + _linear_values(hidden, heads)) * 4
        return total

    encoder_bytes, fields, dimension = _encoder_bytes(config, manifest)
    if config.kind is ModelKind.STEC:
        feedforward = max(dimension * 4, 32)
        layer_values = (
            4 * dimension * dimension + 2 * dimension * feedforward + 9 * dimension + feedforward
        )
        interaction_values = (config.stec_layers + 1) * (fields * fields * dimension + dimension)
        head_values = _head_values(
            dimension * (config.stec_layers + 1),
            config.heads[0].hidden,
        )
        return (
            encoder_bytes
            + (config.stec_layers * layer_values + interaction_values + head_values) * 4
        )

    backbone_values, output_dimension = _backbone_values(config, fields, dimension)
    if config.kind is ModelKind.GATED_DCN:
        return encoder_bytes + (backbone_values + _linear_values(output_dimension, 1)) * 4

    head_values = sum(_head_values(output_dimension, head.hidden) for head in config.heads)
    gate_values = 0
    if config.aggregation is Aggregation.GATED:
        heads = len(config.heads)
        hidden = max(4, heads * 2)
        gate_values = _linear_values(heads, hidden) + _linear_values(hidden, heads)
    mask_bytes = len(config.heads) * fields * 4
    return encoder_bytes + (backbone_values + head_values + gate_values) * 4 + mask_bytes


def validate_weight_budget(
    config: ModelConfig,
    manifest: DatasetManifest,
    maximum_bytes: int,
) -> None:
    estimate = estimate_model_weight_bytes(config, manifest)
    if estimate > maximum_bytes:
        raise ValueError(
            f"estimated model weights require {estimate / 1024**2:.1f} MiB, "
            f"exceeding the {maximum_bytes / 1024**2:.1f} MiB budget"
        )
