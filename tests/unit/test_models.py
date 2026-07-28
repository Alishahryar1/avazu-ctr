from __future__ import annotations

import math
from pathlib import Path
from typing import cast

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from avazu_ctr.config.schema import (
    EnsembleModelConfig,
    ExperimentConfig,
    ModelConfig,
    ModelKind,
    NGPTModelConfig,
    STECModelConfig,
)
from avazu_ctr.contracts import FeatureBatch, ModelOutput
from avazu_ctr.data.dataset import ParquetBatchDataset
from avazu_ctr.data.manifest import load_manifest
from avazu_ctr.models import create_model
from avazu_ctr.models.architectures import DCNModel
from avazu_ctr.models.factory import (
    enforce_weight_budget,
    estimate_model_weight_bytes,
    serialized_weight_bytes,
)
from avazu_ctr.models.layers import LogitGate, NumericalProjection, StableHashEmbedding
from avazu_ctr.models.ngpt import NGPTBlock, NGPTModel
from avazu_ctr.models.state import state_dict_sha256
from avazu_ctr.models.stec import STECBlock, STECModel
from avazu_ctr.objectives import CTRObjective


def _batch(manifest_path: Path) -> FeatureBatch:
    loader = DataLoader(ParquetBatchDataset(manifest_path, "train", 16), batch_size=None)
    return next(iter(loader))


def _model_config(config: ExperimentConfig, kind: ModelKind) -> ModelConfig:
    if kind == ModelKind.DCN:
        return config.model
    if kind == ModelKind.STEC:
        return config.model.model_copy(
            update={
                "kind": kind,
                "dcn": None,
                "stec": STECModelConfig(
                    dimension=8,
                    layers=1,
                    heads=2,
                    ffn_multiplier=2,
                    dropout=0.0,
                    prediction_hidden=(),
                    prediction_dropout=0.0,
                ),
            }
        )
    if kind == ModelKind.NGPT:
        return config.model.model_copy(
            update={
                "kind": kind,
                "dcn": None,
                "ngpt": NGPTModelConfig(
                    dimension=8,
                    layers=1,
                    heads=2,
                    mlp_multiplier=2,
                    dropout=0.0,
                ),
            }
        )
    raise ValueError(kind)


def _assert_unit_rows(values: torch.Tensor, *, skip_first: bool = False) -> None:
    rows = values[1:] if skip_first else values
    expected = torch.ones(rows.shape[0], dtype=rows.dtype, device=rows.device)
    assert torch.allclose(torch.linalg.vector_norm(rows, dim=1), expected, atol=1e-5)


def test_tensor_contracts_reject_invalid_shapes_and_dtypes() -> None:
    categorical = torch.zeros((2, 1), dtype=torch.int64)
    numerical = torch.zeros((2, 1), dtype=torch.float32)
    with pytest.raises(TypeError, match="timestamps"):
        FeatureBatch(categorical, numerical, timestamps=torch.zeros(2))
    with pytest.raises(ValueError, match="aggregate logits"):
        ModelOutput(torch.zeros(2))


@pytest.mark.parametrize("kind", [ModelKind.DCN, ModelKind.STEC, ModelKind.NGPT])
def test_every_architecture_has_full_gradient_coverage(
    processed_project: tuple[ExperimentConfig, Path],
    kind: ModelKind,
) -> None:
    config, manifest_path = processed_project
    manifest = load_manifest(manifest_path)
    model = create_model(_model_config(config, kind), manifest, seed=42)
    batch = _batch(manifest_path)
    output = model(batch)
    loss = CTRObjective(config.objective)(output, cast(torch.Tensor, batch.labels)).total
    loss.backward()
    missing = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    assert output.aggregate_logits.shape == (16, 1)
    assert not missing


def test_ensemble_recurses_and_updates_every_child(
    processed_project: tuple[ExperimentConfig, Path],
) -> None:
    config, manifest_path = processed_project
    manifest = load_manifest(manifest_path)
    children = (
        _model_config(config, ModelKind.DCN),
        _model_config(config, ModelKind.STEC),
    )
    ensemble = config.model.model_copy(
        update={
            "kind": ModelKind.ENSEMBLE,
            "dcn": None,
            "ensemble": EnsembleModelConfig(),
            "children": children,
        }
    )
    model = create_model(ensemble, manifest, seed=42)
    batch = _batch(manifest_path)
    output = model(batch)
    assert len(output.children) == 2
    CTRObjective(config.objective)(
        output,
        cast(torch.Tensor, batch.labels),
    ).total.backward()
    assert all(
        parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad
    )


def test_aggregate_bce_trains_learned_gate(
    processed_project: tuple[ExperimentConfig, Path],
) -> None:
    config, manifest_path = processed_project
    manifest = load_manifest(manifest_path)
    model = create_model(config.model, manifest, seed=42)
    batch = _batch(manifest_path)
    objective = CTRObjective(
        config.objective.model_copy(update={"auxiliary_weight": 0.0, "diversity_weight": 0.0})
    )
    objective(model(batch), cast(torch.Tensor, batch.labels)).total.backward()
    assert isinstance(model, DCNModel)
    assert isinstance(model.gate, LogitGate)
    assert all(parameter.grad is not None for parameter in model.gate.parameters())


def test_one_head_dcn_is_its_only_prediction_head(
    processed_project: tuple[ExperimentConfig, Path],
) -> None:
    config, manifest_path = processed_project
    manifest = load_manifest(manifest_path)
    architecture = config.model.dcn
    if architecture is None:
        raise AssertionError("test fixture must be a DCN")
    one_head = architecture.model_copy(
        update={
            "heads": architecture.heads[:1],
            "aggregation": "mean",
            "feature_bagging": 1.0,
        }
    )
    model_config = config.model.model_copy(update={"dcn": one_head})
    model = create_model(model_config, manifest, seed=42)
    output = model(_batch(manifest_path))
    assert isinstance(model, DCNModel)
    assert model.gate is None
    assert torch.equal(model.feature_masks, torch.ones_like(model.feature_masks))
    assert output.auxiliary_logits is not None
    assert torch.equal(output.aggregate_logits, output.auxiliary_logits[:, 0])


def test_stec_exposes_the_unpooled_interaction_used_by_attention() -> None:
    block = STECBlock(dimension=8, heads=2, dropout=0.0)
    values = torch.randn(3, 4, 8)
    query, key, _, bilinear = block.projected_interactions(values)
    expected = key.unsqueeze(3) * query.unsqueeze(2)
    assert torch.equal(bilinear, expected)
    assert bilinear.shape == (3, 2, 4, 4, 4)
    pooled = bilinear.mean(dim=-1) * math.sqrt(block.head_dimension)
    matrix_product = (key @ query.transpose(-2, -1)) / math.sqrt(block.head_dimension)
    assert torch.allclose(pooled, matrix_product, atol=1e-6)
    _, flattened = block(values)
    assert flattened.shape == (3, 4 * 4 * 8)


def test_stec_collects_one_interaction_per_layer_plus_the_final_state(
    processed_project: tuple[ExperimentConfig, Path],
) -> None:
    config, manifest_path = processed_project
    manifest = load_manifest(manifest_path)
    model = create_model(_model_config(config, ModelKind.STEC), manifest, seed=42)
    assert isinstance(model, STECModel)
    assert len(model.interaction_norms) == len(model.layers) + 1
    interaction_width = model.encoder.fields**2 * 8
    assert all(
        normalization.num_features == interaction_width for normalization in model.interaction_norms
    )


def test_ngpt_has_no_affine_normalization_layers_and_keeps_hidden_states_unit_norm(
    processed_project: tuple[ExperimentConfig, Path],
) -> None:
    config, manifest_path = processed_project
    manifest = load_manifest(manifest_path)
    model = create_model(_model_config(config, ModelKind.NGPT), manifest, seed=42)
    assert isinstance(model, NGPTModel)
    assert not any(isinstance(module, nn.LayerNorm | nn.RMSNorm) for module in model.modules())
    batch = _batch(manifest_path)
    encoded = model.input_projection(model.encoder(batch))
    hidden = torch.cat(
        (
            model.cls_token.expand(encoded.shape[0], -1, -1),
            encoded,
        ),
        dim=1,
    )
    hidden = torch.nn.functional.normalize(hidden, dim=-1)
    for module in model.blocks:
        hidden = cast(NGPTBlock, module)(hidden)
        norms = torch.linalg.vector_norm(hidden, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_ngpt_post_step_normalizes_every_embedding_dimension(
    processed_project: tuple[ExperimentConfig, Path],
) -> None:
    config, manifest_path = processed_project
    manifest = load_manifest(manifest_path)
    model = create_model(_model_config(config, ModelKind.NGPT), manifest, seed=42)
    assert isinstance(model, NGPTModel)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.ndim >= 2:
                parameter.add_(torch.randn_like(parameter))
    model.post_step()

    for module in model.encoder.embeddings.values():
        if isinstance(module, StableHashEmbedding):
            for table in module.tables:
                _assert_unit_rows(cast(nn.Embedding, table).weight)
        else:
            embedding = cast(nn.Embedding, module)
            _assert_unit_rows(embedding.weight, skip_first=embedding.padding_idx == 0)
            if embedding.padding_idx == 0:
                assert torch.count_nonzero(embedding.weight[0]) == 0
    _assert_unit_rows(model.encoder.numerical.weight)
    _assert_unit_rows(model.encoder.numerical.bias)
    assert torch.linalg.vector_norm(model.cls_token).item() == pytest.approx(1.0)

    for module in model.blocks:
        block = cast(NGPTBlock, module)
        for projection in (
            block.attention.query,
            block.attention.key,
            block.attention.value,
            block.feed_forward.up,
            block.feed_forward.gate,
        ):
            _assert_unit_rows(projection.weight)
        for projection in (block.attention.output, block.feed_forward.down):
            expected = torch.ones(projection.weight.shape[1])
            assert torch.allclose(
                torch.linalg.vector_norm(projection.weight, dim=0),
                expected,
                atol=1e-5,
            )
    _assert_unit_rows(model.output.weight)


def test_normalized_numerical_projection_preserves_magnitude_information() -> None:
    projection = NumericalProjection(1, 8, normalized=True)
    values = torch.tensor([[1.0], [2.0]], dtype=torch.float32)
    output = projection(values)
    assert not torch.allclose(output[0], output[1])
    assert torch.linalg.vector_norm(output, dim=-1).squeeze().tolist() == pytest.approx([1.0, 1.0])


def test_hash_state_round_trip_is_exact() -> None:
    first = StableHashEmbedding(31, 8, 3, seed=7)
    second = StableHashEmbedding(31, 8, 3, seed=99)
    second.load_state_dict(first.state_dict(), strict=True)
    values = torch.tensor(
        [0, 1, 17, 999_999, -(2**63), 2**63 - 1],
        dtype=torch.int64,
    )
    assert torch.equal(first(values), second(values))
    assert "coefficients" in first.state_dict()
    assert "offsets" in first.state_dict()


def test_multi_hash_routing_can_separate_primary_bucket_collisions() -> None:
    embedding = StableHashEmbedding(31, 8, 3, seed=7)
    values = torch.tensor([5, 36], dtype=torch.int64)
    locations = embedding.hash_locations(values)
    assert 5 % embedding.buckets == 36 % embedding.buckets
    assert not torch.equal(locations[0], locations[1])


def test_split_optimizer_embedding_group_excludes_numerical_projection(
    processed_project: tuple[ExperimentConfig, Path],
) -> None:
    config, manifest_path = processed_project
    model = create_model(config.model, load_manifest(manifest_path), seed=42)
    assert isinstance(model, DCNModel)
    embedding_ids = {id(parameter) for parameter in model.embedding_parameters()}
    assert id(model.encoder.numerical.weight) not in embedding_ids
    device_embedding = model.encoder.embeddings["device_id"]
    assert isinstance(device_embedding, StableHashEmbedding)
    assert id(device_embedding.mixing_logits) in embedding_ids


@pytest.mark.parametrize("kind", [ModelKind.DCN, ModelKind.STEC, ModelKind.NGPT])
def test_weight_estimate_matches_serialized_state(
    processed_project: tuple[ExperimentConfig, Path],
    kind: ModelKind,
) -> None:
    config, manifest_path = processed_project
    manifest = load_manifest(manifest_path)
    model_config = _model_config(config, kind)
    model = create_model(model_config, manifest, seed=42)
    assert estimate_model_weight_bytes(model_config, manifest) == serialized_weight_bytes(model)


def test_serialized_size_accounts_for_integer_buffers() -> None:
    module = torch.nn.Linear(2, 1)
    module.register_buffer("integer_state", torch.zeros(3, dtype=torch.int64))
    expected = sum(value.numel() * value.element_size() for value in module.state_dict().values())
    assert serialized_weight_bytes(module) == expected


def test_logical_state_hash_covers_values_and_buffers() -> None:
    module = torch.nn.Linear(2, 1)
    module.register_buffer("integer_state", torch.arange(3, dtype=torch.int64))
    before = state_dict_sha256(module.state_dict())
    assert before == state_dict_sha256(dict(reversed(module.state_dict().items())))
    module.get_buffer("integer_state")[0] = 9
    assert state_dict_sha256(module.state_dict()) != before


def test_weight_budget_rejects_before_training(
    processed_project: tuple[ExperimentConfig, Path],
) -> None:
    config, manifest_path = processed_project
    model = create_model(config.model, load_manifest(manifest_path), seed=42)
    with pytest.raises(ValueError, match="exceeding"):
        enforce_weight_budget(model, 100)
