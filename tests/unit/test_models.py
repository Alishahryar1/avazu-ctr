from __future__ import annotations

import math
from pathlib import Path
from typing import cast

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from avazu_ctr.config.schema import (
    DeltaRoutedDCNv2CrossConfig,
    EmbeddingConfig,
    EmbeddingKind,
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
from avazu_ctr.models.layers import (
    DCNv2,
    DeltaRoutedDCNv2,
    FeatureEncoder,
    LogitGate,
    NumericalProjection,
    StableHashEmbedding,
)
from avazu_ctr.models.ngpt import NGPTAttention, NGPTBlock, NGPTModel, unit_norm
from avazu_ctr.models.state import state_dict_sha256
from avazu_ctr.models.stec import STECBatchNorm1d, STECBlock, STECModel
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
    with pytest.raises(ValueError, match="row_ids"):
        FeatureBatch(categorical, numerical, row_ids=["one"])
    with pytest.raises(ValueError, match="aggregate logits"):
        ModelOutput(torch.zeros(2))


def test_feature_batch_pins_every_tensor_lane(monkeypatch: pytest.MonkeyPatch) -> None:
    pinned: list[torch.Tensor] = []

    def record(tensor: torch.Tensor) -> torch.Tensor:
        pinned.append(tensor)
        return tensor

    monkeypatch.setattr(torch.Tensor, "pin_memory", record)
    batch = FeatureBatch(
        categorical=torch.zeros((2, 1), dtype=torch.int64),
        numerical=torch.zeros((2, 1), dtype=torch.float32),
        labels=torch.zeros((2, 1), dtype=torch.float32),
        row_ids=["one", "two"],
    )

    result = batch.pin_memory()

    assert len(pinned) == 3
    assert pinned[0] is batch.categorical
    assert pinned[1] is batch.numerical
    assert pinned[2] is batch.labels
    assert result.row_ids is batch.row_ids


def test_standard_embedding_uses_a_fixed_zero_unknown_row(
    processed_project: tuple[ExperimentConfig, Path],
) -> None:
    config, _ = processed_project
    model_config = config.model.model_copy(
        update={
            "default_embedding": EmbeddingConfig(
                kind=EmbeddingKind.STANDARD,
                dim=8,
            ),
            "feature_embeddings": {},
        }
    )
    encoder = FeatureEncoder(
        ("category",),
        (),
        {"category": 3},
        model_config,
        seed=42,
    )
    embedding = cast(nn.Embedding, encoder.embeddings["category"])
    batch = FeatureBatch(
        categorical=torch.tensor([[0], [1]], dtype=torch.int64),
        numerical=torch.empty((2, 0), dtype=torch.float32),
    )

    assert embedding.padding_idx == 0
    assert torch.equal(embedding.weight[0], torch.zeros_like(embedding.weight[0]))
    assert torch.equal(encoder(batch)[0, 0], torch.zeros(8))

    optimizer = torch.optim.AdamW(encoder.parameters(), lr=0.1, weight_decay=0.1)
    encoder(batch).sum().backward()
    optimizer.step()
    assert torch.equal(embedding.weight[0], torch.zeros_like(embedding.weight[0]))


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
    assert all(normalization.momentum is None for normalization in model.interaction_norms)
    assert model.encoder.numerical.bias is None


def test_stec_inference_is_independent_of_batch_partitioning(
    processed_project: tuple[ExperimentConfig, Path],
) -> None:
    config, manifest_path = processed_project
    manifest = load_manifest(manifest_path)
    model = create_model(_model_config(config, ModelKind.STEC), manifest, seed=42)
    batch = _batch(manifest_path)
    model.train()
    model(batch)
    model.eval()

    midpoint = batch.batch_size // 2
    partitions = (
        FeatureBatch(batch.categorical[:midpoint], batch.numerical[:midpoint]),
        FeatureBatch(batch.categorical[midpoint:], batch.numerical[midpoint:]),
    )
    with torch.inference_mode():
        whole = model(batch).aggregate_logits
        partitioned = torch.cat(
            [model(partition).aggregate_logits for partition in partitions],
            dim=0,
        )
    assert torch.allclose(whole, partitioned, atol=1e-5)


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
            assert embedding.padding_idx == 0
            assert torch.equal(embedding.weight[0], torch.zeros_like(embedding.weight[0]))
            _assert_unit_rows(embedding.weight, skip_first=True)
    _assert_unit_rows(model.encoder.numerical.weight)
    assert model.encoder.numerical.bias is not None
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
    assert model.logit_scale.shape == (2,)


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


def test_split_optimizer_embedding_group_contains_only_lookup_tables(
    processed_project: tuple[ExperimentConfig, Path],
) -> None:
    config, manifest_path = processed_project
    model = create_model(config.model, load_manifest(manifest_path), seed=42)
    assert isinstance(model, DCNModel)
    embedding_ids = {id(parameter) for parameter in model.embedding_table_parameters()}
    assert id(model.encoder.numerical.weight) not in embedding_ids
    device_embedding = model.encoder.embeddings["device_id"]
    assert isinstance(device_embedding, StableHashEmbedding)
    assert id(device_embedding.mixing_logits) not in embedding_ids
    assert all(id(table.weight) in embedding_ids for table in device_embedding.tables)


@pytest.mark.parametrize("rank", [None, 4])
def test_delta_routed_dcnv2_is_exactly_standard_at_initialization(rank: int | None) -> None:
    torch.manual_seed(42)
    standard = DCNv2(dimension=12, layers=5, rank=rank)
    routed = DeltaRoutedDCNv2(dimension=12, layers=5, rank=rank)
    incompatible = routed.load_state_dict(standard.state_dict(), strict=False)
    inputs = torch.randn(16, 12)

    assert incompatible.missing_keys == ["routing_queries"]
    assert not incompatible.unexpected_keys
    assert torch.equal(routed(inputs), standard(inputs))


def test_delta_routing_conserves_residual_mass_and_trains_queries() -> None:
    torch.manual_seed(42)
    routed = DeltaRoutedDCNv2(dimension=12, layers=5, rank=4)
    output, weights_by_layer = routed.forward_with_routing(torch.randn(16, 12))

    assert len(weights_by_layer) == 3
    for source_count, weights in enumerate(weights_by_layer, start=2):
        assert weights.shape == (16, source_count)
        torch.testing.assert_close(
            weights.sum(dim=1),
            torch.ones(16),
        )
        torch.testing.assert_close(
            (weights * source_count - 1.0).sum(dim=1),
            torch.zeros(16),
            atol=1e-6,
            rtol=0.0,
        )

    output.square().mean().backward()
    gradient = routed.routing_queries.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0


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


def test_delta_routed_cross_constructs_and_has_an_exact_weight_estimate(
    processed_project: tuple[ExperimentConfig, Path],
) -> None:
    config, manifest_path = processed_project
    manifest = load_manifest(manifest_path)
    architecture = config.model.dcn
    if architecture is None:
        raise AssertionError("test fixture must be a DCN")
    backbone = architecture.backbone.model_copy(
        update={
            "cross": DeltaRoutedDCNv2CrossConfig(
                layers=4,
                rank=4,
            )
        }
    )
    model_config = config.model.model_copy(
        update={"dcn": architecture.model_copy(update={"backbone": backbone})}
    )
    model = create_model(model_config, manifest, seed=42)

    assert isinstance(model, DCNModel)
    assert isinstance(model.backbone.cross, DeltaRoutedDCNv2)
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


def test_hash_mixer_retains_dominant_parameter_gradient() -> None:
    embedding = StableHashEmbedding(3, 1, 3, seed=42)
    with torch.no_grad():
        embedding.mixing_logits.copy_(torch.tensor([18.0, 0.0, 0.0]))
        for index, table in enumerate(embedding.tables):
            cast(nn.Embedding, table).weight.fill_(float(index == 0))
    embedding(torch.tensor([1])).sum().backward()
    expected = 2 * math.exp(-18) / (1 + 2 * math.exp(-18)) ** 2
    assert embedding.mixing_logits.grad is not None
    assert embedding.mixing_logits.grad[0].item() == pytest.approx(expected, rel=3e-6, abs=0)


def test_head_gate_retains_dominant_bias_gradient() -> None:
    gate = LogitGate(3)
    projection = cast(nn.Linear, gate.network[-1])
    assert projection.bias is not None
    with torch.no_grad():
        projection.weight.zero_()
        projection.bias.copy_(torch.tensor([18.0, 0.0, 0.0]))
    aggregate, _ = gate(torch.tensor([[1.0, 0.0, 0.0]]))
    aggregate.sum().backward()
    expected = 2 * math.exp(-18) / (1 + 2 * math.exp(-18)) ** 2
    assert projection.bias.grad is not None
    assert projection.bias.grad[0].item() == pytest.approx(expected, rel=3e-6, abs=0)


def test_delta_router_retains_dominant_query_gradient() -> None:
    router = DeltaRoutedDCNv2(3, 3, None)
    with torch.no_grad():
        router.routing_queries[0].copy_(torch.tensor([18.0, 0.0, 0.0]))
    sources = list(torch.eye(3).unsqueeze(1))
    _, weights = router._routing_context(torch.zeros(1, 3), sources, router.routing_queries[0])
    weights[0, 0].backward()
    assert router.routing_queries.grad is not None
    assert router.routing_queries.grad[0, 0] > 2e-8
    assert torch.isfinite(router.routing_queries.grad).all()


def test_stec_attention_retains_dominant_query_gradient() -> None:
    block = STECBlock(3, 1, 0.0)
    with torch.no_grad():
        for projection in (block.query, block.key, block.value, block.output):
            projection.weight.copy_(torch.eye(3))
        block.key.weight.mul_(18 * math.sqrt(3))
    attended, _ = block(torch.eye(3).unsqueeze(0))
    attended[0, 0, 0].backward()
    assert block.query.weight.grad is not None
    expected = 18 * 2 * math.exp(-18) / (1 + 2 * math.exp(-18)) ** 2
    assert block.query.weight.grad[0, 0].item() == pytest.approx(expected, rel=5e-6, abs=0)


def test_ngpt_attention_retains_dominant_scale_gradient() -> None:
    block = NGPTAttention(NGPTModelConfig(dimension=3, heads=1))
    captured = []
    handle = block.output.register_forward_pre_hook(
        lambda module, inputs: captured.append(inputs[0])
    )
    with torch.no_grad():
        for projection in (block.query, block.key, block.value):
            projection.weight.copy_(torch.eye(3))
        block.qk_scale.fill_(math.sqrt(18 / math.sqrt(3)) * block.base_scale)
    block(torch.eye(3).unsqueeze(0))
    captured[0][0, 0, 0].backward()
    handle.remove()
    expected = 2 * math.exp(-18) / (1 + 2 * math.exp(-18)) ** 2 * 36 / block.qk_scale[0].item()
    assert block.qk_scale.grad is not None
    assert block.qk_scale.grad[0].item() == pytest.approx(expected, rel=1e-5, abs=0)


@pytest.mark.parametrize("momentum", [None, 0.2])
@pytest.mark.parametrize("spatial", [False, True])
def test_stec_batch_norm_preserves_native_statistics_gradients_and_state(momentum, spatial) -> None:
    torch.manual_seed(12)
    reference = nn.BatchNorm1d(4, momentum=momentum).double()
    candidate = STECBatchNorm1d(4, momentum=momentum).double()
    assert reference.running_mean is not None
    assert reference.running_var is not None
    assert reference.num_batches_tracked is not None
    with torch.no_grad():
        reference.running_mean.copy_(torch.randn(4))
        reference.running_var.copy_(torch.rand(4) + 0.5)
        reference.num_batches_tracked.fill_(7)
        reference.weight.copy_(torch.rand(4) + 0.5)
        reference.bias.copy_(torch.randn(4))
    candidate.load_state_dict(reference.state_dict(), strict=True)
    for rows in (8, 8, 3):
        values = torch.randn((rows, 4, 2) if spatial else (rows, 4), dtype=torch.float64)
        actual_input = values.clone().requires_grad_()
        expected_input = values.clone().requires_grad_()
        actual = candidate(actual_input)
        expected = reference(expected_input)
        upstream = torch.randn_like(actual)
        actual.backward(upstream)
        expected.backward(upstream)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(actual_input.grad, expected_input.grad, rtol=1e-12, atol=1e-12)
        for left, right in zip(candidate.parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(left.grad, right.grad, rtol=1e-12, atol=1e-12)
        for name, value in candidate.state_dict().items():
            torch.testing.assert_close(value, reference.state_dict()[name], rtol=1e-12, atol=1e-12)
    candidate.eval()
    reference.eval()
    before = {name: value.clone() for name, value in candidate.state_dict().items()}
    torch.testing.assert_close(candidate(values), reference(values), rtol=1e-12, atol=1e-12)
    for name, value in candidate.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    reference.load_state_dict(candidate.state_dict(), strict=True)


@pytest.mark.parametrize(("dropout", "training"), [(0.0, True), (0.4, True), (0.4, False)])
def test_ngpt_attention_reference_values_and_gradients(dropout: float, training: bool) -> None:
    torch.manual_seed(42)
    block = NGPTAttention(NGPTModelConfig(dimension=8, heads=2, dropout=dropout)).double()
    block.train(training)
    inputs = torch.randn(2, 3, 8, dtype=torch.float64, requires_grad=True)
    captured = []
    handle = block.output.register_forward_pre_hook(lambda module, args: captured.append(args[0]))
    torch.manual_seed(17)
    actual = block(inputs)
    scale = (block.qk_scale / block.base_scale).view(1, 2, 1, 4)
    query = unit_norm(block._heads(block.query(inputs))) * scale
    key = unit_norm(block._heads(block.key(inputs))) * scale
    value = block._heads(block.value(inputs))
    probabilities = ((query @ key.transpose(-2, -1)) * 2).softmax(-1)
    torch.manual_seed(17)
    probabilities = torch.nn.functional.dropout(probabilities, dropout, training)
    attended = (probabilities @ value).transpose(1, 2).contiguous().view(2, 3, 8)
    torch.testing.assert_close(captured[0], attended, rtol=1e-12, atol=1e-12)
    proposal = unit_norm(block.output(attended))
    alpha = (block.alpha.abs() * (block.alpha_init / block.base_scale)).view(1, 1, -1)
    expected = unit_norm(inputs + alpha * (proposal - inputs))
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    parameters = (inputs, *block.parameters())
    upstream = torch.randn_like(actual)
    actual_gradients = torch.autograd.grad(actual, parameters, upstream)
    expected_gradients = torch.autograd.grad(expected, parameters, upstream)
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=2e-5, atol=1e-7)
    if not training or dropout == 0:
        fused = torch.nn.functional.scaled_dot_product_attention(query, key, value, scale=2)
        torch.testing.assert_close(attended, fused.transpose(1, 2).contiguous().view(2, 3, 8))
        torch.testing.assert_close(block(inputs), actual, rtol=0, atol=0)
    handle.remove()
