from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from avazu_ctr.config.schema import ExperimentConfig, ModelKind
from avazu_ctr.contracts import FeatureBatch, ModelOutput
from avazu_ctr.data.dataset import ParquetBatchDataset
from avazu_ctr.data.manifest import load_manifest
from avazu_ctr.models import create_model
from avazu_ctr.models.architectures import MultiHeadModel
from avazu_ctr.models.factory import enforce_weight_budget, serialized_weight_bytes
from avazu_ctr.models.layers import LogitGate, NumericalProjection, StableHashEmbedding
from avazu_ctr.objectives import CTRObjective


def _batch(manifest_path: Path):
    return next(iter(DataLoader(ParquetBatchDataset(manifest_path, "train", 16), batch_size=None)))


def test_tensor_contracts_reject_invalid_shapes_and_dtypes() -> None:
    categorical = torch.zeros((2, 1), dtype=torch.int64)
    numerical = torch.zeros((2, 1), dtype=torch.float32)
    with pytest.raises(TypeError, match="timestamps"):
        FeatureBatch(categorical, numerical, timestamps=torch.zeros(2))
    with pytest.raises(ValueError, match="aggregate logits"):
        ModelOutput(torch.zeros(2))


@pytest.mark.parametrize(
    "kind",
    [
        ModelKind.GATED_DCN,
        ModelKind.MULTIHEAD,
        ModelKind.NORMALIZED_MULTIHEAD,
        ModelKind.STEC,
    ],
)
def test_every_architecture_has_full_gradient_coverage(
    processed_project: tuple[ExperimentConfig, Path],
    kind: ModelKind,
) -> None:
    config, manifest_path = processed_project
    manifest = load_manifest(manifest_path)
    model_config = config.model.model_copy(update={"kind": kind})
    model = create_model(model_config, manifest, seed=42)
    batch = _batch(manifest_path)
    output = model(batch)
    loss = CTRObjective(config.objective)(output, batch.labels).total
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
        config.model.model_copy(update={"kind": ModelKind.GATED_DCN}),
        config.model.model_copy(update={"kind": ModelKind.STEC}),
    )
    ensemble = config.model.model_copy(update={"kind": ModelKind.ENSEMBLE, "children": children})
    model = create_model(ensemble, manifest, seed=42)
    batch = _batch(manifest_path)
    output = model(batch)
    assert len(output.children) == 2
    CTRObjective(config.objective)(output, batch.labels).total.backward()
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
    objective(model(batch), batch.labels).total.backward()
    assert isinstance(model.gate, LogitGate)
    assert all(parameter.grad is not None for parameter in model.gate.parameters())


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
    assert isinstance(model, MultiHeadModel)
    embedding_ids = {id(parameter) for parameter in model.embedding_parameters()}
    assert id(model.encoder.numerical.weight) not in embedding_ids
    device_embedding = model.encoder.embeddings["device_id"]
    assert isinstance(device_embedding, StableHashEmbedding)
    assert id(device_embedding.mixing_logits) in embedding_ids


def test_serialized_size_accounts_for_integer_buffers() -> None:
    module = torch.nn.Linear(2, 1)
    module.register_buffer("integer_state", torch.zeros(3, dtype=torch.int64))
    expected = sum(value.numel() * value.element_size() for value in module.state_dict().values())
    assert serialized_weight_bytes(module) == expected


def test_weight_budget_rejects_before_training(
    processed_project: tuple[ExperimentConfig, Path],
) -> None:
    config, manifest_path = processed_project
    model = create_model(config.model, load_manifest(manifest_path), seed=42)
    with pytest.raises(ValueError, match="exceeding"):
        enforce_weight_budget(model, 100)
