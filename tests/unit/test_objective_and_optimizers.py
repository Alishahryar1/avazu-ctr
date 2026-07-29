from __future__ import annotations

import torch
from torch import nn

from avazu_ctr.config import load_experiment
from avazu_ctr.contracts import FeatureBatch, ModelOutput
from avazu_ctr.models.base import CTRModel
from avazu_ctr.objectives import CTRObjective
from avazu_ctr.training.optimizers import FTRLProximal, build_optimizer_plan


def test_diversity_penalizes_correlated_residuals() -> None:
    config = load_experiment("configs/champion.yaml")
    objective = CTRObjective(config.objective)
    labels = torch.full((4, 1), 0.5)
    correlated = torch.tensor([[[-2.0], [-2.0]], [[2.0], [2.0]], [[-1.0], [-1.0]], [[1.0], [1.0]]])
    anticorrelated = correlated.clone()
    anticorrelated[:, 1] *= -1
    correlated_loss = objective(ModelOutput(correlated.mean(dim=1), correlated), labels).diversity
    anti_loss = objective(ModelOutput(anticorrelated.mean(dim=1), anticorrelated), labels).diversity
    assert correlated_loss > anti_loss
    assert anti_loss == 0


class SplitModel(CTRModel):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Embedding(10, 2)
        self.output = nn.Linear(2, 1)

    def forward(self, batch: FeatureBatch) -> ModelOutput:
        raise NotImplementedError


def test_identical_embedding_and_dense_configs_still_create_two_optimizers() -> None:
    config = load_experiment("configs/champion.yaml")
    plan = config.training.optimizer.model_copy(
        update={"embeddings": config.training.optimizer.dense}
    )
    bundle = build_optimizer_plan(SplitModel(), plan, total_steps=10)
    assert len(bundle.optimizers) == 2
    assert len(bundle.schedulers) == 2


def test_adamw_stays_unfused_on_cpu() -> None:
    config = load_experiment("configs/champion.yaml")
    bundle = build_optimizer_plan(SplitModel(), config.training.optimizer, total_steps=10)
    assert bundle.optimizers[0].defaults["fused"] is False


def test_ftrl_updates_and_keeps_explicit_state() -> None:
    parameter = nn.Parameter(torch.tensor([1.0]))
    optimizer = FTRLProximal([parameter], alpha=0.1, beta=1.0, l1=0.0, l2=0.0)
    parameter.grad = torch.tensor([0.5])
    optimizer.step()
    assert parameter.item() != 1.0
    assert set(optimizer.state[parameter]) == {"z", "n"}
