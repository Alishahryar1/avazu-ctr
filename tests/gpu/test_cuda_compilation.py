from __future__ import annotations

import pytest
import torch
from torch import nn

from avazu_ctr.config import load_experiment
from avazu_ctr.contracts import FeatureBatch, ModelOutput
from avazu_ctr.inference.execution import InferenceRuntime
from avazu_ctr.models.base import CTRModel
from avazu_ctr.models.compilation import compile_cuda_graph
from avazu_ctr.training.optimizers import build_optimizer_plan

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA compiler contract requires an NVIDIA GPU",
)


class TinyCudaModel(CTRModel):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(32, 4)
        self.output = nn.Linear(6, 1)

    def forward(self, batch: FeatureBatch) -> ModelOutput:
        embedded = self.embedding(batch.categorical[:, 0])
        return ModelOutput(self.output(torch.cat((embedded, batch.numerical), dim=1)))


def make_batch(rows: int) -> FeatureBatch:
    return FeatureBatch(
        categorical=torch.randint(0, 32, (rows, 1), device="cuda"),
        numerical=torch.randn(rows, 2, device="cuda"),
        labels=torch.randint(0, 2, (rows, 1), device="cuda").to(torch.float32),
    )


def test_cuda_compiler_matches_eager_forward_and_backward() -> None:
    torch.manual_seed(42)
    eager = TinyCudaModel().cuda()
    candidate = TinyCudaModel().cuda()
    candidate.load_state_dict(eager.state_dict())
    compiled = compile_cuda_graph(candidate, torch.device("cuda"))
    batch = make_batch(64)
    labels = batch.labels
    if labels is None:
        raise AssertionError("compiled training batch requires labels")

    eager_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        eager(batch).aggregate_logits,
        labels,
    )
    compiled_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        compiled(batch).aggregate_logits,
        labels,
    )
    eager_loss.backward()
    compiled_loss.backward()

    torch.testing.assert_close(compiled_loss, eager_loss)
    for eager_parameter, compiled_parameter in zip(
        eager.parameters(),
        candidate.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(compiled_parameter.grad, eager_parameter.grad)

    dynamic_batch = make_batch(7)
    torch.testing.assert_close(
        compiled(dynamic_batch).aggregate_logits,
        candidate(dynamic_batch).aggregate_logits,
    )


def test_compiled_float16_inference_matches_eager_float32() -> None:
    torch.manual_seed(42)
    eager = TinyCudaModel().cuda().eval()
    candidate = TinyCudaModel().cuda().eval()
    candidate.load_state_dict(eager.state_dict())
    runtime = InferenceRuntime(candidate, torch.device("cuda"))

    for rows in (64, 7):
        host_batch = FeatureBatch(
            categorical=torch.randint(0, 32, (rows, 1)),
            numerical=torch.randn(rows, 2),
        ).pin_memory()
        with torch.inference_mode():
            expected = eager(host_batch.to("cuda")).probabilities().float().cpu()
        actual = runtime.predict(host_batch)

        assert actual.dtype is torch.float32
        torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)


def test_cuda_adamw_uses_the_fused_kernel() -> None:
    model = TinyCudaModel().cuda()
    config = load_experiment("configs/champion.yaml")
    plan = build_optimizer_plan(model, config.training.optimizer, total_steps=10)
    assert plan.optimizers[0].defaults["fused"] is True
