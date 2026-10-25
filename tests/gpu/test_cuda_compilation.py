from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import cast

import pytest
import torch
from torch import nn

from avazu_ctr.config import load_experiment
from avazu_ctr.config.schema import (
    ExperimentConfig,
    ModelKind,
    NGPTModelConfig,
    OptimizerKind,
    SchedulerKind,
    STECModelConfig,
)
from avazu_ctr.contracts import FeatureBatch, ModelOutput
from avazu_ctr.data.manifest import load_manifest
from avazu_ctr.inference.execution import InferenceRuntime
from avazu_ctr.models import create_model
from avazu_ctr.models.base import CTRModel
from avazu_ctr.models.compilation import compile_cuda_graph
from avazu_ctr.models.layers import DeltaRoutedDCNv2
from avazu_ctr.models.ngpt import NGPTAttention
from avazu_ctr.models.stec import STECBatchNorm1d, STECBlock
from avazu_ctr.numerics import log_softmax, soft_target_cross_entropy, softmax
from avazu_ctr.objectives import CTRObjective
from avazu_ctr.training import CandidateTrainer
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


def test_cuda_compiler_trains_delta_routing_under_float16_autocast() -> None:
    torch.manual_seed(42)
    model = DeltaRoutedDCNv2(dimension=24, layers=5, rank=8).cuda()
    compiled = compile_cuda_graph(model, torch.device("cuda"))
    inputs = torch.randn(64, 24, device="cuda")

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        loss = compiled(inputs).square().mean()
    loss.backward()
    torch.cuda.synchronize()

    gradient = model.routing_queries.grad
    assert torch.isfinite(loss)
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0


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


def test_compiled_split_candidate_releases_gradients_before_cpu_transfer(
    processed_project: tuple[ExperimentConfig, Path],
) -> None:
    config, manifest_path = processed_project
    dense = config.training.optimizer.dense
    embeddings = dense.model_copy(
        update={
            "kind": OptimizerKind.ADAGRAD,
            "learning_rate": 0.1,
            "weight_decay": 0.0,
            "scheduler": dense.scheduler.model_copy(update={"kind": SchedulerKind.NONE}),
        }
    )
    optimizer = config.training.optimizer.model_copy(update={"embeddings": embeddings})
    training = config.training.model_copy(
        update={
            "device": "cuda",
            "amp": True,
            "compile_model": True,
            "deterministic_algorithms": False,
            "optimizer": optimizer,
        }
    )
    compiled_config = config.model_copy(
        update={"name": "compiled-split-candidate", "training": training}
    )

    result = CandidateTrainer(compiled_config, manifest_path).fit()

    assert all(parameter.device.type == "cpu" for parameter in result.model.parameters())
    assert all(parameter.grad is None for parameter in result.model.parameters())


class NumericalLossGraph(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.objective = CTRObjective(load_experiment("configs/ngpt.yaml").objective)

    def forward(self, logits, binary, student, teacher):
        return (
            softmax(logits),
            log_softmax(logits),
            self.objective(ModelOutput(binary), torch.ones_like(binary)).total,
            soft_target_cross_entropy(student, teacher),
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_corrected_numerics_survive_autocast_compilation_and_replay(dtype) -> None:
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 is not supported by this GPU")
    graph = NumericalLossGraph().cuda()
    compiled = compile_cuda_graph(graph, torch.device("cuda"))
    scaler = torch.amp.GradScaler("cuda", init_scale=32768)
    for rows in (8, 8, 3):
        torch.compiler.cudagraph_mark_step_begin()
        logits = torch.tensor([[18.0, 0.0, 0.0]] * rows, device="cuda", dtype=dtype)
        binary = torch.full((rows, 1), 18.0, device="cuda", dtype=dtype)
        teacher = torch.tensor([[2.0, 1.0, 0.0]] * rows, device="cuda", dtype=dtype)
        student = teacher.clone()
        # Equality on one row, the smallest representable downward step on others.
        student[1:, 0] = torch.nextafter(student[1:, 0], torch.zeros_like(student[1:, 0]))
        student[-1] = torch.tensor([10000.0, -10000.0, 0.0], device="cuda", dtype=dtype)
        gradients = []
        results = []
        for operation in (graph, compiled):
            inputs = [
                value.detach().clone().requires_grad_()
                for value in (logits, binary, student, teacher)
            ]
            with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
                probabilities, logs, bce, ce = operation(*inputs)
                loss = probabilities[:, 0].mean() + logs[:, 0].mean() + bce + ce
            assert all(value.dtype == torch.float32 for value in (probabilities, logs, bce, ce))
            scaler.scale(loss).backward()
            assert inputs[3].grad is None
            scaled = [cast(torch.Tensor, value.grad).detach().clone() for value in inputs[:3]]
            assert scaled[0][:, 0].ne(0).all()
            assert scaled[1].ne(0).all()
            assert torch.equal(scaled[2][0], torch.zeros_like(scaled[2][0]))
            gradients.append([value.double() / scaler.get_scale() for value in scaled])
            results.append([value.detach().clone() for value in (probabilities, logs, bce, ce)])
        for actual, expected in zip(results[1], results[0], strict=True):
            torch.testing.assert_close(actual, expected, rtol=5e-6, atol=1e-8)
        for actual, expected in zip(gradients[1], gradients[0], strict=True):
            torch.testing.assert_close(actual, expected, rtol=0.02, atol=0)
        tail = math.exp(-18)
        dominant = (2 * tail / (1 + 2 * tail) ** 2 + 2 * tail / (1 + 2 * tail)) / rows
        expected_softmax = torch.tensor(
            [dominant, -dominant / 2, -dominant / 2], device="cuda", dtype=torch.float64
        ).expand(rows, -1)
        torch.testing.assert_close(gradients[1][0], expected_softmax, rtol=0.01, atol=0)
        torch.testing.assert_close(
            gradients[1][1],
            torch.full_like(gradients[1][1], -tail / (1 + tail) / rows),
            rtol=0.01,
            atol=0,
        )
        expected_teacher = (student.double().softmax(-1) - teacher.double().softmax(-1)) / rows
        torch.testing.assert_close(gradients[1][2], expected_teacher, rtol=0.02, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("kind", ["ngpt", "stec"])
def test_explicit_attention_accumulates_in_float32_under_autocast(kind, dtype) -> None:
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 is not supported by this GPU")
    torch.manual_seed(42)
    module = (
        NGPTAttention(NGPTModelConfig(dimension=8, heads=2))
        if kind == "ngpt"
        else STECBlock(8, 2, 0.0)
    ).cuda()
    candidate = copy.deepcopy(module)
    compiled = compile_cuda_graph(candidate, torch.device("cuda"))
    captured = []
    handle = module.output.register_forward_pre_hook(
        lambda module, inputs: captured.append(inputs[0])
    )
    values = torch.randn(8, 5, 8, device="cuda")
    losses = []
    for operation in (module, compiled):
        with torch.autocast("cuda", dtype=dtype):
            output = operation(values)
            attended = output if kind == "ngpt" else output[0]
            loss = attended.float().square().mean()
        loss.backward()
        losses.append(loss.detach().clone())
    assert captured[0].dtype == torch.float32
    handle.remove()
    torch.testing.assert_close(losses[0], losses[1], rtol=0.03, atol=1e-5)
    for eager, actual in zip(module.parameters(), candidate.parameters(), strict=True):
        assert actual.grad is not None
        assert torch.isfinite(actual.grad).all()
        torch.testing.assert_close(actual.grad, eager.grad, rtol=0.05, atol=3e-4)


@pytest.mark.parametrize("kind", [ModelKind.DCN, ModelKind.NGPT, ModelKind.STEC])
def test_corrected_models_compile_train_and_infer_with_strict_state_loading(
    processed_project: tuple[ExperimentConfig, Path], kind: ModelKind
) -> None:
    torch.manual_seed(42)
    config, manifest_path = processed_project
    manifest = load_manifest(manifest_path)
    architecture = config.model
    if kind == ModelKind.NGPT:
        architecture = architecture.model_copy(
            update={
                "kind": kind,
                "dcn": None,
                "ngpt": NGPTModelConfig(dimension=8, heads=2, layers=1),
            }
        )
    elif kind == ModelKind.STEC:
        architecture = architecture.model_copy(
            update={
                "kind": kind,
                "dcn": None,
                "stec": STECModelConfig(
                    dimension=8, heads=2, layers=1, dropout=0.0, prediction_dropout=0.0
                ),
            }
        )
    eager = create_model(architecture, manifest, seed=42).cuda()
    candidate = create_model(architecture, manifest, seed=7).cuda()
    candidate.load_state_dict(eager.state_dict(), strict=True)
    compiled = compile_cuda_graph(candidate, torch.device("cuda"))
    objective = CTRObjective(config.objective)
    batches = []
    for rows in (8, 8, 3):
        torch.compiler.cudagraph_mark_step_begin()
        batch = FeatureBatch(
            categorical=torch.zeros(
                rows, len(manifest.categorical_columns), dtype=torch.int64, device="cuda"
            ),
            numerical=torch.randn(rows, len(manifest.numerical_columns), device="cuda"),
            labels=torch.ones(rows, 1, device="cuda"),
        )
        batches.append(batch)
        losses = []
        for model in (eager, compiled):
            model.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = objective(model(batch), cast(torch.Tensor, batch.labels)).total
            loss.backward()
            losses.append(loss.detach().clone())
        torch.testing.assert_close(losses[0], losses[1], rtol=0.01, atol=1e-4)
        for (name, reference), actual in zip(
            eager.named_parameters(), candidate.parameters(), strict=True
        ):
            assert actual.grad is not None
            assert torch.isfinite(actual.grad).all()
            torch.testing.assert_close(
                actual.grad,
                reference.grad,
                rtol=0.1,
                atol=2e-3,
                msg=lambda message, name=name, rows=rows: f"{name}, rows={rows}: {message}",
            )
    eager.zero_grad(set_to_none=True)
    candidate.zero_grad(set_to_none=True)
    eager.eval()
    runtime = InferenceRuntime(candidate, torch.device("cuda"))
    for batch in batches:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            expected = eager(batch).probabilities().float().cpu()
        actual = runtime.predict(batch)
        torch.testing.assert_close(actual, expected, rtol=0.01, atol=1e-3)
    assert all(parameter.grad is None for parameter in candidate.parameters())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_stec_cumulative_batch_norm_compiles_and_updates_each_replay(dtype) -> None:
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 is not supported by this GPU")
    reference = nn.BatchNorm1d(4, momentum=None).cuda()
    candidate = STECBatchNorm1d(4, momentum=None).cuda()
    assert reference.num_batches_tracked is not None
    assert candidate.num_batches_tracked is not None
    reference.num_batches_tracked.fill_(7)
    candidate.load_state_dict(reference.state_dict(), strict=True)
    compiled = compile_cuda_graph(candidate, torch.device("cuda"))
    for step, rows in enumerate((8, 8, 8, 3), start=8):
        candidate.zero_grad(set_to_none=True)
        reference.zero_grad(set_to_none=True)
        torch.compiler.cudagraph_mark_step_begin()
        values = (torch.randn(rows, 4, device="cuda") + step).to(dtype)
        actual_input = values.clone().requires_grad_()
        expected_input = values.clone().requires_grad_()
        with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
            actual = compiled(actual_input)
            expected = reference(expected_input)
        assert actual.dtype == expected.dtype
        upstream = torch.randn_like(actual)
        actual.backward(upstream)
        expected.backward(upstream)
        torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.01)
        torch.testing.assert_close(actual_input.grad, expected_input.grad, rtol=0.02, atol=0.02)
        for left, right in zip(candidate.parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(left.grad, right.grad, rtol=0.02, atol=0.02)
        for name, value in candidate.state_dict().items():
            torch.testing.assert_close(value, reference.state_dict()[name], rtol=1e-5, atol=1e-6)
        assert candidate.num_batches_tracked.item() == step
