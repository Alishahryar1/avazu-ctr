from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import pytest
import torch

from avazu_ctr.contracts import FeatureBatch, ModelOutput
from avazu_ctr.inference import execution
from avazu_ctr.models.base import CTRModel


class TinyModel(CTRModel):
    def forward(self, batch: FeatureBatch) -> ModelOutput:
        return ModelOutput(batch.numerical[:, :1])


def make_batch() -> FeatureBatch:
    return FeatureBatch(
        categorical=torch.zeros((2, 1), dtype=torch.int64),
        numerical=torch.tensor(((-1.0, 0.0), (1.0, 0.0)), dtype=torch.float32),
    )


def test_inference_graph_returns_float32_probabilities() -> None:
    batch = make_batch()
    probabilities = execution.InferenceGraph(TinyModel())(batch)

    assert probabilities.dtype is torch.float32
    torch.testing.assert_close(probabilities, torch.sigmoid(batch.numerical[:, :1]))


def test_cpu_runtime_stays_eager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_compile(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("CPU inference must stay eager")

    monkeypatch.setattr(execution, "compile_cuda_graph", unexpected_compile)
    runtime = execution.InferenceRuntime(TinyModel(), torch.device("cpu"))

    torch.testing.assert_close(
        runtime.predict(make_batch()),
        torch.sigmoid(make_batch().numerical[:, :1]),
    )


def test_cuda_runtime_compiles_and_uses_nonblocking_float16(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, Any] = {}

    def fake_compile(
        graph: execution.InferenceGraph,
        device: torch.device,
    ) -> execution.InferenceGraph:
        calls["compiled_device"] = device
        return graph

    def fake_to(
        self: FeatureBatch,
        device: torch.device,
        *,
        non_blocking: bool = False,
    ) -> FeatureBatch:
        calls["moved_device"] = device
        calls["non_blocking"] = non_blocking
        return self

    @contextmanager
    def fake_autocast(**options: Any) -> Any:
        calls["autocast"] = options
        yield

    def fake_mark_step() -> None:
        calls["marked_step"] = True

    monkeypatch.setattr(execution, "compile_cuda_graph", fake_compile)
    monkeypatch.setattr(FeatureBatch, "to", fake_to)
    monkeypatch.setattr(execution.torch, "autocast", fake_autocast)
    monkeypatch.setattr(
        execution.torch.compiler,
        "cudagraph_mark_step_begin",
        fake_mark_step,
    )

    runtime = execution.InferenceRuntime(TinyModel(), torch.device("cuda"))
    probabilities = runtime.predict(make_batch())

    assert calls == {
        "compiled_device": torch.device("cuda"),
        "moved_device": torch.device("cuda"),
        "non_blocking": True,
        "marked_step": True,
        "autocast": {
            "device_type": "cuda",
            "dtype": torch.float16,
            "enabled": True,
        },
    }
    assert probabilities.device.type == "cpu"
