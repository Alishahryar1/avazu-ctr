from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

from avazu_ctr.contracts import FeatureBatch, ModelOutput
from avazu_ctr.models import compilation
from avazu_ctr.models.base import CTRModel


class TinyModel(CTRModel):
    def forward(self, batch: FeatureBatch) -> ModelOutput:
        return ModelOutput(batch.numerical[:, :1])


def test_compilation_requires_cuda() -> None:
    with pytest.raises(RuntimeError, match="requires a CUDA device"):
        compilation.compile_cuda_graph(TinyModel(), torch.device("cpu"))


def test_compilation_requires_working_triton(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(compilation, "has_triton", lambda: False)
    with pytest.raises(RuntimeError, match="requires a working Triton"):
        compilation.compile_cuda_graph(TinyModel(), torch.device("cuda"))


def test_compilation_uses_the_full_inductor_fast_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_compile(
        model: nn.Module,
        **options: Any,
    ) -> nn.Module:
        captured.update(options)
        return model

    monkeypatch.setattr(compilation, "has_triton", lambda: True)
    monkeypatch.setattr(compilation.torch, "compile", fake_compile)
    model = TinyModel()
    assert compilation.compile_cuda_graph(model, torch.device("cuda")) is model
    assert captured == {
        "backend": "inductor",
        "mode": "reduce-overhead",
        "fullgraph": True,
    }
