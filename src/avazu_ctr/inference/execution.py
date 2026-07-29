"""Device-specific inference execution."""

from __future__ import annotations

import torch
from torch import nn

from avazu_ctr.contracts import FeatureBatch
from avazu_ctr.models.base import CTRModel
from avazu_ctr.models.compilation import compile_cuda_graph


class InferenceGraph(nn.Module):
    """Return deployment-ready probabilities from the model graph."""

    def __init__(self, model: CTRModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, batch: FeatureBatch) -> torch.Tensor:
        return self.model(batch).probabilities().float()


class InferenceRuntime:
    """Execute eager CPU or compiled float16 CUDA inference."""

    def __init__(self, model: CTRModel, device: torch.device) -> None:
        self.device = device
        self.cuda = device.type == "cuda"
        graph = InferenceGraph(model).eval()
        self.graph = compile_cuda_graph(graph, device) if self.cuda else graph

    @torch.inference_mode()
    def predict(self, batch: FeatureBatch) -> torch.Tensor:
        moved = batch.to(self.device, non_blocking=self.cuda)
        if self.cuda:
            torch.compiler.cudagraph_mark_step_begin()
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.cuda,
        ):
            probabilities = self.graph(moved)
        return probabilities.cpu()
