"""Strict CUDA graph compilation."""

from __future__ import annotations

from typing import cast

import torch
from torch import nn
from torch.utils._triton import has_triton


def compile_cuda_graph[GraphT: nn.Module](graph: GraphT, device: torch.device) -> GraphT:
    """Compile a complete module graph with the CUDA Inductor fast path."""

    if device.type != "cuda":
        raise RuntimeError("graph compilation requires a CUDA device")
    if not has_triton():
        raise RuntimeError(
            "CUDA graph compilation requires a working Triton installation; "
            "install the cu130 project extra"
        )
    return cast(
        GraphT,
        torch.compile(
            graph,
            backend="inductor",
            mode="reduce-overhead",
            fullgraph=True,
        ),
    )
