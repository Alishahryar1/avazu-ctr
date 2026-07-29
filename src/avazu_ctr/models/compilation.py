"""Strict CUDA graph compilation for training and inference."""

from __future__ import annotations

from typing import cast

import torch
from torch.utils._triton import has_triton

from avazu_ctr.models.base import CTRModel


def compile_cuda_model(model: CTRModel, device: torch.device) -> CTRModel:
    """Compile a complete model graph with the CUDA Inductor fast path."""

    if device.type != "cuda":
        raise RuntimeError("model compilation requires a CUDA device")
    if not has_triton():
        raise RuntimeError(
            "CUDA model compilation requires a working Triton installation; "
            "install the cu130 project extra"
        )
    return cast(
        CTRModel,
        torch.compile(
            model,
            backend="inductor",
            mode="reduce-overhead",
            fullgraph=True,
        ),
    )
