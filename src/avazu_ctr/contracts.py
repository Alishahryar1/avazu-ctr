"""Shared tensor and manifest contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


@dataclass(slots=True)
class FeatureBatch:
    """A typed batch shared by training, evaluation, and inference."""

    categorical: torch.Tensor
    numerical: torch.Tensor
    labels: torch.Tensor | None = None
    row_ids: list[str] | None = None
    timestamps: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.categorical.dtype != torch.int64:
            raise TypeError("categorical features must use torch.int64")
        if self.numerical.dtype != torch.float32:
            raise TypeError("numerical features must use torch.float32")
        if self.categorical.ndim != 2 or self.numerical.ndim != 2:
            raise ValueError("categorical and numerical tensors must be rank two")
        if self.categorical.shape[0] != self.numerical.shape[0]:
            raise ValueError("categorical and numerical batch sizes differ")
        if self.labels is not None:
            if self.labels.dtype != torch.float32:
                raise TypeError("labels must use torch.float32")
            if self.labels.shape != (self.categorical.shape[0], 1):
                raise ValueError("labels must have shape [batch, 1]")
        if self.row_ids is not None and len(self.row_ids) != self.categorical.shape[0]:
            raise ValueError("row_ids length differs from the batch size")
        if self.timestamps is not None:
            if self.timestamps.dtype != torch.int64:
                raise TypeError("timestamps must use torch.int64")
            if self.timestamps.shape != (self.categorical.shape[0],):
                raise ValueError("timestamps must have shape [batch]")

    @property
    def batch_size(self) -> int:
        return self.categorical.shape[0]

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> FeatureBatch:
        return FeatureBatch(
            categorical=self.categorical.to(device, non_blocking=non_blocking),
            numerical=self.numerical.to(device, non_blocking=non_blocking),
            labels=(
                self.labels.to(device, non_blocking=non_blocking)
                if self.labels is not None
                else None
            ),
            row_ids=self.row_ids,
            timestamps=(
                self.timestamps.to(device, non_blocking=non_blocking)
                if self.timestamps is not None
                else None
            ),
        )


@dataclass(slots=True)
class ModelOutput:
    """Model output whose aggregate logits are the deployed prediction path."""

    aggregate_logits: torch.Tensor
    auxiliary_logits: torch.Tensor | None = None
    diagnostics: dict[str, torch.Tensor] = field(default_factory=dict)
    children: tuple[ModelOutput, ...] = ()

    def __post_init__(self) -> None:
        if self.aggregate_logits.ndim != 2 or self.aggregate_logits.shape[1] != 1:
            raise ValueError("aggregate logits must have shape [batch, 1]")
        if self.auxiliary_logits is not None and (
            self.auxiliary_logits.ndim != 3
            or self.auxiliary_logits.shape[0] != self.aggregate_logits.shape[0]
            or self.auxiliary_logits.shape[2] != 1
        ):
            raise ValueError("auxiliary logits must have shape [batch, heads, 1]")

    def probabilities(self) -> torch.Tensor:
        return torch.sigmoid(self.aggregate_logits)


@dataclass(frozen=True, slots=True)
class ShardInfo:
    path: Path
    rows: int
    sha256: str


JsonValue = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
Diagnostics = dict[str, Any]
