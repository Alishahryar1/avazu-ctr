"""Single-pass evaluation and row-level comparison data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from torch.utils.data import DataLoader

from avazu_ctr.models.base import CTRModel


@dataclass(slots=True)
class EvaluationResult:
    metrics: dict[str, float]
    probabilities: np.ndarray
    labels: np.ndarray
    row_losses: np.ndarray
    row_ids: list[str]


def expected_calibration_error(
    labels: np.ndarray, probabilities: np.ndarray, bins: int = 15
) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    assignments = np.clip(np.digitize(probabilities, edges) - 1, 0, bins - 1)
    error = 0.0
    for index in range(bins):
        selected = assignments == index
        if selected.any():
            error += selected.mean() * abs(labels[selected].mean() - probabilities[selected].mean())
    return float(error)


@torch.inference_mode()
def evaluate(
    model: CTRModel,
    loader: DataLoader,
    device: torch.device,
    *,
    amp: bool,
    amp_dtype: torch.dtype,
) -> EvaluationResult:
    model.eval()
    probabilities: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    row_ids: list[str] = []
    for batch in loader:
        if batch.labels is None:
            raise ValueError("evaluation requires labels")
        moved = batch.to(device)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp and device.type == "cuda",
        ):
            output = model(moved)
        probabilities.append(output.probabilities().float().cpu().numpy().reshape(-1))
        labels.append(batch.labels.numpy().reshape(-1))
        row_ids.extend(batch.row_ids or [])
    y_true = np.concatenate(labels)
    y_probability = np.concatenate(probabilities).clip(1e-7, 1 - 1e-7)
    row_losses = -(y_true * np.log(y_probability) + (1 - y_true) * np.log(1 - y_probability))
    auc = float("nan")
    if np.unique(y_true).size > 1:
        auc = float(roc_auc_score(y_true, y_probability))
    metrics = {
        "logloss": float(log_loss(y_true, y_probability, labels=[0, 1])),
        "roc_auc": auc,
        "brier": float(brier_score_loss(y_true, y_probability)),
        "ece": expected_calibration_error(y_true, y_probability),
    }
    return EvaluationResult(
        metrics=metrics,
        probabilities=y_probability,
        labels=y_true,
        row_losses=row_losses,
        row_ids=row_ids,
    )
