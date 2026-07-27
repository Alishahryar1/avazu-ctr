"""Paired uncertainty testing and atomic champion replacement."""

from __future__ import annotations

import json
import math
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from avazu_ctr.config.schema import PromotionConfig
from avazu_ctr.inference.bundle import load_bundle
from avazu_ctr.tracking.store import RunStore


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    promoted: bool
    reason: str
    mean_difference: float
    upper_confidence_bound: float
    candidate_fold_mean: float
    incumbent_fold_mean: float
    bootstrap_blocks: int = 0
    bootstrap_samples: int = 0

    def statistics(self) -> dict[str, float | bool | int]:
        return {
            "promoted": self.promoted,
            "mean_difference": self.mean_difference,
            "upper_confidence_bound": self.upper_confidence_bound,
            "candidate_fold_mean": self.candidate_fold_mean,
            "incumbent_fold_mean": self.incumbent_fold_mean,
            "bootstrap_blocks": self.bootstrap_blocks,
            "bootstrap_samples": self.bootstrap_samples,
        }


def decide_promotion(
    candidate_row_losses: np.ndarray,
    incumbent_row_losses: np.ndarray,
    candidate_fold_losses: list[float],
    incumbent_fold_losses: list[float],
    config: PromotionConfig,
    *,
    seed: int,
) -> PromotionDecision:
    if candidate_row_losses.ndim != 1 or incumbent_row_losses.ndim != 1:
        raise ValueError("paired promotion losses must be one-dimensional")
    if candidate_row_losses.shape != incumbent_row_losses.shape:
        raise ValueError("paired promotion losses must have the same shape")
    if candidate_row_losses.size == 0:
        raise ValueError("paired promotion losses cannot be empty")
    if not np.isfinite(candidate_row_losses).all() or not np.isfinite(incumbent_row_losses).all():
        raise ValueError("paired promotion losses must be finite")
    if not candidate_fold_losses or len(candidate_fold_losses) != len(incumbent_fold_losses):
        raise ValueError("candidate and incumbent fold losses must be paired")
    if not np.isfinite(candidate_fold_losses).all() or not np.isfinite(incumbent_fold_losses).all():
        raise ValueError("fold losses must be finite")
    differences = candidate_row_losses - incumbent_row_losses
    mean_difference = float(differences.mean())
    block_count = max(
        1,
        min(
            differences.size,
            math.ceil(differences.size / config.bootstrap_block_rows),
        ),
    )
    boundaries = np.arange(block_count + 1, dtype=np.int64) * differences.size // block_count
    block_sums = np.add.reduceat(differences, boundaries[:-1])
    block_sizes = np.diff(boundaries)
    block_means = block_sums / block_sizes
    rng = np.random.default_rng(seed)
    means = np.empty(config.bootstrap_samples, dtype=np.float64)
    for index in range(config.bootstrap_samples):
        sample = rng.integers(0, block_count, block_count)
        means[index] = block_means[sample].mean()
    upper = float(np.quantile(means, config.confidence))
    candidate_fold_mean = float(np.mean(candidate_fold_losses))
    incumbent_fold_mean = float(np.mean(incumbent_fold_losses))
    if mean_difference >= 0:
        promoted = False
        reason = "candidate did not improve final-holdout logloss"
    elif upper >= 0:
        promoted = False
        reason = "paired bootstrap could not reject a noise-level improvement"
    elif candidate_fold_mean > incumbent_fold_mean + config.fold_guard:
        promoted = False
        reason = "candidate failed the walk-forward fold guard"
    else:
        promoted = True
        reason = "candidate passed final-holdout uncertainty and fold guard"
    return PromotionDecision(
        promoted=promoted,
        reason=reason,
        mean_difference=mean_difference,
        upper_confidence_bound=upper,
        candidate_fold_mean=candidate_fold_mean,
        incumbent_fold_mean=incumbent_fold_mean,
        bootstrap_blocks=block_count,
        bootstrap_samples=config.bootstrap_samples,
    )


def promote_bundle(
    candidate_dir: Path,
    champion_dir: Path,
    decision: PromotionDecision,
    *,
    store: RunStore,
    candidate_run_id: str,
    incumbent_run_id: str | None,
) -> bool:
    candidate_resolved = candidate_dir.resolve()
    champion_resolved = champion_dir.resolve()
    if candidate_resolved == champion_resolved:
        raise ValueError("candidate and champion directories must differ")
    if candidate_resolved == Path(candidate_resolved.anchor):
        raise ValueError("refusing to operate on a filesystem root")
    load_bundle(candidate_resolved)
    store.record_promotion(
        candidate_run_id,
        incumbent_run_id,
        promoted=decision.promoted,
        reason=decision.reason,
        statistics=decision.statistics(),
    )
    if not decision.promoted:
        shutil.rmtree(candidate_resolved)
        return False
    champion_dir.parent.mkdir(parents=True, exist_ok=True)
    backup = champion_dir.parent / f".champion-backup-{uuid.uuid4().hex}"
    if champion_dir.exists():
        champion_dir.replace(backup)
    try:
        candidate_resolved.replace(champion_dir)
        load_bundle(champion_dir)
    except Exception:
        if champion_dir.exists():
            shutil.rmtree(champion_dir)
        if backup.exists():
            backup.replace(champion_dir)
        raise
    if backup.exists():
        shutil.rmtree(backup)
    (champion_dir / "promotion.json").write_text(
        json.dumps(
            {
                "candidate_run_id": candidate_run_id,
                "incumbent_run_id": incumbent_run_id,
                "reason": decision.reason,
                **decision.statistics(),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return True
