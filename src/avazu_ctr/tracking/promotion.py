"""Paired uncertainty testing and atomic selection activation."""

from __future__ import annotations

import json
import math
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from avazu_ctr.config.schema import PromotionConfig
from avazu_ctr.data.manifest import DatasetManifest
from avazu_ctr.tracking.evidence import LoadedSelection, load_selection
from avazu_ctr.tracking.store import RunStore


@dataclass(frozen=True, slots=True)
class SelectionDecision:
    selected: bool
    reason: str
    mean_difference: float | None
    upper_confidence_bound: float | None
    candidate_fold_mean: float
    incumbent_fold_mean: float | None
    bootstrap_blocks: int = 0
    bootstrap_samples: int = 0

    def statistics(self) -> dict[str, float | bool | int | None]:
        return {
            "selected": self.selected,
            "mean_difference": self.mean_difference,
            "upper_confidence_bound": self.upper_confidence_bound,
            "candidate_fold_mean": self.candidate_fold_mean,
            "incumbent_fold_mean": self.incumbent_fold_mean,
            "bootstrap_blocks": self.bootstrap_blocks,
            "bootstrap_samples": self.bootstrap_samples,
        }


def decide_selection(
    candidate_row_losses: np.ndarray,
    incumbent_row_losses: np.ndarray,
    candidate_fold_losses: list[float],
    incumbent_fold_losses: list[float],
    config: PromotionConfig,
    *,
    seed: int,
) -> SelectionDecision:
    if candidate_row_losses.ndim != 1 or incumbent_row_losses.ndim != 1:
        raise ValueError("paired selection losses must be one-dimensional")
    if candidate_row_losses.shape != incumbent_row_losses.shape:
        raise ValueError("paired selection losses must have the same shape")
    if candidate_row_losses.size == 0:
        raise ValueError("paired selection losses cannot be empty")
    if not np.isfinite(candidate_row_losses).all() or not np.isfinite(incumbent_row_losses).all():
        raise ValueError("paired selection losses must be finite")
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
        selected = False
        reason = "candidate did not improve final-holdout logloss"
    elif upper >= 0:
        selected = False
        reason = "paired bootstrap could not reject a noise-level improvement"
    elif candidate_fold_mean > incumbent_fold_mean + config.fold_guard:
        selected = False
        reason = "candidate failed the walk-forward fold guard"
    else:
        selected = True
        reason = "candidate passed final-holdout uncertainty and fold guard"
    return SelectionDecision(
        selected=selected,
        reason=reason,
        mean_difference=mean_difference,
        upper_confidence_bound=upper,
        candidate_fold_mean=candidate_fold_mean,
        incumbent_fold_mean=incumbent_fold_mean,
        bootstrap_blocks=block_count,
        bootstrap_samples=config.bootstrap_samples,
    )


def _assert_comparable(candidate: LoadedSelection, incumbent: LoadedSelection) -> None:
    candidate_evidence = candidate.evidence
    incumbent_evidence = incumbent.evidence
    if (
        candidate_evidence.holdout.population_sha256 != incumbent_evidence.holdout.population_sha256
        or candidate_evidence.holdout.rows != incumbent_evidence.holdout.rows
    ):
        raise ValueError("candidate and incumbent holdout populations differ")
    candidate_folds = candidate_evidence.confirmation.folds
    incumbent_folds = incumbent_evidence.confirmation.folds
    candidate_populations = [
        (fold.window, fold.population_sha256, fold.rows) for fold in candidate_folds
    ]
    incumbent_populations = [
        (fold.window, fold.population_sha256, fold.rows) for fold in incumbent_folds
    ]
    if candidate_populations != incumbent_populations:
        raise ValueError("candidate and incumbent walk-forward populations differ")


def _assert_recorded(selection: LoadedSelection, store: RunStore) -> None:
    evidence = selection.evidence
    expected_config_sha256 = evidence.confirmation.config_sha256

    for fold in evidence.confirmation.folds:
        run = store.run(fold.run_id)
        if (
            run["status"] != "completed"
            or run["kind"] != "confirmation"
            or run["config_sha256"] != expected_config_sha256
        ):
            raise ValueError(f"fold {fold.window} does not match a completed confirmation run")
        plan = json.loads(run["plan_json"])
        summary = json.loads(run["summary_json"])
        manifest = DatasetManifest.model_validate_json(run["dataset_json"])
        if (
            plan.get("mode") != "evaluation"
            or plan.get("early_stopping") is not True
            or plan.get("manifest_sha256") != fold.manifest_sha256
            or manifest.name != fold.window
            or manifest.labelled_source.sha256 != fold.labelled_source_sha256
            or manifest.training_range != fold.training_range
            or manifest.validation_range != fold.validation_range
            or manifest.validation_population_sha256 != fold.population_sha256
            or manifest.validation_rows != fold.rows
            or not math.isclose(
                summary.get("validation", {}).get("logloss", float("nan")),
                fold.logloss,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(f"fold {fold.window} evidence differs from its recorded run")

    holdout = evidence.holdout
    run = store.run(holdout.run_id)
    if (
        run["status"] != "completed"
        or run["kind"] != "candidate"
        or run["config_sha256"] != expected_config_sha256
    ):
        raise ValueError("holdout evidence does not match a completed candidate run")
    plan = json.loads(run["plan_json"])
    summary = json.loads(run["summary_json"])
    manifest = DatasetManifest.model_validate_json(run["dataset_json"])
    if (
        plan.get("mode") != "evaluation"
        or plan.get("early_stopping") is not True
        or plan.get("manifest_sha256") != holdout.manifest_sha256
        or manifest.name != holdout.window
        or manifest.labelled_source.sha256 != holdout.labelled_source_sha256
        or manifest.training_range != holdout.training_range
        or manifest.validation_range != holdout.validation_range
        or manifest.validation_population_sha256 != holdout.population_sha256
        or manifest.validation_rows != holdout.rows
        or summary.get("best_epoch") != holdout.best_epoch
        or not math.isclose(
            summary.get("validation", {}).get("logloss", float("nan")),
            holdout.metrics["logloss"],
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("holdout evidence differs from its recorded run")


def _first_selection(candidate: LoadedSelection) -> SelectionDecision:
    fold_mean = float(np.mean([fold.logloss for fold in candidate.evidence.confirmation.folds]))
    return SelectionDecision(
        selected=True,
        reason="first complete selection evidence",
        mean_difference=None,
        upper_confidence_bound=None,
        candidate_fold_mean=fold_mean,
        incumbent_fold_mean=None,
    )


def activate_selection(
    candidate_path: str | Path,
    active_path: str | Path,
    config: PromotionConfig,
    *,
    seed: int,
    store: RunStore,
) -> SelectionDecision:
    candidate = load_selection(candidate_path)
    candidate_root = candidate.root.resolve()
    active_root = Path(active_path).resolve()
    if candidate_root == active_root:
        raise ValueError("candidate and active selection directories must differ")
    if candidate_root == Path(candidate_root.anchor) or active_root == Path(active_root.anchor):
        raise ValueError("refusing to operate on a filesystem root")
    _assert_recorded(candidate, store)

    incumbent = load_selection(active_root) if active_root.exists() else None
    if incumbent is None:
        decision = _first_selection(candidate)
    else:
        _assert_recorded(incumbent, store)
        _assert_comparable(candidate, incumbent)
        decision = decide_selection(
            candidate.row_losses,
            incumbent.row_losses,
            [fold.logloss for fold in candidate.evidence.confirmation.folds],
            [fold.logloss for fold in incumbent.evidence.confirmation.folds],
            config,
            seed=seed,
        )

    candidate_run_id = candidate.evidence.holdout.run_id
    incumbent_run_id = incumbent.evidence.holdout.run_id if incumbent is not None else None
    if not decision.selected:
        store.record_selection_decision(
            candidate_run_id,
            incumbent_run_id,
            candidate_evidence_sha256=candidate.evidence_sha256,
            incumbent_evidence_sha256=(
                incumbent.evidence_sha256 if incumbent is not None else None
            ),
            selected=False,
            reason=decision.reason,
            statistics=decision.statistics(),
        )
        shutil.rmtree(candidate_root)
        return decision

    active_root.parent.mkdir(parents=True, exist_ok=True)
    backup = active_root.parent / f".selection-backup-{uuid.uuid4().hex}"
    replaced = active_root.exists()
    if replaced:
        active_root.replace(backup)
    try:
        candidate_root.replace(active_root)
        load_selection(active_root)
        store.record_selection_decision(
            candidate_run_id,
            incumbent_run_id,
            candidate_evidence_sha256=candidate.evidence_sha256,
            incumbent_evidence_sha256=(
                incumbent.evidence_sha256 if incumbent is not None else None
            ),
            selected=True,
            reason=decision.reason,
            statistics=decision.statistics(),
        )
    except Exception:
        if active_root.exists():
            shutil.rmtree(active_root)
        if replaced:
            backup.replace(active_root)
        raise
    if replaced:
        shutil.rmtree(backup)
    return decision
