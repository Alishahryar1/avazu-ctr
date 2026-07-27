"""Focused Optuna stages backed by the production trainer."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import optuna

from avazu_ctr.config.loader import resolved_config
from avazu_ctr.config.schema import (
    Aggregation,
    ExperimentConfig,
)
from avazu_ctr.data.manifest import load_manifest
from avazu_ctr.tracking import RunStore
from avazu_ctr.training import Trainer

STAGES = ("optimizer", "capacity", "multihead", "regularization")


@dataclass(frozen=True, slots=True)
class ConfirmationResult:
    config: ExperimentConfig
    fold_losses: tuple[float, ...]

    @property
    def mean_logloss(self) -> float:
        return sum(self.fold_losses) / len(self.fold_losses)


def _sample_config(
    base: ExperimentConfig,
    stage: str,
    trial: optuna.Trial,
) -> ExperimentConfig:
    if stage == "optimizer":
        dense = base.training.optimizer.dense.model_copy(
            update={
                "learning_rate": trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True),
                "weight_decay": trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True),
            }
        )
        scheduler = dense.scheduler.model_copy(
            update={
                "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.15),
                "minimum_lr_ratio": trial.suggest_float("minimum_lr_ratio", 0.001, 0.1, log=True),
            }
        )
        dense = dense.model_copy(update={"scheduler": scheduler})
        optimizer = base.training.optimizer.model_copy(update={"dense": dense})
        training = base.training.model_copy(update={"optimizer": optimizer})
        return base.model_copy(update={"training": training})

    if stage == "capacity":
        backbone = base.model.backbone.model_copy(
            update={
                "dcn_layers": trial.suggest_int("dcn_layers", 2, 8),
                "dcn_rank": trial.suggest_categorical("dcn_rank", [16, 32, 64]),
                "mlp_hidden": {
                    "small": (128, 64),
                    "medium": (256, 128),
                    "large": (512, 256),
                }[trial.suggest_categorical("mlp_hidden", ["small", "medium", "large"])],
            }
        )
        model = base.model.model_copy(update={"backbone": backbone})
        return base.model_copy(update={"model": model})

    if stage == "multihead":
        model = base.model.model_copy(
            update={
                "aggregation": Aggregation(
                    trial.suggest_categorical("aggregation", ["mean", "gated"])
                ),
                "feature_bagging": trial.suggest_float("feature_bagging", 0.6, 1.0),
            }
        )
        objective = base.objective.model_copy(
            update={
                "auxiliary_weight": trial.suggest_float("auxiliary_weight", 0.05, 0.5, log=True),
                "diversity_weight": trial.suggest_float("diversity_weight", 1e-3, 0.2, log=True),
            }
        )
        return base.model_copy(update={"model": model, "objective": objective})

    backbone = base.model.backbone.model_copy(
        update={"dropout": trial.suggest_float("backbone_dropout", 0.0, 0.4)}
    )
    heads = tuple(
        head.model_copy(update={"dropout": trial.suggest_float(f"head_{index}_dropout", 0.0, 0.5)})
        for index, head in enumerate(base.model.heads)
    )
    model = base.model.model_copy(update={"backbone": backbone, "heads": heads})
    return base.model_copy(update={"model": model})


class StagedTuner:
    def __init__(
        self,
        config: ExperimentConfig,
        screening_manifest: str | Path,
        *,
        store: RunStore | None = None,
    ) -> None:
        if not config.tuning.enabled:
            raise ValueError("tuning.enabled must be true for a staged search")
        self.config = config
        self.screening_manifest = Path(screening_manifest)
        self.manifest = load_manifest(self.screening_manifest)
        self.store = store or RunStore(config.tracking.database)
        database = config.tracking.database.resolve().as_posix()
        self.storage = f"sqlite:///{database}"

    def run(self) -> tuple[ExperimentConfig, optuna.Study]:
        parent_run_id = self.store.start_run(
            self.config,
            self.manifest,
            kind="tuning",
        )
        current = self.config
        final_study: optuna.Study | None = None
        try:
            for stage_index, stage in enumerate(STAGES):
                study = optuna.create_study(
                    study_name=f"{self.config.tuning.study_name}-{stage}",
                    storage=self.storage,
                    direction="minimize",
                    load_if_exists=True,
                    sampler=optuna.samplers.TPESampler(
                        seed=self.config.training.seed + stage_index,
                    ),
                    pruner=optuna.pruners.MedianPruner(
                        n_startup_trials=5,
                        n_warmup_steps=1,
                    ),
                )

                def objective(
                    trial: optuna.Trial,
                    base_config: ExperimentConfig = current,
                    stage_name: str = stage,
                ) -> float:
                    sampled = _sample_config(base_config, stage_name, trial)
                    trial.set_user_attr(
                        "resolved_config",
                        json.dumps(resolved_config(sampled), sort_keys=True),
                    )
                    result = Trainer(
                        sampled,
                        self.screening_manifest,
                        store=self.store,
                    ).fit(trial=trial, parent_run_id=parent_run_id)
                    return result.validation.metrics["logloss"]

                study.optimize(
                    objective,
                    n_trials=self.config.tuning.trials_per_stage,
                    timeout=self.config.tuning.timeout_seconds,
                    catch=(ValueError, RuntimeError, FloatingPointError),
                    gc_after_trial=True,
                )
                if not study.best_trials:
                    raise RuntimeError(f"stage {stage} completed without a valid trial")
                current = ExperimentConfig.model_validate_json(
                    study.best_trial.user_attrs["resolved_config"]
                )
                final_study = study
            self.store.finish_run(parent_run_id, status="completed")
        except Exception:
            self.store.finish_run(parent_run_id, status="failed")
            raise
        if final_study is None:
            raise RuntimeError("staged tuning completed without a study")
        return current, final_study

    def confirm(
        self,
        study: optuna.Study,
        manifests: Sequence[str | Path],
    ) -> list[ConfirmationResult]:
        complete_trials = [
            trial
            for trial in study.trials
            if trial.state is optuna.trial.TrialState.COMPLETE and trial.value is not None
        ]

        def completed_value(trial: optuna.trial.FrozenTrial) -> float:
            if trial.value is None:
                raise ValueError("completed trial has no value")
            return trial.value

        complete = sorted(complete_trials, key=completed_value)[
            : self.config.tuning.confirmation_candidates
        ]
        results: list[ConfirmationResult] = []
        for trial in complete:
            candidate = ExperimentConfig.model_validate_json(trial.user_attrs["resolved_config"])
            losses = tuple(
                Trainer(candidate, manifest, store=self.store).fit().validation.metrics["logloss"]
                for manifest in manifests
            )
            results.append(ConfirmationResult(candidate, losses))
        return sorted(results, key=lambda result: result.mean_logloss)
