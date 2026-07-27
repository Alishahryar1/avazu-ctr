"""Early-stopped evaluation training for tuning and model selection."""

from __future__ import annotations

import traceback
from dataclasses import dataclass
from pathlib import Path

import optuna
import torch
from torch.utils.tensorboard import SummaryWriter

from avazu_ctr.config.loader import write_resolved_config
from avazu_ctr.config.schema import ExperimentConfig
from avazu_ctr.data.manifest import (
    DatasetManifest,
    DatasetPurpose,
    load_manifest,
    sha256_file,
    sha256_json,
)
from avazu_ctr.models.base import CTRModel
from avazu_ctr.tracking.store import RunStore
from avazu_ctr.training.engine import (
    OptimizationLoop,
    load_resume,
    make_loader,
    save_resume,
    steps_per_epoch,
    validate_feature_contract,
)
from avazu_ctr.training.evaluation import EvaluationResult, evaluate
from avazu_ctr.training.seed import seed_everything


@dataclass(slots=True)
class CandidateResult:
    run_id: str
    model: CTRModel
    best_epoch: int
    epochs_completed: int
    steps_completed: int
    validation: EvaluationResult
    manifest: DatasetManifest
    manifest_sha256: str


class CandidateTrainer:
    def __init__(
        self,
        config: ExperimentConfig,
        manifest_path: str | Path,
        *,
        store: RunStore | None = None,
    ) -> None:
        self.config = config
        self.manifest_path = Path(manifest_path)
        self.manifest = load_manifest(self.manifest_path, verify_shards=True)
        if self.manifest.purpose is not DatasetPurpose.EVALUATION:
            raise ValueError("candidate training requires an evaluation dataset")
        self.store = store or RunStore(config.tracking.database)
        validate_feature_contract(config, self.manifest)

    def fit(
        self,
        *,
        trial: optuna.Trial | None = None,
        parent_run_id: str | None = None,
        resume_from: str | Path | None = None,
        kind: str = "candidate",
    ) -> CandidateResult:
        if trial is not None:
            kind = "trial"
        elif kind not in {"candidate", "confirmation"}:
            raise ValueError(f"invalid evaluation run kind: {kind}")
        epoch_steps = steps_per_epoch(self.manifest, self.config.training.batch_size)
        manifest_sha256 = sha256_file(self.manifest_path)
        run_id = self.store.start_run(
            self.config,
            self.manifest,
            kind=kind,
            plan={
                "mode": "evaluation",
                "epochs": self.config.training.epochs,
                "steps_per_epoch": epoch_steps,
                "planned_steps": epoch_steps * self.config.training.epochs,
                "early_stopping": True,
                "manifest_sha256": manifest_sha256,
            },
            parent_run_id=parent_run_id,
            study_name=trial.study.study_name if trial is not None else None,
            trial_number=trial.number if trial is not None else None,
        )
        run_dir = self.config.data.artifact_root / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        write_resolved_config(self.config, run_dir / "resolved.yaml")
        writer = (
            SummaryWriter(self.config.tracking.tensorboard_dir / run_id)
            if self.config.tracking.tensorboard
            else None
        )
        try:
            seed_everything(
                self.config.training.seed,
                deterministic_algorithms=self.config.training.deterministic_algorithms,
            )
            result = self._fit(
                run_id,
                run_dir,
                writer,
                trial,
                Path(resume_from) if resume_from is not None else None,
            )
            self.store.finish_run(
                run_id,
                status="completed",
                summary={
                    "best_epoch": result.best_epoch,
                    "epochs_completed": result.epochs_completed,
                    "steps_completed": result.steps_completed,
                    "validation": result.validation.metrics,
                },
            )
            return result
        except optuna.TrialPruned:
            self.store.finish_run(run_id, status="pruned")
            raise
        except Exception:
            self.store.finish_run(run_id, status="failed", error=traceback.format_exc())
            raise
        finally:
            if writer is not None:
                writer.close()

    def _fit(
        self,
        run_id: str,
        run_dir: Path,
        writer: SummaryWriter | None,
        trial: optuna.Trial | None,
        resume_from: Path | None,
    ) -> CandidateResult:
        loop = OptimizationLoop(
            self.config,
            self.manifest,
            self.manifest_path,
            epochs=self.config.training.epochs,
            run_id=run_id,
            store=self.store,
            writer=writer,
        )
        validation_loader = make_loader(
            self.config,
            self.manifest_path,
            "validation",
            shuffle=False,
            pin_memory=loop.device.type == "cuda",
            num_workers=0,
        )
        best_loss = float("inf")
        best_epoch = -1
        best_state: dict[str, torch.Tensor] | None = None
        best_validation: EvaluationResult | None = None
        stale_epochs = 0
        start_epoch = 0
        epochs_completed = 0
        config_sha256 = sha256_json(self.config.model_dump(mode="json"))
        manifest_sha256 = sha256_file(self.manifest_path)
        if resume_from is not None:
            (
                start_epoch,
                best_loss,
                best_epoch,
                best_state,
                stale_epochs,
            ) = load_resume(
                resume_from,
                loop,
                expected_config_sha256=config_sha256,
                expected_manifest_sha256=manifest_sha256,
            )
            epochs_completed = start_epoch

        resume_path = run_dir / "resume.pt"
        for epoch in range(start_epoch, self.config.training.epochs):
            loop.train_epoch(epoch)
            epochs_completed = epoch + 1
            validation = evaluate(
                loop.runtime_model,
                validation_loader,
                loop.device,
                amp=self.config.training.amp,
                amp_dtype=loop.amp_dtype,
            )
            self.store.log_metrics(
                run_id,
                step=epoch,
                split="validation",
                metrics=validation.metrics,
            )
            if writer is not None:
                for name, value in validation.metrics.items():
                    writer.add_scalar(f"validation/{name}", value, epoch)
            if trial is not None:
                trial.report(validation.metrics["logloss"], epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned

            if validation.metrics["logloss"] < best_loss:
                best_loss = validation.metrics["logloss"]
                best_epoch = epoch
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in loop.model.state_dict().items()
                }
                best_validation = validation
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs > self.config.training.early_stopping_patience:
                    break
            if (
                self.config.training.resume_checkpoint
                and trial is None
                and epoch + 1 < self.config.training.epochs
            ):
                save_resume(
                    resume_path,
                    epoch=epoch,
                    best_loss=best_loss,
                    best_epoch=best_epoch,
                    best_state=best_state,
                    stale_epochs=stale_epochs,
                    loop=loop,
                    config_sha256=config_sha256,
                    manifest_sha256=manifest_sha256,
                )

        if best_state is None:
            raise RuntimeError("candidate training did not produce a valid checkpoint")
        loop.model.load_state_dict(best_state, strict=True)
        if best_validation is None:
            best_validation = evaluate(
                loop.runtime_model,
                validation_loader,
                loop.device,
                amp=self.config.training.amp,
                amp_dtype=loop.amp_dtype,
            )
        loop.model.to("cpu")
        resume_path.unlink(missing_ok=True)
        if resume_from is not None and resume_from != resume_path:
            resume_from.unlink(missing_ok=True)
        return CandidateResult(
            run_id=run_id,
            model=loop.model,
            best_epoch=best_epoch,
            epochs_completed=epochs_completed,
            steps_completed=loop.global_step,
            validation=best_validation,
            manifest=self.manifest,
            manifest_sha256=manifest_sha256,
        )
