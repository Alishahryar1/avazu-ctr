"""One training engine for production and tuning."""

from __future__ import annotations

import math
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import optuna
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from avazu_ctr.config.loader import write_resolved_config
from avazu_ctr.config.schema import ExperimentConfig
from avazu_ctr.data.dataset import ParquetBatchDataset
from avazu_ctr.data.manifest import DatasetManifest, load_manifest, sha256_file, sha256_json
from avazu_ctr.models.base import CTRModel
from avazu_ctr.models.factory import (
    create_model,
    enforce_weight_budget,
    validate_weight_budget,
)
from avazu_ctr.objectives import CTRObjective
from avazu_ctr.tracking import RunStore
from avazu_ctr.training.evaluation import EvaluationResult, evaluate
from avazu_ctr.training.optimizers import build_optimizer_plan
from avazu_ctr.training.seed import seed_everything, seed_worker


@dataclass(slots=True)
class TrainingResult:
    run_id: str
    model: CTRModel
    best_epoch: int
    validation: EvaluationResult
    manifest: DatasetManifest


def select_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but this environment has a CPU-only PyTorch build"
            )
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer:
    def __init__(
        self,
        config: ExperimentConfig,
        manifest_path: str | Path,
        *,
        store: RunStore | None = None,
    ) -> None:
        self.config = config
        self.manifest_path = Path(manifest_path)
        self.manifest = load_manifest(self.manifest_path)
        self.store = store or RunStore(config.tracking.database)
        self._validate_feature_contract()

    def _validate_feature_contract(self) -> None:
        model_configs = [self.config.model]
        while model_configs:
            model_config = model_configs.pop()
            for feature in self.manifest.categorical_columns:
                configured = model_config.feature_embeddings.get(
                    feature,
                    model_config.default_embedding,
                )
                if self.manifest.embedding_kinds.get(feature) != configured.kind.value:
                    raise ValueError(
                        f"dataset encodes {feature!r} as "
                        f"{self.manifest.embedding_kinds.get(feature)!r}, "
                        f"but the model config requires {configured.kind.value!r}"
                    )
            model_configs.extend(model_config.children)

    def _loader(self, split: str, *, shuffle: bool, pin_memory: bool) -> DataLoader:
        dataset = ParquetBatchDataset(
            self.manifest_path,
            split,
            self.config.training.batch_size,
            shuffle=shuffle,
            seed=self.config.training.seed,
        )
        generator = torch.Generator().manual_seed(self.config.training.seed)
        return DataLoader(
            dataset,
            batch_size=None,
            num_workers=self.config.training.num_workers,
            worker_init_fn=seed_worker,
            generator=generator,
            pin_memory=pin_memory,
            persistent_workers=False,
        )

    def fit(
        self,
        *,
        trial: optuna.Trial | None = None,
        parent_run_id: str | None = None,
        resume_from: str | Path | None = None,
    ) -> TrainingResult:
        seed_everything(
            self.config.training.seed,
            deterministic_algorithms=self.config.training.deterministic_algorithms,
        )
        run_id = self.store.start_run(
            self.config,
            self.manifest,
            kind="trial" if trial is not None else "train",
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
            result = self._fit(
                run_id,
                run_dir,
                writer,
                trial,
                Path(resume_from) if resume_from is not None else None,
            )
            self.store.finish_run(run_id, status="completed")
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
    ) -> TrainingResult:
        device = select_device(self.config.training.device)
        validate_weight_budget(
            self.config.model,
            self.manifest,
            self.config.promotion.max_weight_bytes,
        )
        model = create_model(
            self.config.model,
            self.manifest,
            seed=self.config.training.seed,
        )
        enforce_weight_budget(model, self.config.promotion.max_weight_bytes)
        model.to(device)
        objective = CTRObjective(self.config.objective).to(device)
        train_loader = self._loader("train", shuffle=True, pin_memory=device.type == "cuda")
        validation_loader = self._loader(
            "validation",
            shuffle=False,
            pin_memory=device.type == "cuda",
        )
        steps_per_epoch = math.ceil(self.manifest.train_rows / self.config.training.batch_size)
        total_steps = steps_per_epoch * self.config.training.epochs
        optimizers = build_optimizer_plan(
            model,
            self.config.training.optimizer,
            total_steps=total_steps,
        )
        runtime_model: CTRModel = model
        if self.config.training.compile_model and hasattr(torch, "compile"):
            runtime_model = cast(CTRModel, torch.compile(model))
        amp_dtype = torch.float16 if self.config.training.amp_dtype == "float16" else torch.bfloat16
        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.config.training.amp and device.type == "cuda",
        )
        best_loss = float("inf")
        best_epoch = -1
        best_state: dict[str, torch.Tensor] | None = None
        best_validation: EvaluationResult | None = None
        stale_epochs = 0
        global_step = 0
        start_epoch = 0
        config_sha256 = sha256_json(self.config.model_dump(mode="json"))
        manifest_sha256 = sha256_file(self.manifest_path)
        if resume_from is not None:
            (
                start_epoch,
                global_step,
                best_loss,
                best_epoch,
                best_state,
                stale_epochs,
            ) = self._load_resume(
                resume_from,
                model,
                optimizers,
                scaler,
                expected_config_sha256=config_sha256,
                expected_manifest_sha256=manifest_sha256,
            )

        resume_path = run_dir / "resume.pt"
        for epoch in range(start_epoch, self.config.training.epochs):
            dataset = train_loader.dataset
            if isinstance(dataset, ParquetBatchDataset):
                dataset.set_epoch(epoch)
            runtime_model.train()
            for batch in train_loader:
                if batch.labels is None:
                    raise ValueError("training batch has no labels")
                moved = batch.to(device, non_blocking=True)
                optimizers.zero_grad()
                with torch.autocast(
                    device_type=device.type,
                    dtype=amp_dtype,
                    enabled=self.config.training.amp and device.type == "cuda",
                ):
                    output = runtime_model(moved)
                    losses = objective(output, moved.labels)
                if not torch.isfinite(losses.total):
                    raise FloatingPointError(f"nonfinite loss at step {global_step}")
                scaled_loss = cast(torch.Tensor, scaler.scale(losses.total))
                scaled_loss.backward()
                for optimizer in optimizers.optimizers:
                    scaler.unscale_(optimizer)
                gradient_norm: torch.Tensor | None = None
                if self.config.training.gradient_clip is not None:
                    gradient_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        self.config.training.gradient_clip,
                    )
                coherent_check = len(optimizers.optimizers) > 1 or not scaler.is_enabled()
                finite_gradients = (
                    bool(torch.isfinite(gradient_norm))
                    if coherent_check and gradient_norm is not None
                    else self._gradients_are_finite(model)
                    if coherent_check
                    else True
                )
                if not finite_gradients and not scaler.is_enabled():
                    raise FloatingPointError(f"nonfinite gradients at step {global_step}")
                previous_scale = scaler.get_scale()
                if finite_gradients:
                    for optimizer in optimizers.optimizers:
                        scaler.step(optimizer)
                scaler.update()
                successful = finite_gradients and scaler.get_scale() >= previous_scale
                if successful:
                    optimizers.step_schedulers()
                    model.post_step()
                global_step += 1
                if global_step % self.config.training.log_every_steps == 0:
                    metrics = {**losses.scalars(), **optimizers.learning_rates()}
                    self.store.log_metrics(
                        run_id,
                        step=global_step,
                        split="train",
                        metrics=metrics,
                    )
                    if writer is not None:
                        for name, value in metrics.items():
                            writer.add_scalar(f"train/{name}", value, global_step)

            validation = evaluate(
                runtime_model,
                validation_loader,
                device,
                amp=self.config.training.amp,
                amp_dtype=amp_dtype,
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
                    name: value.detach().cpu().clone() for name, value in model.state_dict().items()
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
                self._save_resume(
                    resume_path,
                    epoch=epoch,
                    global_step=global_step,
                    best_loss=best_loss,
                    best_epoch=best_epoch,
                    best_state=best_state,
                    stale_epochs=stale_epochs,
                    model=model,
                    optimizers=optimizers,
                    scaler=scaler,
                    config_sha256=config_sha256,
                    manifest_sha256=manifest_sha256,
                )

        if best_state is None or best_validation is None:
            if best_state is None:
                raise RuntimeError("training did not produce a valid checkpoint")
            model.load_state_dict(best_state, strict=True)
            best_validation = evaluate(
                runtime_model,
                validation_loader,
                device,
                amp=self.config.training.amp,
                amp_dtype=amp_dtype,
            )
        model.load_state_dict(best_state, strict=True)
        model.to("cpu")
        resume_path.unlink(missing_ok=True)
        if resume_from is not None and resume_from != resume_path:
            resume_from.unlink(missing_ok=True)
        return TrainingResult(
            run_id=run_id,
            model=model,
            best_epoch=best_epoch,
            validation=best_validation,
            manifest=self.manifest,
        )

    @staticmethod
    def _gradients_are_finite(model: CTRModel) -> bool:
        checks = [
            torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        return not checks or bool(torch.stack(checks).all())

    @staticmethod
    def _save_resume(
        path: Path,
        *,
        epoch: int,
        global_step: int,
        best_loss: float,
        best_epoch: int,
        best_state: dict[str, torch.Tensor] | None,
        stale_epochs: int,
        model: CTRModel,
        optimizers: Any,
        scaler: torch.amp.GradScaler,
        config_sha256: str,
        manifest_sha256: str,
    ) -> None:
        temporary = path.with_suffix(".tmp")
        torch.save(
            {
                "schema_version": 2,
                "config_sha256": config_sha256,
                "manifest_sha256": manifest_sha256,
                "epoch": epoch,
                "global_step": global_step,
                "best_loss": best_loss,
                "best_epoch": best_epoch,
                "best_state": best_state,
                "stale_epochs": stale_epochs,
                "model": model.state_dict(),
                "optimizers": [optimizer.state_dict() for optimizer in optimizers.optimizers],
                "schedulers": [
                    scheduler.state_dict() if scheduler is not None else None
                    for scheduler in optimizers.schedulers
                ],
                "scaler": scaler.state_dict(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            },
            temporary,
        )
        temporary.replace(path)

    @staticmethod
    def _load_resume(
        path: Path,
        model: CTRModel,
        optimizers: Any,
        scaler: torch.amp.GradScaler,
        expected_config_sha256: str,
        expected_manifest_sha256: str,
    ) -> tuple[
        int,
        int,
        float,
        int,
        dict[str, torch.Tensor] | None,
        int,
    ]:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("schema_version") != 2:
            raise ValueError("unsupported resume checkpoint schema")
        if checkpoint.get("config_sha256") != expected_config_sha256:
            raise ValueError("resume checkpoint configuration differs from the active run")
        if checkpoint.get("manifest_sha256") != expected_manifest_sha256:
            raise ValueError("resume checkpoint dataset differs from the active manifest")
        if len(checkpoint["optimizers"]) != len(optimizers.optimizers):
            raise ValueError("resume optimizer plan differs from the active configuration")
        model.load_state_dict(checkpoint["model"], strict=True)
        for optimizer, state in zip(optimizers.optimizers, checkpoint["optimizers"], strict=True):
            optimizer.load_state_dict(state)
        for scheduler, state in zip(optimizers.schedulers, checkpoint["schedulers"], strict=True):
            if scheduler is None and state is not None:
                raise ValueError("resume scheduler plan differs from configuration")
            if scheduler is not None and state is not None:
                scheduler.load_state_dict(state)
        scaler.load_state_dict(checkpoint["scaler"])
        torch.set_rng_state(checkpoint["torch_rng"])
        if torch.cuda.is_available() and checkpoint["cuda_rng"]:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        return (
            int(checkpoint["epoch"]) + 1,
            int(checkpoint["global_step"]),
            float(checkpoint["best_loss"]),
            int(checkpoint["best_epoch"]),
            checkpoint["best_state"],
            int(checkpoint["stale_epochs"]),
        )
