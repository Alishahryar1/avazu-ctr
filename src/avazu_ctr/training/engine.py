"""Shared optimization kernel for evaluation training and production refits."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, cast

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from avazu_ctr.config.schema import ExperimentConfig
from avazu_ctr.data.dataset import ParquetBatchDataset
from avazu_ctr.data.manifest import DatasetManifest
from avazu_ctr.data.preprocessing import feature_config_sha256
from avazu_ctr.models.base import CTRModel
from avazu_ctr.models.factory import (
    create_model,
    enforce_weight_budget,
    validate_weight_budget,
)
from avazu_ctr.objectives import CTRObjective
from avazu_ctr.tracking.store import RunStore
from avazu_ctr.training.optimizers import build_optimizer_plan
from avazu_ctr.training.seed import seed_worker


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


def validate_feature_contract(config: ExperimentConfig, manifest: DatasetManifest) -> None:
    if manifest.feature_config_sha256 != feature_config_sha256(config):
        raise ValueError("dataset feature configuration does not match the active configuration")
    model_configs = [config.model]
    while model_configs:
        model_config = model_configs.pop()
        for feature in manifest.categorical_columns:
            configured = model_config.feature_embeddings.get(
                feature,
                model_config.default_embedding,
            )
            if manifest.embedding_kinds.get(feature) != configured.kind.value:
                raise ValueError(
                    f"dataset encodes {feature!r} as "
                    f"{manifest.embedding_kinds.get(feature)!r}, "
                    f"but the model config requires {configured.kind.value!r}"
                )
        model_configs.extend(model_config.children)


def steps_per_epoch(manifest: DatasetManifest, batch_size: int) -> int:
    """Count actual iterable-dataset batches, including shard boundaries."""

    return sum(math.ceil(shard.rows / batch_size) for shard in manifest.train_shards)


def make_loader(
    config: ExperimentConfig,
    manifest_path: Path,
    split: str,
    *,
    shuffle: bool,
    pin_memory: bool,
    num_workers: int | None = None,
) -> DataLoader:
    dataset = ParquetBatchDataset(
        manifest_path,
        split,
        config.training.batch_size,
        shuffle=shuffle,
        seed=config.training.seed,
    )
    generator = torch.Generator().manual_seed(config.training.seed)
    return DataLoader(
        dataset,
        batch_size=None,
        num_workers=config.training.num_workers if num_workers is None else num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
        pin_memory=pin_memory,
        persistent_workers=False,
    )


class OptimizationLoop:
    """The single batch-level optimization implementation used by every trainer."""

    def __init__(
        self,
        config: ExperimentConfig,
        manifest: DatasetManifest,
        manifest_path: Path,
        *,
        epochs: int,
        run_id: str,
        store: RunStore,
        writer: SummaryWriter | None,
    ) -> None:
        self.config = config
        self.manifest = manifest
        self.manifest_path = manifest_path
        self.run_id = run_id
        self.store = store
        self.writer = writer
        self.device = select_device(config.training.device)
        validate_weight_budget(
            config.model,
            manifest,
            config.deployment.max_weight_bytes,
        )
        self.model = create_model(config.model, manifest, seed=config.training.seed)
        enforce_weight_budget(self.model, config.deployment.max_weight_bytes)
        self.model.to(self.device)
        self.objective = CTRObjective(config.objective).to(self.device)
        self.train_loader = make_loader(
            config,
            manifest_path,
            "train",
            shuffle=True,
            pin_memory=self.device.type == "cuda",
        )
        self.steps_per_epoch = steps_per_epoch(manifest, config.training.batch_size)
        self.total_steps = self.steps_per_epoch * epochs
        self.optimizers = build_optimizer_plan(
            self.model,
            config.training.optimizer,
            total_steps=self.total_steps,
        )
        self.runtime_model: CTRModel = self.model
        if config.training.compile_model and hasattr(torch, "compile"):
            self.runtime_model = cast(CTRModel, torch.compile(self.model))
        self.amp_dtype = torch.float16 if config.training.amp_dtype == "float16" else torch.bfloat16
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=config.training.amp and self.device.type == "cuda",
        )
        self.global_step = 0

    def train_epoch(self, epoch: int) -> None:
        dataset = self.train_loader.dataset
        if isinstance(dataset, ParquetBatchDataset):
            dataset.set_epoch(epoch)
        self.runtime_model.train()
        for batch in self.train_loader:
            if batch.labels is None:
                raise ValueError("training batch has no labels")
            moved = batch.to(self.device, non_blocking=True)
            self.optimizers.zero_grad()
            with torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.config.training.amp and self.device.type == "cuda",
            ):
                output = self.runtime_model(moved)
                losses = self.objective(output, moved.labels)
            if not torch.isfinite(losses.total):
                raise FloatingPointError(f"nonfinite loss at step {self.global_step}")
            scaled_loss = cast(torch.Tensor, self.scaler.scale(losses.total))
            scaled_loss.backward()
            for optimizer in self.optimizers.optimizers:
                self.scaler.unscale_(optimizer)
            gradient_norm: torch.Tensor | None = None
            if self.config.training.gradient_clip is not None:
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.gradient_clip,
                )
            coherent_check = len(self.optimizers.optimizers) > 1 or not self.scaler.is_enabled()
            finite_gradients = (
                bool(torch.isfinite(gradient_norm))
                if coherent_check and gradient_norm is not None
                else self._gradients_are_finite(self.model)
                if coherent_check
                else True
            )
            if not finite_gradients and not self.scaler.is_enabled():
                raise FloatingPointError(f"nonfinite gradients at step {self.global_step}")
            previous_scale = self.scaler.get_scale()
            if finite_gradients:
                for optimizer in self.optimizers.optimizers:
                    self.scaler.step(optimizer)
            self.scaler.update()
            successful = finite_gradients and self.scaler.get_scale() >= previous_scale
            if successful:
                self.optimizers.step_schedulers()
                self.model.post_step()
            self.global_step += 1
            if self.global_step % self.config.training.log_every_steps == 0:
                metrics = {**losses.scalars(), **self.optimizers.learning_rates()}
                self.store.log_metrics(
                    self.run_id,
                    step=self.global_step,
                    split="train",
                    metrics=metrics,
                )
                if self.writer is not None:
                    for name, value in metrics.items():
                        self.writer.add_scalar(f"train/{name}", value, self.global_step)

    @staticmethod
    def _gradients_are_finite(model: CTRModel) -> bool:
        checks = [
            torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        return not checks or bool(torch.stack(checks).all())


def save_resume(
    path: Path,
    *,
    epoch: int,
    best_loss: float,
    best_epoch: int,
    best_state: dict[str, torch.Tensor] | None,
    stale_epochs: int,
    loop: OptimizationLoop,
    config_sha256: str,
    manifest_sha256: str,
) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "schema_version": 4,
            "config_sha256": config_sha256,
            "manifest_sha256": manifest_sha256,
            "epoch": epoch,
            "global_step": loop.global_step,
            "best_loss": best_loss,
            "best_epoch": best_epoch,
            "best_state": best_state,
            "stale_epochs": stale_epochs,
            "model": loop.model.state_dict(),
            "optimizers": [optimizer.state_dict() for optimizer in loop.optimizers.optimizers],
            "schedulers": [
                scheduler.state_dict() if scheduler is not None else None
                for scheduler in loop.optimizers.schedulers
            ],
            "scaler": loop.scaler.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        },
        temporary,
    )
    temporary.replace(path)


def load_resume(
    path: Path,
    loop: OptimizationLoop,
    *,
    expected_config_sha256: str,
    expected_manifest_sha256: str,
) -> tuple[int, float, int, dict[str, torch.Tensor] | None, int]:
    checkpoint: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("schema_version") != 4:
        raise ValueError("unsupported resume checkpoint schema")
    if checkpoint.get("config_sha256") != expected_config_sha256:
        raise ValueError("resume checkpoint configuration differs from the active run")
    if checkpoint.get("manifest_sha256") != expected_manifest_sha256:
        raise ValueError("resume checkpoint dataset differs from the active manifest")
    if len(checkpoint["optimizers"]) != len(loop.optimizers.optimizers):
        raise ValueError("resume optimizer plan differs from the active configuration")
    loop.model.load_state_dict(checkpoint["model"], strict=True)
    for optimizer, state in zip(
        loop.optimizers.optimizers,
        checkpoint["optimizers"],
        strict=True,
    ):
        optimizer.load_state_dict(state)
    for scheduler, state in zip(
        loop.optimizers.schedulers,
        checkpoint["schedulers"],
        strict=True,
    ):
        if scheduler is None and state is not None:
            raise ValueError("resume scheduler plan differs from configuration")
        if scheduler is not None and state is not None:
            scheduler.load_state_dict(state)
    loop.scaler.load_state_dict(checkpoint["scaler"])
    torch.set_rng_state(checkpoint["torch_rng"])
    if torch.cuda.is_available() and checkpoint["cuda_rng"]:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
    loop.global_step = int(checkpoint["global_step"])
    return (
        int(checkpoint["epoch"]) + 1,
        float(checkpoint["best_loss"]),
        int(checkpoint["best_epoch"]),
        checkpoint["best_state"],
        int(checkpoint["stale_epochs"]),
    )
