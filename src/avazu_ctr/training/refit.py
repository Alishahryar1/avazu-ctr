"""Fixed-budget production refit on every labelled row."""

from __future__ import annotations

import traceback
from dataclasses import dataclass
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from avazu_ctr.config.loader import resolved_config, write_resolved_config
from avazu_ctr.config.schema import ExperimentConfig
from avazu_ctr.data.manifest import (
    DatasetManifest,
    DatasetPurpose,
    HourRange,
    load_manifest,
    sha256_file,
    sha256_json,
)
from avazu_ctr.models.base import CTRModel
from avazu_ctr.models.state import state_dict_sha256
from avazu_ctr.tracking.evidence import LoadedSelection, load_selection
from avazu_ctr.tracking.store import RunStore
from avazu_ctr.training.engine import (
    OptimizationLoop,
    steps_per_epoch,
    validate_feature_contract,
)
from avazu_ctr.training.seed import seed_everything


@dataclass(slots=True)
class RefitResult:
    run_id: str
    model: CTRModel
    epochs: int
    steps: int
    manifest: DatasetManifest
    manifest_sha256: str
    selection_id: str
    selection_sha256: str
    model_state_sha256: str


class ProductionRefitter:
    def __init__(
        self,
        manifest_path: str | Path,
        selection_path: str | Path,
        *,
        store: RunStore | None = None,
    ) -> None:
        self.selection: LoadedSelection = load_selection(selection_path)
        self.config: ExperimentConfig = self.selection.evidence.confirmation.config
        self.manifest_path = Path(manifest_path)
        self.manifest = load_manifest(self.manifest_path, verify_shards=True)
        if self.manifest.purpose is not DatasetPurpose.PRODUCTION:
            raise ValueError("production refit requires a production dataset")
        self.store = store or RunStore(self.config.tracking.database)
        validate_feature_contract(self.config, self.manifest)
        active = self.store.active_selection()
        if (
            active is None
            or active["candidate_run_id"] != self.selection.evidence.selection_id
            or active["candidate_evidence_sha256"] != self.selection.evidence_sha256
        ):
            raise ValueError("production refit requires the active recorded selection")
        self._validate_provenance()

    def _validate_provenance(self) -> None:
        config_sha256 = sha256_json(resolved_config(self.config))
        if self.manifest.resolved_config_sha256 != config_sha256:
            raise ValueError("production data was not fitted from the selected configuration")
        holdout = self.selection.evidence.holdout
        expected_range = HourRange(
            start=holdout.training_range.start,
            end=holdout.validation_range.end,
        )
        if (
            self.manifest.labelled_source.sha256 != holdout.labelled_source_sha256
            or self.manifest.training_range != expected_range
        ):
            raise ValueError(
                "production data does not cover the selected holdout's complete labelled source"
            )

    def fit(self) -> RefitResult:
        epochs = self.selection.evidence.holdout.best_epoch + 1
        epoch_steps = steps_per_epoch(
            self.manifest,
            self.config.training.batch_size,
            self.config.training.num_workers,
        )
        manifest_sha256 = sha256_file(self.manifest_path)
        run_id = self.store.start_run(
            self.config,
            self.manifest,
            kind="production_refit",
            plan={
                "mode": "production_refit",
                "epochs": epochs,
                "steps_per_epoch": epoch_steps,
                "planned_steps": epoch_steps * epochs,
                "early_stopping": False,
                "validation": False,
                "manifest_sha256": manifest_sha256,
                "budget_source_run_id": self.selection.evidence.holdout.run_id,
                "budget_source_best_epoch": self.selection.evidence.holdout.best_epoch,
            },
            parent_run_id=self.selection.evidence.holdout.run_id,
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
            loop = OptimizationLoop(
                self.config,
                self.manifest,
                self.manifest_path,
                epochs=epochs,
                run_id=run_id,
                store=self.store,
                writer=writer,
            )
            for epoch in range(epochs):
                loop.train_epoch(epoch)
            if loop.global_step != loop.total_steps:
                raise RuntimeError(
                    f"refit executed {loop.global_step} steps; expected {loop.total_steps}"
                )
            loop.model.to("cpu")
            result = RefitResult(
                run_id=run_id,
                model=loop.model,
                epochs=epochs,
                steps=loop.global_step,
                manifest=self.manifest,
                manifest_sha256=manifest_sha256,
                selection_id=self.selection.evidence.selection_id,
                selection_sha256=self.selection.evidence_sha256,
                model_state_sha256=state_dict_sha256(loop.model.state_dict()),
            )
            self.store.finish_run(
                run_id,
                status="completed",
                summary={
                    "epochs_completed": result.epochs,
                    "steps_completed": result.steps,
                    "selection_id": result.selection_id,
                    "selection_sha256": result.selection_sha256,
                    "model_state_sha256": result.model_state_sha256,
                },
            )
            return result
        except Exception:
            self.store.finish_run(run_id, status="failed", error=traceback.format_exc())
            raise
        finally:
            if writer is not None:
                writer.close()
