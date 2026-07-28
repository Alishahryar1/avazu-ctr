"""Typed, checksummed evidence for model selection."""

from __future__ import annotations

import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, Self

import numpy as np
from pydantic import Field, model_validator

from avazu_ctr.config.loader import resolved_config
from avazu_ctr.config.schema import ExperimentConfig, StrictModel
from avazu_ctr.data.manifest import HourRange, Sha256, sha256_file, sha256_json

FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]


class FoldEvidence(StrictModel):
    window: str
    run_id: str
    manifest_sha256: Sha256
    labelled_source_sha256: Sha256
    training_range: HourRange
    validation_range: HourRange
    population_sha256: Sha256
    rows: Annotated[int, Field(gt=0)]
    logloss: FiniteFloat


class ConfirmationEvidence(StrictModel):
    schema_version: Literal[4] = 4
    config: ExperimentConfig
    config_sha256: Sha256
    folds: tuple[FoldEvidence, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_confirmation(self) -> Self:
        actual_config_sha256 = sha256_json(resolved_config(self.config))
        if self.config_sha256 != actual_config_sha256:
            raise ValueError("confirmation configuration checksum mismatch")
        expected_windows = tuple(
            f"walk_forward_{index}" for index in range(self.config.data.split.walk_forward_folds)
        )
        windows = tuple(fold.window for fold in self.folds)
        if windows != expected_windows:
            raise ValueError(
                f"confirmation folds must be ordered exactly as {list(expected_windows)}"
            )
        if any(fold.training_range.end != fold.validation_range.start for fold in self.folds):
            raise ValueError("fold training and validation ranges must be contiguous")
        fold_hours = self.config.data.split.fold_hours
        for index, fold in enumerate(self.folds):
            if fold.validation_range.end - fold.validation_range.start != fold_hours:
                raise ValueError("confirmation fold width differs from the configured protocol")
            if index:
                previous = self.folds[index - 1]
                if (
                    fold.training_range.start != previous.training_range.start
                    or fold.training_range.end != previous.validation_range.end
                    or fold.validation_range.start != previous.validation_range.end
                ):
                    raise ValueError("confirmation folds are not expanding walk-forward windows")
        if len({fold.labelled_source_sha256 for fold in self.folds}) != 1:
            raise ValueError("confirmation folds must come from one labelled source")
        if len({fold.population_sha256 for fold in self.folds}) != len(self.folds):
            raise ValueError("confirmation folds must contain distinct validation populations")
        return self

    @property
    def mean_logloss(self) -> float:
        return sum(fold.logloss for fold in self.folds) / len(self.folds)


class HoldoutEvidence(StrictModel):
    window: Literal["final_holdout"] = "final_holdout"
    run_id: str
    manifest_sha256: Sha256
    labelled_source_sha256: Sha256
    training_range: HourRange
    validation_range: HourRange
    population_sha256: Sha256
    rows: Annotated[int, Field(gt=0)]
    best_epoch: Annotated[int, Field(ge=0)]
    metrics: dict[str, FiniteFloat]

    @model_validator(mode="after")
    def validate_metrics(self) -> Self:
        if "logloss" not in self.metrics:
            raise ValueError("holdout evidence requires logloss")
        if self.training_range.end != self.validation_range.start:
            raise ValueError("holdout training and validation ranges must be contiguous")
        return self


class SelectionEvidence(StrictModel):
    schema_version: Literal[4] = 4
    selection_id: str
    confirmation: ConfirmationEvidence
    holdout: HoldoutEvidence
    row_losses_path: Literal["holdout-row-losses.npy"] = "holdout-row-losses.npy"
    row_losses_sha256: Sha256

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        if self.selection_id != self.holdout.run_id:
            raise ValueError("selection ID must equal the final-holdout run ID")
        sources = {
            self.holdout.labelled_source_sha256,
            *(fold.labelled_source_sha256 for fold in self.confirmation.folds),
        }
        if len(sources) != 1:
            raise ValueError("selection evidence must come from one labelled source")
        final_fold = self.confirmation.folds[-1]
        holdout_hours = self.confirmation.config.data.split.holdout_hours
        if (
            self.holdout.training_range.start != final_fold.training_range.start
            or self.holdout.training_range.end != final_fold.validation_range.end
            or self.holdout.validation_range.start != final_fold.validation_range.end
            or self.holdout.validation_range.end - self.holdout.validation_range.start
            != holdout_hours
        ):
            raise ValueError("final holdout does not follow the confirmed walk-forward windows")
        return self


@dataclass(frozen=True, slots=True)
class LoadedSelection:
    root: Path
    evidence: SelectionEvidence
    row_losses: np.ndarray
    evidence_sha256: str


def write_confirmation(evidence: ConfirmationEvidence, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}-{uuid.uuid4().hex}.tmp"
    temporary.write_text(evidence.model_dump_json(indent=2), encoding="utf-8")
    temporary.replace(output)
    return output


def load_confirmation(path: str | Path) -> ConfirmationEvidence:
    return ConfirmationEvidence.model_validate_json(Path(path).read_text(encoding="utf-8"))


def write_selection(
    confirmation: ConfirmationEvidence,
    holdout: HoldoutEvidence,
    row_losses: np.ndarray,
    output_dir: str | Path,
) -> Path:
    losses = np.asarray(row_losses)
    if losses.ndim != 1 or losses.size != holdout.rows:
        raise ValueError("holdout row losses must be one-dimensional and match the row count")
    if not np.issubdtype(losses.dtype, np.floating) or not np.isfinite(losses).all():
        raise ValueError("holdout row losses must be finite floating-point values")

    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    try:
        losses_path = staging / "holdout-row-losses.npy"
        np.save(losses_path, losses, allow_pickle=False)
        evidence = SelectionEvidence(
            selection_id=holdout.run_id,
            confirmation=confirmation,
            holdout=holdout,
            row_losses_sha256=sha256_file(losses_path),
        )
        (staging / "selection.json").write_text(
            evidence.model_dump_json(indent=2),
            encoding="utf-8",
        )
        load_selection(staging)
        staging.replace(output)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return output / "selection.json"


def load_selection(path: str | Path) -> LoadedSelection:
    root = Path(path)
    if root.is_file():
        root = root.parent
    evidence_path = root / "selection.json"
    evidence = SelectionEvidence.model_validate_json(evidence_path.read_text(encoding="utf-8"))
    relative_losses = Path(evidence.row_losses_path)
    if relative_losses.is_absolute() or ".." in relative_losses.parts:
        raise ValueError("selection row-loss path must remain inside the evidence directory")
    losses_path = root / relative_losses
    if not losses_path.is_file() or sha256_file(losses_path) != evidence.row_losses_sha256:
        raise ValueError("selection row-loss checksum mismatch")
    losses = np.load(losses_path, allow_pickle=False)
    if (
        losses.ndim != 1
        or losses.size != evidence.holdout.rows
        or not np.issubdtype(losses.dtype, np.floating)
        or not np.isfinite(losses).all()
    ):
        raise ValueError("selection row-loss payload does not match its evidence")
    return LoadedSelection(
        root=root,
        evidence=evidence,
        row_losses=losses,
        evidence_sha256=sha256_file(evidence_path),
    )
