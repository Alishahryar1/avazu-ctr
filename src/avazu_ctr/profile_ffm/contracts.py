"""Checksummed contracts for profile FFM preparation and fitting."""

from __future__ import annotations

import hashlib
from enum import StrEnum
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from avazu_ctr.profile_ffm.config import (
    NativeExecutor,
    ProfileFFMConfig,
    Sha256,
    profile_ffm_config_sha256,
)

PREPARED_ARTIFACT_NAMES = frozenset(
    {
        "train_app_profile",
        "score_app_profile",
        "train_site_profile",
        "score_site_profile",
        "train_app_history",
        "score_app_history",
        "score_app_selector",
        "score_site_selector",
    }
)
PREDICTION_NAMES = frozenset(
    {
        "app_profile",
        "site_profile",
        "site_cold_publisher",
        "app_causal_history",
    }
)
RUN_LOG_NAMES = frozenset(
    f"{prediction}_{stream}" for prediction in PREDICTION_NAMES for stream in ("stdout", "stderr")
)


class StrictManifestModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Inventory(StrEnum):
    APP = "app"
    SITE = "site"


class SourceSplit(StrEnum):
    TRAINING = "training"
    SCORING = "scoring"


def _validate_relative_path(value: str) -> None:
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if (
        not value
        or value != posix.as_posix()
        or "\\" in value
        or posix.is_absolute()
        or windows.is_absolute()
        or ".." in posix.parts
        or ".." in windows.parts
    ):
        raise ValueError("artifact paths must be normalized relative POSIX paths")


class SourceArtifact(StrictManifestModel):
    path: str
    rows: int = Field(gt=0)
    bytes: int = Field(gt=0)
    sha256: Sha256


class FileArtifact(StrictManifestModel):
    path: str
    rows: int = Field(ge=0)
    bytes: int = Field(ge=0)
    sha256: Sha256

    @model_validator(mode="after")
    def validate_path(self) -> Self:
        _validate_relative_path(self.path)
        return self


class PopulationRows(StrictManifestModel):
    training: int = Field(gt=0)
    scoring: int = Field(gt=0)
    training_app: int = Field(gt=0)
    scoring_app: int = Field(gt=0)
    training_site: int = Field(gt=0)
    scoring_site: int = Field(gt=0)
    scoring_app_proxy: int = Field(ge=0)
    scoring_nonempty_history: int = Field(ge=0)
    scoring_cold_site: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_partitions(self) -> Self:
        if self.training_app + self.training_site != self.training:
            raise ValueError("training inventory rows do not cover the population")
        if self.scoring_app + self.scoring_site != self.scoring:
            raise ValueError("scoring inventory rows do not cover the population")
        if self.scoring_app_proxy > self.scoring_app:
            raise ValueError("app selector row counts are inconsistent")
        if self.scoring_nonempty_history > self.scoring_app_proxy:
            raise ValueError("app selector row counts are inconsistent")
        if self.scoring_cold_site > self.scoring_site:
            raise ValueError("cold-site rows exceed the site population")
        return self


class ProfileCoverage(StrictManifestModel):
    users: int = Field(ge=0)
    publisher_id_edges: int = Field(ge=0)
    publisher_domain_edges: int = Field(ge=0)
    training_rows: int = Field(gt=0)
    training_profiled_rows: int = Field(ge=0)
    scoring_rows: int = Field(gt=0)
    scoring_profiled_rows: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_coverage(self) -> Self:
        if self.training_profiled_rows > self.training_rows:
            raise ValueError("profiled training rows exceed their population")
        if self.scoring_profiled_rows > self.scoring_rows:
            raise ValueError("profiled scoring rows exceed their population")
        return self


class PreparationManifest(StrictManifestModel):
    schema_version: Literal[1] = 1
    name: str
    config: ProfileFFMConfig
    config_sha256: Sha256
    sources: dict[SourceSplit, SourceArtifact]
    rows: PopulationRows
    profiles: dict[Inventory, ProfileCoverage]
    artifacts: dict[str, FileArtifact]

    @model_validator(mode="after")
    def validate_contract(self) -> Self:
        if self.name != self.config.name:
            raise ValueError("preparation manifest name does not match its configuration")
        if self.config_sha256 != profile_ffm_config_sha256(self.config):
            raise ValueError("preparation manifest configuration checksum is invalid")
        if set(self.sources) != {SourceSplit.TRAINING, SourceSplit.SCORING}:
            raise ValueError("preparation manifest requires training and scoring sources")
        if self.sources[SourceSplit.TRAINING].rows != self.rows.training:
            raise ValueError("training source rows do not match the prepared population")
        if self.sources[SourceSplit.SCORING].rows != self.rows.scoring:
            raise ValueError("scoring source rows do not match the prepared population")
        if (
            self.config.data.train_sha256 is not None
            and self.sources[SourceSplit.TRAINING].sha256 != self.config.data.train_sha256
        ):
            raise ValueError("training source checksum does not match the configuration")
        if (
            self.config.data.test_sha256 is not None
            and self.sources[SourceSplit.SCORING].sha256 != self.config.data.test_sha256
        ):
            raise ValueError("scoring source checksum does not match the configuration")
        configured_rows = self.config.data.expected_rows
        if configured_rows is not None and self.rows.model_dump() != configured_rows.model_dump():
            raise ValueError("prepared population does not match the configuration")
        if set(self.profiles) != {Inventory.APP, Inventory.SITE}:
            raise ValueError("preparation manifest requires app and site profile coverage")
        profile_rows = {
            Inventory.APP: (self.rows.training_app, self.rows.scoring_app),
            Inventory.SITE: (self.rows.training_site, self.rows.scoring_site),
        }
        for inventory, (training_rows, scoring_rows) in profile_rows.items():
            coverage = self.profiles[inventory]
            if coverage.training_rows != training_rows or coverage.scoring_rows != scoring_rows:
                raise ValueError(f"{inventory.value} profile coverage has inconsistent populations")
        if set(self.artifacts) != PREPARED_ARTIFACT_NAMES:
            raise ValueError("preparation manifest artifacts do not match the sparse contract")
        expected_rows = {
            "train_app_profile": self.rows.training_app,
            "score_app_profile": self.rows.scoring_app,
            "train_site_profile": self.rows.training_site,
            "score_site_profile": self.rows.scoring_site,
            "train_app_history": self.rows.training_app,
            "score_app_history": self.rows.scoring_app,
            "score_app_selector": self.rows.scoring_app,
            "score_site_selector": self.rows.scoring_site,
        }
        for name, rows in expected_rows.items():
            if self.artifacts[name].rows != rows:
                raise ValueError(f"{name} row count does not match its population")
        return self


class NativeSolverEvidence(StrictManifestModel):
    executor: NativeExecutor
    compiler_version: str = Field(min_length=1)
    source_sha256: Sha256
    binary_sha256: Sha256
    build_command: tuple[str, ...] = Field(min_length=1)


class CompositionMetrics(StrictManifestModel):
    rows: int = Field(gt=0)
    app_profile_rows: int = Field(ge=0)
    app_causal_history_rows: int = Field(ge=0)
    site_profile_rows: int = Field(ge=0)
    site_cold_publisher_rows: int = Field(ge=0)
    prediction_minimum: float = Field(gt=0.0, lt=1.0)
    prediction_maximum: float = Field(gt=0.0, lt=1.0)
    prediction_mean: float = Field(gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def validate_partition(self) -> Self:
        if (
            self.app_profile_rows
            + self.app_causal_history_rows
            + self.site_profile_rows
            + self.site_cold_publisher_rows
            != self.rows
        ):
            raise ValueError("prediction sources do not cover the composed submission")
        if self.prediction_minimum > self.prediction_maximum:
            raise ValueError("prediction range is inverted")
        if not self.prediction_minimum <= self.prediction_mean <= self.prediction_maximum:
            raise ValueError("prediction mean falls outside its range")
        return self


class ProfileFFMRunManifest(StrictManifestModel):
    schema_version: Literal[1] = 1
    name: str
    config: ProfileFFMConfig
    config_sha256: Sha256
    preparation_manifest_sha256: Sha256
    rows: PopulationRows
    solver: NativeSolverEvidence
    predictions: dict[str, FileArtifact]
    composition: CompositionMetrics
    submission: FileArtifact
    fit_commands: dict[str, tuple[str, ...]]
    logs: dict[str, FileArtifact]

    @model_validator(mode="after")
    def validate_contract(self) -> Self:
        if self.name != self.config.name:
            raise ValueError("run manifest name does not match its configuration")
        if self.config_sha256 != profile_ffm_config_sha256(self.config):
            raise ValueError("run manifest configuration checksum is invalid")
        if set(self.predictions) != PREDICTION_NAMES:
            raise ValueError("run manifest predictions do not match the fitting contract")
        expected_rows = {
            "app_profile": self.rows.scoring_app,
            "app_causal_history": self.rows.scoring_app,
            "site_profile": self.rows.scoring_site,
            "site_cold_publisher": self.rows.scoring_site,
        }
        for name, rows in expected_rows.items():
            if self.predictions[name].rows != rows:
                raise ValueError(f"{name} prediction rows do not match the run population")
        configured_rows = self.config.data.expected_rows
        if configured_rows is not None and self.rows.model_dump() != configured_rows.model_dump():
            raise ValueError("run population does not match the configuration")
        if (
            self.composition.app_profile_rows + self.composition.app_causal_history_rows
            != self.rows.scoring_app
            or self.composition.site_profile_rows + self.composition.site_cold_publisher_rows
            != self.rows.scoring_site
            or self.composition.rows != self.rows.scoring
        ):
            raise ValueError("composition does not match the run population")
        if set(self.fit_commands) != PREDICTION_NAMES:
            raise ValueError("run manifest commands do not match the fitting contract")
        if any(not command for command in self.fit_commands.values()):
            raise ValueError("run manifest fit commands cannot be empty")
        if set(self.logs) != RUN_LOG_NAMES:
            raise ValueError("run manifest logs do not match the fitting contract")
        if self.submission.rows != self.composition.rows:
            raise ValueError("submission rows do not match composition metrics")
        return self


def sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def source_artifact(
    path: Path,
    *,
    rows: int,
    sha256: str | None = None,
) -> SourceArtifact:
    return SourceArtifact(
        path=str(path.resolve()),
        rows=rows,
        bytes=path.stat().st_size,
        sha256=sha256 or sha256_file(path),
    )


def file_artifact(root: Path, path: Path, *, rows: int) -> FileArtifact:
    relative = path.relative_to(root).as_posix()
    return FileArtifact(
        path=relative,
        rows=rows,
        bytes=path.stat().st_size,
        sha256=sha256_file(path),
    )


def line_file_artifact(root: Path, path: Path) -> FileArtifact:
    with path.open("r", encoding="utf-8") as handle:
        rows = sum(1 for _ in handle)
    return file_artifact(root, path, rows=rows)


def write_preparation_manifest(manifest: PreparationManifest, path: Path) -> None:
    path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")


def load_preparation_manifest(
    path: str | Path,
    *,
    verify_artifacts: bool = False,
) -> PreparationManifest:
    manifest_path = Path(path)
    manifest = PreparationManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    if verify_artifacts:
        for artifact in manifest.artifacts.values():
            artifact_path = manifest_path.parent / artifact.path
            if not artifact_path.is_file():
                raise ValueError(f"missing preparation artifact {artifact_path}")
            if artifact_path.stat().st_size != artifact.bytes:
                raise ValueError(f"size mismatch for {artifact_path}")
            if sha256_file(artifact_path) != artifact.sha256:
                raise ValueError(f"checksum mismatch for {artifact_path}")
    return manifest


def write_run_manifest(manifest: ProfileFFMRunManifest, path: Path) -> None:
    path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
