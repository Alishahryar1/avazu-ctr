"""Processed dataset manifests and integrity verification."""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class StrictManifestModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DatasetPurpose(StrEnum):
    EVALUATION = "evaluation"
    PRODUCTION = "production"


class HourRange(StrictManifestModel):
    start: int
    end: int

    @model_validator(mode="after")
    def validate_order(self) -> Self:
        if self.end <= self.start:
            raise ValueError("hour range end must be greater than its start")
        return self


class RawSource(StrictManifestModel):
    path: str
    sha256: Sha256


class ShardManifest(StrictManifestModel):
    path: str
    rows: int = Field(gt=0)
    sha256: Sha256


class FittedTableManifest(StrictManifestModel):
    feature: str
    kind: str
    path: str
    rows: int = Field(ge=0)
    sha256: Sha256


class DatasetManifest(StrictManifestModel):
    schema_version: Literal[3] = 3
    name: str
    purpose: DatasetPurpose
    labelled_source: RawSource
    prediction_source: RawSource | None = None
    training_range: HourRange
    validation_range: HourRange | None = None
    training_population_sha256: Sha256
    validation_population_sha256: Sha256 | None = None
    test_population_sha256: Sha256 | None = None
    categorical_columns: tuple[str, ...]
    numerical_columns: tuple[str, ...]
    cardinalities: dict[str, int]
    embedding_kinds: dict[str, str]
    train_shards: tuple[ShardManifest, ...] = Field(min_length=1)
    validation_shards: tuple[ShardManifest, ...] = ()
    test_shards: tuple[ShardManifest, ...] = ()
    fitted_tables: tuple[FittedTableManifest, ...] = ()
    resolved_config_sha256: Sha256
    feature_config_sha256: Sha256
    package_lock_sha256: Sha256 | None = None

    @model_validator(mode="after")
    def validate_purpose(self) -> Self:
        if self.purpose is DatasetPurpose.EVALUATION:
            if self.validation_range is None or not self.validation_shards:
                raise ValueError("evaluation datasets require a validation range and shards")
            if self.validation_population_sha256 is None:
                raise ValueError("evaluation datasets require a validation population hash")
            if self.prediction_source is not None or self.test_shards:
                raise ValueError("evaluation datasets cannot contain prediction data")
            if self.test_population_sha256 is not None:
                raise ValueError("evaluation datasets cannot contain a test population hash")
        else:
            if (
                self.validation_range is not None
                or self.validation_shards
                or self.validation_population_sha256 is not None
            ):
                raise ValueError("production datasets cannot contain validation data")
            if self.prediction_source is None or not self.test_shards:
                raise ValueError("production datasets require a prediction source and test shards")
            if self.test_population_sha256 is None:
                raise ValueError("production datasets require a test population hash")
        return self

    @property
    def train_rows(self) -> int:
        return sum(shard.rows for shard in self.train_shards)

    @property
    def validation_rows(self) -> int:
        return sum(shard.rows for shard in self.validation_shards)

    @property
    def test_rows(self) -> int:
        return sum(shard.rows for shard in self.test_shards)

    @property
    def feature_contract_sha256(self) -> str:
        return sha256_json(
            {
                "categorical_columns": self.categorical_columns,
                "numerical_columns": self.numerical_columns,
                "cardinalities": self.cardinalities,
                "embedding_kinds": self.embedding_kinds,
                "fitted_tables": [
                    {
                        "feature": table.feature,
                        "kind": table.kind,
                        "sha256": table.sha256,
                    }
                    for table in self.fitted_tables
                ],
                "feature_config_sha256": self.feature_config_sha256,
            }
        )


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def population_sha256(
    source_sha256: str,
    *,
    split: str,
    hour_range: HourRange | None,
) -> str:
    """Identify an ordered raw population independently of fitted features."""

    return sha256_json(
        {
            "schema_version": 3,
            "source_sha256": source_sha256,
            "split": split,
            "hour_range": hour_range.model_dump() if hour_range is not None else None,
        }
    )


def write_manifest(manifest: DatasetManifest, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")


def load_manifest(path: str | Path, *, verify_shards: bool = False) -> DatasetManifest:
    manifest_path = Path(path)
    manifest = DatasetManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    if verify_shards:
        root = manifest_path.parent
        for artifact in (
            *manifest.train_shards,
            *manifest.validation_shards,
            *manifest.test_shards,
            *manifest.fitted_tables,
        ):
            artifact_path = root / artifact.path
            if not artifact_path.is_file():
                raise ValueError(f"missing manifest artifact {artifact_path}")
            if sha256_file(artifact_path) != artifact.sha256:
                raise ValueError(f"checksum mismatch for {artifact_path}")
    return manifest
