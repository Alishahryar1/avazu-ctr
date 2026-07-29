"""Processed dataset manifests and integrity verification."""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from avazu_ctr.config.schema import FeatureMode

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class StrictManifestModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _validate_artifact_path(value: str) -> None:
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
        raise ValueError("manifest artifact paths must be normalized relative POSIX paths")


class DatasetPurpose(StrEnum):
    EVALUATION = "evaluation"
    PRODUCTION = "production"


class DatasetSplit(StrEnum):
    TRAINING = "training"
    VALIDATION = "validation"
    PREDICTION = "prediction"


class FeatureLane(StrEnum):
    CATEGORICAL = "categorical"
    NUMERICAL = "numerical"


class FeatureFamily(StrEnum):
    RAW = "raw"
    TIME = "time"
    CONTEXT = "context"
    CROSS = "cross"
    BUCKET = "bucket"
    FREQUENCY = "frequency"
    DISTINCT_COUNT = "distinct_count"
    HISTORY = "history"
    TARGET = "target"


class FittedTableKind(StrEnum):
    VOCABULARY = "vocabulary"
    COVARIATE_LOOKUP = "covariate_lookup"
    TARGET_ENCODING = "target_encoding"
    TEMPORAL_TARGET_ENCODING = "temporal_target_encoding"


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

    @model_validator(mode="after")
    def validate_path(self) -> Self:
        _validate_artifact_path(self.path)
        return self


class FittedTableManifest(StrictManifestModel):
    kind: FittedTableKind
    join_keys: tuple[str, ...] = Field(min_length=1)
    outputs: tuple[str, ...] = Field(min_length=1)
    path: str
    rows: int = Field(ge=0)
    sha256: Sha256
    sources: tuple[DatasetSplit, ...] = Field(min_length=1)
    uses_labels: bool

    @model_validator(mode="after")
    def validate_shape(self) -> Self:
        if len(set(self.join_keys)) != len(self.join_keys):
            raise ValueError("fitted-table join keys must be unique")
        if len(set(self.outputs)) != len(self.outputs):
            raise ValueError("fitted-table outputs must be unique")
        if set(self.join_keys).intersection(self.outputs):
            raise ValueError("fitted-table join keys and outputs must be disjoint")
        _validate_artifact_path(self.path)

        primary = self.join_keys[0]
        expected: tuple[str, ...] | None = None
        if self.kind is FittedTableKind.VOCABULARY:
            if len(self.join_keys) != 1:
                raise ValueError("vocabulary tables require exactly one join key")
            expected = (f"{primary}__id",)
        elif self.kind is FittedTableKind.COVARIATE_LOOKUP:
            if len(self.join_keys) != 1:
                raise ValueError("covariate lookup tables require exactly one join key")
            frequency_output = f"{primary}__frequency"
            if any(
                output != frequency_output and not output.endswith("__raw")
                for output in self.outputs
            ):
                raise ValueError(
                    "covariate lookup outputs must be the key frequency or raw aggregates"
                )
        elif self.kind is FittedTableKind.TARGET_ENCODING:
            if len(self.join_keys) != 1:
                raise ValueError("target-encoding tables require exactly one join key")
            expected = ("_positive", "_count")
        elif self.kind is FittedTableKind.TEMPORAL_TARGET_ENCODING:
            if self.join_keys != (primary, "_te_block"):
                raise ValueError(
                    "temporal target-encoding tables require a feature and _te_block join key"
                )
            expected = (
                "_previous_positive",
                "_previous_count",
                "_prior_count",
                "_block_prior",
            )
        if expected is not None and self.outputs != expected:
            raise ValueError(f"{self.kind.value} table outputs must be {expected}")
        return self


class CategoricalEncodingContract(StrictManifestModel):
    vocabulary_sources: tuple[DatasetSplit, ...] = (DatasetSplit.TRAINING,)
    unknown_id: Literal[0] = 0
    unknown_embedding: Literal["zero"] = "zero"

    @model_validator(mode="after")
    def validate_vocabulary_sources(self) -> Self:
        if self.vocabulary_sources != (DatasetSplit.TRAINING,):
            raise ValueError("categorical vocabularies must be fitted from training only")
        return self


class FeatureDefinition(StrictManifestModel):
    name: str
    lane: FeatureLane
    family: FeatureFamily
    inputs: tuple[str, ...] = Field(min_length=1)
    uses_labels: bool = False


class OovStatistic(StrictManifestModel):
    rows: int = Field(gt=0)
    unknown_rows: int = Field(ge=0)
    rate: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_rate(self) -> Self:
        if self.unknown_rows > self.rows:
            raise ValueError("unknown rows cannot exceed total rows")
        expected = self.unknown_rows / self.rows
        if abs(self.rate - expected) > 1e-12:
            raise ValueError("OOV rate does not match its row counts")
        return self


class SplitDiagnostics(StrictManifestModel):
    rows: int = Field(gt=0)
    categorical_oov: dict[str, OovStatistic]


class DatasetManifest(StrictManifestModel):
    schema_version: Literal[6] = 6
    name: str
    purpose: DatasetPurpose
    feature_mode: FeatureMode
    categorical_encoding: CategoricalEncodingContract
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
    features: tuple[FeatureDefinition, ...]
    diagnostics: dict[DatasetSplit, SplitDiagnostics]
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

        categorical_features = tuple(
            feature.name for feature in self.features if feature.lane is FeatureLane.CATEGORICAL
        )
        numerical_features = tuple(
            feature.name for feature in self.features if feature.lane is FeatureLane.NUMERICAL
        )
        if categorical_features != self.categorical_columns:
            raise ValueError("categorical feature definitions do not match the ordered columns")
        if numerical_features != self.numerical_columns:
            raise ValueError("numerical feature definitions do not match the ordered columns")
        names = tuple(feature.name for feature in self.features)
        if len(set(names)) != len(names):
            raise ValueError("feature definitions contain duplicate names")
        if set(self.cardinalities) != set(self.categorical_columns):
            raise ValueError("cardinalities must cover exactly the categorical columns")
        if any(cardinality <= 0 for cardinality in self.cardinalities.values()):
            raise ValueError("categorical cardinalities must be positive")
        if set(self.embedding_kinds) != set(self.categorical_columns):
            raise ValueError("embedding kinds must cover exactly the categorical columns")
        if not set(self.embedding_kinds.values()).issubset({"hash", "standard"}):
            raise ValueError("embedding kinds must be hash or standard")
        if set(self.categorical_columns).intersection(self.numerical_columns):
            raise ValueError("categorical and numerical columns must be disjoint")

        scoring_split = (
            DatasetSplit.VALIDATION
            if self.purpose is DatasetPurpose.EVALUATION
            else DatasetSplit.PREDICTION
        )
        expected_splits = {DatasetSplit.TRAINING, scoring_split}
        if set(self.diagnostics) != expected_splits:
            raise ValueError("dataset diagnostics do not match the manifest purpose")
        expected_rows = {
            DatasetSplit.TRAINING: self.train_rows,
            scoring_split: (
                self.validation_rows
                if self.purpose is DatasetPurpose.EVALUATION
                else self.test_rows
            ),
        }
        vocabulary_features = {
            feature for feature, kind in self.embedding_kinds.items() if kind == "standard"
        }
        for split, diagnostics in self.diagnostics.items():
            if diagnostics.rows != expected_rows[split]:
                raise ValueError(f"{split.value} diagnostics have an incorrect row count")
            if set(diagnostics.categorical_oov) != vocabulary_features:
                raise ValueError(f"{split.value} diagnostics must cover every vocabulary feature")
            if any(item.rows != diagnostics.rows for item in diagnostics.categorical_oov.values()):
                raise ValueError(f"{split.value} OOV diagnostics have inconsistent row counts")

        transductive_sources = (
            (DatasetSplit.TRAINING,)
            if self.feature_mode is FeatureMode.INDUCTIVE
            else (DatasetSplit.TRAINING, scoring_split)
        )
        label_kinds = {
            FittedTableKind.TARGET_ENCODING,
            FittedTableKind.TEMPORAL_TARGET_ENCODING,
        }
        expected_contracts: dict[
            tuple[FittedTableKind, tuple[str, ...]],
            tuple[str, ...],
        ] = {
            (FittedTableKind.VOCABULARY, (feature,)): (f"{feature}__id",)
            for feature, kind in self.embedding_kinds.items()
            if kind == "standard"
        }
        covariate_outputs: dict[str, list[str]] = {}
        target_features: dict[str, None] = {}
        for feature in self.features:
            if feature.family is FeatureFamily.FREQUENCY:
                key = feature.inputs[0]
                covariate_outputs.setdefault(key, []).append(f"{key}__frequency")
            elif feature.family is FeatureFamily.DISTINCT_COUNT:
                key = feature.inputs[0]
                covariate_outputs.setdefault(key, []).append(f"{feature.name}__raw")
            elif feature.family is FeatureFamily.TARGET:
                target_features.setdefault(feature.inputs[0], None)
        expected_contracts.update(
            {
                (FittedTableKind.COVARIATE_LOOKUP, (key,)): tuple(outputs)
                for key, outputs in covariate_outputs.items()
            }
        )
        for feature in target_features:
            expected_contracts[(FittedTableKind.TARGET_ENCODING, (feature,))] = (
                "_positive",
                "_count",
            )
            expected_contracts[
                (
                    FittedTableKind.TEMPORAL_TARGET_ENCODING,
                    (feature, "_te_block"),
                )
            ] = (
                "_previous_positive",
                "_previous_count",
                "_prior_count",
                "_block_prior",
            )

        table_paths: set[str] = set()
        table_contracts: set[tuple[FittedTableKind, tuple[str, ...]]] = set()
        actual_contracts: dict[
            tuple[FittedTableKind, tuple[str, ...]],
            tuple[str, ...],
        ] = {}
        for table in self.fitted_tables:
            if table.path in table_paths:
                raise ValueError(f"fitted-table path {table.path!r} is repeated")
            table_paths.add(table.path)
            table_contract = (table.kind, table.join_keys)
            if table_contract in table_contracts:
                raise ValueError(
                    f"fitted-table contract {table.kind.value} {table.join_keys} is repeated"
                )
            table_contracts.add(table_contract)
            actual_contracts[table_contract] = table.outputs
            if len(set(table.sources)) != len(table.sources):
                raise ValueError(f"fitted table {table.path!r} repeats a source")
            if not set(table.sources).issubset(expected_splits):
                raise ValueError(f"fitted table {table.path!r} has an invalid source")
            if table.uses_labels != (table.kind in label_kinds):
                raise ValueError(f"fitted table {table.path!r} has inconsistent label provenance")
            expected_sources = (
                self.categorical_encoding.vocabulary_sources
                if table.kind is FittedTableKind.VOCABULARY
                else (DatasetSplit.TRAINING,)
                if table.kind in label_kinds
                else transductive_sources
            )
            if table.sources != expected_sources:
                raise ValueError(
                    f"fitted table {table.path!r} has sources {table.sources}, "
                    f"expected {expected_sources}"
                )
        if actual_contracts != expected_contracts:
            raise ValueError(
                "fitted tables do not exactly match the declared feature and embedding contract"
            )
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
                "feature_mode": self.feature_mode,
                "categorical_encoding": self.categorical_encoding.model_dump(mode="json"),
                "features": [feature.model_dump(mode="json") for feature in self.features],
                "fitted_tables": [
                    {
                        "kind": table.kind,
                        "join_keys": table.join_keys,
                        "outputs": table.outputs,
                        "sha256": table.sha256,
                        "sources": table.sources,
                        "uses_labels": table.uses_labels,
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
            "schema_version": 4,
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
