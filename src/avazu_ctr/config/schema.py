"""Pydantic schemas for Avazu CTR experiments."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelKind(StrEnum):
    GATED_DCN = "gated_dcn"
    MULTIHEAD = "multihead"
    NORMALIZED_MULTIHEAD = "normalized_multihead"
    STEC = "stec"
    ENSEMBLE = "ensemble"


class Aggregation(StrEnum):
    MEAN = "mean"
    GATED = "gated"


class EmbeddingKind(StrEnum):
    STANDARD = "standard"
    HASH = "hash"


class SchedulerKind(StrEnum):
    NONE = "none"
    COSINE = "cosine"


class OptimizerKind(StrEnum):
    ADAMW = "adamw"
    ADAGRAD = "adagrad"
    FTRL = "ftrl"


AVAZU_CATEGORICAL_COLUMNS = (
    "C1",
    "banner_pos",
    "site_id",
    "site_domain",
    "site_category",
    "app_id",
    "app_domain",
    "app_category",
    "device_id",
    "device_ip",
    "device_model",
    "device_type",
    "device_conn_type",
    "C14",
    "C15",
    "C16",
    "C17",
    "C18",
    "C19",
    "C20",
    "C21",
)
ENGINEERED_CATEGORICAL_COLUMNS = ("hour_of_day", "day_of_week")


class EmbeddingConfig(StrictModel):
    kind: EmbeddingKind = EmbeddingKind.STANDARD
    dim: Annotated[int, Field(gt=0)] = 16
    buckets: Annotated[int, Field(gt=1)] = 100_000
    hashes: Annotated[int, Field(ge=1, le=8)] = 2


class SENetConfig(StrictModel):
    enabled: bool = True
    reduction_ratio: Annotated[int, Field(gt=0)] = 3
    activation: Literal["relu", "gelu", "silu"] = "gelu"


class BackboneConfig(StrictModel):
    senet: SENetConfig = SENetConfig()
    dcn_layers: Annotated[int, Field(ge=0, le=16)] = 4
    dcn_rank: Annotated[int | None, Field(gt=0)] = 32
    mlp_hidden: tuple[Annotated[int, Field(gt=0)], ...] = (256, 128)
    activation: Literal["relu", "gelu", "silu"] = "gelu"
    dropout: Annotated[float, Field(ge=0.0, lt=1.0)] = 0.1
    layer_norm: bool = True


class HeadConfig(StrictModel):
    hidden: tuple[Annotated[int, Field(gt=0)], ...] = (64,)
    dropout: Annotated[float, Field(ge=0.0, lt=1.0)] = 0.1


class ModelConfig(StrictModel):
    kind: ModelKind
    default_embedding: EmbeddingConfig = EmbeddingConfig()
    feature_embeddings: dict[str, EmbeddingConfig] = Field(default_factory=dict)
    backbone: BackboneConfig = BackboneConfig()
    heads: tuple[HeadConfig, ...] = (HeadConfig(), HeadConfig(), HeadConfig())
    aggregation: Aggregation = Aggregation.GATED
    feature_bagging: Annotated[float, Field(gt=0.0, le=1.0)] = 0.8
    stec_layers: Annotated[int, Field(ge=1, le=12)] = 2
    stec_heads: Annotated[int, Field(ge=1, le=16)] = 4
    children: tuple[ModelConfig, ...] = ()

    @model_validator(mode="after")
    def validate_shape(self) -> Self:
        if (
            self.kind in {ModelKind.MULTIHEAD, ModelKind.NORMALIZED_MULTIHEAD}
            and len(self.heads) < 2
        ):
            raise ValueError("multihead models require at least two heads")
        if self.kind is ModelKind.STEC:
            if not self.heads:
                raise ValueError("STEC requires one prediction-head configuration")
            dim = self.default_embedding.dim
            if dim % self.stec_heads:
                raise ValueError("STEC embedding dimension must be divisible by stec_heads")
        if self.kind is ModelKind.ENSEMBLE and not self.children:
            raise ValueError("ensemble requires at least one child model")
        if self.kind is not ModelKind.ENSEMBLE and self.children:
            raise ValueError("only ensemble models may define children")
        return self


class ObjectiveConfig(StrictModel):
    aggregate_weight: Annotated[float, Field(gt=0.0)] = 1.0
    auxiliary_weight: Annotated[float, Field(ge=0.0)] = 0.25
    diversity_weight: Annotated[float, Field(ge=0.0)] = 0.05
    child_weight: Annotated[float, Field(ge=0.0)] = 0.1
    correlation_epsilon: Annotated[float, Field(gt=0.0)] = 1e-6


class TemporalSplitConfig(StrictModel):
    holdout_hours: Annotated[int, Field(gt=0)] = 24
    fold_hours: Annotated[int, Field(gt=0)] = 24
    walk_forward_folds: Annotated[int, Field(ge=1)] = 3
    minimum_train_hours: Annotated[int, Field(gt=0)] = 24


class TargetEncodingConfig(StrictModel):
    enabled: bool = True
    columns: tuple[str, ...] = ("site_id", "app_id")
    blocks: Annotated[int, Field(ge=2)] = 5
    smoothing: Annotated[float, Field(gt=0.0)] = 20.0
    neutral_prior: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.5


class DataConfig(StrictModel):
    train_path: Path = Path("data/raw/train.gz")
    test_path: Path = Path("data/raw/test.gz")
    artifact_root: Path = Path("artifacts")
    shard_rows: Annotated[int, Field(gt=0)] = 1_000_000
    vocabulary_limit: Annotated[int, Field(gt=1)] = 1_000_000
    minimum_frequency: Annotated[int, Field(ge=1)] = 2
    categorical_columns: tuple[str, ...] = AVAZU_CATEGORICAL_COLUMNS
    count_columns: tuple[str, ...] = ("device_ip", "device_id", "app_id", "site_id")
    split: TemporalSplitConfig = TemporalSplitConfig()
    target_encoding: TargetEncodingConfig = TargetEncodingConfig()


class SchedulerConfig(StrictModel):
    kind: SchedulerKind = SchedulerKind.COSINE
    warmup_ratio: Annotated[float, Field(ge=0.0, lt=1.0)] = 0.05
    minimum_lr_ratio: Annotated[float, Field(ge=0.0, le=1.0)] = 0.01


class OptimizerConfig(StrictModel):
    kind: OptimizerKind = OptimizerKind.ADAMW
    learning_rate: Annotated[float, Field(gt=0.0)] = 1e-3
    weight_decay: Annotated[float, Field(ge=0.0)] = 1e-5
    beta1: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.9
    beta2: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.999
    ftrl_beta: Annotated[float, Field(ge=0.0)] = 1.0
    l1: Annotated[float, Field(ge=0.0)] = 0.0
    l2: Annotated[float, Field(ge=0.0)] = 0.0
    scheduler: SchedulerConfig = SchedulerConfig()


class OptimizerPlanConfig(StrictModel):
    dense: OptimizerConfig = OptimizerConfig()
    embeddings: OptimizerConfig | None = None


class TrainingConfig(StrictModel):
    seed: int = 42
    epochs: Annotated[int, Field(gt=0)] = 10
    batch_size: Annotated[int, Field(gt=0)] = 4096
    num_workers: Annotated[int, Field(ge=0)] = 0
    device: Literal["auto", "cpu", "cuda"] = "auto"
    amp: bool = True
    amp_dtype: Literal["float16", "bfloat16"] = "float16"
    compile_model: bool = False
    deterministic_algorithms: bool = False
    gradient_clip: Annotated[float | None, Field(gt=0.0)] = 5.0
    early_stopping_patience: Annotated[int, Field(ge=0)] = 3
    log_every_steps: Annotated[int, Field(gt=0)] = 50
    optimizer: OptimizerPlanConfig = OptimizerPlanConfig()
    resume_checkpoint: bool = False


class TrackingConfig(StrictModel):
    database: Path = Path("artifacts/experiments.sqlite3")
    tensorboard_dir: Path = Path("artifacts/tensorboard")
    champion_dir: Path = Path("artifacts/champion")
    tensorboard: bool = True


class TuningConfig(StrictModel):
    enabled: bool = False
    study_name: str = "avazu-ctr"
    trials_per_stage: Annotated[int, Field(gt=0)] = 30
    confirmation_candidates: Annotated[int, Field(gt=0)] = 5
    timeout_seconds: Annotated[int | None, Field(gt=0)] = None


class PromotionConfig(StrictModel):
    bootstrap_samples: Annotated[int, Field(ge=100)] = 2_000
    bootstrap_block_rows: Annotated[int, Field(gt=0)] = 100_000
    confidence: Annotated[float, Field(gt=0.5, lt=1.0)] = 0.95
    fold_guard: Annotated[float, Field(ge=0.0)] = 0.0001
    max_weight_bytes: Annotated[int, Field(gt=0)] = 512 * 1024 * 1024


class ExperimentConfig(StrictModel):
    schema_version: Literal[2]
    name: str
    data: DataConfig = DataConfig()
    model: ModelConfig
    objective: ObjectiveConfig = ObjectiveConfig()
    training: TrainingConfig = TrainingConfig()
    tracking: TrackingConfig = TrackingConfig()
    tuning: TuningConfig = TuningConfig()
    promotion: PromotionConfig = PromotionConfig()

    @model_validator(mode="after")
    def validate_feature_configuration(self) -> Self:
        allowed_raw = set(AVAZU_CATEGORICAL_COLUMNS)
        categorical = set(self.data.categorical_columns)
        if len(categorical) != len(self.data.categorical_columns):
            raise ValueError("data.categorical_columns contains duplicates")
        unknown_categorical = categorical.difference(allowed_raw)
        if unknown_categorical:
            raise ValueError(f"unknown Avazu categorical columns: {sorted(unknown_categorical)}")
        for name, features in (
            ("data.count_columns", self.data.count_columns),
            ("data.target_encoding.columns", self.data.target_encoding.columns),
        ):
            if len(set(features)) != len(features):
                raise ValueError(f"{name} contains duplicates")
            unknown = set(features).difference(allowed_raw)
            if unknown:
                raise ValueError(f"{name} contains unknown Avazu columns: {sorted(unknown)}")
        model_features = categorical.union(ENGINEERED_CATEGORICAL_COLUMNS)
        model_configs = [self.model]
        while model_configs:
            model_config = model_configs.pop()
            unknown_embeddings = set(model_config.feature_embeddings).difference(model_features)
            if unknown_embeddings:
                raise ValueError(
                    f"model.feature_embeddings contains inactive features: "
                    f"{sorted(unknown_embeddings)}"
                )
            for feature in model_features:
                root_kind = self.model.feature_embeddings.get(
                    feature,
                    self.model.default_embedding,
                ).kind
                nested_kind = model_config.feature_embeddings.get(
                    feature,
                    model_config.default_embedding,
                ).kind
                if nested_kind is not root_kind:
                    raise ValueError(
                        f"ensemble children must encode {feature!r} as {root_kind.value!r}"
                    )
            model_configs.extend(model_config.children)
        return self
