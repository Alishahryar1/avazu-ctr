"""Pydantic schemas for Avazu CTR experiments."""

from __future__ import annotations

import math
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelKind(StrEnum):
    DCN = "dcn"
    STEC = "stec"
    NGPT = "ngpt"
    ENSEMBLE = "ensemble"


class Aggregation(StrEnum):
    MEAN = "mean"
    GATED = "gated"


class EmbeddingKind(StrEnum):
    STANDARD = "standard"
    HASH = "hash"


class FeatureMode(StrEnum):
    INDUCTIVE = "inductive"
    COMPETITION_TRANSDUCTIVE = "competition_transductive"


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
TIME_CATEGORICAL_COLUMNS = (
    "hour_of_day",
    "day_of_week",
    "day_of_month",
    "hour_of_week",
)
TIME_NUMERICAL_COLUMNS = ("hour_sin", "hour_cos")
CONTEXT_CATEGORICAL_COLUMNS = (
    "inventory_type",
    "publisher_id",
    "publisher_domain",
    "publisher_category",
    "identity_kind",
    "user_id",
)
FeatureName = Annotated[str, Field(pattern=r"^[A-Za-z][A-Za-z0-9_]*$")]


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


class DCNModelConfig(StrictModel):
    backbone: BackboneConfig = BackboneConfig()
    heads: Annotated[tuple[HeadConfig, ...], Field(min_length=1)] = (
        HeadConfig(),
        HeadConfig(),
        HeadConfig(),
    )
    aggregation: Aggregation = Aggregation.GATED
    feature_bagging: Annotated[float, Field(gt=0.0, le=1.0)] = 0.8

    @model_validator(mode="after")
    def validate_heads(self) -> Self:
        if len(self.heads) == 1:
            if self.aggregation != Aggregation.MEAN:
                raise ValueError("a one-head DCN requires mean aggregation")
            if self.feature_bagging != 1.0:
                raise ValueError("a one-head DCN requires feature_bagging=1")
        return self


class STECModelConfig(StrictModel):
    dimension: Annotated[int, Field(gt=0)] = 16
    layers: Annotated[int, Field(ge=1, le=12)] = 2
    heads: Annotated[int, Field(ge=1, le=16)] = 4
    ffn_multiplier: Annotated[int, Field(ge=1, le=16)] = 4
    dropout: Annotated[float, Field(ge=0.0, lt=1.0)] = 0.1
    batch_norm_momentum: Annotated[float | None, Field(gt=0.0, le=1.0)] = None
    batch_norm_epsilon: Annotated[float, Field(gt=0.0)] = 1e-5
    prediction_hidden: tuple[Annotated[int, Field(gt=0)], ...] = (256, 128)
    prediction_dropout: Annotated[float, Field(ge=0.0, lt=1.0)] = 0.1

    @model_validator(mode="after")
    def validate_attention_shape(self) -> Self:
        if self.dimension % self.heads:
            raise ValueError("STEC dimension must be divisible by its attention heads")
        return self


class NGPTModelConfig(StrictModel):
    dimension: Annotated[int, Field(gt=0)] = 32
    layers: Annotated[int, Field(ge=1, le=24)] = 2
    heads: Annotated[int, Field(ge=1, le=16)] = 4
    mlp_multiplier: Annotated[int, Field(ge=1, le=16)] = 4
    alpha_init: Annotated[float, Field(gt=0.0, le=1.0)] = 0.05
    dropout: Annotated[float, Field(ge=0.0, lt=1.0)] = 0.0

    @model_validator(mode="after")
    def validate_attention_shape(self) -> Self:
        if self.dimension % self.heads:
            raise ValueError("nGPT dimension must be divisible by its attention heads")
        return self


class EnsembleModelConfig(StrictModel):
    aggregation: Aggregation = Aggregation.MEAN


class ModelConfig(StrictModel):
    kind: ModelKind
    default_embedding: EmbeddingConfig = EmbeddingConfig()
    feature_embeddings: dict[str, EmbeddingConfig] = Field(default_factory=dict)
    dcn: DCNModelConfig | None = None
    stec: STECModelConfig | None = None
    ngpt: NGPTModelConfig | None = None
    ensemble: EnsembleModelConfig | None = None
    children: tuple[ModelConfig, ...] = ()

    @model_validator(mode="after")
    def validate_shape(self) -> Self:
        configured = {
            ModelKind.DCN: self.dcn is not None,
            ModelKind.STEC: self.stec is not None,
            ModelKind.NGPT: self.ngpt is not None,
            ModelKind.ENSEMBLE: self.ensemble is not None,
        }
        if self.kind == ModelKind.ENSEMBLE:
            if not self.children:
                raise ValueError("ensemble requires at least one child model")
        elif self.children:
            raise ValueError("only ensemble models may define children")
        if not configured[self.kind]:
            raise ValueError(f"{self.kind.value} requires its matching architecture payload")
        inactive = [
            kind.value for kind, active in configured.items() if active and kind != self.kind
        ]
        if inactive:
            raise ValueError(f"{self.kind.value} cannot define inactive payloads: {inactive}")
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


class CrossFeatureConfig(StrictModel):
    name: FeatureName
    columns: Annotated[tuple[FeatureName, ...], Field(min_length=2)]


class DistinctCountFeatureConfig(StrictModel):
    name: FeatureName
    group_by: FeatureName
    value: FeatureName


class ContextFeatureConfig(StrictModel):
    enabled: bool = False
    app_site_sentinel: str = "85f751fd"
    unknown_device_id: str = "a99f214a"


class HistoryFeatureConfig(StrictModel):
    key: FeatureName
    within_hour: bool = False
    clicks: bool = False
    click_pattern_bits: Annotated[int, Field(ge=0, le=8)] = 0

    @model_validator(mode="after")
    def validate_click_pattern(self) -> Self:
        if self.click_pattern_bits and not self.clicks:
            raise ValueError("history click_pattern_bits requires clicks=true")
        return self


class BucketScale(StrEnum):
    IDENTITY = "identity"
    LOG1P = "log1p"


class BucketFeatureConfig(StrictModel):
    name: FeatureName
    source: FeatureName
    boundaries: Annotated[tuple[float, ...], Field(min_length=1)]
    source_scale: BucketScale = BucketScale.IDENTITY

    @model_validator(mode="after")
    def validate_boundaries(self) -> Self:
        if not all(math.isfinite(value) for value in self.boundaries):
            raise ValueError(f"bucket {self.name!r} boundaries must be finite")
        if any(
            left >= right
            for left, right in zip(
                self.boundaries,
                self.boundaries[1:],
                strict=False,
            )
        ):
            raise ValueError(f"bucket {self.name!r} boundaries must be strictly increasing")
        if self.source_scale is BucketScale.LOG1P and self.boundaries[0] < 0.0:
            raise ValueError(f"bucket {self.name!r} log1p boundaries cannot be negative")
        return self


class TargetEncodingConfig(StrictModel):
    columns: tuple[FeatureName, ...] = (
        "app_id",
        "site_id",
        "site_domain",
        "app_domain",
        "C14",
        "C17",
    )
    blocks: Annotated[int, Field(ge=2)] = 10
    smoothing: Annotated[float, Field(gt=0.0)] = 20.0
    probability_clip: Annotated[float, Field(gt=0.0, lt=0.5)] = 1e-5


class FeatureSetConfig(StrictModel):
    mode: FeatureMode = FeatureMode.INDUCTIVE
    raw_categorical_columns: tuple[FeatureName, ...] = AVAZU_CATEGORICAL_COLUMNS
    context: ContextFeatureConfig = ContextFeatureConfig()
    crosses: tuple[CrossFeatureConfig, ...] = (
        CrossFeatureConfig(
            name="user_proxy",
            columns=("device_ip", "device_model"),
        ),
        CrossFeatureConfig(
            name="device_id_x_app_id",
            columns=("device_id", "app_id"),
        ),
        CrossFeatureConfig(
            name="device_ip_x_C14",
            columns=("device_ip", "C14"),
        ),
        CrossFeatureConfig(
            name="user_proxy_x_app_id",
            columns=("user_proxy", "app_id"),
        ),
        CrossFeatureConfig(
            name="user_proxy_x_site_id",
            columns=("user_proxy", "site_id"),
        ),
        CrossFeatureConfig(
            name="site_id_x_C14",
            columns=("site_id", "C14"),
        ),
        CrossFeatureConfig(
            name="app_id_x_C14",
            columns=("app_id", "C14"),
        ),
    )
    frequency_columns: tuple[FeatureName, ...] = (
        "device_ip",
        "device_id",
        "user_proxy",
        "app_id",
        "site_id",
        "C14",
        "C17",
        "C21",
    )
    distinct_counts: tuple[DistinctCountFeatureConfig, ...] = (
        DistinctCountFeatureConfig(
            name="device_ip_distinct_apps_log1p",
            group_by="device_ip",
            value="app_id",
        ),
        DistinctCountFeatureConfig(
            name="device_ip_distinct_sites_log1p",
            group_by="device_ip",
            value="site_id",
        ),
        DistinctCountFeatureConfig(
            name="user_proxy_distinct_apps_log1p",
            group_by="user_proxy",
            value="app_id",
        ),
        DistinctCountFeatureConfig(
            name="user_proxy_distinct_sites_log1p",
            group_by="user_proxy",
            value="site_id",
        ),
    )
    history: tuple[HistoryFeatureConfig, ...] = (
        HistoryFeatureConfig(key="user_proxy", within_hour=True),
        HistoryFeatureConfig(key="device_ip"),
    )
    target_encoding: TargetEncodingConfig = TargetEncodingConfig()
    buckets: tuple[BucketFeatureConfig, ...] = ()

    @property
    def context_categorical_columns(self) -> tuple[str, ...]:
        return CONTEXT_CATEGORICAL_COLUMNS if self.context.enabled else ()

    @property
    def pre_transform_categorical_columns(self) -> tuple[str, ...]:
        return (
            *self.raw_categorical_columns,
            *TIME_CATEGORICAL_COLUMNS,
            *self.context_categorical_columns,
            *(cross.name for cross in self.crosses),
        )

    @property
    def history_categorical_columns(self) -> tuple[str, ...]:
        return tuple(
            f"{feature.key}_recent_click_pattern"
            for feature in self.history
            if feature.click_pattern_bits
        )

    @property
    def categorical_columns(self) -> tuple[str, ...]:
        return (
            *self.pre_transform_categorical_columns,
            *(feature.name for feature in self.buckets),
            *self.history_categorical_columns,
        )

    @property
    def history_numerical_columns(self) -> tuple[str, ...]:
        history: list[str] = []
        for feature in self.history:
            history.extend(
                (
                    f"{feature.key}_prior_impressions_log1p",
                    f"{feature.key}_hours_since_previous_impression_log1p",
                )
            )
            if feature.within_hour:
                history.append(f"{feature.key}_prior_hour_impressions_log1p")
            if feature.clicks:
                history.extend(
                    (
                        f"{feature.key}_prior_clicks_log1p",
                        f"{feature.key}_prior_nonclicks_log1p",
                        f"{feature.key}_prior_ctr_logit_lift",
                        f"{feature.key}_hours_since_last_click_log1p",
                        f"{feature.key}_impressions_since_last_click_log1p",
                    )
                )
        return tuple(history)

    @property
    def target_numerical_columns(self) -> tuple[str, ...]:
        target: list[str] = []
        for feature in self.target_encoding.columns:
            target.extend(
                (
                    f"{feature}_target_logit_lift",
                    f"{feature}_target_evidence_log1p",
                )
            )
        return tuple(target)

    @property
    def numerical_columns(self) -> tuple[str, ...]:
        return (
            *TIME_NUMERICAL_COLUMNS,
            *(f"{feature}_frequency_log1p" for feature in self.frequency_columns),
            *(feature.name for feature in self.distinct_counts),
            *self.history_numerical_columns,
            *self.target_numerical_columns,
        )

    @property
    def label_dependent_columns(self) -> frozenset[str]:
        history = {
            name
            for feature in self.history
            if feature.clicks
            for name in (
                f"{feature.key}_prior_clicks_log1p",
                f"{feature.key}_prior_nonclicks_log1p",
                f"{feature.key}_prior_ctr_logit_lift",
                f"{feature.key}_hours_since_last_click_log1p",
                f"{feature.key}_impressions_since_last_click_log1p",
                *((f"{feature.key}_recent_click_pattern",) if feature.click_pattern_bits else ()),
            )
        }
        return frozenset((*history, *self.target_numerical_columns))

    @property
    def post_transform_categorical_columns(self) -> tuple[str, ...]:
        return (
            *(feature.name for feature in self.buckets),
            *self.history_categorical_columns,
        )

    @model_validator(mode="after")
    def validate_feature_plan(self) -> Self:
        raw = self.raw_categorical_columns
        if len(set(raw)) != len(raw):
            raise ValueError("features.raw_categorical_columns contains duplicates")
        unknown_raw = set(raw).difference(AVAZU_CATEGORICAL_COLUMNS)
        if unknown_raw:
            raise ValueError(f"unknown Avazu categorical columns: {sorted(unknown_raw)}")

        available = set(raw).union(TIME_CATEGORICAL_COLUMNS)
        if self.context.enabled:
            required = {
                "site_id",
                "site_domain",
                "site_category",
                "app_id",
                "app_domain",
                "app_category",
                "device_id",
                "device_ip",
                "device_model",
            }
            unknown = required.difference(available)
            if unknown:
                raise ValueError(
                    f"features.context requires unavailable columns: {sorted(unknown)}"
                )
            available.update(CONTEXT_CATEGORICAL_COLUMNS)
        for cross in self.crosses:
            if cross.name in available:
                raise ValueError(f"feature name {cross.name!r} is duplicated")
            if len(set(cross.columns)) != len(cross.columns):
                raise ValueError(f"cross {cross.name!r} repeats an input column")
            unknown = set(cross.columns).difference(available)
            if unknown:
                raise ValueError(
                    f"cross {cross.name!r} references unavailable columns: {sorted(unknown)}"
                )
            available.add(cross.name)

        for field_name, features in (
            ("features.frequency_columns", self.frequency_columns),
            ("features.history", tuple(feature.key for feature in self.history)),
            ("features.target_encoding.columns", self.target_encoding.columns),
        ):
            if len(set(features)) != len(features):
                raise ValueError(f"{field_name} contains duplicates")
            unknown = set(features).difference(available)
            if unknown:
                raise ValueError(f"{field_name} references unavailable columns: {sorted(unknown)}")

        for feature in self.distinct_counts:
            if feature.name in available:
                raise ValueError(f"feature name {feature.name!r} is duplicated")
            if feature.group_by == feature.value:
                raise ValueError(f"distinct count {feature.name!r} requires two different columns")
            unknown = {feature.group_by, feature.value}.difference(available)
            if unknown:
                raise ValueError(
                    f"distinct count {feature.name!r} references unavailable columns: "
                    f"{sorted(unknown)}"
                )

        numerical = self.numerical_columns
        if len(set(numerical)) != len(numerical):
            raise ValueError("engineered numerical feature names must be unique")
        collisions = set(numerical).intersection(available)
        if collisions:
            raise ValueError(
                f"feature names span both categorical and numerical lanes: {collisions}"
            )
        bucket_names: set[str] = set()
        numerical_set = set(numerical)
        for bucket in self.buckets:
            if (
                bucket.name in available
                or bucket.name in numerical_set
                or bucket.name in bucket_names
            ):
                raise ValueError(f"feature name {bucket.name!r} is duplicated")
            if bucket.source not in numerical_set:
                raise ValueError(
                    f"bucket {bucket.name!r} references unavailable numerical feature "
                    f"{bucket.source!r}"
                )
            bucket_names.add(bucket.name)
        history_categories = set(self.history_categorical_columns)
        categorical = self.categorical_columns
        if len(set(categorical)) != len(categorical):
            raise ValueError("engineered categorical feature names must be unique")
        if history_categories.intersection(available | numerical_set | bucket_names):
            raise ValueError("history categorical feature names collide with another feature")
        return self


class DataConfig(StrictModel):
    train_path: Path = Path("data/raw/train.gz")
    test_path: Path = Path("data/raw/test.gz")
    artifact_root: Path = Path("artifacts")
    shard_rows: Annotated[int, Field(gt=0)] = 250_000
    vocabulary_limit: Annotated[int, Field(gt=1)] = 1_000_000
    minimum_frequency: Annotated[int, Field(ge=1)] = 2
    split: TemporalSplitConfig = TemporalSplitConfig()
    features: FeatureSetConfig = FeatureSetConfig()


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
    selection_dir: Path = Path("artifacts/selection")
    tensorboard: bool = True


class DeploymentConfig(StrictModel):
    champion_dir: Path = Path("artifacts/champion")
    max_weight_bytes: Annotated[int, Field(gt=0)] = 512 * 1024 * 1024


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


class ExperimentConfig(StrictModel):
    schema_version: Literal[5]
    name: str
    data: DataConfig = DataConfig()
    model: ModelConfig
    objective: ObjectiveConfig = ObjectiveConfig()
    training: TrainingConfig = TrainingConfig()
    tracking: TrackingConfig = TrackingConfig()
    deployment: DeploymentConfig = DeploymentConfig()
    tuning: TuningConfig = TuningConfig()
    promotion: PromotionConfig = PromotionConfig()

    @model_validator(mode="after")
    def validate_feature_configuration(self) -> Self:
        if self.tracking.selection_dir.resolve() == self.deployment.champion_dir.resolve():
            raise ValueError("selection and champion directories must differ")
        model_features = set(self.data.features.categorical_columns)
        cross_features = {feature.name for feature in self.data.features.crosses}
        post_transform_features = set(self.data.features.post_transform_categorical_columns)
        model_configs = [self.model]
        contains_ngpt = False
        while model_configs:
            model_config = model_configs.pop()
            contains_ngpt |= model_config.kind == ModelKind.NGPT
            unknown_embeddings = set(model_config.feature_embeddings).difference(model_features)
            if unknown_embeddings:
                raise ValueError(
                    f"model.feature_embeddings contains inactive features: "
                    f"{sorted(unknown_embeddings)}"
                )
            non_hashed_crosses = {
                feature
                for feature in cross_features
                if model_config.feature_embeddings.get(
                    feature,
                    model_config.default_embedding,
                ).kind
                is not EmbeddingKind.HASH
            }
            if non_hashed_crosses:
                raise ValueError(
                    f"cross features require bounded hash embeddings: {sorted(non_hashed_crosses)}"
                )
            non_hashed_post_transform = {
                feature
                for feature in post_transform_features
                if model_config.feature_embeddings.get(
                    feature,
                    model_config.default_embedding,
                ).kind
                is not EmbeddingKind.HASH
            }
            if non_hashed_post_transform:
                raise ValueError(
                    "post-transform categorical features require bounded hash embeddings: "
                    f"{sorted(non_hashed_post_transform)}"
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
                if nested_kind != root_kind:
                    raise ValueError(
                        f"ensemble children must encode {feature!r} as {root_kind.value!r}"
                    )
            model_configs.extend(model_config.children)
        if contains_ngpt:
            optimizer = self.training.optimizer
            if optimizer.embeddings is not None:
                raise ValueError("nGPT requires one optimizer for coherent hyperspherical updates")
            if optimizer.dense.kind != OptimizerKind.ADAMW:
                raise ValueError("nGPT requires Adam")
            if optimizer.dense.weight_decay != 0.0:
                raise ValueError("nGPT requires zero weight decay")
            if optimizer.dense.scheduler.warmup_ratio != 0.0:
                raise ValueError("nGPT requires zero learning-rate warmup")
        return self
