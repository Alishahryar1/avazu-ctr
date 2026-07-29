"""Fit compact, key-centric feature state from declared data populations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

from avazu_ctr.config.schema import EmbeddingKind, ExperimentConfig
from avazu_ctr.data.features.plan import feature_definitions
from avazu_ctr.data.manifest import (
    DatasetSplit,
    FeatureDefinition,
    FittedTableKind,
    FittedTableManifest,
    HourRange,
    sha256_file,
)

_MAX_MULTIPLEXED_SINKS = 4


@dataclass(slots=True)
class FittedFeatureState:
    training_rows: int
    categorical_columns: tuple[str, ...]
    numerical_columns: tuple[str, ...]
    cardinalities: dict[str, int]
    embedding_kinds: dict[str, str]
    vocabularies: dict[str, Path]
    covariate_lookups: dict[str, Path]
    target_encodings: dict[str, Path]
    temporal_target_encodings: dict[str, Path]
    global_prior: float
    definitions: tuple[FeatureDefinition, ...]
    tables: list[FittedTableManifest]


@dataclass(frozen=True, slots=True)
class _FittedSink:
    frame: pl.LazyFrame
    path: Path
    kind: FittedTableKind
    join_keys: tuple[str, ...]
    outputs: tuple[str, ...]
    sources: tuple[DatasetSplit, ...]
    uses_labels: bool


def _finalize_fitted_sink(sink: _FittedSink) -> FittedTableManifest:
    metadata = pq.ParquetFile(sink.path).metadata
    if metadata is None:
        raise ValueError(f"fitted table {sink.path} has no Parquet metadata")
    return FittedTableManifest(
        kind=sink.kind,
        join_keys=sink.join_keys,
        outputs=sink.outputs,
        path=sink.path.name,
        rows=metadata.num_rows,
        sha256=sha256_file(sink.path),
        sources=sink.sources,
        uses_labels=sink.uses_labels,
    )


def _write_fitted_tables(
    sinks: list[_FittedSink],
    *,
    max_concurrent: int = _MAX_MULTIPLEXED_SINKS,
) -> list[FittedTableManifest]:
    """Write bounded groups of lazy sinks so compatible plans share source scans."""

    if max_concurrent <= 0:
        raise ValueError("max_concurrent must be positive")
    manifests: list[FittedTableManifest] = []
    for start in range(0, len(sinks), max_concurrent):
        batch = sinks[start : start + max_concurrent]
        plans: list[pl.LazyFrame] = []
        for sink in batch:
            sink.path.parent.mkdir(parents=True, exist_ok=True)
            plan = sink.frame.sink_parquet(
                sink.path,
                compression="zstd",
                statistics=True,
                maintain_order=True,
                lazy=True,
            )
            if plan is None:
                raise RuntimeError("lazy fitted-table sink did not return a query plan")
            plans.append(plan)
        pl.collect_all(plans, engine="streaming")
        manifests.extend(_finalize_fitted_sink(sink) for sink in batch)
    return manifests


def _vocabulary_sinks(
    train: pl.LazyFrame,
    config: ExperimentConfig,
    state_dir: Path,
) -> tuple[list[_FittedSink], dict[str, str]]:
    sinks: list[_FittedSink] = []
    kinds: dict[str, str] = {}
    for feature in config.data.features.categorical_columns:
        embedding = config.model.feature_embeddings.get(feature, config.model.default_embedding)
        kinds[feature] = embedding.kind.value
        if embedding.kind is EmbeddingKind.HASH:
            continue
        output = f"{feature}__id"
        vocabulary = (
            train.group_by(feature)
            .len(name="_frequency")
            .filter(pl.col("_frequency") >= config.data.minimum_frequency)
            .sort(feature)
            .with_row_index(output, offset=1)
            .select(feature, pl.col(output).cast(pl.Int64))
        )
        sinks.append(
            _FittedSink(
                frame=vocabulary,
                path=state_dir / f"vocabulary_{feature}.parquet",
                kind=FittedTableKind.VOCABULARY,
                join_keys=(feature,),
                outputs=(output,),
                sources=(DatasetSplit.TRAINING,),
                uses_labels=False,
            )
        )
    return sinks, kinds


def _covariate_sinks(
    covariates: pl.LazyFrame,
    config: ExperimentConfig,
    state_dir: Path,
    *,
    sources: tuple[DatasetSplit, ...],
) -> list[_FittedSink]:
    aggregations: dict[str, list[pl.Expr]] = {}
    outputs: dict[str, list[str]] = {}

    for feature in config.data.features.frequency_columns:
        output = f"{feature}__frequency"
        aggregations.setdefault(feature, []).append(pl.len().alias(output))
        outputs.setdefault(feature, []).append(output)
    for feature in config.data.features.distinct_counts:
        output = f"{feature.name}__raw"
        aggregations.setdefault(feature.group_by, []).append(
            pl.col(feature.value).n_unique().alias(output)
        )
        outputs.setdefault(feature.group_by, []).append(output)

    return [
        _FittedSink(
            frame=covariates.group_by(key).agg(*expressions).sort(key),
            path=state_dir / f"covariate_{key}.parquet",
            kind=FittedTableKind.COVARIATE_LOOKUP,
            join_keys=(key,),
            outputs=tuple(outputs[key]),
            sources=sources,
            uses_labels=False,
        )
        for key, expressions in aggregations.items()
    ]


def _target_sinks(
    train: pl.LazyFrame,
    config: ExperimentConfig,
    state_dir: Path,
    *,
    training_range: HourRange,
) -> list[_FittedSink]:
    target = config.data.features.target_encoding
    width = max(
        1,
        (training_range.end - training_range.start + target.blocks - 1) // target.blocks,
    )
    blocked = train.with_columns(
        ((pl.col("_timestamp_hour") - training_range.start) // width)
        .clip(0, target.blocks - 1)
        .alias("_te_block")
    )
    global_blocks = (
        blocked.group_by("_te_block")
        .agg(pl.col("click").sum().alias("_block_positive"), pl.len().alias("_block_count"))
        .sort("_te_block")
        .with_columns(
            (pl.col("_block_positive").cum_sum() - pl.col("_block_positive")).alias(
                "_prior_positive"
            ),
            (pl.col("_block_count").cum_sum() - pl.col("_block_count")).alias("_prior_count"),
        )
        .with_columns(
            pl.when(pl.col("_prior_count") > 0)
            .then(pl.col("_prior_positive") / pl.col("_prior_count"))
            .otherwise(None)
            .alias("_block_prior")
        )
        .select("_te_block", "_prior_count", "_block_prior")
    )

    sinks: list[_FittedSink] = []
    for feature in target.columns:
        stats = (
            train.group_by(feature)
            .agg(
                pl.col("click").sum().cast(pl.Int64).alias("_positive"),
                pl.len().cast(pl.Int64).alias("_count"),
            )
            .sort(feature)
        )
        sinks.append(
            _FittedSink(
                frame=stats,
                path=state_dir / f"target_encoding_{feature}.parquet",
                kind=FittedTableKind.TARGET_ENCODING,
                join_keys=(feature,),
                outputs=("_positive", "_count"),
                sources=(DatasetSplit.TRAINING,),
                uses_labels=True,
            )
        )
        temporal_stats = (
            blocked.group_by(feature, "_te_block")
            .agg(pl.col("click").sum().alias("_positive"), pl.len().alias("_count"))
            .sort(feature, "_te_block")
            .with_columns(
                (pl.col("_positive").cum_sum().over(feature) - pl.col("_positive")).alias(
                    "_previous_positive"
                ),
                (pl.col("_count").cum_sum().over(feature) - pl.col("_count")).alias(
                    "_previous_count"
                ),
            )
            .select(feature, "_te_block", "_previous_positive", "_previous_count")
            .join(
                global_blocks,
                on="_te_block",
                how="left",
                maintain_order="left",
            )
            .sort(feature, "_te_block")
        )
        sinks.append(
            _FittedSink(
                frame=temporal_stats,
                path=state_dir / f"temporal_target_encoding_{feature}.parquet",
                kind=FittedTableKind.TEMPORAL_TARGET_ENCODING,
                join_keys=(feature, "_te_block"),
                outputs=(
                    "_previous_positive",
                    "_previous_count",
                    "_prior_count",
                    "_block_prior",
                ),
                sources=(DatasetSplit.TRAINING,),
                uses_labels=True,
            )
        )
    return sinks


def fit_feature_state(
    train: pl.LazyFrame,
    covariates: pl.LazyFrame,
    config: ExperimentConfig,
    state_dir: Path,
    *,
    covariate_sources: tuple[DatasetSplit, ...],
    training_range: HourRange,
) -> FittedFeatureState:
    """Fit compact lookup state from explicit covariate and label populations."""

    features = config.data.features
    vocabulary_sinks, kinds = _vocabulary_sinks(train, config, state_dir)
    vocabulary_tables = _write_fitted_tables(vocabulary_sinks)
    vocabularies = {sink.join_keys[0]: sink.path for sink in vocabulary_sinks}
    cardinalities = {
        feature: config.model.feature_embeddings.get(
            feature, config.model.default_embedding
        ).buckets
        for feature in features.categorical_columns
        if kinds[feature] == EmbeddingKind.HASH.value
    }
    for table in vocabulary_tables:
        feature = table.join_keys[0]
        if table.rows > config.data.vocabulary_limit:
            raise ValueError(
                f"{feature} has {table.rows} retained values; configure a hash embedding "
                f"or raise data.vocabulary_limit"
            )
        cardinalities[feature] = table.rows + 1

    covariate_sinks = _covariate_sinks(
        covariates,
        config,
        state_dir,
        sources=covariate_sources,
    )
    covariate_tables = _write_fitted_tables(covariate_sinks)
    covariate_lookups = {sink.join_keys[0]: sink.path for sink in covariate_sinks}

    label_summary = train.select(
        pl.col("click").sum().alias("positive"),
        pl.len().alias("count"),
    ).collect(engine="streaming")
    positives = int(label_summary["positive"][0])
    label_count = int(label_summary["count"][0])
    if label_count == 0:
        raise ValueError("training window is empty")
    global_prior = positives / label_count

    target_sinks = _target_sinks(
        train,
        config,
        state_dir,
        training_range=training_range,
    )
    target_tables = _write_fitted_tables(target_sinks)
    target_encodings = {
        sink.join_keys[0]: sink.path
        for sink in target_sinks
        if sink.kind is FittedTableKind.TARGET_ENCODING
    }
    temporal_target_encodings = {
        sink.join_keys[0]: sink.path
        for sink in target_sinks
        if sink.kind is FittedTableKind.TEMPORAL_TARGET_ENCODING
    }

    return FittedFeatureState(
        training_rows=label_count,
        categorical_columns=features.categorical_columns,
        numerical_columns=features.numerical_columns,
        cardinalities=cardinalities,
        embedding_kinds=kinds,
        vocabularies=vocabularies,
        covariate_lookups=covariate_lookups,
        target_encodings=target_encodings,
        temporal_target_encodings=temporal_target_encodings,
        global_prior=global_prior,
        definitions=feature_definitions(config),
        tables=[*vocabulary_tables, *covariate_tables, *target_tables],
    )
