"""Fit feature state from explicitly declared covariate and label populations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from avazu_ctr.config.schema import EmbeddingKind, ExperimentConfig
from avazu_ctr.data.features.plan import feature_definitions
from avazu_ctr.data.manifest import (
    DatasetSplit,
    FeatureDefinition,
    FittedTableManifest,
    HourRange,
    sha256_file,
)


@dataclass(slots=True)
class FittedFeatureState:
    training_rows: int
    categorical_columns: tuple[str, ...]
    numerical_columns: tuple[str, ...]
    cardinalities: dict[str, int]
    embedding_kinds: dict[str, str]
    vocabularies: dict[str, Path]
    frequencies: dict[str, Path]
    distinct_counts: dict[str, Path]
    target_encodings: dict[str, Path]
    temporal_target_encodings: dict[str, Path]
    global_prior: float
    definitions: tuple[FeatureDefinition, ...]
    tables: list[FittedTableManifest]


def _write_fitted_table(
    frame: pl.LazyFrame,
    path: Path,
    *,
    feature: str,
    kind: str,
    sources: tuple[DatasetSplit, ...],
    uses_labels: bool,
) -> FittedTableManifest:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.sink_parquet(
        path,
        compression="zstd",
        statistics=True,
        maintain_order=True,
        engine="streaming",
    )
    rows = int(
        pl.scan_parquet(path).select(pl.len().alias("rows")).collect(engine="streaming")["rows"][0]
    )
    return FittedTableManifest(
        feature=feature,
        kind=kind,
        path=path.name,
        rows=rows,
        sha256=sha256_file(path),
        sources=sources,
        uses_labels=uses_labels,
    )


def fit_feature_state(
    train: pl.LazyFrame,
    covariates: pl.LazyFrame,
    config: ExperimentConfig,
    state_dir: Path,
    *,
    covariate_sources: tuple[DatasetSplit, ...],
    training_range: HourRange,
) -> FittedFeatureState:
    """Fit covariate state from the declared scope and label state from training only."""

    features = config.data.features
    categorical = features.categorical_columns
    cardinalities: dict[str, int] = {}
    kinds: dict[str, str] = {}
    vocabularies: dict[str, Path] = {}
    frequencies: dict[str, Path] = {}
    distinct_counts: dict[str, Path] = {}
    target_encodings: dict[str, Path] = {}
    temporal_target_encodings: dict[str, Path] = {}
    tables: list[FittedTableManifest] = []

    for feature in categorical:
        embedding = config.model.feature_embeddings.get(feature, config.model.default_embedding)
        kinds[feature] = embedding.kind.value
        if embedding.kind is EmbeddingKind.HASH:
            cardinalities[feature] = embedding.buckets
            continue
        vocabulary = (
            covariates.group_by(feature)
            .len(name="_frequency")
            .filter(pl.col("_frequency") >= config.data.minimum_frequency)
            .sort(feature)
            .with_row_index(f"{feature}__id", offset=1)
            .select(feature, pl.col(f"{feature}__id").cast(pl.Int64))
        )
        path = state_dir / f"vocabulary_{feature}.parquet"
        table = _write_fitted_table(
            vocabulary,
            path,
            feature=feature,
            kind="vocabulary",
            sources=covariate_sources,
            uses_labels=False,
        )
        if table.rows > config.data.vocabulary_limit:
            raise ValueError(
                f"{feature} has {table.rows} retained values; configure a hash embedding "
                f"or raise data.vocabulary_limit"
            )
        tables.append(table)
        vocabularies[feature] = path
        cardinalities[feature] = table.rows + 1

    for feature in features.frequency_columns:
        output = f"{feature}__frequency"
        counts = covariates.group_by(feature).len(name=output).sort(feature)
        path = state_dir / f"frequency_{feature}.parquet"
        tables.append(
            _write_fitted_table(
                counts,
                path,
                feature=feature,
                kind="covariate_frequency",
                sources=covariate_sources,
                uses_labels=False,
            )
        )
        frequencies[feature] = path

    for feature in features.distinct_counts:
        raw_name = f"{feature.name}__raw"
        counts = (
            covariates.group_by(feature.group_by)
            .agg(pl.col(feature.value).n_unique().alias(raw_name))
            .sort(feature.group_by)
        )
        path = state_dir / f"distinct_{feature.name}.parquet"
        tables.append(
            _write_fitted_table(
                counts,
                path,
                feature=feature.name,
                kind="covariate_distinct_count",
                sources=covariate_sources,
                uses_labels=False,
            )
        )
        distinct_counts[feature.name] = path

    label_summary = train.select(
        pl.col("click").sum().alias("positive"),
        pl.len().alias("count"),
    ).collect(engine="streaming")
    positives = int(label_summary["positive"][0])
    label_count = int(label_summary["count"][0])
    if label_count == 0:
        raise ValueError("training window is empty")
    global_prior = positives / label_count

    target = features.target_encoding
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

    for feature in features.target_encoding.columns:
        stats = (
            train.group_by(feature)
            .agg(
                pl.col("click").sum().cast(pl.Int64).alias("_positive"),
                pl.len().cast(pl.Int64).alias("_count"),
            )
            .sort(feature)
        )
        path = state_dir / f"target_encoding_{feature}.parquet"
        tables.append(
            _write_fitted_table(
                stats,
                path,
                feature=feature,
                kind="target_encoding",
                sources=(DatasetSplit.TRAINING,),
                uses_labels=True,
            )
        )
        target_encodings[feature] = path
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
        temporal_path = state_dir / f"temporal_target_encoding_{feature}.parquet"
        tables.append(
            _write_fitted_table(
                temporal_stats,
                temporal_path,
                feature=feature,
                kind="temporal_target_encoding",
                sources=(DatasetSplit.TRAINING,),
                uses_labels=True,
            )
        )
        temporal_target_encodings[feature] = temporal_path

    return FittedFeatureState(
        training_rows=label_count,
        categorical_columns=categorical,
        numerical_columns=features.numerical_columns,
        cardinalities=cardinalities,
        embedding_kinds=kinds,
        vocabularies=vocabularies,
        frequencies=frequencies,
        distinct_counts=distinct_counts,
        target_encodings=target_encodings,
        temporal_target_encodings=temporal_target_encodings,
        global_prior=global_prior,
        definitions=feature_definitions(config),
        tables=tables,
    )
