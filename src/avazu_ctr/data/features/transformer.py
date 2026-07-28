"""Compile fitted Parquet state into one optimized lazy feature plan."""

from __future__ import annotations

import math

import polars as pl

from avazu_ctr.config.schema import EmbeddingKind, ExperimentConfig
from avazu_ctr.data.features.fitting import FittedFeatureState
from avazu_ctr.data.manifest import HourRange


def _logit(probability: pl.Expr, clip: float) -> pl.Expr:
    bounded = probability.clip(clip, 1.0 - clip)
    return (bounded / (1.0 - bounded)).log()


class FittedFeatureTransformer:
    """Apply the fitted feature contract without materializing state in Python."""

    def __init__(
        self,
        state: FittedFeatureState,
        config: ExperimentConfig,
        training_range: HourRange,
    ) -> None:
        self.state = state
        self.config = config
        self.training_range = training_range
        self.vocabularies = {
            feature: pl.scan_parquet(path) for feature, path in state.vocabularies.items()
        }
        self.frequencies = {
            feature: pl.scan_parquet(path) for feature, path in state.frequencies.items()
        }
        self.distinct_counts = {
            feature: pl.scan_parquet(path) for feature, path in state.distinct_counts.items()
        }
        self.target_encodings = {
            feature: pl.scan_parquet(path) for feature, path in state.target_encodings.items()
        }
        self.temporal_target_encodings = {
            feature: pl.scan_parquet(path)
            for feature, path in state.temporal_target_encodings.items()
        }

    def transform(self, frame: pl.LazyFrame, *, training: bool) -> pl.LazyFrame:
        transformed = self._apply_covariate_statistics(frame)
        transformed = (
            self._apply_training_target_features(transformed)
            if training
            else self._apply_scoring_target_features(transformed)
        )
        return self._apply_categories(transformed)

    def _apply_covariate_statistics(self, frame: pl.LazyFrame) -> pl.LazyFrame:
        transformed = frame
        for feature, table in self.frequencies.items():
            raw_name = f"{feature}__frequency"
            transformed = transformed.join(
                table,
                on=feature,
                how="left",
                maintain_order="left",
            ).with_columns(
                pl.col(raw_name)
                .fill_null(0)
                .cast(pl.Float64)
                .log1p()
                .cast(pl.Float32)
                .alias(f"{feature}_frequency_log1p")
            )

        configured = {
            feature.name: feature for feature in self.config.data.features.distinct_counts
        }
        for name, table in self.distinct_counts.items():
            feature = configured[name]
            raw_name = f"{name}__raw"
            transformed = transformed.join(
                table,
                on=feature.group_by,
                how="left",
                maintain_order="left",
            ).with_columns(
                pl.col(raw_name).fill_null(0).cast(pl.Float64).log1p().cast(pl.Float32).alias(name)
            )
        return transformed

    def _apply_training_target_features(self, frame: pl.LazyFrame) -> pl.LazyFrame:
        target = self.config.data.features.target_encoding
        if not target.columns:
            return frame
        width = max(
            1,
            (self.training_range.end - self.training_range.start + target.blocks - 1)
            // target.blocks,
        )
        transformed = frame.with_columns(
            ((pl.col("_timestamp_hour") - self.training_range.start) // width)
            .clip(0, target.blocks - 1)
            .alias("_te_block")
        )
        for feature, table in self.temporal_target_encodings.items():
            previous_positive = pl.col("_previous_positive").fill_null(0)
            previous_count = pl.col("_previous_count").fill_null(0)
            prior = pl.col("_block_prior").fill_null(0.5)
            posterior = (previous_positive + target.smoothing * prior) / (
                previous_count + target.smoothing
            )
            transformed = (
                transformed.join(
                    table,
                    on=[feature, "_te_block"],
                    how="left",
                    maintain_order="left",
                )
                .with_columns(
                    pl.when(pl.col("_prior_count") > 0)
                    .then(
                        _logit(posterior, target.probability_clip)
                        - _logit(prior, target.probability_clip)
                    )
                    .otherwise(0.0)
                    .cast(pl.Float32)
                    .alias(f"{feature}_target_logit_lift"),
                    previous_count.cast(pl.Float64)
                    .log1p()
                    .cast(pl.Float32)
                    .alias(f"{feature}_target_evidence_log1p"),
                )
                .drop(
                    "_previous_positive",
                    "_previous_count",
                    "_prior_count",
                    "_block_prior",
                )
            )
        return transformed.drop("_te_block")

    def _apply_scoring_target_features(self, frame: pl.LazyFrame) -> pl.LazyFrame:
        target = self.config.data.features.target_encoding
        if not target.columns:
            return frame
        prior = min(
            max(self.state.global_prior, target.probability_clip),
            1.0 - target.probability_clip,
        )
        prior_logit = math.log(prior / (1.0 - prior))
        transformed = frame
        for feature, table in self.target_encodings.items():
            count = pl.col("_count").fill_null(0)
            posterior = (
                pl.col("_positive").fill_null(0) + target.smoothing * self.state.global_prior
            ) / (count + target.smoothing)
            transformed = (
                transformed.join(
                    table,
                    on=feature,
                    how="left",
                    maintain_order="left",
                )
                .with_columns(
                    (_logit(posterior, target.probability_clip) - prior_logit)
                    .cast(pl.Float32)
                    .alias(f"{feature}_target_logit_lift"),
                    count.cast(pl.Float64)
                    .log1p()
                    .cast(pl.Float32)
                    .alias(f"{feature}_target_evidence_log1p"),
                )
                .drop("_positive", "_count")
            )
        return transformed

    def _apply_categories(self, frame: pl.LazyFrame) -> pl.LazyFrame:
        transformed = frame
        for feature in self.state.categorical_columns:
            embedding = self.config.model.feature_embeddings.get(
                feature,
                self.config.model.default_embedding,
            )
            if embedding.kind is EmbeddingKind.HASH:
                transformed = transformed.with_columns(
                    pl.col(feature)
                    .cast(pl.String)
                    .hash(
                        seed=self.config.training.seed,
                        seed_1=self.config.training.seed + 1,
                        seed_2=self.config.training.seed + 2,
                        seed_3=self.config.training.seed + 3,
                    )
                    .reinterpret(signed=True)
                    .alias(feature)
                )
                continue
            id_column = f"{feature}__id"
            transformed = (
                transformed.join(
                    self.vocabularies[feature],
                    on=feature,
                    how="left",
                    maintain_order="left",
                )
                .drop(feature)
                .rename({id_column: feature})
                .with_columns(pl.col(feature).fill_null(0).cast(pl.Int64))
            )
        return transformed
