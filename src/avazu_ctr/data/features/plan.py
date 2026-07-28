"""Compile strict configuration recipes into an ordered feature contract."""

from __future__ import annotations

import polars as pl

from avazu_ctr.config.schema import (
    TIME_CATEGORICAL_COLUMNS,
    TIME_NUMERICAL_COLUMNS,
    ExperimentConfig,
)
from avazu_ctr.data.manifest import (
    FeatureDefinition,
    FeatureFamily,
    FeatureLane,
)

_CROSS_SEPARATOR = "\x1f"


def feature_definitions(config: ExperimentConfig) -> tuple[FeatureDefinition, ...]:
    """Compile configured recipes into one ordered, auditable model contract."""

    features = config.data.features
    definitions: list[FeatureDefinition] = [
        *(
            FeatureDefinition(
                name=name,
                lane=FeatureLane.CATEGORICAL,
                family=FeatureFamily.RAW,
                inputs=(name,),
            )
            for name in features.raw_categorical_columns
        ),
        *(
            FeatureDefinition(
                name=name,
                lane=FeatureLane.CATEGORICAL,
                family=FeatureFamily.TIME,
                inputs=("hour",),
            )
            for name in TIME_CATEGORICAL_COLUMNS
        ),
        *(
            FeatureDefinition(
                name=cross.name,
                lane=FeatureLane.CATEGORICAL,
                family=FeatureFamily.CROSS,
                inputs=cross.columns,
            )
            for cross in features.crosses
        ),
        *(
            FeatureDefinition(
                name=name,
                lane=FeatureLane.NUMERICAL,
                family=FeatureFamily.TIME,
                inputs=("hour",),
            )
            for name in TIME_NUMERICAL_COLUMNS
        ),
        *(
            FeatureDefinition(
                name=f"{name}_frequency_log1p",
                lane=FeatureLane.NUMERICAL,
                family=FeatureFamily.FREQUENCY,
                inputs=(name,),
            )
            for name in features.frequency_columns
        ),
        *(
            FeatureDefinition(
                name=feature.name,
                lane=FeatureLane.NUMERICAL,
                family=FeatureFamily.DISTINCT_COUNT,
                inputs=(feature.group_by, feature.value),
            )
            for feature in features.distinct_counts
        ),
    ]
    for history in features.history:
        definitions.extend(
            (
                FeatureDefinition(
                    name=f"{history.key}_prior_impressions_log1p",
                    lane=FeatureLane.NUMERICAL,
                    family=FeatureFamily.HISTORY,
                    inputs=(history.key, "_timestamp_hour"),
                ),
                FeatureDefinition(
                    name=f"{history.key}_hours_since_previous_impression_log1p",
                    lane=FeatureLane.NUMERICAL,
                    family=FeatureFamily.HISTORY,
                    inputs=(history.key, "_timestamp_hour"),
                ),
            )
        )
        if history.within_hour:
            definitions.append(
                FeatureDefinition(
                    name=f"{history.key}_prior_hour_impressions_log1p",
                    lane=FeatureLane.NUMERICAL,
                    family=FeatureFamily.HISTORY,
                    inputs=(history.key, "_timestamp_hour"),
                )
            )
    for name in features.target_encoding.columns:
        definitions.extend(
            (
                FeatureDefinition(
                    name=f"{name}_target_logit_lift",
                    lane=FeatureLane.NUMERICAL,
                    family=FeatureFamily.TARGET,
                    inputs=(name, "click", "_timestamp_hour"),
                    uses_labels=True,
                ),
                FeatureDefinition(
                    name=f"{name}_target_evidence_log1p",
                    lane=FeatureLane.NUMERICAL,
                    family=FeatureFamily.TARGET,
                    inputs=(name, "click", "_timestamp_hour"),
                    uses_labels=True,
                ),
            )
        )
    return tuple(definitions)


def derive_categorical_features(
    frame: pl.LazyFrame,
    config: ExperimentConfig,
) -> pl.LazyFrame:
    """Create deterministic row-local categorical crosses in dependency order."""

    transformed = frame
    for cross in config.data.features.crosses:
        transformed = transformed.with_columns(
            pl.concat_str(
                [
                    pl.col(column).cast(pl.String).fill_null("__MISSING__")
                    for column in cross.columns
                ],
                separator=_CROSS_SEPARATOR,
            ).alias(cross.name)
        )
    return transformed
