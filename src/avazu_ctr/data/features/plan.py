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
_CONTEXT_INPUTS = {
    "inventory_type": ("site_id",),
    "publisher_id": ("site_id", "app_id"),
    "publisher_domain": ("site_domain", "app_domain"),
    "publisher_category": ("site_category", "app_category"),
    "identity_kind": ("device_id",),
    "user_id": ("device_id", "device_ip", "device_model"),
}


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
                name=name,
                lane=FeatureLane.CATEGORICAL,
                family=FeatureFamily.CONTEXT,
                inputs=_CONTEXT_INPUTS[name],
            )
            for name in features.context_categorical_columns
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
                name=bucket.name,
                lane=FeatureLane.CATEGORICAL,
                family=FeatureFamily.BUCKET,
                inputs=(bucket.source,),
                uses_labels=bucket.source in features.label_dependent_columns,
            )
            for bucket in features.buckets
        ),
        *(
            FeatureDefinition(
                name=name,
                lane=FeatureLane.CATEGORICAL,
                family=FeatureFamily.HISTORY,
                inputs=(history.key, "click", "_timestamp_hour"),
                uses_labels=True,
            )
            for history in features.history
            if history.click_pattern_bits
            for name in (f"{history.key}_recent_click_pattern",)
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
        if history.clicks:
            definitions.extend(
                FeatureDefinition(
                    name=f"{history.key}_{suffix}",
                    lane=FeatureLane.NUMERICAL,
                    family=FeatureFamily.HISTORY,
                    inputs=(history.key, "click", "_timestamp_hour"),
                    uses_labels=True,
                )
                for suffix in (
                    "prior_clicks_log1p",
                    "prior_nonclicks_log1p",
                    "prior_ctr_logit_lift",
                    "hours_since_last_click_log1p",
                    "impressions_since_last_click_log1p",
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
    context = config.data.features.context
    if context.enabled:
        is_app = pl.col("site_id") == context.app_site_sentinel
        known_device = pl.col("device_id") != context.unknown_device_id
        proxy_id = pl.concat_str(
            (
                pl.lit("proxy"),
                pl.col("device_ip").cast(pl.String).fill_null("__MISSING__"),
                pl.col("device_model").cast(pl.String).fill_null("__MISSING__"),
            ),
            separator=_CROSS_SEPARATOR,
        )
        device_id = pl.concat_str(
            (
                pl.lit("device"),
                pl.col("device_id").cast(pl.String).fill_null("__MISSING__"),
            ),
            separator=_CROSS_SEPARATOR,
        )
        transformed = transformed.with_columns(
            pl.when(is_app).then(pl.lit("app")).otherwise(pl.lit("site")).alias("inventory_type"),
            pl.when(is_app)
            .then(pl.col("app_id"))
            .otherwise(pl.col("site_id"))
            .alias("publisher_id"),
            pl.when(is_app)
            .then(pl.col("app_domain"))
            .otherwise(pl.col("site_domain"))
            .alias("publisher_domain"),
            pl.when(is_app)
            .then(pl.col("app_category"))
            .otherwise(pl.col("site_category"))
            .alias("publisher_category"),
            pl.when(known_device)
            .then(pl.lit("device"))
            .otherwise(pl.lit("proxy"))
            .alias("identity_kind"),
            pl.when(known_device).then(device_id).otherwise(proxy_id).alias("user_id"),
        )
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
