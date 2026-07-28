"""Raw Avazu schema and deterministic feature expressions."""

from __future__ import annotations

import math
from pathlib import Path

import polars as pl

from avazu_ctr.config.schema import (
    AVAZU_CATEGORICAL_COLUMNS,
)
from avazu_ctr.config.schema import (
    TIME_CATEGORICAL_COLUMNS as TIME_CATEGORICAL_COLUMNS,
)
from avazu_ctr.config.schema import (
    TIME_NUMERICAL_COLUMNS as TIME_NUMERICAL_COLUMNS,
)

RAW_CATEGORICAL_COLUMNS = AVAZU_CATEGORICAL_COLUMNS
REQUIRED_TRAIN_COLUMNS = ("id", "click", "hour", *RAW_CATEGORICAL_COLUMNS)
REQUIRED_TEST_COLUMNS = ("id", "hour", *RAW_CATEGORICAL_COLUMNS)


def scan_raw(path: Path, *, labelled: bool) -> pl.LazyFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    schema: dict[str, type[pl.DataType]] = dict.fromkeys(REQUIRED_TEST_COLUMNS, pl.String)
    if labelled:
        schema["click"] = pl.Int8
    frame = pl.scan_csv(
        path,
        schema_overrides=schema,
        infer_schema=False,
        row_index_name="_row_index",
    )
    expected = set(REQUIRED_TRAIN_COLUMNS if labelled else REQUIRED_TEST_COLUMNS)
    missing = expected.difference(frame.collect_schema().names())
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return add_time_features(frame)


def add_time_features(frame: pl.LazyFrame) -> pl.LazyFrame:
    parsed = (pl.col("hour") + "00").str.strptime(pl.Datetime("us"), "%y%m%d%H%M", strict=True)
    return frame.with_columns(
        parsed.alias("_timestamp"),
        (parsed.dt.epoch("s") // 3600).cast(pl.Int64).alias("_timestamp_hour"),
        parsed.dt.hour().cast(pl.Int64).alias("hour_of_day"),
        parsed.dt.weekday().cast(pl.Int64).alias("day_of_week"),
        parsed.dt.day().cast(pl.Int64).alias("day_of_month"),
        ((parsed.dt.weekday().cast(pl.Int64) - 1) * 24 + parsed.dt.hour().cast(pl.Int64)).alias(
            "hour_of_week"
        ),
        (pl.col("hour").cast(pl.Int64) % 100).cast(pl.Int16).alias("_raw_hour"),
    ).with_columns(
        (pl.col("_raw_hour") * (2.0 * math.pi / 24.0)).sin().cast(pl.Float32).alias("hour_sin"),
        (pl.col("_raw_hour") * (2.0 * math.pi / 24.0)).cos().cast(pl.Float32).alias("hour_cos"),
    )
