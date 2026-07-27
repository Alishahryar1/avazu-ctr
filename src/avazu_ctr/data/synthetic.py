"""Schema-faithful deterministic fixtures for tests and smoke runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from avazu_ctr.data.schema import RAW_CATEGORICAL_COLUMNS


def write_synthetic_avazu(
    directory: Path,
    *,
    hours: int = 120,
    rows_per_hour: int = 20,
    seed: int = 42,
) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    start = np.datetime64("2014-10-21T00")
    rows = hours * rows_per_hour
    timestamps = start + np.repeat(np.arange(hours), rows_per_hour).astype("timedelta64[h]")
    hour_strings = [
        str(value.astype("datetime64[h]")).replace("-", "").replace("T", "")[2:]
        for value in timestamps
    ]
    data: dict[str, Any] = {
        "id": [f"{index:020d}" for index in range(rows)],
        "hour": hour_strings,
    }
    for index, column in enumerate(RAW_CATEGORICAL_COLUMNS):
        cardinality = 3 + index % 11
        data[column] = [f"{column}_{value}" for value in rng.integers(0, cardinality, rows)]
    signal = (
        np.asarray([int(value.rsplit("_", 1)[1]) for value in data["site_id"]])
        + np.asarray([int(value.rsplit("_", 1)[1]) for value in data["app_id"]])
        + np.arange(rows) // rows_per_hour
    )
    probabilities = 1.0 / (1.0 + np.exp(-(-2.0 + 0.15 * (signal % 7))))
    data["click"] = rng.binomial(1, probabilities).astype(np.int8)
    train = pl.DataFrame(data)
    train_path = directory / "train.csv"
    train.write_csv(train_path)

    test = (
        train.tail(rows_per_hour * 2)
        .drop("click")
        .with_columns(pl.Series("id", [f"test-{index:020d}" for index in range(rows_per_hour * 2)]))
    )
    test_path = directory / "test.csv"
    test.write_csv(test_path)
    return train_path, test_path
