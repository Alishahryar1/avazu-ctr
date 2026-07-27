from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest
import torch
from torch.utils.data import DataLoader

from avazu_ctr.config.schema import ExperimentConfig, TemporalSplitConfig
from avazu_ctr.data.dataset import ParquetBatchDataset
from avazu_ctr.data.manifest import load_manifest
from avazu_ctr.data.preprocessing import CANONICAL_SCHEMA_VERSION, preprocess, scan_raw
from avazu_ctr.data.split import build_temporal_windows
from avazu_ctr.data.synthetic import write_synthetic_avazu


def test_expanding_windows_leave_final_day_untouched() -> None:
    windows = build_temporal_windows(list(range(240)), TemporalSplitConfig())
    assert [window.name for window in windows] == [
        "walk_forward_0",
        "walk_forward_1",
        "walk_forward_2",
        "final_holdout",
    ]
    assert windows[-1].train_end == 216
    assert (windows[-1].valid_start, windows[-1].valid_end) == (216, 240)
    assert all(window.train_end == window.valid_start for window in windows)


def test_missing_hours_are_rejected() -> None:
    with pytest.raises(ValueError, match="missing hours"):
        build_temporal_windows([0, 1, 3, 4], TemporalSplitConfig(walk_forward_folds=1))


def test_legacy_manifest_schema_is_rejected(
    processed_project: tuple[ExperimentConfig, Path],
    tmp_path: Path,
) -> None:
    _, manifest_path = processed_project
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw["schema_version"] = 1
    legacy = tmp_path / "manifest.json"
    legacy.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="Input should be 2"):
        load_manifest(legacy)


def test_processed_batches_have_typed_lanes(
    processed_project: tuple[ExperimentConfig, Path],
) -> None:
    config, manifest_path = processed_project
    batch = next(iter(DataLoader(ParquetBatchDataset(manifest_path, "train", 16), batch_size=None)))
    assert batch.categorical.dtype is torch.int64
    assert batch.numerical.dtype is torch.float32
    assert batch.labels is not None
    assert batch.labels.dtype is torch.float32
    assert batch.categorical.shape[0] == 16
    manifest = load_manifest(manifest_path)
    device_id = manifest.categorical_columns.index("device_id")
    buckets = config.model.feature_embeddings["device_id"].buckets
    assert torch.any(batch.categorical[:, device_id].abs() >= buckets)


def test_first_target_encoding_block_uses_neutral_prior(
    processed_project: tuple[ExperimentConfig, Path],
) -> None:
    _, manifest_path = processed_project
    manifest = load_manifest(manifest_path)
    first = pl.read_parquet(manifest_path.parent / manifest.train_shards[0].path)
    first_hour = first["_timestamp_hour"].min()
    initial = first.filter(pl.col("_timestamp_hour") == first_hour)
    for column in ("site_id__temporal_target_rate", "app_id__temporal_target_rate"):
        assert initial[column].to_list() == pytest.approx([0.5] * initial.height)


def test_validation_labels_and_covariates_cannot_change_fitted_training_state(
    tmp_path: Path,
    config_factory: Callable[..., ExperimentConfig],
) -> None:
    original_train, original_test = write_synthetic_avazu(
        tmp_path / "original", hours=120, rows_per_hour=4
    )
    original = pl.read_csv(
        original_train,
        schema_overrides={"id": pl.String, "hour": pl.String, "click": pl.Int8},
    )
    cutoff = original["hour"].unique().sort().tail(24).min()
    changed = original.with_columns(
        pl.when(pl.col("hour") >= cutoff)
        .then(1 - pl.col("click"))
        .otherwise(pl.col("click"))
        .alias("click"),
        pl.when(pl.col("hour") >= cutoff)
        .then(pl.lit("future-only-site"))
        .otherwise(pl.col("site_id"))
        .alias("site_id"),
    )
    changed_train = tmp_path / "changed" / "train.csv"
    changed_train.parent.mkdir()
    changed.write_csv(changed_train)
    changed_test = tmp_path / "changed" / "test.csv"
    pl.read_csv(
        original_test,
        schema_overrides={"id": pl.String, "hour": pl.String},
    ).write_csv(changed_test)

    config_a = config_factory(tmp_path / "artifacts-a", original_train, original_test, name="a")
    config_b = config_factory(tmp_path / "artifacts-b", changed_train, changed_test, name="b")
    manifest_a = load_manifest(preprocess(config_a))
    manifest_b = load_manifest(preprocess(config_b))
    assert [(item.feature, item.kind, item.sha256) for item in manifest_a.fitted_tables] == [
        (item.feature, item.kind, item.sha256) for item in manifest_b.fitted_tables
    ]
    assert [item.sha256 for item in manifest_a.train_shards] == [
        item.sha256 for item in manifest_b.train_shards
    ]


def test_canonical_cache_is_reused_and_invalidated_by_raw_checksum(
    tmp_path: Path,
    config_factory: Callable[..., ExperimentConfig],
) -> None:
    train, test = write_synthetic_avazu(tmp_path / "raw", hours=120, rows_per_hour=2)
    config = config_factory(tmp_path / "artifacts", train, test)
    preprocess(config)

    cache = config.data.artifact_root / "cache" / "canonical" / f"v{CANONICAL_SCHEMA_VERSION}"
    assert len(list(cache.glob("train-*.parquet"))) == 1

    with patch("avazu_ctr.data.preprocessing.scan_raw", wraps=scan_raw) as cached_scan:
        preprocess(config, overwrite=True)
    cached_scan.assert_not_called()

    (
        pl.read_csv(train, infer_schema=False)
        .with_row_index("_row")
        .with_columns(
            pl.when(pl.col("_row") == 0)
            .then(pl.lit("changed-id"))
            .otherwise(pl.col("id"))
            .alias("id")
        )
        .drop("_row")
        .write_csv(train)
    )
    with patch("avazu_ctr.data.preprocessing.scan_raw", wraps=scan_raw) as refreshed_scan:
        preprocess(config, overwrite=True)
    assert refreshed_scan.call_count == 1
    assert len(list(cache.glob("train-*.parquet"))) == 2


def test_nonbinary_labels_are_rejected_before_feature_fitting(
    tmp_path: Path,
    config_factory: Callable[..., ExperimentConfig],
) -> None:
    train, test = write_synthetic_avazu(tmp_path / "raw", hours=120, rows_per_hour=2)
    invalid = (
        pl.read_csv(train, infer_schema=False)
        .with_row_index("_row")
        .with_columns(
            pl.when(pl.col("_row") == 0).then(pl.lit("2")).otherwise(pl.col("click")).alias("click")
        )
        .drop("_row")
    )
    invalid.write_csv(train)
    config = config_factory(tmp_path / "artifacts", train, test)
    with pytest.raises(ValueError, match="binary"):
        preprocess(config)
