from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import polars as pl
import pytest
import torch
from polars.testing import assert_frame_equal
from torch.utils.data import DataLoader

from avazu_ctr.config import load_experiment
from avazu_ctr.config.schema import ExperimentConfig, FeatureMode, TemporalSplitConfig
from avazu_ctr.data.dataset import ParquetBatchDataset
from avazu_ctr.data.features import (
    HistoryState,
    add_causal_history,
    derive_categorical_features,
)
from avazu_ctr.data.manifest import (
    DatasetManifest,
    DatasetPurpose,
    DatasetSplit,
    FittedTableKind,
    load_manifest,
)
from avazu_ctr.data.preprocessing import (
    CANONICAL_SCHEMA_VERSION,
    preprocess_evaluation,
    scan_raw,
)
from avazu_ctr.data.split import build_temporal_windows
from avazu_ctr.data.synthetic import write_synthetic_avazu
from avazu_ctr.training.engine import steps_per_epoch


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


def test_unsupported_manifest_schema_is_rejected(
    processed_project: tuple[ExperimentConfig, Path],
    tmp_path: Path,
) -> None:
    _, manifest_path = processed_project
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw["schema_version"] = 1
    unsupported = tmp_path / "manifest.json"
    unsupported.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="Input should be 6"):
        load_manifest(unsupported)


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
    assert batch.row_ids is None
    manifest = load_manifest(manifest_path)
    assert manifest.purpose is DatasetPurpose.EVALUATION
    assert not manifest.test_shards
    device_id = manifest.categorical_columns.index("device_id")
    buckets = config.model.feature_embeddings["device_id"].buckets
    assert torch.any(batch.categorical[:, device_id].abs() >= buckets)


def test_covariate_state_is_one_wide_lookup_per_join_key(
    processed_project: tuple[ExperimentConfig, Path],
) -> None:
    config, manifest_path = processed_project
    manifest = load_manifest(manifest_path)
    lookups = [
        table for table in manifest.fitted_tables if table.kind is FittedTableKind.COVARIATE_LOOKUP
    ]

    assert len(lookups) == 1
    lookup = lookups[0]
    assert lookup.join_keys == ("device_ip",)
    assert lookup.outputs == (
        "device_ip__frequency",
        "device_ip_distinct_apps_log1p__raw",
    )
    table = pl.read_parquet(manifest_path.parent / lookup.path)
    assert tuple(table.columns) == (*lookup.join_keys, *lookup.outputs)

    source = derive_categorical_features(
        scan_raw(config.data.train_path, labelled=True).filter(
            (pl.col("_timestamp_hour") >= manifest.training_range.start)
            & (pl.col("_timestamp_hour") < manifest.training_range.end)
        ),
        config,
    ).collect()
    expected = (
        source.group_by("device_ip")
        .agg(
            pl.len().alias("device_ip__frequency"),
            pl.col("app_id").n_unique().alias("device_ip_distinct_apps_log1p__raw"),
        )
        .sort("device_ip")
    )
    assert_frame_equal(table, expected)

    expected_features = (
        source.select("_row_index", "device_ip")
        .join(expected, on="device_ip", how="left")
        .select(
            pl.col("_row_index").cast(pl.Int64),
            pl.col("device_ip__frequency")
            .cast(pl.Float64)
            .log1p()
            .cast(pl.Float32)
            .alias("device_ip_frequency_log1p"),
            pl.col("device_ip_distinct_apps_log1p__raw")
            .cast(pl.Float64)
            .log1p()
            .cast(pl.Float32)
            .alias("device_ip_distinct_apps_log1p"),
        )
        .sort("_row_index")
    )
    actual_features = pl.concat(
        pl.read_parquet(
            manifest_path.parent / shard.path,
            columns=[
                "_row_index",
                "device_ip_frequency_log1p",
                "device_ip_distinct_apps_log1p",
            ],
        )
        for shard in manifest.train_shards
    ).sort("_row_index")
    assert_frame_equal(actual_features, expected_features)


def test_fitted_queries_are_multiplexed_in_bounded_groups(
    tmp_path: Path,
    config_factory: Callable[..., ExperimentConfig],
) -> None:
    train, test = write_synthetic_avazu(tmp_path / "raw", hours=120, rows_per_hour=2)
    config = config_factory(tmp_path / "artifacts", train, test)

    with patch(
        "avazu_ctr.data.features.fitting.pl.collect_all",
        wraps=pl.collect_all,
    ) as collect_all:
        preprocess_evaluation(config)

    widths = [len(call.args[0]) for call in collect_all.call_args_list]
    assert widths
    assert max(widths) <= 4
    assert any(width > 1 for width in widths)


def test_native_shards_are_bounded_contiguous_and_ordered(
    processed_project: tuple[ExperimentConfig, Path],
) -> None:
    config, manifest_path = processed_project
    manifest = load_manifest(manifest_path)
    assert [Path(shard.path).name for shard in manifest.train_shards] == [
        f"part-{index:05d}.parquet" for index in range(len(manifest.train_shards))
    ]
    assert all(shard.rows <= config.data.shard_rows for shard in manifest.train_shards)
    rows = pl.concat(
        pl.read_parquet(manifest_path.parent / shard.path, columns=["_row_index"])
        for shard in manifest.train_shards
    )
    assert rows["_row_index"].is_sorted()


def test_loader_projects_metadata_and_coalesces_across_shards(
    processed_project: tuple[ExperimentConfig, Path],
) -> None:
    _, manifest_path = processed_project
    manifest = load_manifest(manifest_path)
    batch_size = 100
    dataset = ParquetBatchDataset(
        manifest_path,
        "train",
        batch_size,
        include_row_ids=True,
    )

    with patch("avazu_ctr.data.dataset.pl.read_parquet", wraps=pl.read_parquet) as read:
        batches = list(dataset)

    assert [batch.batch_size for batch in batches] == [
        *([batch_size] * (manifest.train_rows // batch_size)),
        manifest.train_rows % batch_size,
    ]
    assert all(batch.row_ids is not None for batch in batches)
    projected = read.call_args_list[0].kwargs["columns"]
    assert "id" in projected
    assert "_timestamp_hour" not in projected
    expected_ids = [
        row_id
        for shard in manifest.train_shards
        for row_id in pl.read_parquet(
            manifest_path.parent / shard.path,
            columns=["id"],
        )["id"].to_list()
    ]
    assert [
        row_id
        for batch in batches
        for row_id in (batch.row_ids if batch.row_ids is not None else [])
    ] == expected_ids


def test_step_budget_matches_coalesced_worker_partitions(
    processed_project: tuple[ExperimentConfig, Path],
) -> None:
    _, manifest_path = processed_project
    manifest = load_manifest(manifest_path)
    batch_size = 100

    assert steps_per_epoch(manifest, batch_size) == math.ceil(manifest.train_rows / batch_size)
    expected_two_workers = sum(
        math.ceil(sum(shard.rows for shard in manifest.train_shards[worker_id::2]) / batch_size)
        for worker_id in range(2)
    )
    assert steps_per_epoch(manifest, batch_size, 2) == expected_two_workers
    with pytest.raises(ValueError, match="num_workers"):
        steps_per_epoch(manifest, batch_size, -1)


def test_worker_partitions_emit_every_row_once_and_match_the_step_budget(
    processed_project: tuple[ExperimentConfig, Path],
) -> None:
    _, manifest_path = processed_project
    manifest = load_manifest(manifest_path)
    batch_size = 100
    batches = []
    for worker_id in range(2):
        dataset = ParquetBatchDataset(
            manifest_path,
            "train",
            batch_size,
            shuffle=True,
            include_row_ids=True,
        )
        worker = SimpleNamespace(id=worker_id, num_workers=2)
        with patch(
            "avazu_ctr.data.dataset.torch.utils.data.get_worker_info",
            return_value=worker,
        ):
            batches.extend(dataset)

    assert len(batches) == steps_per_epoch(manifest, batch_size, 2)
    assert sum(batch.batch_size for batch in batches) == manifest.train_rows
    actual_ids = [
        row_id
        for batch in batches
        for row_id in (batch.row_ids if batch.row_ids is not None else [])
    ]
    expected_ids = [
        row_id
        for shard in manifest.train_shards
        for row_id in pl.read_parquet(
            manifest_path.parent / shard.path,
            columns=["id"],
        )["id"].to_list()
    ]
    assert sorted(actual_ids) == sorted(expected_ids)


def test_manifest_rejects_ambiguous_fitted_table_columns(
    processed_project: tuple[ExperimentConfig, Path],
) -> None:
    _, manifest_path = processed_project
    manifest = load_manifest(manifest_path)
    lookup = next(
        table for table in manifest.fitted_tables if table.kind is FittedTableKind.COVARIATE_LOOKUP
    )
    invalid = lookup.model_copy(update={"outputs": lookup.join_keys})
    fitted_tables = tuple(invalid if table is lookup else table for table in manifest.fitted_tables)

    with pytest.raises(ValueError, match="must be disjoint"):
        DatasetManifest.model_validate(
            manifest.model_copy(update={"fitted_tables": fitted_tables}).model_dump(mode="json")
        )

    duplicate = lookup.model_copy(update={"path": "state/duplicate.parquet"})
    with pytest.raises(ValueError, match=r"contract.*repeated"):
        DatasetManifest.model_validate(
            manifest.model_copy(
                update={"fitted_tables": (*manifest.fitted_tables, duplicate)}
            ).model_dump(mode="json")
        )

    with pytest.raises(ValueError, match="do not exactly match"):
        DatasetManifest.model_validate(
            manifest.model_copy(
                update={
                    "fitted_tables": tuple(
                        table for table in manifest.fitted_tables if table is not lookup
                    )
                }
            ).model_dump(mode="json")
        )

    escaped_shard = manifest.train_shards[0].model_copy(update={"path": "../outside.parquet"})
    with pytest.raises(ValueError, match="normalized relative"):
        DatasetManifest.model_validate(
            manifest.model_copy(
                update={"train_shards": (escaped_shard, *manifest.train_shards[1:])}
            ).model_dump(mode="json")
        )


def test_first_target_encoding_block_has_zero_label_evidence(
    processed_project: tuple[ExperimentConfig, Path],
) -> None:
    _, manifest_path = processed_project
    manifest = load_manifest(manifest_path)
    first = pl.read_parquet(manifest_path.parent / manifest.train_shards[0].path)
    first_hour = first["_timestamp_hour"].min()
    initial = first.filter(pl.col("_timestamp_hour") == first_hour)
    for feature in ("site_id", "app_id"):
        assert initial[f"{feature}_target_logit_lift"].to_list() == pytest.approx(
            [0.0] * initial.height
        )
        assert initial[f"{feature}_target_evidence_log1p"].to_list() == pytest.approx(
            [0.0] * initial.height
        )


def test_causal_history_uses_event_order_and_truthful_impression_semantics() -> None:
    config = load_experiment("configs/champion.yaml")
    frame = pl.DataFrame(
        {
            "_row_index": [2, 0, 1],
            "_timestamp_hour": [12, 10, 10],
            "device_ip": ["ip", "ip", "ip"],
            "device_model": ["model", "model", "model"],
            "device_id": ["device", "device", "device"],
            "app_id": ["app", "app", "app"],
            "site_id": ["site", "site", "site"],
            "C14": ["ad", "ad", "ad"],
        }
    ).lazy()
    result = add_causal_history(
        derive_categorical_features(frame, config)
        .collect()
        .sort(
            "_timestamp_hour",
            "_row_index",
        ),
        config,
        HistoryState(),
    )
    assert result["user_proxy_prior_impressions_log1p"].to_list() == pytest.approx(
        [0.0, math.log(2), math.log(3)]
    )
    assert result["user_proxy_hours_since_previous_impression_log1p"].to_list() == pytest.approx(
        [0.0, 0.0, math.log(3)]
    )
    assert result["user_proxy_prior_hour_impressions_log1p"].to_list() == pytest.approx(
        [0.0, math.log(2), 0.0]
    )


def test_causal_history_is_identical_across_batch_boundaries_and_table_growth() -> None:
    config = load_experiment("configs/champion.yaml")
    frame = derive_categorical_features(
        pl.DataFrame(
            {
                "_row_index": list(range(10)),
                "_timestamp_hour": [10, 10, 10, 11, 11, 12, 12, 12, 13, 14],
                "device_ip": ["a", "b", "a", "c", "a", "b", "d", "a", "e", "a"],
                "device_model": ["m", "m", "m", "n", "m", "m", "n", "m", "m", "m"],
                "device_id": ["device"] * 10,
                "app_id": ["app"] * 10,
                "site_id": ["site"] * 10,
                "C14": ["ad"] * 10,
            }
        ).lazy(),
        config,
    ).collect()
    expected = add_causal_history(frame, config, HistoryState(initial_capacity=2))
    state = HistoryState(initial_capacity=2)
    actual = pl.concat(
        (
            add_causal_history(frame[:3], config, state),
            add_causal_history(frame[3:7], config, state),
            add_causal_history(frame[7:], config, state),
        )
    )
    history_columns = [
        column for column in expected.columns if "_prior_" in column or "_hours_since_" in column
    ]
    assert_frame_equal(actual.select(history_columns), expected.select(history_columns))
    assert all(table.capacity > 2 for table in state.tables.values())


def test_causal_history_rejects_a_partition_older_than_its_state() -> None:
    config = load_experiment("configs/champion.yaml")
    state = HistoryState()
    base = pl.DataFrame(
        {
            "_row_index": [0],
            "_timestamp_hour": [12],
            "device_ip": ["ip"],
            "device_model": ["model"],
            "device_id": ["device"],
            "app_id": ["app"],
            "site_id": ["site"],
            "C14": ["ad"],
        }
    )
    current = derive_categorical_features(base.lazy(), config).collect()
    add_causal_history(current, config, state)
    older = current.with_columns(pl.lit(11).alias("_timestamp_hour"))
    with pytest.raises(ValueError, match="temporally ordered"):
        add_causal_history(older, config, state)


def _full_feature_history_frame() -> tuple[ExperimentConfig, pl.DataFrame]:
    config = load_experiment("configs/full_features.yaml")
    frame = pl.DataFrame(
        {
            "_row_index": [0, 1, 2, 3],
            "_timestamp_hour": [10, 10, 11, 12],
            "click": [1, 0, 0, 1],
            "device_ip": ["ip"] * 4,
            "device_model": ["model"] * 4,
            "device_id": ["a99f214a"] * 4,
            "app_id": ["app"] * 4,
            "app_domain": ["app-domain"] * 4,
            "app_category": ["app-category"] * 4,
            "site_id": ["85f751fd"] * 4,
            "site_domain": ["site-domain"] * 4,
            "site_category": ["site-category"] * 4,
            "C14": ["ad"] * 4,
            "hour_of_week": [1, 1, 2, 3],
        }
    )
    return config, derive_categorical_features(frame.lazy(), config).collect()


def test_context_features_unify_inventory_and_reliable_identity() -> None:
    config = load_experiment("configs/full_features.yaml")
    frame = pl.DataFrame(
        {
            "site_id": ["85f751fd", "site"],
            "site_domain": ["site-domain", "site-domain"],
            "site_category": ["site-category", "site-category"],
            "app_id": ["app", "ecad2386"],
            "app_domain": ["app-domain", "app-domain"],
            "app_category": ["app-category", "app-category"],
            "device_id": ["a99f214a", "known-device"],
            "device_ip": ["ip-a", "ip-b"],
            "device_model": ["model-a", "model-b"],
            "C14": ["ad-a", "ad-b"],
            "hour_of_week": [1, 2],
        }
    )
    derived = derive_categorical_features(frame.lazy(), config).collect()
    assert derived["inventory_type"].to_list() == ["app", "site"]
    assert derived["publisher_id"].to_list() == ["app", "site"]
    assert derived["publisher_domain"].to_list() == ["app-domain", "site-domain"]
    assert derived["identity_kind"].to_list() == ["proxy", "device"]
    assert derived["user_id"][0].startswith("proxy\x1f")
    assert derived["user_id"][1] == "device\x1fknown-device"


def test_click_history_uses_completed_hours_across_batch_boundaries() -> None:
    config, frame = _full_feature_history_frame()
    state = HistoryState(global_prior=0.25)
    first = add_causal_history(frame[:1], config, state, use_labels=True)
    middle = add_causal_history(frame[1:3], config, state, use_labels=True)
    final = add_causal_history(frame[3:], config, state)
    result = pl.concat((first, middle, final))

    assert result["user_id_prior_clicks_log1p"].to_list() == pytest.approx(
        [0.0, 0.0, math.log(2), math.log(2)]
    )
    assert result["user_id_prior_nonclicks_log1p"].to_list() == pytest.approx(
        [0.0, 0.0, math.log(2), math.log(3)]
    )
    assert result["user_id_recent_click_pattern"].to_list() == [0, 0, 17, 34]
    assert result["user_id_hours_since_last_click_log1p"].to_list() == pytest.approx(
        [0.0, 0.0, math.log(2), math.log(3)]
    )


def test_scoring_labels_never_update_click_history() -> None:
    config, frame = _full_feature_history_frame()
    state = HistoryState(global_prior=0.25)
    add_causal_history(frame[:2], config, state, use_labels=True)
    scoring = add_causal_history(frame[2:3], config, state, use_labels=False)
    later = add_causal_history(frame[3:], config, state, use_labels=False)

    assert scoring["user_id_recent_click_pattern"].to_list() == [17]
    assert later["user_id_recent_click_pattern"].to_list() == [17]
    assert later["user_id_prior_clicks_log1p"].to_list() == pytest.approx([math.log(2)])
    assert later["user_id_prior_nonclicks_log1p"].to_list() == pytest.approx([math.log(2)])


def test_manifest_records_feature_lineage_fit_sources_and_coverage(
    processed_project: tuple[ExperimentConfig, Path],
) -> None:
    _, manifest_path = processed_project
    manifest = load_manifest(manifest_path)
    assert manifest.feature_mode is FeatureMode.INDUCTIVE
    assert tuple(feature.name for feature in manifest.features) == (
        *manifest.categorical_columns,
        *manifest.numerical_columns,
    )
    assert all(table.sources == (DatasetSplit.TRAINING,) for table in manifest.fitted_tables)
    assert set(manifest.diagnostics) == {
        DatasetSplit.TRAINING,
        DatasetSplit.VALIDATION,
    }
    assert manifest.diagnostics[DatasetSplit.TRAINING].rows == manifest.train_rows
    assert manifest.diagnostics[DatasetSplit.VALIDATION].rows == manifest.validation_rows
    assert any(feature.uses_labels for feature in manifest.features)
    assert all(
        table.uses_labels for table in manifest.fitted_tables if table.kind == "target_encoding"
    )


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
    manifest_a = load_manifest(preprocess_evaluation(config_a))
    manifest_b = load_manifest(preprocess_evaluation(config_b))
    assert [
        (item.join_keys, item.outputs, item.kind, item.sha256) for item in manifest_a.fitted_tables
    ] == [
        (item.join_keys, item.outputs, item.kind, item.sha256) for item in manifest_b.fitted_tables
    ]
    assert [item.sha256 for item in manifest_a.train_shards] == [
        item.sha256 for item in manifest_b.train_shards
    ]


def test_competition_transduction_is_explicit_label_free_and_reported(
    tmp_path: Path,
    config_factory: Callable[..., ExperimentConfig],
) -> None:
    train, test = write_synthetic_avazu(tmp_path / "raw", hours=120, rows_per_hour=2)
    labelled = pl.read_csv(
        train,
        schema_overrides={"id": pl.String, "hour": pl.String, "click": pl.Int8},
    )
    cutoff = labelled["hour"].unique().sort().tail(24).min()
    changed = labelled.with_columns(
        pl.when(pl.col("hour") >= cutoff)
        .then(pl.lit("future-only-domain"))
        .otherwise(pl.col("site_domain"))
        .alias("site_domain")
    )
    changed_path = tmp_path / "raw" / "train-with-future-domain.csv"
    changed.write_csv(changed_path)

    inductive = config_factory(tmp_path / "inductive", changed_path, test, name="inductive")
    inductive_raw = inductive.model_dump(mode="json")
    inductive_raw["model"]["feature_embeddings"]["site_domain"] = {
        "kind": "standard",
        "dim": 8,
        "buckets": 127,
        "hashes": 1,
    }
    inductive = ExperimentConfig.model_validate(inductive_raw)

    transductive_raw = inductive.model_dump(mode="json")
    transductive_raw["name"] = "transductive"
    transductive_raw["data"]["artifact_root"] = str(tmp_path / "transductive")
    transductive_raw["data"]["features"]["mode"] = FeatureMode.COMPETITION_TRANSDUCTIVE
    transductive = ExperimentConfig.model_validate(transductive_raw)

    inductive_manifest = load_manifest(preprocess_evaluation(inductive))
    transductive_manifest = load_manifest(preprocess_evaluation(transductive))
    inductive_oov = inductive_manifest.diagnostics[DatasetSplit.VALIDATION].categorical_oov
    transductive_oov = transductive_manifest.diagnostics[DatasetSplit.VALIDATION].categorical_oov
    assert inductive_oov["site_domain"].rate == 1.0
    assert transductive_oov["site_domain"].rate == 1.0
    for table in transductive_manifest.fitted_tables:
        expected = (
            (DatasetSplit.TRAINING,)
            if table.uses_labels or table.kind is FittedTableKind.VOCABULARY
            else (DatasetSplit.TRAINING, DatasetSplit.VALIDATION)
        )
        assert table.sources == expected
    assert any(
        table.sources == (DatasetSplit.TRAINING, DatasetSplit.VALIDATION)
        for table in transductive_manifest.fitted_tables
    )


def test_manifest_rejects_a_scoring_fitted_vocabulary(
    tmp_path: Path,
    config_factory: Callable[..., ExperimentConfig],
) -> None:
    train, test = write_synthetic_avazu(tmp_path / "raw", hours=120, rows_per_hour=2)
    config = config_factory(tmp_path / "artifacts", train, test)
    raw = config.model_dump(mode="json")
    raw["model"]["feature_embeddings"]["site_domain"] = {
        "kind": "standard",
        "dim": 8,
        "buckets": 127,
        "hashes": 1,
    }
    config = ExperimentConfig.model_validate(raw)
    manifest = load_manifest(preprocess_evaluation(config))
    fitted_tables = tuple(
        table.model_copy(
            update={
                "sources": (DatasetSplit.TRAINING, DatasetSplit.VALIDATION),
            }
        )
        if table.kind is FittedTableKind.VOCABULARY
        else table
        for table in manifest.fitted_tables
    )

    with pytest.raises(ValueError, match=r"expected.*training"):
        DatasetManifest.model_validate(
            manifest.model_copy(update={"fitted_tables": fitted_tables}).model_dump(mode="json")
        )


def test_canonical_cache_is_reused_and_invalidated_by_raw_checksum(
    tmp_path: Path,
    config_factory: Callable[..., ExperimentConfig],
) -> None:
    train, test = write_synthetic_avazu(tmp_path / "raw", hours=120, rows_per_hour=2)
    config = config_factory(tmp_path / "artifacts", train, test)
    preprocess_evaluation(config)

    cache = config.data.artifact_root / "cache" / "canonical" / f"v{CANONICAL_SCHEMA_VERSION}"
    assert len(list(cache.glob("train-*.parquet"))) == 1

    with patch("avazu_ctr.data.preprocessing.scan_raw", wraps=scan_raw) as cached_scan:
        preprocess_evaluation(config, overwrite=True)
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
        preprocess_evaluation(config, overwrite=True)
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
        preprocess_evaluation(config)


def test_production_dataset_uses_all_labels_and_has_no_validation(
    processed_project: tuple[ExperimentConfig, Path],
    production_project: tuple[ExperimentConfig, Path],
) -> None:
    _, evaluation_path = processed_project
    _, production_path = production_project
    evaluation = load_manifest(evaluation_path)
    production = load_manifest(production_path)

    assert production.purpose is DatasetPurpose.PRODUCTION
    assert not production.validation_shards
    assert production.validation_range is None
    assert production.test_rows > 0
    assert production.train_rows == evaluation.train_rows + evaluation.validation_rows
    assert production.training_range.start == evaluation.training_range.start
    assert evaluation.validation_range is not None
    assert production.training_range.end == evaluation.validation_range.end
