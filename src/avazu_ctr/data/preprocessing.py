"""Window-scoped, leakage-safe preprocessing to typed Parquet shards."""

from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import polars as pl

from avazu_ctr.config.loader import resolved_config
from avazu_ctr.config.schema import EmbeddingKind, ExperimentConfig
from avazu_ctr.data.manifest import (
    DatasetManifest,
    FittedTableManifest,
    ShardManifest,
    sha256_file,
    sha256_json,
    write_manifest,
)
from avazu_ctr.data.schema import ENGINEERED_CATEGORICAL_COLUMNS, scan_raw
from avazu_ctr.data.split import TemporalWindow, build_temporal_windows

NUMERICAL_BASE = ("hour_sin", "hour_cos")
CANONICAL_SCHEMA_VERSION = 2


@dataclass(slots=True)
class FittedState:
    categorical_columns: tuple[str, ...]
    numerical_columns: tuple[str, ...]
    cardinalities: dict[str, int]
    embedding_kinds: dict[str, str]
    vocabularies: dict[str, Path]
    counts: dict[str, Path]
    target_encodings: dict[str, Path]
    global_prior: float
    tables: list[FittedTableManifest]


@lru_cache(maxsize=8)
def _checksum_for_file_state(path: Path, size: int, modified_ns: int) -> str:
    del size, modified_ns
    return sha256_file(path)


def _raw_checksum(path: Path) -> str:
    state = path.stat()
    return _checksum_for_file_state(path.resolve(), state.st_size, state.st_mtime_ns)


def _canonical_scan(
    config: ExperimentConfig,
    path: Path,
    *,
    labelled: bool,
) -> tuple[pl.LazyFrame, str]:
    raw_sha256 = _raw_checksum(path)
    cache_dir = config.data.artifact_root / "cache" / "canonical" / f"v{CANONICAL_SCHEMA_VERSION}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    split = "train" if labelled else "test"
    canonical_path = cache_dir / f"{split}-{raw_sha256}.parquet"
    if not canonical_path.exists():
        temporary = cache_dir / f".{canonical_path.name}.{uuid.uuid4().hex}.tmp"
        try:
            scan_raw(path, labelled=labelled).sink_parquet(
                temporary,
                compression="zstd",
                statistics=True,
                row_group_size=config.data.shard_rows,
                maintain_order=True,
                engine="streaming",
            )
            temporary.replace(canonical_path)
        finally:
            temporary.unlink(missing_ok=True)
    return pl.scan_parquet(canonical_path), raw_sha256


def _within(frame: pl.LazyFrame, start: int, end: int) -> pl.LazyFrame:
    return frame.filter((pl.col("_timestamp_hour") >= start) & (pl.col("_timestamp_hour") < end))


def _temporal_windows(
    raw: pl.LazyFrame,
    config: ExperimentConfig,
) -> tuple[TemporalWindow, ...]:
    hours = (
        raw.select("_timestamp_hour")
        .unique()
        .sort("_timestamp_hour")
        .collect(engine="streaming")["_timestamp_hour"]
        .to_list()
    )
    return build_temporal_windows(hours, config.data.split)


def _validate_labels(raw: pl.LazyFrame) -> None:
    summary = raw.select(
        pl.col("click").min().alias("minimum"),
        pl.col("click").max().alias("maximum"),
        pl.col("click").null_count().alias("nulls"),
    ).collect(engine="streaming")
    if (
        summary["nulls"][0] != 0
        or summary["minimum"][0] not in {0, 1}
        or summary["maximum"][0] not in {0, 1}
    ):
        raise ValueError("raw click labels must be non-null binary values")


def temporal_windows(config: ExperimentConfig) -> tuple[TemporalWindow, ...]:
    """Return validated windows from the reusable canonical training scan."""

    raw, _ = _canonical_scan(config, config.data.train_path, labelled=True)
    return _temporal_windows(raw, config)


def _write_table(frame: pl.DataFrame, path: Path, feature: str, kind: str) -> FittedTableManifest:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path, compression="zstd", statistics=True)
    return FittedTableManifest(
        feature=feature,
        kind=kind,
        path=path.name,
        rows=frame.height,
        sha256=sha256_file(path),
    )


def _fit_state(
    train: pl.LazyFrame,
    config: ExperimentConfig,
    state_dir: Path,
) -> FittedState:
    categorical = (*config.data.categorical_columns, *ENGINEERED_CATEGORICAL_COLUMNS)
    cardinalities: dict[str, int] = {}
    kinds: dict[str, str] = {}
    vocabularies: dict[str, Path] = {}
    count_tables: dict[str, Path] = {}
    target_tables: dict[str, Path] = {}
    tables: list[FittedTableManifest] = []

    for feature in categorical:
        embedding = config.model.feature_embeddings.get(feature, config.model.default_embedding)
        kinds[feature] = embedding.kind.value
        if embedding.kind is EmbeddingKind.HASH:
            cardinalities[feature] = embedding.buckets
            continue
        counts = (
            train.group_by(feature)
            .len(name="_count")
            .filter(pl.col("_count") >= config.data.minimum_frequency)
            .sort(feature)
            .collect(engine="streaming")
            .with_row_index(f"{feature}__id", offset=1)
            .select(feature, pl.col(f"{feature}__id").cast(pl.Int64))
        )
        if counts.height > config.data.vocabulary_limit:
            raise ValueError(
                f"{feature} has {counts.height} retained values; configure a hash embedding "
                f"or raise data.vocabulary_limit"
            )
        vocab_path = state_dir / f"vocabulary_{feature}.parquet"
        tables.append(_write_table(counts, vocab_path, feature, "vocabulary"))
        vocabularies[feature] = vocab_path
        cardinalities[feature] = counts.height + 1

    for feature in config.data.count_columns:
        counts = (
            train.group_by(feature)
            .len(name=f"{feature}__training_impressions")
            .sort(feature)
            .collect(engine="streaming")
        )
        table_path = state_dir / f"counts_{feature}.parquet"
        tables.append(_write_table(counts, table_path, feature, "training_impression_count"))
        count_tables[feature] = table_path

    label_summary = train.select(
        pl.col("click").sum().alias("positive"),
        pl.len().alias("count"),
    ).collect(engine="streaming")
    positives = int(label_summary["positive"][0])
    label_count = int(label_summary["count"][0])
    if label_count == 0:
        raise ValueError("training window is empty")
    global_prior = positives / label_count

    if config.data.target_encoding.enabled:
        for feature in config.data.target_encoding.columns:
            stats = (
                train.group_by(feature)
                .agg(
                    pl.col("click").sum().cast(pl.Int64).alias("_positive"),
                    pl.len().cast(pl.Int64).alias("_count"),
                )
                .sort(feature)
                .collect(engine="streaming")
            )
            table_path = state_dir / f"target_encoding_{feature}.parquet"
            tables.append(_write_table(stats, table_path, feature, "target_encoding"))
            target_tables[feature] = table_path

    numerical = [
        *NUMERICAL_BASE,
        *(f"{name}__training_impressions_log1p" for name in config.data.count_columns),
    ]
    if config.data.target_encoding.enabled:
        numerical.extend(
            f"{name}__temporal_target_rate" for name in config.data.target_encoding.columns
        )
    return FittedState(
        categorical_columns=tuple(categorical),
        numerical_columns=tuple(numerical),
        cardinalities=cardinalities,
        embedding_kinds=kinds,
        vocabularies=vocabularies,
        counts=count_tables,
        target_encodings=target_tables,
        global_prior=global_prior,
        tables=tables,
    )


def _apply_categories(
    frame: pl.LazyFrame,
    state: FittedState,
    config: ExperimentConfig,
) -> pl.LazyFrame:
    transformed = frame
    for feature in state.categorical_columns:
        embedding = config.model.feature_embeddings.get(feature, config.model.default_embedding)
        if embedding.kind is EmbeddingKind.HASH:
            transformed = transformed.with_columns(
                pl.col(feature)
                .cast(pl.String)
                .hash(
                    seed=config.training.seed,
                    seed_1=config.training.seed + 1,
                    seed_2=config.training.seed + 2,
                    seed_3=config.training.seed + 3,
                )
                .reinterpret(signed=True)
                .alias(feature)
            )
            continue
        id_column = f"{feature}__id"
        transformed = (
            transformed.join(
                pl.scan_parquet(state.vocabularies[feature]),
                on=feature,
                how="left",
                maintain_order="left",
            )
            .drop(feature)
            .rename({id_column: feature})
            .with_columns(pl.col(feature).fill_null(0).cast(pl.Int64))
        )
    return transformed


def _apply_counts(frame: pl.LazyFrame, state: FittedState) -> pl.LazyFrame:
    transformed = frame
    for feature, path in state.counts.items():
        count_column = f"{feature}__training_impressions"
        transformed = transformed.join(
            pl.scan_parquet(path),
            on=feature,
            how="left",
            maintain_order="left",
        ).with_columns(
            pl.col(count_column)
            .fill_null(0)
            .cast(pl.Float32)
            .log1p()
            .alias(f"{count_column}_log1p")
        )
    return transformed


def _training_target_encodings(
    train: pl.LazyFrame,
    state: FittedState,
    config: ExperimentConfig,
    window: TemporalWindow,
) -> pl.LazyFrame:
    target = config.data.target_encoding
    if not target.enabled:
        return train
    width = max(1, (window.train_end - window.train_start + target.blocks - 1) // target.blocks)
    transformed = train.with_columns(
        ((pl.col("_timestamp_hour") - window.train_start) // width)
        .clip(0, target.blocks - 1)
        .alias("_te_block")
    )
    global_blocks = (
        transformed.group_by("_te_block")
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
            .otherwise(target.neutral_prior)
            .alias("_block_prior")
        )
        .select("_te_block", "_block_prior")
    )
    transformed = transformed.join(
        global_blocks,
        on="_te_block",
        how="left",
        maintain_order="left",
    )

    for feature in target.columns:
        stats = (
            transformed.group_by(feature, "_te_block")
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
        )
        output = f"{feature}__temporal_target_rate"
        transformed = (
            transformed.join(
                stats,
                on=[feature, "_te_block"],
                how="left",
                maintain_order="left",
            )
            .with_columns(
                (
                    (
                        pl.col("_previous_positive").fill_null(0)
                        + target.smoothing * pl.col("_block_prior")
                    )
                    / (pl.col("_previous_count").fill_null(0) + target.smoothing)
                )
                .cast(pl.Float32)
                .alias(output)
            )
            .drop("_previous_positive", "_previous_count")
        )
    return transformed.drop("_te_block", "_block_prior")


def _validation_target_encodings(
    frame: pl.LazyFrame,
    state: FittedState,
    config: ExperimentConfig,
) -> pl.LazyFrame:
    target = config.data.target_encoding
    transformed = frame
    if not target.enabled:
        return transformed
    for feature, path in state.target_encodings.items():
        output = f"{feature}__temporal_target_rate"
        transformed = (
            transformed.join(
                pl.scan_parquet(path),
                on=feature,
                how="left",
                maintain_order="left",
            )
            .with_columns(
                (
                    (pl.col("_positive").fill_null(0) + target.smoothing * state.global_prior)
                    / (pl.col("_count").fill_null(0) + target.smoothing)
                )
                .cast(pl.Float32)
                .alias(output)
            )
            .drop("_positive", "_count")
        )
    return transformed


def _finalize(
    frame: pl.LazyFrame,
    state: FittedState,
    config: ExperimentConfig,
    *,
    labelled: bool,
    training: bool,
    window: TemporalWindow,
) -> pl.LazyFrame:
    transformed = _apply_counts(frame, state)
    transformed = (
        _training_target_encodings(transformed, state, config, window)
        if training
        else _validation_target_encodings(transformed, state, config)
    )
    transformed = _apply_categories(transformed, state, config)
    selected: list[pl.Expr | str] = [
        pl.col("_row_index").cast(pl.Int64),
        pl.col("id").cast(pl.String),
        pl.col("_timestamp_hour").cast(pl.Int64),
        *(pl.col(name).cast(pl.Int64) for name in state.categorical_columns),
        *(
            pl.col(name).fill_nan(0.0).fill_null(0.0).cast(pl.Float32)
            for name in state.numerical_columns
        ),
    ]
    if labelled:
        selected.append(pl.col("click").cast(pl.Float32))
    return transformed.select(selected)


def _write_shards(
    frame: pl.LazyFrame,
    output_dir: Path,
    shard_rows: int,
) -> tuple[ShardManifest, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    shards: list[ShardManifest] = []
    for index, batch in enumerate(
        frame.collect_batches(chunk_size=shard_rows, maintain_order=True, engine="streaming")
    ):
        if batch.is_empty():
            continue
        path = output_dir / f"part-{index:05d}.parquet"
        batch.write_parquet(path, compression="zstd", statistics=True)
        shards.append(
            ShardManifest(
                path=path.as_posix(),
                rows=batch.height,
                sha256=sha256_file(path),
            )
        )
    if not shards:
        raise ValueError(f"no rows were written to {output_dir}")
    return tuple(shards)


def _relative_shards(shards: tuple[ShardManifest, ...], root: Path) -> tuple[ShardManifest, ...]:
    return tuple(
        shard.model_copy(update={"path": Path(shard.path).relative_to(root).as_posix()})
        for shard in shards
    )


def preprocess(
    config: ExperimentConfig,
    *,
    window_name: str = "final_holdout",
    include_test: bool = True,
    overwrite: bool = False,
) -> Path:
    raw, raw_sha256 = _canonical_scan(
        config,
        config.data.train_path,
        labelled=True,
    )
    _validate_labels(raw)
    windows = {window.name: window for window in _temporal_windows(raw, config)}
    if window_name not in windows:
        raise ValueError(f"unknown window {window_name!r}; choose one of {sorted(windows)}")
    window = windows[window_name]

    root = config.data.artifact_root / "datasets" / config.name / window.name
    if root.exists():
        if not overwrite:
            raise FileExistsError(f"{root} already exists; pass overwrite=True to replace it")
        resolved_root = root.resolve()
        artifact_root = config.data.artifact_root.resolve()
        if artifact_root not in resolved_root.parents:
            raise ValueError(f"refusing to remove output outside artifact root: {resolved_root}")
        shutil.rmtree(resolved_root)
    root.mkdir(parents=True)
    state_dir = root / "state"

    train = _within(raw, window.train_start, window.train_end)
    validation = _within(raw, window.valid_start, window.valid_end)
    state = _fit_state(train, config, state_dir)

    train_frame = _finalize(
        train,
        state,
        config,
        labelled=True,
        training=True,
        window=window,
    )
    valid_frame = _finalize(
        validation,
        state,
        config,
        labelled=True,
        training=False,
        window=window,
    )
    train_shards = _write_shards(train_frame, root / "train", config.data.shard_rows)
    validation_shards = _write_shards(valid_frame, root / "validation", config.data.shard_rows)

    test_shards: tuple[ShardManifest, ...] = ()
    if include_test and window.final_holdout and config.data.test_path.exists():
        test, _ = _canonical_scan(
            config,
            config.data.test_path,
            labelled=False,
        )
        test_frame = _finalize(
            test,
            state,
            config,
            labelled=False,
            training=False,
            window=window,
        )
        test_shards = _write_shards(test_frame, root / "test", config.data.shard_rows)

    lock_path = Path("uv.lock")
    config_dict = resolved_config(config)
    manifest = DatasetManifest(
        name=window.name,
        raw_path=str(config.data.train_path),
        raw_sha256=raw_sha256,
        train_start=window.train_start,
        train_end=window.train_end,
        valid_start=window.valid_start,
        valid_end=window.valid_end,
        categorical_columns=state.categorical_columns,
        numerical_columns=state.numerical_columns,
        cardinalities=state.cardinalities,
        embedding_kinds=state.embedding_kinds,
        train_shards=_relative_shards(train_shards, root),
        validation_shards=_relative_shards(validation_shards, root),
        test_shards=_relative_shards(test_shards, root),
        fitted_tables=tuple(
            table.model_copy(update={"path": f"state/{table.path}"}) for table in state.tables
        ),
        config_sha256=sha256_json(config_dict),
        package_lock_sha256=sha256_file(lock_path) if lock_path.exists() else None,
    )
    manifest_path = root / "manifest.json"
    write_manifest(manifest, manifest_path)
    (state_dir / "preprocessor.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "global_prior": state.global_prior,
                "categorical_columns": state.categorical_columns,
                "numerical_columns": state.numerical_columns,
                "cardinalities": state.cardinalities,
                "embedding_kinds": state.embedding_kinds,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return manifest_path
