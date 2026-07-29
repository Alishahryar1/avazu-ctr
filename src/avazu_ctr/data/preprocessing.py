"""Window-scoped, leakage-safe preprocessing to typed Parquet shards."""

from __future__ import annotations

import json
import shutil
import uuid
from functools import lru_cache
from pathlib import Path

import polars as pl

from avazu_ctr.config.loader import resolved_config
from avazu_ctr.config.schema import EmbeddingKind, ExperimentConfig, FeatureMode
from avazu_ctr.data.features import (
    FittedFeatureState,
    FittedFeatureTransformer,
    HistoryState,
    derive_categorical_features,
    feature_definitions,
    fit_feature_state,
    scan_with_causal_history,
)
from avazu_ctr.data.manifest import (
    CategoricalEncodingContract,
    DatasetManifest,
    DatasetPurpose,
    DatasetSplit,
    HourRange,
    OovStatistic,
    RawSource,
    ShardManifest,
    SplitDiagnostics,
    load_manifest,
    population_sha256,
    sha256_file,
    sha256_json,
    write_manifest,
)
from avazu_ctr.data.schema import scan_raw
from avazu_ctr.data.split import TemporalWindow, build_temporal_windows

CANONICAL_SCHEMA_VERSION = 4
FEATURE_SCHEMA_VERSION = 5


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


def feature_config_sha256(config: ExperimentConfig) -> str:
    """Hash only configuration fields that change processed feature values."""

    categorical = config.data.features.categorical_columns
    feature_recipe = config.data.features.model_dump(mode="json")
    if not config.data.features.context.enabled:
        feature_recipe.pop("context")
    if not config.data.features.buckets:
        feature_recipe.pop("buckets")
    for history in feature_recipe["history"]:
        if not history["clicks"]:
            history.pop("clicks")
        if not history["click_pattern_bits"]:
            history.pop("click_pattern_bits")
    embedding_kinds = {
        feature: config.model.feature_embeddings.get(
            feature,
            config.model.default_embedding,
        ).kind.value
        for feature in categorical
    }
    return sha256_json(
        {
            "schema_version": FEATURE_SCHEMA_VERSION,
            "categorical_encoding": CategoricalEncodingContract().model_dump(mode="json"),
            "features": feature_recipe,
            "compiled_features": [
                feature.model_dump(mode="json") for feature in feature_definitions(config)
            ],
            "minimum_frequency": config.data.minimum_frequency,
            "vocabulary_limit": config.data.vocabulary_limit,
            "embedding_kinds": embedding_kinds,
            "hash_seed": config.training.seed,
        }
    )


def _select_model_columns(
    transformed: pl.LazyFrame,
    state: FittedFeatureState,
    *,
    labelled: bool,
) -> pl.LazyFrame:
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


def _write_feature_shards(
    frame: pl.LazyFrame,
    output_dir: Path,
    shard_rows: int,
    *,
    state: FittedFeatureState,
    transformer: FittedFeatureTransformer,
    history: HistoryState,
    config: ExperimentConfig,
    labelled: bool,
    training: bool,
) -> tuple[ShardManifest, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    shards: list[ShardManifest] = []
    transformed = _select_model_columns(
        transformer.transform(
            scan_with_causal_history(
                frame,
                config,
                history,
                chunk_size=shard_rows,
                use_labels=training,
            ),
            training=training,
        ),
        state,
        labelled=labelled,
    )
    for index, batch in enumerate(
        transformed.collect_batches(
            chunk_size=shard_rows,
            maintain_order=True,
            engine="streaming",
        )
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
        del batch
    if not shards:
        raise ValueError(f"no rows were written to {output_dir}")
    return tuple(shards)


def _covariate_sources(
    mode: FeatureMode,
    scoring_split: DatasetSplit,
) -> tuple[DatasetSplit, ...]:
    return (
        (DatasetSplit.TRAINING,)
        if mode is FeatureMode.INDUCTIVE
        else (DatasetSplit.TRAINING, scoring_split)
    )


def _covariate_reference(
    train: pl.LazyFrame,
    scoring: pl.LazyFrame,
    config: ExperimentConfig,
) -> pl.LazyFrame:
    columns = config.data.features.pre_transform_categorical_columns
    if config.data.features.mode is FeatureMode.INDUCTIVE:
        return train.select(columns)
    return pl.concat(
        (train.select(columns), scoring.select(columns)),
        how="vertical",
    )


def _split_diagnostics(
    shards: tuple[ShardManifest, ...],
    state: FittedFeatureState,
) -> SplitDiagnostics:
    rows = sum(shard.rows for shard in shards)
    vocabulary_features = tuple(
        feature
        for feature, kind in state.embedding_kinds.items()
        if kind == EmbeddingKind.STANDARD.value
    )
    if not vocabulary_features:
        return SplitDiagnostics(rows=rows, categorical_oov={})
    summary = (
        pl.scan_parquet([Path(shard.path) for shard in shards])
        .select(*((pl.col(feature) == 0).sum().alias(feature) for feature in vocabulary_features))
        .collect(engine="streaming")
    )
    return SplitDiagnostics(
        rows=rows,
        categorical_oov={
            feature: OovStatistic(
                rows=rows,
                unknown_rows=int(summary[feature][0]),
                rate=int(summary[feature][0]) / rows,
            )
            for feature in vocabulary_features
        },
    )


def _relative_shards(shards: tuple[ShardManifest, ...], root: Path) -> tuple[ShardManifest, ...]:
    return tuple(
        shard.model_copy(update={"path": Path(shard.path).relative_to(root).as_posix()})
        for shard in shards
    )


def _staging_root(root: Path, artifact_root: Path, *, overwrite: bool) -> Path:
    resolved_root = root.resolve()
    resolved_artifact_root = artifact_root.resolve()
    if (
        resolved_root == resolved_artifact_root
        or resolved_artifact_root not in resolved_root.parents
    ):
        raise ValueError(f"dataset output must be below the artifact root: {resolved_root}")
    if root.exists() and not overwrite:
        raise FileExistsError(f"{root} already exists; pass overwrite=True to replace it")
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = root.parent / f".{root.name}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    return staging


def _publish_dataset(staging: Path, root: Path) -> Path:
    backup = root.parent / f".{root.name}-{uuid.uuid4().hex}.backup"
    replaced = root.exists()
    if replaced:
        root.replace(backup)
    try:
        staging.replace(root)
        load_manifest(root / "manifest.json", verify_shards=True)
    except Exception:
        if root.exists():
            shutil.rmtree(root)
        if replaced:
            backup.replace(root)
        raise
    if replaced:
        shutil.rmtree(backup)
    return root / "manifest.json"


def _write_preprocessor(
    state: FittedFeatureState,
    path: Path,
    *,
    purpose: DatasetPurpose,
    feature_mode: FeatureMode,
) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 5,
                "purpose": purpose.value,
                "feature_mode": feature_mode.value,
                "categorical_encoding": CategoricalEncodingContract().model_dump(mode="json"),
                "global_prior": state.global_prior,
                "categorical_columns": state.categorical_columns,
                "numerical_columns": state.numerical_columns,
                "cardinalities": state.cardinalities,
                "embedding_kinds": state.embedding_kinds,
                "features": [feature.model_dump(mode="json") for feature in state.definitions],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _package_lock_sha256() -> str | None:
    lock_path = Path("uv.lock")
    return sha256_file(lock_path) if lock_path.exists() else None


def preprocess_evaluation(
    config: ExperimentConfig,
    *,
    window_name: str = "final_holdout",
    overwrite: bool = False,
) -> Path:
    """Fit one train/validation window without touching competition test data."""

    raw, raw_sha256 = _canonical_scan(config, config.data.train_path, labelled=True)
    _validate_labels(raw)
    windows = {window.name: window for window in _temporal_windows(raw, config)}
    if window_name not in windows:
        raise ValueError(f"unknown window {window_name!r}; choose one of {sorted(windows)}")
    window = windows[window_name]
    training_range = HourRange(start=window.train_start, end=window.train_end)
    validation_range = HourRange(start=window.valid_start, end=window.valid_end)

    root = config.data.artifact_root / "datasets" / config.name / window.name
    staging = _staging_root(root, config.data.artifact_root, overwrite=overwrite)
    try:
        state_dir = staging / "state"
        train = derive_categorical_features(
            _within(raw, training_range.start, training_range.end),
            config,
        )
        validation = derive_categorical_features(
            _within(raw, validation_range.start, validation_range.end),
            config,
        )
        covariate_sources = _covariate_sources(
            config.data.features.mode,
            DatasetSplit.VALIDATION,
        )
        state = fit_feature_state(
            train,
            _covariate_reference(train, validation, config),
            config,
            state_dir,
            covariate_sources=covariate_sources,
            training_range=training_range,
        )
        transformer = FittedFeatureTransformer(state, config, training_range)
        history = HistoryState.for_expected_rows(
            state.training_rows,
            global_prior=state.global_prior,
        )
        train_shards = _write_feature_shards(
            train,
            staging / "train",
            config.data.shard_rows,
            state=state,
            transformer=transformer,
            history=history,
            config=config,
            labelled=True,
            training=True,
        )
        validation_shards = _write_feature_shards(
            validation,
            staging / "validation",
            config.data.shard_rows,
            state=state,
            transformer=transformer,
            history=history,
            config=config,
            labelled=True,
            training=False,
        )
        manifest = DatasetManifest(
            name=window.name,
            purpose=DatasetPurpose.EVALUATION,
            feature_mode=config.data.features.mode,
            categorical_encoding=CategoricalEncodingContract(),
            labelled_source=RawSource(path=str(config.data.train_path), sha256=raw_sha256),
            training_range=training_range,
            validation_range=validation_range,
            training_population_sha256=population_sha256(
                raw_sha256,
                split="labelled",
                hour_range=training_range,
            ),
            validation_population_sha256=population_sha256(
                raw_sha256,
                split="labelled",
                hour_range=validation_range,
            ),
            categorical_columns=state.categorical_columns,
            numerical_columns=state.numerical_columns,
            cardinalities=state.cardinalities,
            embedding_kinds=state.embedding_kinds,
            features=state.definitions,
            diagnostics={
                DatasetSplit.TRAINING: _split_diagnostics(train_shards, state),
                DatasetSplit.VALIDATION: _split_diagnostics(validation_shards, state),
            },
            train_shards=_relative_shards(train_shards, staging),
            validation_shards=_relative_shards(validation_shards, staging),
            fitted_tables=tuple(
                table.model_copy(update={"path": f"state/{table.path}"}) for table in state.tables
            ),
            resolved_config_sha256=sha256_json(resolved_config(config)),
            feature_config_sha256=feature_config_sha256(config),
            package_lock_sha256=_package_lock_sha256(),
        )
        write_manifest(manifest, staging / "manifest.json")
        _write_preprocessor(
            state,
            state_dir / "preprocessor.json",
            purpose=DatasetPurpose.EVALUATION,
            feature_mode=config.data.features.mode,
        )
        return _publish_dataset(staging, root)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def preprocess_production(
    config: ExperimentConfig,
    *,
    overwrite: bool = False,
) -> Path:
    """Fit deployable features on labels and the explicitly declared covariate scope."""

    raw, raw_sha256 = _canonical_scan(config, config.data.train_path, labelled=True)
    _validate_labels(raw)
    windows = _temporal_windows(raw, config)
    final_window = windows[-1]
    training_range = HourRange(start=final_window.train_start, end=final_window.valid_end)
    test, test_sha256 = _canonical_scan(config, config.data.test_path, labelled=False)

    root = config.data.artifact_root / "datasets" / config.name / "production"
    staging = _staging_root(root, config.data.artifact_root, overwrite=overwrite)
    try:
        state_dir = staging / "state"
        train = derive_categorical_features(
            _within(raw, training_range.start, training_range.end),
            config,
        )
        test = derive_categorical_features(test, config)
        covariate_sources = _covariate_sources(
            config.data.features.mode,
            DatasetSplit.PREDICTION,
        )
        state = fit_feature_state(
            train,
            _covariate_reference(train, test, config),
            config,
            state_dir,
            covariate_sources=covariate_sources,
            training_range=training_range,
        )
        transformer = FittedFeatureTransformer(state, config, training_range)
        history = HistoryState.for_expected_rows(
            state.training_rows,
            global_prior=state.global_prior,
        )
        train_shards = _write_feature_shards(
            train,
            staging / "train",
            config.data.shard_rows,
            state=state,
            transformer=transformer,
            history=history,
            config=config,
            labelled=True,
            training=True,
        )
        test_shards = _write_feature_shards(
            test,
            staging / "test",
            config.data.shard_rows,
            state=state,
            transformer=transformer,
            history=history,
            config=config,
            labelled=False,
            training=False,
        )
        manifest = DatasetManifest(
            name="production",
            purpose=DatasetPurpose.PRODUCTION,
            feature_mode=config.data.features.mode,
            categorical_encoding=CategoricalEncodingContract(),
            labelled_source=RawSource(path=str(config.data.train_path), sha256=raw_sha256),
            prediction_source=RawSource(path=str(config.data.test_path), sha256=test_sha256),
            training_range=training_range,
            training_population_sha256=population_sha256(
                raw_sha256,
                split="labelled",
                hour_range=training_range,
            ),
            test_population_sha256=population_sha256(
                test_sha256,
                split="prediction",
                hour_range=None,
            ),
            categorical_columns=state.categorical_columns,
            numerical_columns=state.numerical_columns,
            cardinalities=state.cardinalities,
            embedding_kinds=state.embedding_kinds,
            features=state.definitions,
            diagnostics={
                DatasetSplit.TRAINING: _split_diagnostics(train_shards, state),
                DatasetSplit.PREDICTION: _split_diagnostics(test_shards, state),
            },
            train_shards=_relative_shards(train_shards, staging),
            test_shards=_relative_shards(test_shards, staging),
            fitted_tables=tuple(
                table.model_copy(update={"path": f"state/{table.path}"}) for table in state.tables
            ),
            resolved_config_sha256=sha256_json(resolved_config(config)),
            feature_config_sha256=feature_config_sha256(config),
            package_lock_sha256=_package_lock_sha256(),
        )
        write_manifest(manifest, staging / "manifest.json")
        _write_preprocessor(
            state,
            state_dir / "preprocessor.json",
            purpose=DatasetPurpose.PRODUCTION,
            feature_mode=config.data.features.mode,
        )
        return _publish_dataset(staging, root)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
