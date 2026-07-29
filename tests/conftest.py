from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from avazu_ctr.config import load_experiment
from avazu_ctr.config.schema import (
    EmbeddingConfig,
    EmbeddingKind,
    ExperimentConfig,
    FeatureMode,
)
from avazu_ctr.data.preprocessing import (
    preprocess_evaluation,
    preprocess_production,
    temporal_windows,
)
from avazu_ctr.data.synthetic import write_synthetic_avazu


def small_config(
    root: Path,
    train_path: Path,
    test_path: Path,
    *,
    name: str = "test",
) -> ExperimentConfig:
    config = load_experiment("configs/champion.yaml")
    source_features = config.data.features
    features = source_features.model_copy(
        update={
            "mode": FeatureMode.INDUCTIVE,
            "crosses": source_features.crosses[:1],
            "frequency_columns": ("device_ip",),
            "distinct_counts": source_features.distinct_counts[:1],
            "history": source_features.history[:1],
            "target_encoding": source_features.target_encoding.model_copy(
                update={"columns": ("site_id", "app_id"), "blocks": 2}
            ),
        }
    )
    data = config.data.model_copy(
        update={
            "train_path": train_path,
            "test_path": test_path,
            "artifact_root": root,
            "shard_rows": 128,
            "minimum_frequency": 1,
            "features": features,
        }
    )
    default = EmbeddingConfig(
        kind=EmbeddingKind.HASH,
        dim=8,
        buckets=127,
        hashes=1,
    )
    active_categorical = set(features.categorical_columns)
    embeddings = {
        feature: value.model_copy(update={"buckets": 127, "dim": 8})
        for feature, value in config.model.feature_embeddings.items()
        if feature in active_categorical
    }
    architecture = config.model.dcn
    if architecture is None:
        raise AssertionError("the shipped champion must be a DCN")
    backbone = architecture.backbone.model_copy(
        update={
            "dcn_layers": 1,
            "dcn_rank": 4,
            "mlp_hidden": (16,),
            "dropout": 0.0,
        }
    )
    heads = tuple(
        head.model_copy(update={"hidden": (8,), "dropout": 0.0}) for head in architecture.heads
    )
    architecture = architecture.model_copy(
        update={
            "backbone": backbone,
            "heads": heads,
        }
    )
    model = config.model.model_copy(
        update={
            "default_embedding": default,
            "feature_embeddings": embeddings,
            "dcn": architecture,
        }
    )
    training = config.training.model_copy(
        update={
            "epochs": 1,
            "batch_size": 32,
            "device": "cpu",
            "amp": False,
            "compile_model": False,
            "deterministic_algorithms": True,
            "log_every_steps": 1,
            "early_stopping_patience": 0,
        }
    )
    tracking = config.tracking.model_copy(
        update={
            "database": root / "experiments.sqlite3",
            "tensorboard_dir": root / "tensorboard",
            "selection_dir": root / "selection",
        }
    )
    deployment = config.deployment.model_copy(update={"champion_dir": root / "champion"})
    return config.model_copy(
        update={
            "name": name,
            "data": data,
            "model": model,
            "training": training,
            "tracking": tracking,
            "deployment": deployment,
        }
    )


@pytest.fixture(scope="session")
def evaluation_project(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[ExperimentConfig, dict[str, Path]]:
    root = tmp_path_factory.mktemp("processed")
    train, test = write_synthetic_avazu(root / "raw", hours=120, rows_per_hour=6)
    config = small_config(root, train, test)
    manifests = {
        window.name: preprocess_evaluation(config, window_name=window.name)
        for window in temporal_windows(config)
    }
    return config, manifests


@pytest.fixture(scope="session")
def processed_project(
    evaluation_project: tuple[ExperimentConfig, dict[str, Path]],
) -> tuple[ExperimentConfig, Path]:
    config, manifests = evaluation_project
    return config, manifests["final_holdout"]


@pytest.fixture(scope="session")
def production_project(
    evaluation_project: tuple[ExperimentConfig, dict[str, Path]],
) -> tuple[ExperimentConfig, Path]:
    config, _ = evaluation_project
    return config, preprocess_production(config)


@pytest.fixture
def config_factory() -> Callable[..., ExperimentConfig]:
    return small_config
