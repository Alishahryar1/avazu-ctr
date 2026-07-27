from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from avazu_ctr.config import load_experiment
from avazu_ctr.config.schema import EmbeddingConfig, EmbeddingKind, ExperimentConfig
from avazu_ctr.data.preprocessing import preprocess
from avazu_ctr.data.synthetic import write_synthetic_avazu


def small_config(
    root: Path,
    train_path: Path,
    test_path: Path,
    *,
    name: str = "test",
) -> ExperimentConfig:
    config = load_experiment("configs/champion.yaml")
    data = config.data.model_copy(
        update={
            "train_path": train_path,
            "test_path": test_path,
            "artifact_root": root,
            "shard_rows": 128,
            "minimum_frequency": 1,
        }
    )
    default = EmbeddingConfig(kind=EmbeddingKind.STANDARD, dim=8)
    embeddings = {
        feature: value.model_copy(update={"buckets": 127, "dim": 8})
        for feature, value in config.model.feature_embeddings.items()
    }
    backbone = config.model.backbone.model_copy(
        update={
            "dcn_layers": 1,
            "dcn_rank": 4,
            "mlp_hidden": (16,),
            "dropout": 0.0,
        }
    )
    heads = tuple(
        head.model_copy(update={"hidden": (8,), "dropout": 0.0}) for head in config.model.heads
    )
    model = config.model.model_copy(
        update={
            "default_embedding": default,
            "feature_embeddings": embeddings,
            "backbone": backbone,
            "heads": heads,
        }
    )
    training = config.training.model_copy(
        update={
            "epochs": 1,
            "batch_size": 32,
            "device": "cpu",
            "amp": False,
            "deterministic_algorithms": True,
            "log_every_steps": 1,
            "early_stopping_patience": 0,
        }
    )
    tracking = config.tracking.model_copy(
        update={
            "database": root / "experiments.sqlite3",
            "tensorboard_dir": root / "tensorboard",
            "champion_dir": root / "champion",
        }
    )
    return config.model_copy(
        update={
            "name": name,
            "data": data,
            "model": model,
            "training": training,
            "tracking": tracking,
        }
    )


@pytest.fixture(scope="session")
def processed_project(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[ExperimentConfig, Path]:
    root = tmp_path_factory.mktemp("processed")
    train, test = write_synthetic_avazu(root / "raw", hours=120, rows_per_hour=6)
    config = small_config(root, train, test)
    manifest = preprocess(config)
    return config, manifest


@pytest.fixture
def config_factory() -> Callable[..., ExperimentConfig]:
    return small_config
