from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from avazu_ctr.cli import app
from avazu_ctr.config import load_experiment
from avazu_ctr.config.schema import ExperimentConfig, FeatureMode, ModelKind


@pytest.mark.parametrize(
    "path",
    ["configs/baseline.yaml", "configs/champion.yaml", "configs/tuning.yaml"],
)
def test_shipped_configs_are_strict_and_current(path: str) -> None:
    config = load_experiment(path)
    assert config.schema_version == 4
    assert config.data.features.mode is FeatureMode.COMPETITION_TRANSDUCTIVE


def test_unknown_fields_and_schema_versions_are_rejected() -> None:
    raw = load_experiment("configs/champion.yaml").model_dump(mode="json")
    raw["unsupported_checkpoint"] = True
    with pytest.raises(ValidationError, match="Extra inputs"):
        ExperimentConfig.model_validate(raw)
    raw.pop("unsupported_checkpoint")
    raw["schema_version"] = 1
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(raw)


def test_stec_dimension_mismatch_fails_before_model_construction() -> None:
    raw = load_experiment("configs/champion.yaml").model_dump(mode="json")
    raw["model"]["kind"] = ModelKind.STEC
    raw["model"]["default_embedding"]["dim"] = 10
    raw["model"]["stec_heads"] = 4
    with pytest.raises(ValidationError, match="divisible"):
        ExperimentConfig.model_validate(raw)


def test_unknown_embedding_feature_fails_during_config_validation() -> None:
    raw = load_experiment("configs/champion.yaml").model_dump(mode="json")
    raw["model"]["feature_embeddings"]["device_typo"] = {
        "kind": "hash",
        "dim": 8,
        "buckets": 100,
        "hashes": 2,
    }
    with pytest.raises(ValidationError, match="inactive features"):
        ExperimentConfig.model_validate(raw)


def test_shipped_feature_plan_is_complete_and_crosses_are_bounded() -> None:
    config = load_experiment("configs/champion.yaml")
    features = config.data.features
    assert len(features.categorical_columns) == 32
    assert len(features.numerical_columns) == 31
    assert len({*features.categorical_columns, *features.numerical_columns}) == 63
    for cross in features.crosses:
        assert config.model.feature_embeddings[cross.name].kind.value == "hash"


def test_crosses_must_follow_dependency_order() -> None:
    raw = load_experiment("configs/champion.yaml").model_dump(mode="json")
    raw["data"]["features"]["crosses"][0]["columns"] = ["future_cross", "device_ip"]
    with pytest.raises(ValidationError, match="unavailable columns"):
        ExperimentConfig.model_validate(raw)


def test_crosses_cannot_repeat_an_input() -> None:
    raw = load_experiment("configs/champion.yaml").model_dump(mode="json")
    raw["data"]["features"]["crosses"][0]["columns"] = ["device_ip", "device_ip"]
    with pytest.raises(ValidationError, match="repeats an input"):
        ExperimentConfig.model_validate(raw)


def test_crosses_cannot_use_unbounded_vocabulary_embeddings() -> None:
    raw = load_experiment("configs/champion.yaml").model_dump(mode="json")
    raw["model"]["feature_embeddings"]["user_proxy"]["kind"] = "standard"
    with pytest.raises(ValidationError, match="require bounded hash embeddings"):
        ExperimentConfig.model_validate(raw)


def test_selection_and_champion_directories_must_differ() -> None:
    raw = load_experiment("configs/champion.yaml").model_dump(mode="json")
    raw["tracking"]["selection_dir"] = raw["deployment"]["champion_dir"]
    with pytest.raises(ValidationError, match="must differ"):
        ExperimentConfig.model_validate(raw)


def test_ensemble_children_must_share_the_dataset_encoding_kind() -> None:
    raw = load_experiment("configs/champion.yaml").model_dump(mode="json")
    child = deepcopy(raw["model"])
    child["default_embedding"]["kind"] = "hash"
    raw["model"]["kind"] = "ensemble"
    raw["model"]["children"] = [child]
    with pytest.raises(ValidationError, match="ensemble children"):
        ExperimentConfig.model_validate(raw)


def test_cli_exposes_final_commands() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in (
        "preprocess",
        "confirm",
        "tune",
        "candidate",
        "promote",
        "prepare-production",
        "refit",
        "predict",
        "report",
        "tensorboard",
    ):
        assert command in result.stdout


def test_python_policy_is_312() -> None:
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    assert 'requires-python = ">=3.12,<3.13"' in pyproject
    assert "torchvision" not in pyproject
    assert "pyperclip" not in pyproject
