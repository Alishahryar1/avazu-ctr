from __future__ import annotations

from copy import deepcopy
from inspect import signature
from pathlib import Path

import optuna
import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from avazu_ctr.cli import app, predict_command
from avazu_ctr.config import load_experiment
from avazu_ctr.config.schema import ExperimentConfig, FeatureMode, ModelKind
from avazu_ctr.tuning.study import _sample_config, tuning_stages


@pytest.mark.parametrize(
    "path",
    [
        "configs/baseline.yaml",
        "configs/champion.yaml",
        "configs/full_features.yaml",
        "configs/ngpt.yaml",
        "configs/stec.yaml",
        "configs/tuning.yaml",
    ],
)
def test_shipped_configs_are_strict_and_current(path: str) -> None:
    config = load_experiment(path)
    assert config.schema_version == 5
    assert config.data.features.mode is FeatureMode.COMPETITION_TRANSDUCTIVE
    for feature in ("hour_of_day", "day_of_week", "day_of_month", "hour_of_week"):
        assert config.model.feature_embeddings[feature].kind.value == "hash"


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
    raw["model"]["dcn"] = None
    raw["model"]["stec"] = {"dimension": 10, "heads": 4}
    with pytest.raises(ValidationError, match="divisible"):
        ExperimentConfig.model_validate(raw)


def test_stec_rejects_frozen_batch_norm_statistics() -> None:
    raw = load_experiment("configs/stec.yaml").model_dump(mode="json")
    raw["model"]["stec"]["batch_norm_momentum"] = 0.0
    with pytest.raises(ValidationError, match="greater than 0"):
        ExperimentConfig.model_validate(raw)


def test_architecture_payloads_are_explicit_and_exclusive() -> None:
    raw = load_experiment("configs/champion.yaml").model_dump(mode="json")
    raw["model"]["kind"] = ModelKind.STEC
    raw["model"]["stec"] = {"dimension": 16, "heads": 4}
    with pytest.raises(ValidationError, match="inactive payloads"):
        ExperimentConfig.model_validate(raw)


def test_ngpt_enforces_its_optimizer_recipe() -> None:
    raw = load_experiment("configs/ngpt.yaml").model_dump(mode="json")
    raw["training"]["optimizer"]["dense"]["weight_decay"] = 1e-5
    with pytest.raises(ValidationError, match="zero weight decay"):
        ExperimentConfig.model_validate(raw)


@pytest.mark.parametrize(
    "path", ["configs/champion.yaml", "configs/stec.yaml", "configs/ngpt.yaml"]
)
def test_tuning_stages_preserve_valid_architecture_configs(path: str) -> None:
    config = load_experiment(path)
    for index, stage in enumerate(tuning_stages(config)):
        study = optuna.create_study(
            sampler=optuna.samplers.RandomSampler(seed=index),
        )
        sampled = _sample_config(config, stage, study.ask())
        ExperimentConfig.model_validate(sampled.model_dump())


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


def test_full_feature_candidate_adds_every_planned_information_family() -> None:
    config = load_experiment("configs/full_features.yaml")
    features = config.data.features
    assert config.training.compile_model
    assert len(features.categorical_columns) == 53
    assert len(features.numerical_columns) == 57
    assert len({*features.categorical_columns, *features.numerical_columns}) == 110
    assert features.context.enabled
    assert any(feature.clicks for feature in features.history)
    assert features.buckets
    assert {
        "publisher_id",
        "user_id",
        "user_id_recent_click_pattern",
        "publisher_id_target_lift_bin",
    }.issubset(features.categorical_columns)
    assert {
        "user_id_prior_ctr_logit_lift",
        "publisher_id_target_logit_lift",
        "publisher_id_distinct_users_log1p",
    }.issubset(features.numerical_columns)


def test_post_transform_categories_require_bounded_hash_embeddings() -> None:
    raw = load_experiment("configs/full_features.yaml").model_dump(mode="json")
    raw["model"]["feature_embeddings"]["user_id_prior_clicks_bin"]["kind"] = "standard"
    with pytest.raises(ValidationError, match="post-transform categorical"):
        ExperimentConfig.model_validate(raw)


def test_buckets_must_reference_compiled_numerical_features() -> None:
    raw = load_experiment("configs/full_features.yaml").model_dump(mode="json")
    raw["data"]["features"]["buckets"][0]["source"] = "future_numerical_feature"
    with pytest.raises(ValidationError, match="unavailable numerical feature"):
        ExperimentConfig.model_validate(raw)


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
    raw["model"]["dcn"] = None
    raw["model"]["ensemble"] = {"aggregation": "mean"}
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


def test_cuda_prediction_has_one_automatic_fast_path() -> None:
    parameters = signature(predict_command).parameters
    assert "device" in parameters
    assert "compile_model" not in parameters


def test_python_policy_is_312() -> None:
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    assert 'requires-python = ">=3.12,<3.13"' in pyproject
    assert "torchvision" not in pyproject
    assert "pyperclip" not in pyproject
