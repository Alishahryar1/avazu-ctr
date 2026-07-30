from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path

import pytest

from avazu_ctr.config import load_experiment
from avazu_ctr.data.preprocessing import feature_config_sha256
from avazu_ctr.profile_ffm.config import load_profile_ffm, profile_ffm_config_sha256

BENCHMARK_PATH = Path("benchmarks/champion.json")
NEURAL_BENCHMARK_PATH = Path("benchmarks/senet_dcnv2.json")


def test_senet_dcnv2_recipe_matches_its_contract() -> None:
    record = json.loads(NEURAL_BENCHMARK_PATH.read_text(encoding="utf-8"))
    recipe = record["current_recipe"]
    champion = load_experiment(recipe["configuration"])
    baseline = load_experiment(recipe["baseline_configuration"])
    budget = record["training_budget"]

    assert record["schema_version"] == 2
    assert champion.training.epochs == baseline.training.epochs == budget["epochs"] == 1
    assert champion.training.seed == baseline.training.seed == budget["seed"]
    assert champion.training.early_stopping_patience == 0
    assert baseline.training.early_stopping_patience == 0
    assert champion.training.amp_dtype == baseline.training.amp_dtype == "float16"
    assert feature_config_sha256(champion) == recipe["feature_contract"]["sha256"]
    assert feature_config_sha256(baseline) == recipe["feature_contract"]["sha256"]
    assert (
        len(champion.data.features.categorical_columns)
        == recipe["feature_contract"]["categorical_features"]
    )
    assert (
        len(champion.data.features.numerical_columns)
        == recipe["feature_contract"]["numerical_features"]
    )


def test_recorded_selection_evidence_is_internally_consistent() -> None:
    record = json.loads(NEURAL_BENCHMARK_PATH.read_text(encoding="utf-8"))
    recorded = record["recorded_champion"]
    selection = recorded["selection"]
    incumbent = selection["incumbent"]
    candidate = selection["candidate"]
    promotion = selection["promotion"]

    for run_evidence in (incumbent, candidate):
        fold_losses = list(run_evidence["walk_forward_logloss"].values())
        assert run_evidence["mean_walk_forward_logloss"] == pytest.approx(sum(fold_losses) / 3)

    assert candidate["mean_walk_forward_logloss"] <= (
        incumbent["mean_walk_forward_logloss"] + promotion["fold_guard"]
    )
    assert promotion["mean_difference"] == pytest.approx(
        candidate["final_holdout_logloss"] - incumbent["final_holdout_logloss"],
        abs=2e-8,
    )
    assert promotion["mean_difference"] < 0
    assert promotion["upper_confidence_bound"] < 0
    assert recorded["production"]["selection_id"] == candidate["run_id"]
    for digest, expected_length in (
        (recorded["source_commit"], 40),
        (recorded["feature_contract"]["sha256"], 64),
        (incumbent["resolved_config_sha256"], 64),
        (candidate["resolved_config_sha256"], 64),
        (recorded["production"]["selection_evidence_sha256"], 64),
        (recorded["production"]["model_state_sha256"], 64),
        (recorded["production"]["bundle_sha256"], 64),
        (recorded["kaggle"]["submission_sha256"], 64),
    ):
        assert len(digest) == expected_length
        assert all(character in "0123456789abcdef" for character in digest)


def test_profile_ffm_champion_record_is_internally_consistent() -> None:
    record = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    config = load_profile_ffm(record["configuration"])
    population = record["population"]
    composition = record["prediction_composition"]
    progression = record["score_progression"]
    kaggle = record["kaggle"]

    assert record["schema_version"] == 1
    assert record["model_family"] == "profile-ffm"
    assert profile_ffm_config_sha256(config) == record["config_sha256"]
    assert config.training.rank == record["training"]["rank"]
    assert config.training.learning_rate == record["training"]["learning_rate"]
    assert config.training.l2 == record["training"]["l2"]
    assert config.training.epochs == record["training"]["epochs"]
    assert (
        population["training_app_rows"] + population["training_site_rows"]
        == population["training_rows"]
    )
    assert (
        population["scoring_app_rows"] + population["scoring_site_rows"]
        == population["scoring_rows"]
    )
    assert (
        composition["app_profile_rows"] + composition["app_causal_history_rows"]
        == population["scoring_app_rows"]
    )
    assert (
        composition["site_profile_rows"] + composition["site_cold_publisher_rows"]
        == population["scoring_site_rows"]
    )
    assert all(
        later["public_logloss"] < earlier["public_logloss"]
        and later["private_logloss"] < earlier["private_logloss"]
        for earlier, later in pairwise(progression)
    )
    assert progression[-1]["submission_id"] == kaggle["submission_id"]
    assert progression[-1]["public_logloss"] == kaggle["public_logloss"]
    assert progression[-1]["private_logloss"] == kaggle["private_logloss"]
    assert kaggle["submission_rows"] == population["scoring_rows"]
    for digest in (record["config_sha256"], kaggle["submission_sha256"]):
        assert len(digest) == 64
        assert all(character in "0123456789abcdef" for character in digest)


def test_readme_reports_the_profile_ffm_champion() -> None:
    record = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    kaggle = record["kaggle"]
    readme = Path("README.md").read_text(encoding="utf-8")

    assert f"**{kaggle['private_logloss']:.5f}**" in readme
    assert f"**{kaggle['public_logloss']:.5f}**" in readme
    assert f"`{kaggle['submission_id']}`" in readme
