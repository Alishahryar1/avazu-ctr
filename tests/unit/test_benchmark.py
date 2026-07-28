from __future__ import annotations

import json
from pathlib import Path

import pytest

from avazu_ctr.config import load_experiment
from avazu_ctr.data.preprocessing import feature_config_sha256

BENCHMARK_PATH = Path("benchmarks/champion.json")


def test_recorded_champion_matches_the_shipped_recipes() -> None:
    record = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    champion = load_experiment(record["configuration"])
    baseline = load_experiment(record["baseline_configuration"])
    budget = record["training_budget"]

    assert record["schema_version"] == 1
    assert champion.training.epochs == baseline.training.epochs == budget["epochs"] == 1
    assert champion.training.seed == baseline.training.seed == budget["seed"]
    assert champion.training.early_stopping_patience == 0
    assert baseline.training.early_stopping_patience == 0
    assert champion.training.amp_dtype == baseline.training.amp_dtype == "float16"
    assert feature_config_sha256(champion) == record["feature_contract"]["sha256"]
    assert feature_config_sha256(baseline) == record["feature_contract"]["sha256"]
    assert (
        len(champion.data.features.categorical_columns)
        == record["feature_contract"]["categorical_features"]
    )
    assert (
        len(champion.data.features.numerical_columns)
        == record["feature_contract"]["numerical_features"]
    )


def test_recorded_selection_evidence_is_internally_consistent() -> None:
    record = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    selection = record["selection"]
    incumbent = selection["incumbent"]
    candidate = selection["candidate"]
    promotion = selection["promotion"]

    for evidence in (incumbent, candidate):
        fold_losses = list(evidence["walk_forward_logloss"].values())
        assert evidence["mean_walk_forward_logloss"] == pytest.approx(sum(fold_losses) / 3)

    assert candidate["mean_walk_forward_logloss"] <= (
        incumbent["mean_walk_forward_logloss"] + promotion["fold_guard"]
    )
    assert promotion["mean_difference"] == pytest.approx(
        candidate["final_holdout_logloss"] - incumbent["final_holdout_logloss"],
        abs=2e-8,
    )
    assert promotion["mean_difference"] < 0
    assert promotion["upper_confidence_bound"] < 0
    assert record["production"]["selection_id"] == candidate["run_id"]
    for digest, expected_length in (
        (record["source_commit"], 40),
        (record["feature_contract"]["sha256"], 64),
        (incumbent["resolved_config_sha256"], 64),
        (candidate["resolved_config_sha256"], 64),
        (record["production"]["selection_evidence_sha256"], 64),
        (record["production"]["model_state_sha256"], 64),
        (record["production"]["bundle_sha256"], 64),
        (record["kaggle"]["submission_sha256"], 64),
    ):
        assert len(digest) == expected_length
        assert all(character in "0123456789abcdef" for character in digest)


def test_readme_reports_the_recorded_late_submission() -> None:
    record = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    kaggle = record["kaggle"]
    readme = Path("README.md").read_text(encoding="utf-8")

    assert kaggle["late_submission"] is True
    assert kaggle["officially_ranked"] is False
    assert f"**{kaggle['private_logloss']:.5f}**" in readme
    assert f"**{kaggle['public_logloss']:.5f}**" in readme
    assert f"`{kaggle['submission_id']}`" in readme
