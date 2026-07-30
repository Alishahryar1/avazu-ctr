from __future__ import annotations

import csv
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from avazu_ctr.cli import app
from avazu_ctr.profile_ffm.artifacts import copy_file, publish_directory
from avazu_ctr.profile_ffm.config import (
    ExpectedRows,
    ProfileFFMConfig,
    load_profile_ffm,
    resolved_profile_ffm_config,
)
from avazu_ctr.profile_ffm.contracts import PopulationRows
from avazu_ctr.profile_ffm.hashing import (
    hash_profile_token,
    hash_token,
    publisher_masked,
)
from avazu_ctr.profile_ffm.history import CausalHistory, history_hashes, history_tokens
from avazu_ctr.profile_ffm.pipeline import compose_profile_ffm_predictions


def test_shipped_profile_ffm_config_is_the_recorded_recipe() -> None:
    config = load_profile_ffm("configs/profile_ffm.yaml")
    expected = config.data.expected_rows

    assert config.schema_version == 1
    assert config.training.rank == 4
    assert config.training.epochs == 6
    assert config.training.learning_rate == 0.05
    assert config.training.l2 == 0.00002
    assert config.cold_publisher.training_mask_basis_points == 1800
    assert expected is not None
    assert expected.training == 40_428_967
    assert expected.scoring == 4_577_464
    assert expected.scoring_nonempty_history == 360_442
    assert expected.scoring_cold_site == 818_259
    resolved = resolved_profile_ffm_config(config)
    assert resolved["data"]["train_path"] == "data/raw/train.gz"
    assert resolved["data"]["test_path"] == "data/raw/test.gz"
    assert resolved["data"]["artifact_root"] == "artifacts/profile-ffm"


def test_profile_ffm_config_rejects_inconsistent_populations_and_rank() -> None:
    with pytest.raises(ValidationError, match="training inventory rows"):
        ExpectedRows(
            training=3,
            scoring=2,
            training_app=1,
            training_site=1,
            scoring_app=1,
            scoring_site=1,
            scoring_app_proxy=1,
            scoring_nonempty_history=1,
            scoring_cold_site=1,
        )

    raw = load_profile_ffm("configs/profile_ffm.yaml").model_dump(mode="json")
    raw["training"]["rank"] = 3
    with pytest.raises(ValidationError, match="multiple of four"):
        ProfileFFMConfig.model_validate(raw)

    raw = load_profile_ffm("configs/profile_ffm.yaml").model_dump(mode="json")
    raw["cold_publisher"]["training_mask_basis_points"] = 0
    with pytest.raises(ValidationError, match="greater than 0"):
        ProfileFFMConfig.model_validate(raw)


def test_hashing_matches_the_sparse_feature_contract() -> None:
    assert hash_token("pub_id-learned-cold") == 926_508
    assert hash_token("user_click_history2-4-101") == 931_812
    assert hash_profile_token("app_id-app") == 390_134
    assert not publisher_masked(0, 1800)
    assert sum(publisher_masked(row, 1800) for row in range(128)) == 26
    with pytest.raises(ValueError, match="basis_points"):
        publisher_masked(0, 10_001)


def test_completed_hour_history_never_exposes_current_hour_labels() -> None:
    state = CausalHistory()

    assert state.advance(hour="14102100", label="1", update_labels=True, completed_events=4) == ""
    assert state.advance(hour="14102100", label="0", update_labels=True, completed_events=4) == ""
    assert state.advance(hour="14102101", label="1", update_labels=True, completed_events=4) == "10"
    assert (
        state.advance(hour="14102200", label="0", update_labels=False, completed_events=4) == "101"
    )
    assert (
        state.advance(hour="14102201", label="1", update_labels=False, completed_events=4) == "101"
    )

    assert history_tokens(4, "101", count_threshold=30) == (
        "user_click_history2-4-101",
        "user_count-4",
    )
    assert history_tokens(31, "101", count_threshold=30) == (
        "user_click_history-31",
        "user_count-31",
    )
    assert history_hashes(4, "101", count_threshold=30, bins=1_000_000)[0] == 931_812
    with pytest.raises(ValueError, match="0 or 1"):
        CausalHistory().advance(
            hour="14102100",
            label="x",
            update_labels=True,
            completed_events=4,
        )
    with pytest.raises(ValueError, match="nondecreasing"):
        state.advance(
            hour="14102023",
            label="0",
            update_labels=True,
            completed_events=4,
        )


def _write_selector(path: Path, header: str, rows: list[tuple[str, int]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("id", header))
        writer.writerows(rows)


def test_prediction_composition_preserves_source_order_and_selectors(
    tmp_path: Path,
) -> None:
    app_selector = tmp_path / "app.csv"
    site_selector = tmp_path / "site.csv"
    _write_selector(app_selector, "use_history", [("a", 0), ("b", 1)])
    _write_selector(site_selector, "use_cold_publisher", [("c", 1), ("d", 0)])
    app_profile = tmp_path / "app-profile.txt"
    app_history = tmp_path / "app-history.txt"
    site_profile = tmp_path / "site-profile.txt"
    site_cold = tmp_path / "site-cold.txt"
    app_profile.write_text("0.100000\n0.200000\n", encoding="utf-8")
    app_history.write_text("0.300000\n0.400000\n", encoding="utf-8")
    site_profile.write_text("0.500000\n0.600000\n", encoding="utf-8")
    site_cold.write_text("0.700000\n0.800000\n", encoding="utf-8")
    output = tmp_path / "submission.csv"

    metrics = compose_profile_ffm_predictions(
        app_selector=app_selector,
        site_selector=site_selector,
        app_profile=app_profile,
        app_history=app_history,
        site_profile=site_profile,
        site_cold=site_cold,
        output=output,
    )

    assert output.read_text(encoding="utf-8") == (
        "id,click\na,0.100000\nb,0.400000\nc,0.700000\nd,0.600000\n"
    )
    assert metrics.rows == 4
    assert metrics.app_profile_rows == 1
    assert metrics.app_causal_history_rows == 1
    assert metrics.site_profile_rows == 1
    assert metrics.site_cold_publisher_rows == 1
    assert metrics.prediction_minimum == pytest.approx(0.1)
    assert metrics.prediction_maximum == pytest.approx(0.7)
    assert metrics.prediction_mean == pytest.approx(0.45)


def test_population_contract_rejects_invalid_selector_counts() -> None:
    with pytest.raises(ValidationError, match="app selector"):
        PopulationRows(
            training=2,
            scoring=2,
            training_app=1,
            training_site=1,
            scoring_app=1,
            scoring_site=1,
            scoring_app_proxy=2,
            scoring_nonempty_history=2,
            scoring_cold_site=1,
        )


def test_profile_ffm_artifacts_publish_atomically(tmp_path: Path) -> None:
    target = tmp_path / "published"
    first_stage = tmp_path / ".first.staging"
    first_stage.mkdir()
    (first_stage / "value.txt").write_text("first\n", encoding="utf-8")

    publish_directory(first_stage, target, overwrite=False)
    assert (target / "value.txt").read_text(encoding="utf-8") == "first\n"

    rejected_stage = tmp_path / ".rejected.staging"
    rejected_stage.mkdir()
    with pytest.raises(FileExistsError, match="published"):
        publish_directory(rejected_stage, target, overwrite=False)
    assert rejected_stage.is_dir()

    replacement = tmp_path / ".replacement.staging"
    replacement.mkdir()
    (replacement / "value.txt").write_text("second\n", encoding="utf-8")
    publish_directory(replacement, target, overwrite=True)
    assert (target / "value.txt").read_text(encoding="utf-8") == "second\n"
    assert not list(tmp_path.glob("*.backup"))

    source = tmp_path / "source.txt"
    destination = tmp_path / "destination.txt"
    source.write_text("source\n", encoding="utf-8")
    destination.write_text("existing\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match=r"destination\.txt"):
        copy_file(source, destination, overwrite=False)
    assert copy_file(source, destination, overwrite=True) == destination
    assert destination.read_text(encoding="utf-8") == "source\n"
    assert copy_file(destination, destination, overwrite=False) == destination


def test_profile_ffm_cli_composes_each_workflow_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_profile_ffm("configs/profile_ffm.yaml")
    config_path = tmp_path / "profile_ffm.yaml"
    config_path.write_text("{}\n", encoding="utf-8")
    preparation = tmp_path / "prepared" / "manifest.json"
    preparation.parent.mkdir()
    preparation.write_text("{}\n", encoding="utf-8")
    submission = tmp_path / "submission.csv"
    calls: list[str] = []

    def fake_load(_path: Path) -> ProfileFFMConfig:
        calls.append("load")
        return config

    def fake_prepare(
        _config: ProfileFFMConfig,
        *,
        overwrite: bool = False,
        progress: object | None = None,
    ) -> Path:
        assert overwrite is False
        assert progress is not None
        calls.append("prepare")
        return preparation

    def fake_fit(
        _config: ProfileFFMConfig,
        *,
        preparation_manifest: str | Path | None = None,
        output: str | Path | None = None,
        overwrite: bool = False,
        clean_prepared: bool = False,
        progress: object | None = None,
    ) -> Path:
        assert overwrite is False
        assert progress is not None
        assert preparation_manifest in {None, preparation}
        assert output in {None, submission}
        assert clean_prepared is False
        calls.append("fit")
        return submission

    monkeypatch.setattr("avazu_ctr.profile_ffm.cli.load_profile_ffm", fake_load)
    monkeypatch.setattr("avazu_ctr.profile_ffm.cli.prepare_profile_ffm", fake_prepare)
    monkeypatch.setattr("avazu_ctr.profile_ffm.cli.fit_predict_profile_ffm", fake_fit)
    runner = CliRunner()

    prepared = runner.invoke(app, ["profile-ffm", "prepare", str(config_path)])
    assert prepared.exit_code == 0, prepared.output
    assert "Prepared profile FFM inputs" in prepared.output

    fitted = runner.invoke(
        app,
        [
            "profile-ffm",
            "fit-predict",
            str(config_path),
            "--preparation-manifest",
            str(preparation),
            "--output",
            str(submission),
        ],
    )
    assert fitted.exit_code == 0, fitted.output
    assert "Wrote profile FFM submission" in fitted.output

    reproduced = runner.invoke(
        app,
        ["profile-ffm", "reproduce", str(config_path), "--output", str(submission)],
    )
    assert reproduced.exit_code == 0, reproduced.output
    assert calls == ["load", "prepare", "load", "fit", "load", "prepare", "fit"]
