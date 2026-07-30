from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

from avazu_ctr.profile_ffm.config import ProfileFFMConfig
from avazu_ctr.profile_ffm.contracts import (
    Inventory,
    NativeSolverEvidence,
    ProfileFFMRunManifest,
    load_preparation_manifest,
    sha256_file,
)
from avazu_ctr.profile_ffm.hashing import hash_token
from avazu_ctr.profile_ffm.pipeline import fit_predict_profile_ffm
from avazu_ctr.profile_ffm.preprocessing import prepare_profile_ffm
from avazu_ctr.profile_ffm.solver import (
    SolverBuild,
    SolverJob,
    build_solver,
    run_solver_job,
)

RAW_FIELDS = (
    "id",
    "click",
    "hour",
    "C1",
    "banner_pos",
    "site_id",
    "site_domain",
    "site_category",
    "app_id",
    "app_domain",
    "app_category",
    "device_id",
    "device_ip",
    "device_model",
    "device_type",
    "device_conn_type",
    "C14",
    "C15",
    "C16",
    "C17",
    "C18",
    "C19",
    "C20",
    "C21",
)


def _row(
    row_id: str,
    *,
    hour: str,
    click: str = "0",
    app: bool,
    publisher: str,
    device_id: str,
    device_ip: str,
    device_model: str = "model",
) -> dict[str, str]:
    return {
        "id": row_id,
        "click": click,
        "hour": hour,
        "C1": "1005",
        "banner_pos": "0",
        "site_id": "85f751fd" if app else publisher,
        "site_domain": "site-domain",
        "site_category": "site-category",
        "app_id": publisher if app else "ecad2386",
        "app_domain": "app-domain",
        "app_category": "app-category",
        "device_id": device_id,
        "device_ip": device_ip,
        "device_model": device_model,
        "device_type": "1",
        "device_conn_type": "0",
        "C14": "15706",
        "C15": "320",
        "C16": "50",
        "C17": "1722",
        "C18": "0",
        "C19": "35",
        "C20": "-1",
        "C21": "79",
    }


def _write_raw(path: Path, rows: list[dict[str, str]], *, labelled: bool) -> None:
    fields = RAW_FIELDS if labelled else tuple(field for field in RAW_FIELDS if field != "click")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def _fixture_config(tmp_path: Path) -> ProfileFFMConfig:
    train = [
        _row(
            "tr-app-proxy-0",
            hour="14102100",
            click="1",
            app=True,
            publisher="app-seen",
            device_id="a99f214a",
            device_ip="proxy-a",
        ),
        _row(
            "tr-app-proxy-1",
            hour="14102100",
            click="0",
            app=True,
            publisher="app-seen",
            device_id="a99f214a",
            device_ip="proxy-a",
        ),
        _row(
            "tr-app-proxy-2",
            hour="14102101",
            click="1",
            app=True,
            publisher="app-seen",
            device_id="a99f214a",
            device_ip="proxy-a",
        ),
        _row(
            "tr-app-known-0",
            hour="14102100",
            click="0",
            app=True,
            publisher="app-other",
            device_id="known-a",
            device_ip="known-ip",
        ),
        _row(
            "tr-app-known-1",
            hour="14102101",
            click="1",
            app=True,
            publisher="app-other",
            device_id="known-a",
            device_ip="known-ip",
        ),
        _row(
            "tr-site-seen-0",
            hour="14102100",
            click="0",
            app=False,
            publisher="site-seen",
            device_id="site-user-a",
            device_ip="site-ip-a",
        ),
        _row(
            "tr-site-seen-1",
            hour="14102101",
            click="1",
            app=False,
            publisher="site-seen",
            device_id="site-user-b",
            device_ip="site-ip-b",
        ),
        _row(
            "tr-site-other",
            hour="14102101",
            click="0",
            app=False,
            publisher="site-other",
            device_id="site-user-c",
            device_ip="site-ip-c",
        ),
    ]
    test = [
        _row(
            "te-app-proxy-history",
            hour="14102200",
            app=True,
            publisher="app-seen",
            device_id="a99f214a",
            device_ip="proxy-a",
        ),
        _row(
            "te-app-proxy-empty",
            hour="14102200",
            app=True,
            publisher="app-new",
            device_id="a99f214a",
            device_ip="proxy-b",
        ),
        _row(
            "te-app-known",
            hour="14102200",
            app=True,
            publisher="app-other",
            device_id="known-a",
            device_ip="known-ip",
        ),
        _row(
            "te-site-seen",
            hour="14102200",
            app=False,
            publisher="site-seen",
            device_id="site-score-a",
            device_ip="site-score-ip-a",
        ),
        _row(
            "te-site-cold",
            hour="14102200",
            app=False,
            publisher="site-new",
            device_id="site-score-b",
            device_ip="site-score-ip-b",
        ),
        _row(
            "te-site-other",
            hour="14102200",
            app=False,
            publisher="site-other",
            device_id="site-score-c",
            device_ip="site-score-ip-c",
        ),
    ]
    train.sort(key=lambda row: row["hour"])
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    _write_raw(train_path, train, labelled=True)
    _write_raw(test_path, test, labelled=False)
    return ProfileFFMConfig.model_validate(
        {
            "schema_version": 1,
            "name": "profile-ffm-fixture",
            "data": {
                "train_path": train_path,
                "test_path": test_path,
                "artifact_root": tmp_path / "artifacts",
                "expected_rows": {
                    "training": 8,
                    "scoring": 6,
                    "training_app": 5,
                    "scoring_app": 3,
                    "training_site": 3,
                    "scoring_site": 3,
                    "scoring_app_proxy": 2,
                    "scoring_nonempty_history": 1,
                    "scoring_cold_site": 1,
                },
            },
            "cold_publisher": {
                "training_mask_basis_points": 10_000,
                "token": "pub_id-learned-cold",
            },
            "training": {
                "rank": 4,
                "learning_rate": 0.05,
                "l2": 0.00002,
                "epochs": 1,
                "executor": "auto",
            },
        }
    )


def test_preparation_preserves_profiles_history_and_publisher_selectors(
    tmp_path: Path,
) -> None:
    config = _fixture_config(tmp_path)
    manifest_path = prepare_profile_ffm(config)
    manifest = load_preparation_manifest(manifest_path, verify_artifacts=True)
    expected_rows = config.data.expected_rows

    assert expected_rows is not None
    assert manifest.config == config
    assert manifest.rows.model_dump() == expected_rows.model_dump()
    assert manifest.profiles[Inventory.APP].training_profiled_rows == 5
    assert manifest.profiles[Inventory.APP].scoring_profiled_rows == 3
    assert manifest.profiles[Inventory.SITE].training_profiled_rows == 3
    assert not (manifest_path.parent / "work").exists()

    app_selector = (manifest_path.parent / manifest.artifacts["score_app_selector"].path).read_text(
        encoding="utf-8"
    )
    assert app_selector == (
        "id,use_history\nte-app-proxy-history,1\nte-app-proxy-empty,0\nte-app-known,0\n"
    )
    site_selector = (
        manifest_path.parent / manifest.artifacts["score_site_selector"].path
    ).read_text(encoding="utf-8")
    assert site_selector == (
        "id,use_cold_publisher\nte-site-seen,0\nte-site-cold,1\nte-site-other,0\n"
    )
    history_sparse = (
        (manifest_path.parent / manifest.artifacts["score_app_history"].path)
        .read_text(encoding="utf-8")
        .splitlines()
    )
    expected_history_hash = hash_token("user_click_history2-4-101")
    assert f" 18:{expected_history_hash}:" in history_sparse[0]
    assert " 19:" in history_sparse[0]
    assert " 18:" in history_sparse[1]
    assert " 18:" not in history_sparse[2]
    with pytest.raises(FileExistsError):
        prepare_profile_ffm(config)
    profile_path = manifest_path.parent / manifest.artifacts["train_app_profile"].path
    profile_path.write_text(
        profile_path.read_text(encoding="utf-8") + "tampered\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="size mismatch"):
        load_preparation_manifest(manifest_path, verify_artifacts=True)


def test_fit_predict_publishes_checked_composition_without_model_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _fixture_config(tmp_path)
    preparation_path = prepare_profile_ffm(config)

    def fake_build(
        _config: ProfileFFMConfig,
        destination: Path,
    ) -> SolverBuild:
        destination.parent.mkdir(parents=True)
        destination.write_bytes(b"solver")
        return SolverBuild(
            binary=destination,
            evidence=NativeSolverEvidence(
                executor="native",
                compiler_version="g++ fixture",
                source_sha256="a" * 64,
                binary_sha256="b" * 64,
                build_command=("g++", "solver.cpp"),
            ),
        )

    values = {
        "app_profile": "0.100000\n",
        "app_causal_history": "0.400000\n",
        "site_profile": "0.600000\n",
        "site_cold_publisher": "0.700000\n",
    }

    def fake_run(
        _build: SolverBuild,
        job: SolverJob,
        _config: ProfileFFMConfig,
        *,
        stdout_path: Path,
        stderr_path: Path,
    ) -> tuple[str, ...]:
        rows = len(job.scoring.read_text(encoding="utf-8").splitlines())
        job.output.parent.mkdir(parents=True, exist_ok=True)
        job.output.write_text(values[job.name] * rows, encoding="utf-8")
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_path.write_text("epoch train_logloss\n", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return ("profile-ffm-solver", job.name)

    monkeypatch.setattr("avazu_ctr.profile_ffm.pipeline.build_solver", fake_build)
    monkeypatch.setattr("avazu_ctr.profile_ffm.pipeline.run_solver_job", fake_run)
    output = tmp_path / "submission.csv"
    output.write_text("existing\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match=r"submission\.csv"):
        fit_predict_profile_ffm(
            config,
            preparation_manifest=preparation_path,
            output=output,
        )
    output.unlink()
    with pytest.raises(ValueError, match="inside a preparation"):
        fit_predict_profile_ffm(
            config,
            preparation_manifest=preparation_path,
            output=preparation_path.parent / "submission.csv",
            clean_prepared=True,
        )

    written = fit_predict_profile_ffm(
        config,
        preparation_manifest=preparation_path,
        output=output,
    )

    assert written == output
    assert output.read_text(encoding="utf-8") == (
        "id,click\n"
        "te-app-proxy-history,0.400000\n"
        "te-app-proxy-empty,0.100000\n"
        "te-app-known,0.100000\n"
        "te-site-seen,0.600000\n"
        "te-site-cold,0.700000\n"
        "te-site-other,0.600000\n"
    )
    run_root = config.data.artifact_root / "run"
    manifest = ProfileFFMRunManifest.model_validate_json(
        (run_root / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest.config == config
    assert manifest.composition.app_causal_history_rows == 1
    assert manifest.composition.site_cold_publisher_rows == 1
    assert set(manifest.logs) == {
        f"{prediction}_{stream}"
        for prediction in manifest.predictions
        for stream in ("stdout", "stderr")
    }
    for artifact in (*manifest.predictions.values(), *manifest.logs.values()):
        path = run_root / artifact.path
        assert path.stat().st_size == artifact.bytes
        assert sha256_file(path) == artifact.sha256
    assert not list(run_root.rglob("*.pt"))
    assert not list(run_root.rglob("*.safetensors"))
    with pytest.raises(FileExistsError):
        fit_predict_profile_ffm(config, preparation_manifest=preparation_path)
    clean_output = tmp_path / "clean-submission.csv"
    assert (
        fit_predict_profile_ffm(
            config,
            preparation_manifest=preparation_path,
            output=clean_output,
            overwrite=True,
            clean_prepared=True,
        )
        == clean_output
    )
    assert not preparation_path.parent.exists()


def test_packaged_solver_compiles_and_is_deterministic(tmp_path: Path) -> None:
    config = _fixture_config(tmp_path)
    sparse = (
        "0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16:20:0.50000\n"
        "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17:21:0.50000\n"
    )
    training = tmp_path / "training.ffm"
    scoring = tmp_path / "scoring.ffm"
    training.write_text(sparse, encoding="utf-8")
    scoring.write_text(sparse, encoding="utf-8")
    build = build_solver(config, tmp_path / "native" / "profile_ffm_solver")
    outputs = []
    for index in range(2):
        output = tmp_path / f"predictions-{index}.txt"
        run_solver_job(
            build,
            SolverJob(
                name=f"determinism-{index}",
                training=training,
                scoring=scoring,
                output=output,
            ),
            config,
            stdout_path=tmp_path / f"stdout-{index}.log",
            stderr_path=tmp_path / f"stderr-{index}.log",
        )
        outputs.append(output.read_text(encoding="utf-8"))

    assert outputs[0] == outputs[1]
    values = [float(value) for value in outputs[0].splitlines()]
    assert len(values) == 2
    assert all(math.isfinite(value) and 0.0 < value < 1.0 for value in values)
    curve = (tmp_path / "stdout-0.log").read_text(encoding="utf-8").splitlines()
    assert curve[0] == "epoch train_logloss"
    assert len(curve) == 2
    epoch, loss = curve[1].split()
    assert epoch == "0"
    assert math.isfinite(float(loss))
