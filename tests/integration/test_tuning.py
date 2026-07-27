from __future__ import annotations

from pathlib import Path

from avazu_ctr.config.schema import ExperimentConfig
from avazu_ctr.tuning import StagedTuner


def test_staged_tuning_and_confirmation_use_the_production_trainer(
    processed_project: tuple[ExperimentConfig, Path],
    tmp_path: Path,
) -> None:
    config, manifest = processed_project
    tuning = config.tuning.model_copy(
        update={
            "enabled": True,
            "study_name": "integration",
            "trials_per_stage": 1,
            "confirmation_candidates": 1,
        }
    )
    tracking = config.tracking.model_copy(
        update={
            "database": tmp_path / "experiments.sqlite3",
            "tensorboard_dir": tmp_path / "tensorboard",
            "champion_dir": tmp_path / "champion",
            "tensorboard": False,
        }
    )
    data = config.data.model_copy(update={"artifact_root": tmp_path})
    tuned_config = config.model_copy(
        update={
            "name": "tuning-integration",
            "data": data,
            "tracking": tracking,
            "tuning": tuning,
        }
    )

    tuner = StagedTuner(tuned_config, manifest)
    best, study = tuner.run()
    confirmation = tuner.confirm(study, [manifest])

    assert study.best_value > 0
    assert best.tuning.study_name == "integration"
    assert len(confirmation) == 1
    assert confirmation[0].mean_logloss > 0
