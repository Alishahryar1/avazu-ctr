from __future__ import annotations

from pathlib import Path

from avazu_ctr.config.schema import ExperimentConfig
from avazu_ctr.tuning import StagedTuner


def test_staged_tuning_and_confirmation_use_the_candidate_trainer(
    evaluation_project: tuple[ExperimentConfig, dict[str, Path]],
    tmp_path: Path,
) -> None:
    config, manifests = evaluation_project
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
            "selection_dir": tmp_path / "selection",
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

    tuner = StagedTuner(tuned_config, manifests["walk_forward_0"])
    best, study = tuner.run()
    confirmation = tuner.confirm(
        study,
        [manifests[f"walk_forward_{index}"] for index in range(3)],
    )

    assert study.best_value > 0
    assert best.tuning.study_name == "integration"
    assert len(confirmation) == 1
    assert confirmation[0].mean_logloss > 0
    assert [fold.window for fold in confirmation[0].folds] == [
        "walk_forward_0",
        "walk_forward_1",
        "walk_forward_2",
    ]
