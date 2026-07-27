from pathlib import Path

from avazu_ctr.config.schema import ExperimentConfig
from avazu_ctr.exploration import dataset_report, run_report
from avazu_ctr.tracking import RunStore


def test_reports_are_json_and_self_contained_html(
    processed_project: tuple[ExperimentConfig, Path],
    tmp_path: Path,
) -> None:
    config, manifest = processed_project
    json_path, html_path = dataset_report(manifest, tmp_path / "dataset")
    assert json_path.exists()
    assert "<!doctype html>" in html_path.read_text(encoding="utf-8")
    RunStore(config.tracking.database)
    run_json, run_html = run_report(config.tracking.database, tmp_path / "runs")
    assert run_json.exists()
    assert run_html.exists()
