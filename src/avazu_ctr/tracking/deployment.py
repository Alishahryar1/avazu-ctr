"""Atomic deployment of a validated production bundle."""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

from avazu_ctr.data.manifest import DatasetManifest, sha256_file
from avazu_ctr.inference.bundle import LoadedBundle, load_bundle
from avazu_ctr.tracking.evidence import load_selection
from avazu_ctr.tracking.store import RunStore


def deploy_bundle(
    staged_path: str | Path,
    champion_path: str | Path,
    *,
    selection_path: str | Path,
    store: RunStore,
) -> LoadedBundle:
    staged = Path(staged_path)
    if staged.is_file():
        staged = staged.parent
    staged = staged.resolve()
    champion = Path(champion_path).resolve()
    if staged == champion:
        raise ValueError("staged and champion directories must differ")
    if staged == Path(staged.anchor) or champion == Path(champion.anchor):
        raise ValueError("refusing to operate on a filesystem root")
    candidate = load_bundle(staged)
    selection = load_selection(selection_path)
    active_selection = store.active_selection()
    if (
        active_selection is None
        or active_selection["candidate_run_id"] != candidate.metadata["selection_id"]
        or active_selection["candidate_evidence_sha256"] != candidate.metadata["selection_sha256"]
        or selection.evidence.selection_id != candidate.metadata["selection_id"]
        or selection.evidence_sha256 != candidate.metadata["selection_sha256"]
    ):
        raise ValueError("active selection changed before deployment")
    refit_plan = candidate.metadata["refit_plan"]
    refit_run = store.run(str(candidate.metadata["refit_run_id"]))
    recorded_plan = json.loads(refit_run["plan_json"])
    recorded_summary = json.loads(refit_run["summary_json"])
    recorded_manifest = DatasetManifest.model_validate_json(refit_run["dataset_json"])
    if (
        refit_run["status"] != "completed"
        or refit_run["kind"] != "production_refit"
        or refit_run["parent_run_id"] != selection.evidence.holdout.run_id
        or refit_run["config_sha256"] != selection.evidence.confirmation.config_sha256
        or recorded_manifest != candidate.manifest
        or recorded_plan.get("mode") != "production_refit"
        or recorded_plan.get("validation") is not False
        or recorded_plan.get("early_stopping") is not False
        or recorded_plan.get("budget_source_run_id") != selection.evidence.holdout.run_id
        or recorded_plan.get("budget_source_best_epoch") != selection.evidence.holdout.best_epoch
        or recorded_plan.get("manifest_sha256") != candidate.metadata["source_manifest_sha256"]
        or recorded_plan.get("epochs") != refit_plan["epochs"]
        or recorded_plan.get("planned_steps") != refit_plan["steps"]
        or recorded_summary.get("epochs_completed") != refit_plan["epochs"]
        or recorded_summary.get("steps_completed") != refit_plan["steps"]
        or recorded_summary.get("model_state_sha256") != candidate.metadata["model_state_sha256"]
        or recorded_summary.get("selection_id") != selection.evidence.selection_id
        or recorded_summary.get("selection_sha256") != selection.evidence_sha256
        or refit_plan["epochs"] != selection.evidence.holdout.best_epoch + 1
    ):
        raise ValueError("bundle provenance differs from the completed production refit")
    incumbent = load_bundle(champion) if champion.exists() else None

    champion.parent.mkdir(parents=True, exist_ok=True)
    backup = champion.parent / f".champion-backup-{uuid.uuid4().hex}"
    replaced = champion.exists()
    if replaced:
        champion.replace(backup)
    try:
        staged.replace(champion)
        deployed = load_bundle(champion)
        bundle_path = champion / "bundle.json"
        store.record_deployment(
            selection_run_id=str(candidate.metadata["selection_id"]),
            refit_run_id=str(candidate.metadata["refit_run_id"]),
            replaced_refit_run_id=(
                str(incumbent.metadata["refit_run_id"]) if incumbent is not None else None
            ),
            bundle_path=bundle_path,
            bundle_sha256=sha256_file(bundle_path),
        )
    except Exception:
        if champion.exists():
            shutil.rmtree(champion)
        if replaced:
            backup.replace(champion)
        raise
    if replaced:
        shutil.rmtree(backup)
    return deployed
