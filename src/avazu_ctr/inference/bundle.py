"""Inference-only safetensors bundles."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from avazu_ctr.config.loader import resolved_config
from avazu_ctr.config.schema import ExperimentConfig
from avazu_ctr.data.manifest import (
    DatasetManifest,
    sha256_file,
    sha256_json,
)
from avazu_ctr.models.base import CTRModel
from avazu_ctr.models.factory import create_model

BUNDLE_SCHEMA_VERSION = 2


@dataclass(slots=True)
class LoadedBundle:
    model: CTRModel
    config: ExperimentConfig
    manifest: DatasetManifest
    metadata: dict[str, Any]


def export_bundle(
    model: CTRModel,
    config: ExperimentConfig,
    manifest: DatasetManifest,
    source_manifest_path: Path,
    output_dir: Path,
    *,
    run_id: str,
    validation_metrics: dict[str, float] | None = None,
) -> Path:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    weights_path = output_dir / "model.safetensors"
    state = {name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()}
    save_file(state, weights_path)
    weights_bytes = weights_path.stat().st_size
    if weights_bytes > config.promotion.max_weight_bytes:
        shutil.rmtree(output_dir)
        raise ValueError(
            f"serialized weights are {weights_bytes} bytes, "
            f"over the {config.promotion.max_weight_bytes} byte cap"
        )

    state_source = source_manifest_path.parent / "state"
    preprocessor_source = state_source / "preprocessor.json"
    if not preprocessor_source.exists():
        shutil.rmtree(output_dir)
        raise ValueError("dataset manifest has no fitted preprocessor state")
    shutil.copytree(state_source, output_dir / "preprocessor")
    preprocessor_path = output_dir / "preprocessor" / "preprocessor.json"
    bundle = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "run_id": run_id,
        "config": resolved_config(config),
        "dataset_manifest": json.loads(manifest.model_dump_json()),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "weights_sha256": sha256_file(weights_path),
        "weights_bytes": weights_bytes,
        "preprocessor_sha256": sha256_file(preprocessor_path),
        "state_schema_sha256": sha256_json(
            {
                name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for name, value in state.items()
            }
        ),
        "validation_metrics": validation_metrics or {},
    }
    bundle_path = output_dir / "bundle.json"
    bundle_path.write_text(json.dumps(bundle, indent=2, sort_keys=True), encoding="utf-8")
    return bundle_path


def load_bundle(path: str | Path, *, device: str | torch.device = "cpu") -> LoadedBundle:
    root = Path(path)
    if root.is_file():
        root = root.parent
    metadata_path = root / "bundle.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        raise ValueError("unsupported bundle schema")
    weights_path = root / "model.safetensors"
    if sha256_file(weights_path) != metadata["weights_sha256"]:
        raise ValueError("model weight checksum mismatch")
    config = ExperimentConfig.model_validate(metadata["config"])
    if weights_path.stat().st_size > config.promotion.max_weight_bytes:
        raise ValueError("bundle exceeds the configured model-weight budget")
    manifest = DatasetManifest.model_validate(metadata["dataset_manifest"])
    preprocessor_path = root / "preprocessor" / "preprocessor.json"
    if (
        not preprocessor_path.exists()
        or sha256_file(preprocessor_path) != metadata["preprocessor_sha256"]
    ):
        raise ValueError("preprocessor metadata checksum mismatch")
    for table in manifest.fitted_tables:
        table_path = root / "preprocessor" / Path(table.path).name
        if not table_path.exists() or sha256_file(table_path) != table.sha256:
            raise ValueError(f"preprocessor checksum mismatch for {table.feature}")
    model = create_model(config.model, manifest, seed=config.training.seed)
    state = load_file(weights_path, device=str(device))
    state_schema = sha256_json(
        {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in state.items()
        }
    )
    if state_schema != metadata["state_schema_sha256"]:
        raise ValueError("model state schema mismatch")
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return LoadedBundle(
        model=model,
        config=config,
        manifest=manifest,
        metadata=metadata,
    )
