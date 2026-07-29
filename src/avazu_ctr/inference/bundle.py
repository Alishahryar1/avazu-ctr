"""Strict production-only safetensors bundles."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

import torch
from pydantic import Field
from safetensors.torch import load_file, save_file

from avazu_ctr.config.loader import resolved_config
from avazu_ctr.config.schema import ExperimentConfig, StrictModel
from avazu_ctr.data.manifest import (
    DatasetManifest,
    DatasetPurpose,
    Sha256,
    sha256_file,
    sha256_json,
)
from avazu_ctr.models.base import CTRModel
from avazu_ctr.models.factory import create_model
from avazu_ctr.models.state import state_dict_sha256


@dataclass(slots=True)
class LoadedBundle:
    model: CTRModel
    config: ExperimentConfig
    manifest: DatasetManifest
    metadata: dict[str, Any]


class RefitPlan(StrictModel):
    epochs: Annotated[int, Field(gt=0)]
    steps: Annotated[int, Field(gt=0)]
    validation: Literal[False] = False
    early_stopping: Literal[False] = False


class BundleMetadata(StrictModel):
    schema_version: Literal[5] = 5
    role: Literal["production"] = "production"
    refit_run_id: Annotated[str, Field(min_length=1)]
    selection_id: Annotated[str, Field(min_length=1)]
    selection_sha256: Sha256
    config: ExperimentConfig
    dataset_manifest: DatasetManifest
    source_manifest_sha256: Sha256
    feature_contract_sha256: Sha256
    weights_sha256: Sha256
    weights_bytes: Annotated[int, Field(gt=0)]
    preprocessor_sha256: Sha256
    state_schema_sha256: Sha256
    model_state_sha256: Sha256
    refit_plan: RefitPlan


def export_production_bundle(
    model: CTRModel,
    config: ExperimentConfig,
    manifest: DatasetManifest,
    source_manifest_path: Path,
    output_dir: Path,
    *,
    refit_run_id: str,
    selection_id: str,
    selection_sha256: str,
    epochs: int,
    steps: int,
) -> Path:
    if manifest.purpose is not DatasetPurpose.PRODUCTION:
        raise ValueError("only a production refit can be exported for deployment")
    if epochs <= 0 or steps <= 0:
        raise ValueError("production refit provenance requires positive epochs and steps")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    weights_path = output_dir / "model.safetensors"
    try:
        state = {
            name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()
        }
        save_file(state, weights_path)
        weights_bytes = weights_path.stat().st_size
        if weights_bytes > config.deployment.max_weight_bytes:
            raise ValueError(
                f"serialized weights are {weights_bytes} bytes, "
                f"over the {config.deployment.max_weight_bytes} byte cap"
            )

        state_source = source_manifest_path.parent / "state"
        preprocessor_source = state_source / "preprocessor.json"
        if not preprocessor_source.exists():
            raise ValueError("production manifest has no fitted preprocessor state")
        shutil.copytree(state_source, output_dir / "preprocessor")
        preprocessor_path = output_dir / "preprocessor" / "preprocessor.json"
        bundle = BundleMetadata(
            refit_run_id=refit_run_id,
            selection_id=selection_id,
            selection_sha256=selection_sha256,
            config=config,
            dataset_manifest=manifest,
            source_manifest_sha256=sha256_file(source_manifest_path),
            feature_contract_sha256=manifest.feature_contract_sha256,
            weights_sha256=sha256_file(weights_path),
            weights_bytes=weights_bytes,
            preprocessor_sha256=sha256_file(preprocessor_path),
            state_schema_sha256=sha256_json(
                {
                    name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                    for name, value in state.items()
                }
            ),
            model_state_sha256=state_dict_sha256(state),
            refit_plan=RefitPlan(epochs=epochs, steps=steps),
        )
        bundle_path = output_dir / "bundle.json"
        bundle_path.write_text(bundle.model_dump_json(indent=2), encoding="utf-8")
        load_bundle(output_dir)
        return bundle_path
    except Exception:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        raise


def load_bundle(path: str | Path, *, device: str | torch.device = "cpu") -> LoadedBundle:
    root = Path(path)
    if root.is_file():
        root = root.parent
    metadata_path = root / "bundle.json"
    parsed = BundleMetadata.model_validate_json(metadata_path.read_text(encoding="utf-8"))
    metadata = json.loads(parsed.model_dump_json())

    weights_path = root / "model.safetensors"
    if sha256_file(weights_path) != parsed.weights_sha256:
        raise ValueError("model weight checksum mismatch")
    config = parsed.config
    if parsed.weights_bytes != weights_path.stat().st_size:
        raise ValueError("model weight size mismatch")
    if weights_path.stat().st_size > config.deployment.max_weight_bytes:
        raise ValueError("bundle exceeds the configured model-weight budget")
    manifest = parsed.dataset_manifest
    if manifest.purpose is not DatasetPurpose.PRODUCTION:
        raise ValueError("bundle embeds a non-production dataset")
    if manifest.feature_contract_sha256 != parsed.feature_contract_sha256:
        raise ValueError("bundle feature-contract checksum mismatch")
    if manifest.resolved_config_sha256 != sha256_json(resolved_config(config)):
        raise ValueError("bundle configuration does not match its production dataset")

    preprocessor_path = root / "preprocessor" / "preprocessor.json"
    if (
        not preprocessor_path.exists()
        or sha256_file(preprocessor_path) != parsed.preprocessor_sha256
    ):
        raise ValueError("preprocessor metadata checksum mismatch")
    preprocessor = json.loads(preprocessor_path.read_text(encoding="utf-8"))
    if (
        set(preprocessor)
        != {
            "schema_version",
            "purpose",
            "feature_mode",
            "categorical_encoding",
            "global_prior",
            "categorical_columns",
            "numerical_columns",
            "cardinalities",
            "embedding_kinds",
            "features",
        }
        or preprocessor.get("schema_version") != 5
        or preprocessor.get("purpose") != "production"
        or preprocessor.get("feature_mode") != manifest.feature_mode.value
        or preprocessor.get("categorical_encoding")
        != manifest.categorical_encoding.model_dump(mode="json")
        or not isinstance(preprocessor.get("global_prior"), int | float)
        or isinstance(preprocessor.get("global_prior"), bool)
        or not 0.0 <= preprocessor["global_prior"] <= 1.0
        or tuple(preprocessor.get("categorical_columns", ())) != manifest.categorical_columns
        or tuple(preprocessor.get("numerical_columns", ())) != manifest.numerical_columns
        or preprocessor.get("cardinalities") != manifest.cardinalities
        or preprocessor.get("embedding_kinds") != manifest.embedding_kinds
        or preprocessor.get("features")
        != [feature.model_dump(mode="json") for feature in manifest.features]
    ):
        raise ValueError("bundle has an invalid production preprocessor")
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
    if state_schema != parsed.state_schema_sha256:
        raise ValueError("model state schema mismatch")
    if state_dict_sha256(state) != parsed.model_state_sha256:
        raise ValueError("logical model state checksum mismatch")
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return LoadedBundle(
        model=model,
        config=config,
        manifest=manifest,
        metadata=metadata,
    )
