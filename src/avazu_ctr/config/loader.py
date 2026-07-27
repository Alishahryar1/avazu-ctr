"""YAML configuration loading and resolved snapshots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from avazu_ctr.config.schema import ExperimentConfig


def load_experiment(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{config_path} must contain a YAML mapping")
    return ExperimentConfig.model_validate(raw)


def resolved_config(config: ExperimentConfig) -> dict[str, Any]:
    return json.loads(config.model_dump_json())


def write_resolved_config(config: ExperimentConfig, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        yaml.safe_dump(resolved_config(config), sort_keys=False),
        encoding="utf-8",
    )
