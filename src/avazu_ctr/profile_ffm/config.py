"""Strict configuration for the profile FFM workflow."""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class NativeExecutor(StrEnum):
    AUTO = "auto"
    NATIVE = "native"
    WSL = "wsl"


class ExpectedRows(StrictModel):
    training: Annotated[int, Field(gt=0)]
    scoring: Annotated[int, Field(gt=0)]
    training_app: Annotated[int, Field(gt=0)]
    scoring_app: Annotated[int, Field(gt=0)]
    training_site: Annotated[int, Field(gt=0)]
    scoring_site: Annotated[int, Field(gt=0)]
    scoring_app_proxy: Annotated[int, Field(ge=0)]
    scoring_nonempty_history: Annotated[int, Field(ge=0)]
    scoring_cold_site: Annotated[int, Field(ge=0)]

    @model_validator(mode="after")
    def validate_partitions(self) -> Self:
        if self.training_app + self.training_site != self.training:
            raise ValueError("training inventory rows must sum to the training population")
        if self.scoring_app + self.scoring_site != self.scoring:
            raise ValueError("scoring inventory rows must sum to the scoring population")
        if self.scoring_app_proxy > self.scoring_app:
            raise ValueError("app-proxy scoring rows cannot exceed app scoring rows")
        if self.scoring_nonempty_history > self.scoring_app_proxy:
            raise ValueError("nonempty-history rows cannot exceed app-proxy scoring rows")
        if self.scoring_cold_site > self.scoring_site:
            raise ValueError("cold-site rows cannot exceed site scoring rows")
        return self


class ProfileFFMDataConfig(StrictModel):
    train_path: Path
    test_path: Path
    artifact_root: Path
    train_sha256: Sha256 | None = None
    test_sha256: Sha256 | None = None
    expected_rows: ExpectedRows | None = None


class ProfileFeaturesConfig(StrictModel):
    hash_bins: Literal[1_000_000] = 1_000_000
    app_site_sentinel: str = "85f751fd"
    unknown_device_id: str = "a99f214a"
    frequency_identity_threshold: Annotated[int, Field(gt=0)] = 1_000
    profile_identity_threshold: Annotated[int, Field(gt=1)] = 100
    profile_max_user_rows_exclusive: Annotated[int, Field(gt=1)] = 100
    profile_l2_norm: Annotated[float, Field(gt=0.0)] = 0.5
    history_count_threshold: Annotated[int, Field(gt=0)] = 30
    completed_history_events: Annotated[int, Field(gt=0)] = 4


class ColdPublisherConfig(StrictModel):
    training_mask_basis_points: Annotated[int, Field(gt=0, le=10_000)] = 1_800
    token: Annotated[str, Field(min_length=1)] = "pub_id-learned-cold"


class ProfileFFMTrainingConfig(StrictModel):
    rank: Annotated[int, Field(gt=0)] = 4
    learning_rate: Annotated[float, Field(gt=0.0)] = 0.05
    l2: Annotated[float, Field(ge=0.0)] = 0.00002
    epochs: Annotated[int, Field(gt=0)] = 6
    executor: NativeExecutor = NativeExecutor.AUTO

    @model_validator(mode="after")
    def validate_rank(self) -> Self:
        if self.rank % 4:
            raise ValueError("profile FFM rank must be a multiple of four")
        return self


class ProfileFFMConfig(StrictModel):
    schema_version: Literal[1]
    name: Annotated[str, Field(min_length=1)]
    data: ProfileFFMDataConfig
    features: ProfileFeaturesConfig = ProfileFeaturesConfig()
    cold_publisher: ColdPublisherConfig = ColdPublisherConfig()
    training: ProfileFFMTrainingConfig = ProfileFFMTrainingConfig()


def load_profile_ffm(path: str | Path) -> ProfileFFMConfig:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{config_path} must contain a YAML mapping")
    return ProfileFFMConfig.model_validate(raw)


def resolved_profile_ffm_config(config: ProfileFFMConfig) -> dict[str, Any]:
    return json.loads(config.model_dump_json())


def profile_ffm_config_sha256(config: ProfileFFMConfig) -> str:
    encoded = json.dumps(
        resolved_profile_ffm_config(config),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()
