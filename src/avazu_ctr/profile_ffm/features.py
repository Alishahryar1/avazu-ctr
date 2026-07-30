"""Exact sparse feature construction for profile FFM."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from avazu_ctr.profile_ffm.config import ProfileFFMConfig
from avazu_ctr.profile_ffm.contracts import Inventory
from avazu_ctr.profile_ffm.hashing import hash_profile_token, hash_token

BASE_FIELDS = (
    "pub_id",
    "pub_domain",
    "pub_category",
    "banner_pos",
    "device_model",
    "device_conn_type",
    "C14",
    "C17",
    "C20",
    "C21",
)
ROW_SIDECAR_FIELDS = (
    "id",
    "click",
    "hour",
    "device_id",
    "device_ip",
    "device_model",
    "pub_id",
    "pub_domain",
)
PROFILE_FIELD_INDEX = {"pub_id": 16, "pub_domain": 17}
HISTORY_FIELD_INDEX = 18
USER_COUNT_FIELD_INDEX = 19
BASE_FIELD_COUNT = 15
SINGLETON_WEIGHT = math.sqrt(2.0 / BASE_FIELD_COUNT)


@dataclass
class CovariateCounts:
    device_ids: Counter[str] = field(default_factory=Counter)
    device_ips: Counter[str] = field(default_factory=Counter)
    users: Counter[str] = field(default_factory=Counter)
    user_hours: Counter[str] = field(default_factory=Counter)

    def add(self, row: dict[str, str], *, unknown_device_id: str) -> None:
        user = base_user(row, unknown_device_id=unknown_device_id)
        self.device_ids[row["device_id"]] += 1
        self.device_ips[row["device_ip"]] += 1
        self.users[user] += 1
        self.user_hours[f"{user}-{row['hour']}"] += 1


def inventory_for(row: dict[str, str], *, app_site_sentinel: str) -> Inventory:
    return Inventory.APP if row["site_id"] == app_site_sentinel else Inventory.SITE


def publisher_fields(
    row: dict[str, str],
    *,
    inventory: Inventory,
) -> tuple[str, str, str]:
    prefix = inventory.value
    return (
        row[f"{prefix}_id"],
        row[f"{prefix}_domain"],
        row[f"{prefix}_category"],
    )


def base_user(row: dict[str, str], *, unknown_device_id: str) -> str:
    if row["device_id"] == unknown_device_id:
        return f"ip-{row['device_ip']}-{row['device_model']}"
    return f"id-{row['device_id']}"


def profile_user(row: dict[str, str], *, unknown_device_id: str) -> str:
    if row["device_id"] == unknown_device_id:
        return f"{row['device_ip']}-{row['device_model']}"
    return row["device_id"]


def proxy_user(row: dict[str, str]) -> str:
    return f"ip-{row['device_ip']}-{row['device_model']}"


def base_feature_tokens(
    row: dict[str, str],
    counts: CovariateCounts,
    *,
    history: str,
    config: ProfileFFMConfig,
) -> tuple[int, ...]:
    features = config.features
    bins = features.hash_bins
    raw = [hash_token(f"{name}-{row[name]}", bins=bins) for name in BASE_FIELDS]
    raw.append(hash_token(f"hour-{row['hour'][-2:]}", bins=bins))

    device_ip_count = counts.device_ips[row["device_ip"]]
    if device_ip_count > features.frequency_identity_threshold:
        device_ip_token = f"device_ip-{row['device_ip']}"
    else:
        device_ip_token = f"device_ip-less-{device_ip_count}"
    raw.append(hash_token(device_ip_token, bins=bins))

    device_id_count = counts.device_ids[row["device_id"]]
    if device_id_count > features.frequency_identity_threshold:
        device_id_token = f"device_id-{row['device_id']}"
    else:
        device_id_token = f"device_id-less-{device_id_count}"
    raw.append(hash_token(device_id_token, bins=bins))

    user = base_user(row, unknown_device_id=features.unknown_device_id)
    user_hour_count = counts.user_hours[f"{user}-{row['hour']}"]
    smooth_count = 0 if user_hour_count > features.history_count_threshold else user_hour_count
    raw.append(hash_token(f"smooth_user_hour_count-{smooth_count}", bins=bins))

    user_count = counts.users[user]
    if user_count > features.history_count_threshold:
        history_token = f"user_click_histroy-{user_count}"
    else:
        history_token = f"user_click_histroy-{user_count}-{history}"
    raw.append(hash_token(history_token, bins=bins))
    if len(raw) != BASE_FIELD_COUNT:
        raise AssertionError("base feature construction must emit exactly 15 fields")
    return tuple(raw)


def _population(
    train_rows: Path,
    score_rows: Path,
    *,
    unknown_device_id: str,
) -> pl.LazyFrame:
    selected = (
        "id",
        "device_id",
        "device_ip",
        "device_model",
        "pub_id",
        "pub_domain",
    )
    return (
        pl.concat(
            [
                pl.scan_csv(
                    path,
                    infer_schema=False,
                    low_memory=True,
                    rechunk=False,
                ).select(selected)
                for path in (train_rows, score_rows)
            ]
        )
        .with_row_index("__population_row")
        .with_columns(
            pl.when(pl.col("device_id") != unknown_device_id)
            .then(pl.col("device_id"))
            .otherwise(
                pl.concat_str(
                    (pl.col("device_ip"), pl.col("device_model")),
                    separator="-",
                )
            )
            .alias("user")
        )
    )


def build_profile_edges(
    train_rows: Path,
    score_rows: Path,
    output_root: Path,
    *,
    inventory: Inventory,
    config: ProfileFFMConfig,
) -> tuple[Path, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    population = _population(
        train_rows,
        score_rows,
        unknown_device_id=config.features.unknown_device_id,
    )
    outputs: list[Path] = []
    for value_field in ("pub_id", "pub_domain"):
        eligible_users = (
            population.group_by("user")
            .len(name="group_rows")
            .filter(pl.col("group_rows") < config.features.profile_max_user_rows_exclusive)
            .select("user")
        )
        global_counts = population.group_by(value_field).len(name="global_count")
        raw_name = {
            (Inventory.APP, "pub_id"): "app_id",
            (Inventory.APP, "pub_domain"): "app_domain",
            (Inventory.SITE, "pub_id"): "site_id",
            (Inventory.SITE, "pub_domain"): "site_domain",
        }[(inventory, value_field)]
        token = (
            pl.when(pl.col("global_count") >= config.features.profile_identity_threshold)
            .then(pl.concat_str((pl.lit(f"{raw_name}-"), pl.col(value_field))))
            .when(pl.col("global_count") >= 2)
            .then(
                pl.concat_str(
                    (
                        pl.lit(f"{raw_name}-less-"),
                        pl.col("global_count").cast(pl.String),
                    )
                )
            )
            .otherwise(pl.lit(f"{raw_name}-less"))
            .alias("token")
        )
        counted = (
            population.select("__population_row", "user", value_field)
            .join(global_counts, on=value_field, how="left")
            .with_columns(token)
            .join(eligible_users, on="user", how="inner")
            .group_by("user", "token")
            .agg(
                pl.len().alias("token_count"),
                pl.col("__population_row").min().alias("first_occurrence"),
            )
        )
        norms = counted.group_by("user").agg(
            pl.col("token_count").cast(pl.Float64).pow(2).sum().sqrt().alias("l2_norm")
        )
        edges = (
            counted.join(norms, on="user", how="left")
            .with_columns(
                (
                    config.features.profile_l2_norm
                    * pl.col("token_count").cast(pl.Float64)
                    / pl.col("l2_norm")
                ).alias("weight"),
                pl.lit(
                    PROFILE_FIELD_INDEX[value_field],
                    dtype=pl.UInt8,
                ).alias("field_index"),
            )
            .select(
                "user",
                "field_index",
                "token",
                "weight",
                "first_occurrence",
            )
            .sort("user", "field_index", "first_occurrence")
        )
        output = output_root / f"{value_field}.parquet"
        edges.sink_parquet(
            output,
            compression="zstd",
            compression_level=3,
            maintain_order=True,
            mkdir=True,
        )
        outputs.append(output)
    return outputs[0], outputs[1]


def load_profiles(paths: tuple[Path, Path]) -> dict[str, str]:
    profiles: dict[str, list[tuple[int, int, float]]] = {}
    for path in paths:
        rows = (
            pl.scan_parquet(path)
            .sort("user", "field_index", "first_occurrence")
            .select("user", "field_index", "token", "weight")
            .collect(engine="streaming")
            .iter_rows()
        )
        for user, field_index, token, weight in rows:
            profiles.setdefault(str(user), []).append(
                (
                    int(field_index),
                    hash_profile_token(str(token)),
                    float(weight),
                )
            )
    return {
        user: "".join(
            f" {field_index}:{token_id}:{weight:.5f}" for field_index, token_id, weight in values
        )
        for user, values in profiles.items()
    }
