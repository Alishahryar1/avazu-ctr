"""Raw-to-sparse preparation for the profile FFM workflow."""

from __future__ import annotations

import csv
import gc
import gzip
import shutil
import uuid
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import IO, Any

import polars as pl

from avazu_ctr.profile_ffm.artifacts import publish_directory
from avazu_ctr.profile_ffm.config import ProfileFFMConfig, profile_ffm_config_sha256
from avazu_ctr.profile_ffm.contracts import (
    FileArtifact,
    Inventory,
    PopulationRows,
    PreparationManifest,
    ProfileCoverage,
    SourceSplit,
    file_artifact,
    sha256_file,
    source_artifact,
    write_preparation_manifest,
)
from avazu_ctr.profile_ffm.features import (
    BASE_FIELD_COUNT,
    HISTORY_FIELD_INDEX,
    ROW_SIDECAR_FIELDS,
    SINGLETON_WEIGHT,
    USER_COUNT_FIELD_INDEX,
    CovariateCounts,
    base_feature_tokens,
    base_user,
    build_profile_edges,
    inventory_for,
    load_profiles,
    profile_user,
    proxy_user,
    publisher_fields,
)
from avazu_ctr.profile_ffm.history import CausalHistory, history_hashes

Progress = Callable[[str], None]
RAW_COLUMNS = (
    "id",
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


@contextmanager
def _open_text(path: Path) -> Iterator[IO[str]]:
    if path.suffix == ".gz":
        with gzip.open(path, mode="rt", encoding="utf-8", newline="") as handle:
            yield handle
    else:
        with path.open("r", encoding="utf-8", newline="") as handle:
            yield handle


def _raw_rows(path: Path, *, labelled: bool) -> Iterator[dict[str, str]]:
    with _open_text(path) as handle:
        reader = csv.DictReader(handle)
        expected = {"click", *RAW_COLUMNS} if labelled else set(RAW_COLUMNS)
        missing = expected.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        previous_hour = ""
        for row in reader:
            hour = row["hour"]
            if len(hour) != 8 or not hour.isdigit():
                raise ValueError(f"{path} contains an invalid Avazu hour")
            if previous_hour and hour < previous_hour:
                raise ValueError(f"{path} rows must be ordered by nondecreasing hour")
            previous_hour = hour
            if labelled:
                if row["click"] not in {"0", "1"}:
                    raise ValueError(f"{path} contains a non-binary click label")
            else:
                row["click"] = "0"
            yield row


def _source_digest(path: Path, expected: str | None) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = sha256_file(path)
    if expected is not None and digest != expected:
        raise ValueError(f"source checksum mismatch for {path}")
    return digest


def _count_covariates(
    config: ProfileFFMConfig,
    progress: Progress,
) -> tuple[CovariateCounts, dict[SourceSplit, int]]:
    counts = CovariateCounts()
    rows: dict[SourceSplit, int] = {}
    for split, path, labelled in (
        (SourceSplit.TRAINING, config.data.train_path, True),
        (SourceSplit.SCORING, config.data.test_path, False),
    ):
        progress(f"Counting {split.value} covariates")
        split_rows = 0
        for row in _raw_rows(path, labelled=labelled):
            counts.add(
                row,
                unknown_device_id=config.features.unknown_device_id,
            )
            split_rows += 1
        if not split_rows:
            raise ValueError(f"{path} contains no rows")
        rows[split] = split_rows
    return counts, rows


def _working_paths(
    root: Path,
) -> tuple[
    dict[tuple[SourceSplit, Inventory], Path],
    dict[tuple[SourceSplit, Inventory], Path],
]:
    rows = {
        (split, inventory): root / "work" / "rows" / f"{split.value}.{inventory.value}.csv"
        for split in SourceSplit
        for inventory in Inventory
    }
    base = {
        (split, inventory): root / "work" / "base" / f"{split.value}.{inventory.value}.ffm"
        for split in SourceSplit
        for inventory in Inventory
    }
    return rows, base


def _encode_base_rows(
    config: ProfileFFMConfig,
    counts: CovariateCounts,
    root: Path,
    progress: Progress,
) -> tuple[
    dict[tuple[SourceSplit, Inventory], Path],
    dict[tuple[SourceSplit, Inventory], Path],
    dict[tuple[SourceSplit, Inventory], int],
]:
    row_paths, base_paths = _working_paths(root)
    for path in (*row_paths.values(), *base_paths.values()):
        path.parent.mkdir(parents=True, exist_ok=True)

    metrics = {(split, inventory): 0 for split in SourceSplit for inventory in Inventory}
    histories: dict[str, CausalHistory] = {}
    with ExitStack() as stack:
        row_writers: dict[tuple[SourceSplit, Inventory], csv.DictWriter] = {}
        sparse_writers: dict[tuple[SourceSplit, Inventory], IO[str]] = {}
        for key, path in row_paths.items():
            handle = stack.enter_context(path.open("w", encoding="utf-8", newline=""))
            writer = csv.DictWriter(
                handle,
                fieldnames=ROW_SIDECAR_FIELDS,
                lineterminator="\n",
            )
            writer.writeheader()
            row_writers[key] = writer
        for key, path in base_paths.items():
            sparse_writers[key] = stack.enter_context(
                path.open("w", encoding="utf-8", newline="\n")
            )

        for split, path, labelled in (
            (SourceSplit.TRAINING, config.data.train_path, True),
            (SourceSplit.SCORING, config.data.test_path, False),
        ):
            progress(f"Encoding {split.value} base fields")
            for row in _raw_rows(path, labelled=labelled):
                inventory = inventory_for(
                    row,
                    app_site_sentinel=config.features.app_site_sentinel,
                )
                pub_id, pub_domain, pub_category = publisher_fields(
                    row,
                    inventory=inventory,
                )
                encoded_row = {
                    **row,
                    "pub_id": pub_id,
                    "pub_domain": pub_domain,
                    "pub_category": pub_category,
                }
                history = ""
                if row["device_id"] != config.features.unknown_device_id:
                    user = base_user(
                        row,
                        unknown_device_id=config.features.unknown_device_id,
                    )
                    state = histories.setdefault(user, CausalHistory())
                    history = state.advance(
                        hour=row["hour"],
                        label=row["click"],
                        update_labels=split is SourceSplit.TRAINING,
                        completed_events=config.features.completed_history_events,
                    )
                tokens = base_feature_tokens(
                    encoded_row,
                    counts,
                    history=history,
                    config=config,
                )
                key = (split, inventory)
                row_writers[key].writerow(
                    {
                        "id": row["id"],
                        "click": row["click"],
                        "hour": row["hour"],
                        "device_id": row["device_id"],
                        "device_ip": row["device_ip"],
                        "device_model": row["device_model"],
                        "pub_id": pub_id,
                        "pub_domain": pub_domain,
                    }
                )
                sparse_writers[key].write(
                    f"{row['id']} {row['click']} " + " ".join(str(token) for token in tokens) + "\n"
                )
                metrics[key] += 1
    return row_paths, base_paths, metrics


def _append_profiles(
    row_path: Path,
    base_path: Path,
    output: Path,
    profiles: dict[str, str],
    *,
    unknown_device_id: str,
) -> tuple[int, int]:
    rows = 0
    profiled_rows = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with (
        row_path.open("r", encoding="utf-8", newline="") as rows_file,
        base_path.open("r", encoding="utf-8") as base_file,
        output.open("w", encoding="utf-8", newline="\n") as destination,
    ):
        for row, sparse_line in zip(
            csv.DictReader(rows_file),
            base_file,
            strict=True,
        ):
            sparse_id, instance = sparse_line.rstrip("\n").split(" ", 1)
            if sparse_id != row["id"]:
                raise RuntimeError("base sparse rows do not align with their sidecar")
            if len(instance.split()) != BASE_FIELD_COUNT + 1:
                raise RuntimeError("base sparse row does not contain 15 feature hashes")
            suffix = profiles.get(
                profile_user(row, unknown_device_id=unknown_device_id),
                "",
            )
            if suffix:
                profiled_rows += 1
            destination.write(instance + suffix + "\n")
            rows += 1
    return rows, profiled_rows


def _prepare_profiles(
    config: ProfileFFMConfig,
    root: Path,
    row_paths: dict[tuple[SourceSplit, Inventory], Path],
    base_paths: dict[tuple[SourceSplit, Inventory], Path],
    progress: Progress,
) -> tuple[dict[str, Path], dict[Inventory, ProfileCoverage]]:
    sparse_root = root / "sparse"
    outputs: dict[str, Path] = {}
    coverage: dict[Inventory, ProfileCoverage] = {}
    for inventory in Inventory:
        progress(f"Building {inventory.value} publisher profiles")
        edge_paths = build_profile_edges(
            row_paths[(SourceSplit.TRAINING, inventory)],
            row_paths[(SourceSplit.SCORING, inventory)],
            root / "work" / "profiles" / inventory.value,
            inventory=inventory,
            config=config,
        )
        profiles = load_profiles(edge_paths)
        split_metrics: dict[SourceSplit, tuple[int, int]] = {}
        for split in SourceSplit:
            name = (
                f"{'train' if split is SourceSplit.TRAINING else 'score'}_{inventory.value}_profile"
            )
            output = sparse_root / f"{split.value}.{inventory.value}.profile.ffm"
            split_metrics[split] = _append_profiles(
                row_paths[(split, inventory)],
                base_paths[(split, inventory)],
                output,
                profiles,
                unknown_device_id=config.features.unknown_device_id,
            )
            outputs[name] = output
        coverage[inventory] = ProfileCoverage(
            users=len(profiles),
            publisher_id_edges=(pl.scan_parquet(edge_paths[0]).select(pl.len()).collect().item()),
            publisher_domain_edges=(
                pl.scan_parquet(edge_paths[1]).select(pl.len()).collect().item()
            ),
            training_rows=split_metrics[SourceSplit.TRAINING][0],
            training_profiled_rows=split_metrics[SourceSplit.TRAINING][1],
            scoring_rows=split_metrics[SourceSplit.SCORING][0],
            scoring_profiled_rows=split_metrics[SourceSplit.SCORING][1],
        )
        del profiles
        gc.collect()
    return outputs, coverage


def _build_proxy_counts(
    train_rows: Path,
    score_rows: Path,
    output: Path,
    *,
    unknown_device_id: str,
) -> dict[str, int]:
    scans = [
        (
            pl.scan_csv(
                path,
                infer_schema=False,
                low_memory=True,
                rechunk=False,
            )
            .filter(pl.col("device_id") == unknown_device_id)
            .select(
                pl.concat_str(
                    (
                        pl.lit("ip-"),
                        pl.col("device_ip"),
                        pl.lit("-"),
                        pl.col("device_model"),
                    )
                ).alias("user")
            )
        )
        for path in (train_rows, score_rows)
    ]
    pl.concat(scans).group_by("user").len(name="user_count").sort("user").sink_parquet(
        output,
        compression="zstd",
        compression_level=3,
        maintain_order=True,
        mkdir=True,
    )
    return {
        str(user): int(count)
        for user, count in pl.read_parquet(output).select("user", "user_count").iter_rows()
    }


def _append_history(
    row_path: Path,
    profile_path: Path,
    output: Path,
    *,
    counts: dict[str, int],
    states: dict[str, CausalHistory],
    config: ProfileFFMConfig,
    update_labels: bool,
    selector_path: Path | None,
) -> tuple[int, int, int]:
    rows = 0
    proxy_rows = 0
    nonempty_rows = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        row_file = stack.enter_context(row_path.open("r", encoding="utf-8", newline=""))
        profile_file = stack.enter_context(profile_path.open("r", encoding="utf-8"))
        destination = stack.enter_context(output.open("w", encoding="utf-8", newline="\n"))
        selector: Any | None = None
        if selector_path is not None:
            selector_path.parent.mkdir(parents=True, exist_ok=True)
            selector_file = stack.enter_context(
                selector_path.open("w", encoding="utf-8", newline="")
            )
            selector = csv.writer(selector_file, lineterminator="\n")
            selector.writerow(("id", "use_history"))

        for row, profile_line in zip(
            csv.DictReader(row_file),
            profile_file,
            strict=True,
        ):
            instance = profile_line.rstrip("\n")
            if instance.split(" ", 1)[0] != row["click"]:
                raise RuntimeError("profile sparse rows do not align with their sidecar")
            use_history = False
            if row["device_id"] == config.features.unknown_device_id:
                user = proxy_user(row)
                state = states.setdefault(user, CausalHistory())
                history = state.advance(
                    hour=row["hour"],
                    label=row["click"],
                    update_labels=update_labels,
                    completed_events=config.features.completed_history_events,
                )
                history_hash, count_hash = history_hashes(
                    counts[user],
                    history,
                    count_threshold=config.features.history_count_threshold,
                    bins=config.features.hash_bins,
                )
                instance += (
                    f" {HISTORY_FIELD_INDEX}:{history_hash}:{SINGLETON_WEIGHT:.20f}"
                    f" {USER_COUNT_FIELD_INDEX}:{count_hash}:{SINGLETON_WEIGHT:.20f}"
                )
                proxy_rows += 1
                use_history = bool(history)
                nonempty_rows += use_history
            if selector is not None:
                selector.writerow((row["id"], int(use_history)))
            destination.write(instance + "\n")
            rows += 1
    return rows, proxy_rows, nonempty_rows


def _prepare_history(
    config: ProfileFFMConfig,
    root: Path,
    row_paths: dict[tuple[SourceSplit, Inventory], Path],
    sparse: dict[str, Path],
    progress: Progress,
) -> tuple[Path, Path, Path, int, int]:
    progress("Building app-proxy completed-hour history")
    counts = _build_proxy_counts(
        row_paths[(SourceSplit.TRAINING, Inventory.APP)],
        row_paths[(SourceSplit.SCORING, Inventory.APP)],
        root / "work" / "proxy_user_counts.parquet",
        unknown_device_id=config.features.unknown_device_id,
    )
    states: dict[str, CausalHistory] = {}
    train_output = root / "sparse" / "training.app.history.ffm"
    score_output = root / "sparse" / "scoring.app.history.ffm"
    selector = root / "selectors" / "scoring.app.csv"
    train_metrics = _append_history(
        row_paths[(SourceSplit.TRAINING, Inventory.APP)],
        sparse["train_app_profile"],
        train_output,
        counts=counts,
        states=states,
        config=config,
        update_labels=True,
        selector_path=None,
    )
    score_metrics = _append_history(
        row_paths[(SourceSplit.SCORING, Inventory.APP)],
        sparse["score_app_profile"],
        score_output,
        counts=counts,
        states=states,
        config=config,
        update_labels=False,
        selector_path=selector,
    )
    if train_metrics[0] == 0 or score_metrics[0] == 0:
        raise ValueError("history preparation requires nonempty app populations")
    return (
        train_output,
        score_output,
        selector,
        score_metrics[1],
        score_metrics[2],
    )


def _prepare_site_selector(
    train_rows: Path,
    score_rows: Path,
    output: Path,
) -> tuple[int, int]:
    seen_publishers: set[str] = set()
    with train_rows.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            seen_publishers.add(row["pub_id"])
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    cold_rows = 0
    with (
        score_rows.open("r", encoding="utf-8", newline="") as source,
        output.open("w", encoding="utf-8", newline="") as destination,
    ):
        writer = csv.writer(destination, lineterminator="\n")
        writer.writerow(("id", "use_cold_publisher"))
        for row in csv.DictReader(source):
            use_cold = row["pub_id"] not in seen_publishers
            writer.writerow((row["id"], int(use_cold)))
            rows += 1
            cold_rows += use_cold
    return rows, cold_rows


def _validate_expected_rows(config: ProfileFFMConfig, rows: PopulationRows) -> None:
    expected = config.data.expected_rows
    if expected is None:
        return
    if rows.model_dump() != expected.model_dump():
        raise ValueError(
            "prepared population does not match configured row expectations: "
            f"{rows.model_dump()} != {expected.model_dump()}"
        )


def _prepare(
    config: ProfileFFMConfig,
    root: Path,
    progress: Progress,
) -> PreparationManifest:
    train_digest = _source_digest(
        config.data.train_path,
        config.data.train_sha256,
    )
    test_digest = _source_digest(
        config.data.test_path,
        config.data.test_sha256,
    )
    counts, counted_rows = _count_covariates(config, progress)
    row_paths, base_paths, inventory_rows = _encode_base_rows(
        config,
        counts,
        root,
        progress,
    )
    if (
        sum(inventory_rows[(SourceSplit.TRAINING, inventory)] for inventory in Inventory)
        != counted_rows[SourceSplit.TRAINING]
        or sum(inventory_rows[(SourceSplit.SCORING, inventory)] for inventory in Inventory)
        != counted_rows[SourceSplit.SCORING]
    ):
        raise RuntimeError("encoded inventories do not cover their raw populations")
    del counts
    gc.collect()

    sparse, profiles = _prepare_profiles(
        config,
        root,
        row_paths,
        base_paths,
        progress,
    )
    (
        train_history,
        score_history,
        app_selector,
        app_proxy_rows,
        nonempty_history_rows,
    ) = _prepare_history(config, root, row_paths, sparse, progress)
    progress("Building site publisher selector")
    site_selector = root / "selectors" / "scoring.site.csv"
    site_selector_rows, cold_site_rows = _prepare_site_selector(
        row_paths[(SourceSplit.TRAINING, Inventory.SITE)],
        row_paths[(SourceSplit.SCORING, Inventory.SITE)],
        site_selector,
    )
    rows = PopulationRows(
        training=counted_rows[SourceSplit.TRAINING],
        scoring=counted_rows[SourceSplit.SCORING],
        training_app=inventory_rows[(SourceSplit.TRAINING, Inventory.APP)],
        scoring_app=inventory_rows[(SourceSplit.SCORING, Inventory.APP)],
        training_site=inventory_rows[(SourceSplit.TRAINING, Inventory.SITE)],
        scoring_site=inventory_rows[(SourceSplit.SCORING, Inventory.SITE)],
        scoring_app_proxy=app_proxy_rows,
        scoring_nonempty_history=nonempty_history_rows,
        scoring_cold_site=cold_site_rows,
    )
    if site_selector_rows != rows.scoring_site:
        raise RuntimeError("site selector does not align with the scoring population")
    _validate_expected_rows(config, rows)

    artifacts_paths = {
        **sparse,
        "train_app_history": train_history,
        "score_app_history": score_history,
        "score_app_selector": app_selector,
        "score_site_selector": site_selector,
    }
    artifacts: dict[str, FileArtifact] = {}
    artifact_rows = {
        "train_app_profile": rows.training_app,
        "score_app_profile": rows.scoring_app,
        "train_site_profile": rows.training_site,
        "score_site_profile": rows.scoring_site,
        "train_app_history": rows.training_app,
        "score_app_history": rows.scoring_app,
        "score_app_selector": rows.scoring_app,
        "score_site_selector": rows.scoring_site,
    }
    progress("Checksumming prepared sparse artifacts")
    for name, path in artifacts_paths.items():
        artifacts[name] = file_artifact(
            root,
            path,
            rows=artifact_rows[name],
        )

    shutil.rmtree(root / "work")
    return PreparationManifest(
        name=config.name,
        config=config,
        config_sha256=profile_ffm_config_sha256(config),
        sources={
            SourceSplit.TRAINING: source_artifact(
                config.data.train_path,
                rows=rows.training,
                sha256=train_digest,
            ),
            SourceSplit.SCORING: source_artifact(
                config.data.test_path,
                rows=rows.scoring,
                sha256=test_digest,
            ),
        },
        rows=rows,
        profiles=profiles,
        artifacts=artifacts,
    )


def prepare_profile_ffm(
    config: ProfileFFMConfig,
    *,
    overwrite: bool = False,
    progress: Progress | None = None,
) -> Path:
    """Prepare solver-ready profile FFM files and publish their manifest."""

    emit = progress or (lambda _: None)
    target = config.data.artifact_root / "prepared"
    if target.exists() and not overwrite:
        raise FileExistsError(f"{target} already exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = target.parent / f".prepared-{uuid.uuid4().hex}.staging"
    stage.mkdir()
    try:
        manifest = _prepare(config, stage, emit)
        write_preparation_manifest(manifest, stage / "manifest.json")
        publish_directory(stage, target, overwrite=overwrite)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return target / "manifest.json"
