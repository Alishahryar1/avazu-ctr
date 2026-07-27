"""JSON and self-contained HTML diagnostics."""

from __future__ import annotations

import html
import json
import sqlite3
from pathlib import Path
from typing import Any

import polars as pl

from avazu_ctr.data.manifest import load_manifest
from avazu_ctr.data.schema import RAW_CATEGORICAL_COLUMNS, scan_raw


def _write_report(payload: dict[str, Any], output: Path, title: str) -> tuple[Path, Path]:
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "report.json"
    html_path = output / "report.html"
    rendered = json.dumps(payload, indent=2, sort_keys=True, default=str)
    json_path.write_text(rendered, encoding="utf-8")
    html_path.write_text(
        "\n".join(
            (
                "<!doctype html>",
                '<html lang="en"><meta charset="utf-8">',
                f"<title>{html.escape(title)}</title>",
                "<style>body{font:15px system-ui;max-width:1100px;margin:2rem auto;"
                "padding:0 1rem}pre{background:#111;color:#eee;padding:1rem;"
                "overflow:auto;border-radius:.5rem}</style>",
                f"<h1>{html.escape(title)}</h1>",
                f"<pre>{html.escape(rendered)}</pre>",
                "</html>",
            )
        ),
        encoding="utf-8",
    )
    return json_path, html_path


def dataset_report(manifest_path: str | Path, output: str | Path) -> tuple[Path, Path]:
    path = Path(manifest_path)
    manifest = load_manifest(path, verify_shards=True)
    validation = pl.scan_parquet([path.parent / shard.path for shard in manifest.validation_shards])
    summary = validation.select(
        pl.len().alias("rows"),
        pl.col("click").mean().alias("click_rate"),
        pl.col("_timestamp_hour").min().alias("first_hour"),
        pl.col("_timestamp_hour").max().alias("last_hour"),
    ).collect(engine="streaming")
    payload = {
        "manifest": json.loads(manifest.model_dump_json()),
        "validation_summary": summary.to_dicts()[0],
        "weight_cardinality_estimate": sum(manifest.cardinalities.values()),
    }
    return _write_report(payload, Path(output), f"Dataset report: {manifest.name}")


def raw_report(
    raw_path: str | Path,
    output: str | Path,
    *,
    labelled: bool = True,
) -> tuple[Path, Path]:
    frame = scan_raw(Path(raw_path), labelled=labelled)
    expressions: list[pl.Expr] = [
        pl.len().alias("rows"),
        pl.col("_timestamp_hour").min().alias("first_hour"),
        pl.col("_timestamp_hour").max().alias("last_hour"),
        *(
            pl.col(column).n_unique().alias(f"{column}__cardinality")
            for column in RAW_CATEGORICAL_COLUMNS
        ),
    ]
    if labelled:
        expressions.append(pl.col("click").mean().alias("click_rate"))
    summary = frame.select(expressions).collect(engine="streaming").to_dicts()[0]
    return _write_report(
        {"raw_path": str(raw_path), "summary": summary},
        Path(output),
        "Raw Avazu data report",
    )


def run_report(database: str | Path, output: str | Path) -> tuple[Path, Path]:
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    try:
        runs = [dict(row) for row in connection.execute("SELECT * FROM runs ORDER BY started_at")]
        metrics = [
            dict(row)
            for row in connection.execute(
                "SELECT * FROM metrics ORDER BY run_id, step, split, name"
            )
        ]
        promotions = [
            dict(row)
            for row in connection.execute("SELECT * FROM promotions ORDER BY promotion_id")
        ]
    finally:
        connection.close()
    return _write_report(
        {"runs": runs, "metrics": metrics, "promotions": promotions},
        Path(output),
        "Experiment comparison",
    )
