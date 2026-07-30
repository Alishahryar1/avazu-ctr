"""Atomic publication helpers for profile FFM artifacts."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path


def publish_directory(stage: Path, target: Path, *, overwrite: bool) -> None:
    """Atomically publish a sibling staging directory with rollback."""

    if stage.parent.resolve() != target.parent.resolve():
        raise ValueError("staging and target directories must share a parent")
    if not stage.is_dir():
        raise FileNotFoundError(stage)
    if target.exists() and not target.is_dir():
        raise NotADirectoryError(target)
    if target.exists() and not overwrite:
        raise FileExistsError(f"{target} already exists")

    backup = target.parent / f".{target.name}-{uuid.uuid4().hex}.backup"
    replaced = False
    try:
        if target.exists():
            target.replace(backup)
            replaced = True
        stage.replace(target)
    except BaseException:
        if replaced and backup.exists() and not target.exists():
            backup.replace(target)
        raise
    else:
        if backup.exists():
            shutil.rmtree(backup)


def copy_file(source: Path, destination: Path, *, overwrite: bool) -> Path:
    """Copy a file through a sibling temporary path and publish it atomically."""

    if not source.is_file():
        raise FileNotFoundError(source)
    if source.resolve() == destination.resolve():
        return source
    if destination.exists() and not overwrite:
        raise FileExistsError(f"{destination} already exists")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}-{uuid.uuid4().hex}.tmp"
    try:
        shutil.copyfile(source, temporary)
        if destination.exists() and not overwrite:
            raise FileExistsError(f"{destination} already exists")
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination
