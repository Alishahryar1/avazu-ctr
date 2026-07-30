"""Exact token hashing and deterministic publisher masking."""

from __future__ import annotations

import hashlib


def hash_token(token: str, *, bins: int = 1_000_000) -> int:
    digest = hashlib.md5(token.encode("utf-8"), usedforsecurity=False).hexdigest()
    return int(digest, 16) % (bins - 1) + 1


def hash_profile_token(token: str) -> int:
    digest = hashlib.md5(
        f"group-{token}".encode(),
        usedforsecurity=False,
    ).hexdigest()
    return int(int(digest, 16) % (1e6 - 1) + 1)


def publisher_masked(row: int, basis_points: int) -> bool:
    if not 0 <= basis_points <= 10_000:
        raise ValueError("basis_points must be in [0, 10000]")
    value = (row + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    value ^= value >> 31
    return value % 10_000 < basis_points
