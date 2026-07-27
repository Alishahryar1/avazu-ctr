"""Deterministic logical hashes for model state."""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Mapping

import torch


def state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        encoded_name = name.encode("utf-8")
        encoded_dtype = str(tensor.dtype).encode("ascii")
        digest.update(struct.pack(">I", len(encoded_name)))
        digest.update(encoded_name)
        digest.update(struct.pack(">I", len(encoded_dtype)))
        digest.update(encoded_dtype)
        digest.update(struct.pack(">I", tensor.ndim))
        for dimension in tensor.shape:
            digest.update(struct.pack(">Q", dimension))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy())
    return digest.hexdigest()
