"""JSON-safe encoding of observations and actions crossing the UniVTAC <-> lerobot env boundary.

Both processes import this module: the Isaac side has torch but no lerobot,
the lerobot side has no Isaac. Arrays travel as base64 bytes with dtype and
shape; everything else stays plain JSON.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import numpy as np


def to_wire(value: Any) -> Any:
    """Tensors and arrays -> ``{"__ndarray__": ...}``; dicts, lists and scalars recurse."""
    try:
        import torch

        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
    except ImportError:
        pass
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {
            "__ndarray__": True,
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "data": base64.b64encode(array.tobytes()).decode("ascii"),
        }
    if isinstance(value, dict):
        return {str(k): to_wire(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_wire(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return to_wire(np.asarray(value))


def from_wire(value: Any) -> Any:
    """Inverse of :func:`to_wire`; arrays come back as writable numpy arrays."""
    if isinstance(value, dict):
        if value.get("__ndarray__") is True:
            data = base64.b64decode(value["data"].encode("ascii"))
            return np.frombuffer(data, dtype=np.dtype(value["dtype"])).reshape(value["shape"]).copy()
        return {k: from_wire(v) for k, v in value.items()}
    if isinstance(value, list):
        return [from_wire(v) for v in value]
    return value
