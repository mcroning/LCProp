"""Shared PR-only helpers for portable transport payload arrays."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Mapping

import numpy as np

from lcprop.core.backend import asnumpy
from lcprop.transport.envelopes import TransportCodecError


ARRAY_MARKER = "__lcprop_array__"


def pack_portable(value: Any, arrays: dict[str, np.ndarray], path: str) -> Any:
    """Move arrays into NPZ storage while retaining JSON-shaped metadata."""

    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        key = path.replace(".", "__")
        arrays[key] = np.asarray(asnumpy(value)).copy()
        return {ARRAY_MARKER: key}
    if isinstance(value, Mapping):
        return {
            str(key): pack_portable(item, arrays, f"{path}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [
            pack_portable(item, arrays, f"{path}.{index}")
            for index, item in enumerate(value)
        ]
    return value


def unpack_portable(value: Any, arrays: Mapping[str, np.ndarray]) -> Any:
    """Restore arrays from a verified portable NPZ payload."""

    if isinstance(value, Mapping):
        if set(value) == {ARRAY_MARKER}:
            key = value[ARRAY_MARKER]
            if key not in arrays:
                raise TransportCodecError(
                    f"missing transported PR array {key!r}"
                )
            return np.asarray(arrays[key]).copy()
        return {
            str(key): unpack_portable(item, arrays)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [unpack_portable(item, arrays) for item in value]
    return value


__all__ = ["ARRAY_MARKER", "pack_portable", "unpack_portable"]
