"""Shared grid specification with historical LC compatibility exports.

``GridSpec`` is the canonical material-neutral API in this module. The LC
names exposed through ``__getattr__`` and ``__all__`` are retained for import
compatibility; new LC code should import them from :mod:`lcprop.lc`.
"""

from dataclasses import dataclass
import math
from numbers import Integral, Real
from importlib import import_module


@dataclass(frozen=True)
class GridSpec:
    """Human-facing transverse/z discretization choices."""

    Nx: int = 256
    Ny: int = 256
    dz_um: float = 5.0
    x_aperture_um: float = 75.0
    y_aperture_um: float = 100.0
    z_length_um: float = 3000.0

    def validate(self) -> None:
        for name in ("Nx", "Ny"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 1:
                raise ValueError("Nx and Ny must be integers > 1")
        for name in ("dz_um", "x_aperture_um", "y_aperture_um", "z_length_um"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
                or value <= 0.0
            ):
                raise ValueError(f"{name} must be finite and positive")


_LC_COMPATIBILITY_EXPORTS = {"BiasSpec", "LCContext", "LCMaterial"}
_LC_COMPATIBILITY_EXPORTS.add("TimeSpec")


def __getattr__(name: str):
    if name not in _LC_COMPATIBILITY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name = (
        "lcprop.lc.normalization" if name == "TimeSpec" else "lcprop.lc.specs"
    )
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | _LC_COMPATIBILITY_EXPORTS)


__all__ = [
    "BiasSpec",
    "GridSpec",
    "LCContext",
    "LCMaterial",
    "TimeSpec",
]
