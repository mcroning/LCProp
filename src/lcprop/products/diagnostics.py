"""Material-neutral optical diagnostics.

These functions operate on already-prepared arrays and runtime grids. They do
not know how to build experiments, run algorithms, save files, or plot.
"""

from __future__ import annotations

from typing import Any

from lcprop.core.backend import asnumpy

Array = Any


def scalar_float(x: Any) -> float:
    """Convert backend scalar/array scalar to Python float."""
    return float(asnumpy(x))


def normalized_field_integral(I: Array, grid: Any) -> float:
    """Return the dimensionless normalized intensity integral."""
    xp = grid.xp
    p = xp.sum(I) * float(grid.dx_um) * float(grid.dy_um)
    return scalar_float(p)


def total_power(I: Array, grid: Any) -> float:
    """Compatibility alias for :func:`normalized_field_integral`."""
    return normalized_field_integral(I, grid)


def centroid(I: Array, grid: Any) -> tuple[float, float]:
    """Return intensity centroid (xc_um, yc_um)."""
    xp = grid.xp
    raw = xp.sum(I) + xp.asarray(1e-300, dtype=I.dtype)
    xc = xp.sum(I * grid.x_um[:, None]) / raw
    yc = xp.sum(I * grid.y_um[None, :]) / raw
    return scalar_float(xc), scalar_float(yc)


def rms_widths(I: Array, grid: Any) -> tuple[float, float]:
    """Return intensity RMS widths (sx_um, sy_um)."""
    xp = grid.xp
    raw = xp.sum(I) + xp.asarray(1e-300, dtype=I.dtype)
    xc = xp.sum(I * grid.x_um[:, None]) / raw
    yc = xp.sum(I * grid.y_um[None, :]) / raw
    sx = xp.sqrt(xp.sum(I * (grid.x_um[:, None] - xc) ** 2) / raw)
    sy = xp.sqrt(xp.sum(I * (grid.y_um[None, :] - yc) ** 2) / raw)
    return scalar_float(sx), scalar_float(sy)


def intensity_metrics(I: Array, grid: Any) -> dict[str, float]:
    """Return standard scalar metrics for one 2-D intensity frame."""
    xp = grid.xp
    xc, yc = centroid(I, grid)
    sx, sy = rms_widths(I, grid)
    return {
        "normalized_field_integral": normalized_field_integral(I, grid),
        "Imax": scalar_float(xp.max(I)),
        "sx_um": sx,
        "sy_um": sy,
        "xc_um": xc,
        "yc_um": yc,
    }



_LC_COMPAT_EXPORTS = (
    "residual_theta_static",
    "theta_metrics",
    "theta_update_metrics",
)


def __getattr__(name: str):
    """Resolve legacy LC diagnostics from their LC-owned module."""
    if name not in _LC_COMPAT_EXPORTS:
        raise AttributeError(name)
    from lcprop.lc import diagnostics as lc_diagnostics

    return getattr(lc_diagnostics, name)


__all__ = [
    "Array",
    "scalar_float",
    "normalized_field_integral",
    "total_power",
    "centroid",
    "rms_widths",
    "intensity_metrics",
    *_LC_COMPAT_EXPORTS,
]
