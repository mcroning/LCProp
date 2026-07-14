"""Shared diagnostics for LC workflows and products.

These functions operate on already-prepared arrays and runtime grids. They do
not know how to build experiments, run algorithms, save files, or plot.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from lcprop.core.backend import asnumpy
from lcprop.algorithms.theta_cn import static_director_residual_metrics

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


def theta_metrics(theta: Array) -> dict[str, float]:
    """Return min/max/rms metrics for a theta array."""
    try:
        xp = theta.__array_namespace__()  # type: ignore[attr-defined]
    except Exception:
        xp = None

    arr = asnumpy(theta)
    return {
        "theta_min": float(np.min(arr)),
        "theta_max": float(np.max(arr)),
        "theta_rms": float(np.sqrt(np.mean(arr * arr))),
    }


def theta_update_metrics(theta: Array, theta_prev: Array) -> dict[str, float]:
    """Return RMS and max update between two theta arrays."""
    d = asnumpy(theta - theta_prev)
    return {
        "dtheta_rms": float(np.sqrt(np.mean(d * d))),
        "dtheta_max": float(np.max(np.abs(d))),
    }


def residual_theta_static(
    theta: Array,
    intensity: Array,
    *,
    b: float,
    bi: float,
    dx: float,
    dy: float,
    theta_bc: float,
    xp: Any | None = None,
) -> dict[str, float]:
    """Compatibility wrapper for the canonical static residual metrics."""
    return static_director_residual_metrics(
        theta,
        intensity,
        b=b,
        bi=bi,
        dx=dx,
        dy=dy,
        xp=xp,
    )


__all__ = [
    "Array",
    "scalar_float",
    "normalized_field_integral",
    "total_power",
    "centroid",
    "rms_widths",
    "intensity_metrics",
    "theta_metrics",
    "theta_update_metrics",
    "residual_theta_static",
]
