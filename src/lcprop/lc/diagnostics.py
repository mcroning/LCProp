"""Liquid-crystal director diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np

from lcprop.algorithms.theta_cn import static_director_residual_metrics
from lcprop.core.backend import asnumpy

Array = Any


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
    "residual_theta_static",
    "theta_metrics",
    "theta_update_metrics",
]
