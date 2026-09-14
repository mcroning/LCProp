"""Compact longitudinal optical-intensity retention for PR Fast results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class PRLongitudinalIntensityCuts:
    """Physical optical intensity on the transverse samples nearest zero."""

    xz: np.ndarray
    yz: np.ndarray
    x_cut_um: float
    y_cut_um: float


def centered_transverse_coordinates(
    grid_summary: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Return the canonical cell-centered transverse sample coordinates."""

    nx = int(grid_summary["Nx"])
    ny = int(grid_summary["Ny"])
    dx_um = float(grid_summary["dx_um"])
    dy_um = float(grid_summary["dy_um"])
    x = (np.arange(nx) - 0.5 * (nx - 1)) * dx_um
    y = (np.arange(ny) - 0.5 * (ny - 1)) * dy_um
    return x, y


def extract_longitudinal_optical_intensity_cuts(
    source_intensity_stack: Any,
    *,
    grid_summary: Mapping[str, Any],
    peak_intensity_reference: float,
    background_intensity: float,
) -> PRLongitudinalIntensityCuts:
    """Extract physical ``(z, x)`` and ``(z, y)`` cuts nearest zero.

    ``source_intensity_stack`` is the dimensionless PR-driving intensity.
    Removing its uniform material background and restoring the launch peak
    reference gives the optical intensity used by the existing Full-result
    longitudinal presentation.
    """

    source = np.asarray(source_intensity_stack)
    nx = int(grid_summary["Nx"])
    ny = int(grid_summary["Ny"])
    if source.ndim != 3 or source.shape[1:] != (nx, ny):
        raise ValueError(
            "source_intensity_stack must have shape (Nz, Nx, Ny) consistent "
            "with grid_summary"
        )
    if source.dtype.kind != "f" or source.dtype.hasobject:
        raise TypeError("source_intensity_stack must be a real floating array")
    reference = float(peak_intensity_reference)
    background = float(background_intensity)
    if not np.isfinite(reference) or reference <= 0.0:
        raise ValueError("peak_intensity_reference must be finite and positive")
    if not np.isfinite(background) or background < 0.0:
        raise ValueError("background_intensity must be finite and nonnegative")

    x, y = centered_transverse_coordinates(grid_summary)
    ix = int(np.argmin(np.abs(x)))
    iy = int(np.argmin(np.abs(y)))
    return PRLongitudinalIntensityCuts(
        xz=np.asarray((source[:, :, iy] - background) * reference).copy(),
        yz=np.asarray((source[:, ix, :] - background) * reference).copy(),
        x_cut_um=float(x[ix]),
        y_cut_um=float(y[iy]),
    )


def extract_backend_longitudinal_optical_intensity_cuts(
    source_intensity_stack: Any,
    *,
    grid_summary: Mapping[str, Any],
    peak_intensity_reference: float,
    background_intensity: float,
    asnumpy,
) -> PRLongitudinalIntensityCuts:
    """Extract backend-resident cuts before transferring them to the host."""

    nx = int(grid_summary["Nx"])
    ny = int(grid_summary["Ny"])
    shape = getattr(source_intensity_stack, "shape", None)
    if shape is None or len(shape) != 3 or tuple(shape[1:]) != (nx, ny):
        raise ValueError(
            "source_intensity_stack must have shape (Nz, Nx, Ny) consistent "
            "with grid_summary"
        )
    reference = float(peak_intensity_reference)
    background = float(background_intensity)
    if not np.isfinite(reference) or reference <= 0.0:
        raise ValueError("peak_intensity_reference must be finite and positive")
    if not np.isfinite(background) or background < 0.0:
        raise ValueError("background_intensity must be finite and nonnegative")
    x, y = centered_transverse_coordinates(grid_summary)
    ix = int(np.argmin(np.abs(x)))
    iy = int(np.argmin(np.abs(y)))
    return PRLongitudinalIntensityCuts(
        xz=np.asarray(asnumpy(
            (source_intensity_stack[:, :, iy] - background) * reference
        )).copy(),
        yz=np.asarray(asnumpy(
            (source_intensity_stack[:, ix, :] - background) * reference
        )).copy(),
        x_cut_um=float(x[ix]),
        y_cut_um=float(y[iy]),
    )


def fast_retention_summary(
    omitted_fields: tuple[str, ...],
    cuts: PRLongitudinalIntensityCuts,
) -> dict[str, Any]:
    """Describe the compact retained Fast longitudinal products."""

    return {
        "policy": "fast",
        "omitted_fields": list(omitted_fields),
        "retained_fields": [
            "longitudinal_intensity_xz",
            "longitudinal_intensity_yz",
            "x_cut_um",
            "y_cut_um",
        ],
        "longitudinal_cut_selection": "nearest_transverse_sample_to_zero",
        "x_cut_um": cuts.x_cut_um,
        "y_cut_um": cuts.y_cut_um,
    }


def retained_longitudinal_intensity_cuts(result: Any) -> PRLongitudinalIntensityCuts:
    """Recover an already projected Fast cut pair for codec revalidation."""

    xz = getattr(result, "longitudinal_intensity_xz", None)
    yz = getattr(result, "longitudinal_intensity_yz", None)
    x_cut_um = getattr(result, "x_cut_um", None)
    y_cut_um = getattr(result, "y_cut_um", None)
    if xz is None or yz is None or x_cut_um is None or y_cut_um is None:
        raise ValueError("Fast result lacks retained longitudinal intensity cuts")
    return PRLongitudinalIntensityCuts(
        xz=np.asarray(xz),
        yz=np.asarray(yz),
        x_cut_um=float(x_cut_um),
        y_cut_um=float(y_cut_um),
    )


def validate_longitudinal_cut_coordinates(
    values: Mapping[str, Any], grid_summary: Mapping[str, Any]
) -> bool:
    """Validate optional paired cuts and return whether they are present."""

    xz = values.get("longitudinal_intensity_xz")
    yz = values.get("longitudinal_intensity_yz")
    x_cut = values.get("x_cut_um")
    y_cut = values.get("y_cut_um")
    absent = xz is None and yz is None and x_cut is None and y_cut is None
    if absent:
        return False
    if xz is None or yz is None or x_cut is None or y_cut is None:
        raise ValueError("retained longitudinal cuts and coordinates must be paired")
    x, y = centered_transverse_coordinates(grid_summary)
    expected_x = float(x[int(np.argmin(np.abs(x)))])
    expected_y = float(y[int(np.argmin(np.abs(y)))])
    if not np.isfinite(float(x_cut)) or not np.isfinite(float(y_cut)):
        raise ValueError("retained longitudinal cut coordinates must be finite")
    if float(x_cut) != expected_x or float(y_cut) != expected_y:
        raise ValueError(
            "retained longitudinal cut coordinates are not the nearest-zero samples"
        )
    return True


__all__ = [
    "PRLongitudinalIntensityCuts",
    "centered_transverse_coordinates",
    "extract_longitudinal_optical_intensity_cuts",
    "extract_backend_longitudinal_optical_intensity_cuts",
    "fast_retention_summary",
    "retained_longitudinal_intensity_cuts",
    "validate_longitudinal_cut_coordinates",
]
