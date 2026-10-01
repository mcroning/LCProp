"""Compact longitudinal optical-intensity retention for PR Fast results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


# One complex128 chunk is at most 1 MiB; never export an endpoint plane.
_PRESENTATION_CHUNK_ELEMENTS = 65536


def presentation_peak_intensity_reference(launch, *, asnumpy=np.asarray) -> float:
    """Historical NumPy presentation reference, independent of solver reference.

    Evaluate abs(A)**2 in the launch dtype on bounded host chunks, take each
    channel maximum, then perform the original NumPy sum in channel order.
    Existing host launch storage is viewed rather than copied.
    """
    def host_chunk(index):
        chunk = launch[index]
        return np.asarray(chunk if isinstance(chunk, np.ndarray) else asnumpy(chunk))
    return _presentation_reference_from_chunks(launch, host_chunk)


def copy_host_launch_for_presentation(launch, *, xp, synchronize):
    """Accept exact host bytes into owned device storage and retain one scalar.

    Each bounded host snapshot supplies both the upload and NumPy arithmetic.
    Finish its upload before releasing it; no device-to-host copy is needed.
    Caller mutation after acceptance cannot change either accepted output.
    """
    accepted = xp.empty(launch.shape, dtype=launch.dtype)
    def host_chunk(index):
        owned = np.array(launch[index], copy=True, order="C")
        accepted[index] = xp.asarray(owned)
        synchronize()
        return owned
    reference = _presentation_reference_from_chunks(launch, host_chunk)
    return accepted, reference


def _presentation_reference_from_chunks(launch, host_chunk):
    if launch.ndim != 3 or any(size == 0 for size in launch.shape):
        raise ValueError("launch must have nonempty shape (Nch, Nx, Ny)")
    _, nx, ny = launch.shape
    rows = max(1, _PRESENTATION_CHUNK_ELEMENTS // ny)
    maxima = []
    for channel in range(launch.shape[0]):
        peak = None
        for ix in range(0, nx, rows):
            for iy in range(0, ny, _PRESENTATION_CHUNK_ELEMENTS):
                index = (channel, slice(ix, ix + rows),
                         slice(iy, iy + _PRESENTATION_CHUNK_ELEMENTS))
                host = host_chunk(index)
                value = np.max(np.abs(host) ** 2)
                peak = value if peak is None else np.maximum(peak, value)
                del host
        maxima.append(peak)
    reference = float(np.sum(np.asarray(maxima)))
    if not np.isfinite(reference) or reference <= 0.0:
        raise ValueError("presentation peak intensity reference must be finite and positive")
    return reference


def _normalize_presentation_cut(raw, reference, background):
    """Shared NumPy arithmetic for Full projection and backend raw cuts."""
    return np.asarray((np.asarray(raw) - background) * reference).copy()


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
        xz=_normalize_presentation_cut(source[:, :, iy], reference, background),
        yz=_normalize_presentation_cut(source[:, ix, :], reference, background),
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
    """Transfer only raw cuts, then use the authoritative NumPy arithmetic."""

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
        xz=_normalize_presentation_cut(
            asnumpy(source_intensity_stack[:, :, iy]), reference, background),
        yz=_normalize_presentation_cut(
            asnumpy(source_intensity_stack[:, ix, :]), reference, background),
        x_cut_um=float(x[ix]),
        y_cut_um=float(y[iy]),
    )


def fast_retention_summary(
    omitted_fields: tuple[str, ...],
    cuts: PRLongitudinalIntensityCuts | None,
    *,
    intensity_preview_metadata: Mapping[str, Any] | None = None,
    additional_retained_fields: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Describe the compact retained Fast longitudinal products."""

    retained = []
    if cuts is not None:
        retained.extend([
            "longitudinal_intensity_xz",
            "longitudinal_intensity_yz",
            "x_cut_um",
            "y_cut_um",
        ])
    if intensity_preview_metadata is not None:
        retained.extend(["intensity_preview", "intensity_preview_metadata"])
    retained.extend(additional_retained_fields)
    summary = {
        "policy": "fast",
        "omitted_fields": list(omitted_fields),
        "retained_fields": retained,
    }
    if cuts is not None:
        summary.update({
            "longitudinal_cut_selection": "nearest_transverse_sample_to_zero",
            "x_cut_um": cuts.x_cut_um,
            "y_cut_um": cuts.y_cut_um,
        })
    if intensity_preview_metadata is not None:
        summary["intensity_preview"] = dict(intensity_preview_metadata)
    return summary


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
    "presentation_peak_intensity_reference",
    "extract_longitudinal_optical_intensity_cuts",
    "extract_backend_longitudinal_optical_intensity_cuts",
    "fast_retention_summary",
    "retained_longitudinal_intensity_cuts",
    "validate_longitudinal_cut_coordinates",
]
