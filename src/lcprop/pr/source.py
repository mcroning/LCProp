"""Optical-field to photorefractive-source mappings."""

from __future__ import annotations

from typing import Any

from lcprop.optics.splitstep import total_intensity


def channel_peak_intensity_reference(A, *, xp: Any) -> float:
    """Return the sum of individual channel peak intensities.

    This is the explicit legacy PR channel-peak normalization. Coherent
    interference is deliberately excluded from this historical reference.
    """

    if A.ndim != 3:
        raise ValueError("A must have shape (Nch, Nx, Ny)")
    reference = xp.sum(xp.max(xp.abs(A) ** 2, axis=(-2, -1)))
    value = float(reference.item() if hasattr(reference, "item") else reference)
    if value <= 0.0:
        raise ValueError("channel peak intensity reference must be positive")
    return value


def pr_driving_intensity(
    A,
    *,
    peak_intensity_reference: float,
    background_intensity: float,
    coherence_groups: tuple[str, ...] | list[str] | None = None,
    xp: Any,
):
    """Return normalized optical intensity plus total uniform background."""

    from .illumination import PhysicalIlluminationReference
    if isinstance(peak_intensity_reference, PhysicalIlluminationReference):
        if float(background_intensity) != peak_intensity_reference.background_fraction:
            raise ValueError("physical reference/background fractions disagree")
        return peak_intensity_reference.driving_intensity(
            A, coherence_groups=coherence_groups, xp=xp)

    reference = float(peak_intensity_reference)
    if reference <= 0.0:
        raise ValueError("peak_intensity_reference must be positive")
    if float(background_intensity) < 0.0:
        raise ValueError("background_intensity must be nonnegative")
    optical = total_intensity(
        A,
        coherence_groups=coherence_groups,
        xp=xp,
    )
    return optical / reference + float(background_intensity)


__all__ = ["channel_peak_intensity_reference", "pr_driving_intensity"]
