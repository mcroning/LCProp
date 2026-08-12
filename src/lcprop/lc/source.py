"""Liquid-crystal source construction from propagated optical channels."""

from __future__ import annotations

from typing import Any

from lcprop.lc import propagation
from lcprop.optics import splitstep

Array = Any


def director_driving_intensity(
    A: Array,
    *,
    coherent: bool = False,
    coherence_groups: tuple[str, ...] | list[str] | None = None,
    xp: Any | None = None,
) -> Array:
    """Construct the validated LC director-driving optical intensity.

    The canonical LC model uses the sum of coherent-group intensities. The
    director equation applies its physical coupling coefficient separately.
    """

    return splitstep.total_intensity(
        A,
        coherent=coherent,
        coherence_groups=coherence_groups,
        xp=xp,
    )


def advance_slice_with_midpoint_source(
    A: Array,
    theta: Array,
    *,
    kernel: Array,
    dz: float,
    wavelength: float,
    n_ref: float,
    ne: float,
    no: float,
    Nsub: int = 1,
    coherent: bool = False,
    coherence_groups: tuple[str, ...] | list[str] | None = None,
    xp: Any | None = None,
) -> tuple[Array, Array, Array, Array]:
    """Advance one LC slice and return entrance, exit, and midpoint sources."""

    I_before = director_driving_intensity(
        A,
        coherent=coherent,
        coherence_groups=coherence_groups,
        xp=xp,
    )
    propagation.advance_slice(
        A,
        theta,
        kernel=kernel,
        dz=dz,
        wavelength=wavelength,
        n_ref=n_ref,
        ne=ne,
        no=no,
        Nsub=Nsub,
        xp=xp,
    )
    I_after = director_driving_intensity(
        A,
        coherent=coherent,
        coherence_groups=coherence_groups,
        xp=xp,
    )
    I_mid = 0.5 * (I_before + I_after)
    return A, I_before, I_after, I_mid


__all__ = [
    "advance_slice_with_midpoint_source",
    "director_driving_intensity",
]
