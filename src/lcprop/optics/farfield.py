"""Material-neutral angular-spectrum presentation diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from lcprop.core.beams import normalize_coherence_groups


@dataclass(frozen=True)
class DirectionCosineSpectrum:
    """Power density on transverse in-medium direction-cosine axes."""

    intensity: Any
    s_x: Any
    s_y: Any


def direction_cosine_spectrum(
    A,
    *,
    dx_um: float,
    dy_um: float,
    wavelength_um: float,
    refractive_index: float,
    coherence_groups: tuple[str, ...] | list[str] | None = None,
    xp: Any = np,
) -> DirectionCosineSpectrum:
    """Return coherence-aware far-field power density versus ``s_x,s_y``.

    The transverse direction cosines inside the launch medium are

    ``s_x = wavelength_um * f_x / refractive_index`` and likewise for y.

    The continuous-transform normalization makes the numerical integral of
    the returned density over ``ds_x ds_y`` equal the transverse optical
    power integral over ``dx dy`` (up to FFT roundoff).
    """

    if A.ndim != 3:
        raise ValueError("A must have shape (Nch, Nx, Ny)")
    if float(dx_um) <= 0.0 or float(dy_um) <= 0.0:
        raise ValueError("dx_um and dy_um must be positive")
    if float(wavelength_um) <= 0.0 or float(refractive_index) <= 0.0:
        raise ValueError("wavelength_um and refractive_index must be positive")

    names = normalize_coherence_groups(
        A.shape[0],
        coherent=False,
        coherence_groups=coherence_groups,
    )
    grouped: dict[str, list[int]] = {}
    for index, name in enumerate(names):
        grouped.setdefault(name, []).append(index)

    nx, ny = A.shape[-2:]
    density = xp.zeros((nx, ny), dtype=xp.abs(A[0]).dtype)
    transform_scale = float(dx_um) * float(dy_um)
    density_scale = (float(refractive_index) / float(wavelength_um)) ** 2
    for indices in grouped.values():
        group_field = xp.sum(A[indices], axis=0)
        transformed = (
            xp.fft.fftshift(xp.fft.fft2(group_field)) * transform_scale
        )
        density += xp.abs(transformed) ** 2 * density_scale

    angular_scale = float(wavelength_um) / float(refractive_index)
    s_x = xp.fft.fftshift(xp.fft.fftfreq(nx, d=float(dx_um))) * angular_scale
    s_y = xp.fft.fftshift(xp.fft.fftfreq(ny, d=float(dy_um))) * angular_scale
    return DirectionCosineSpectrum(intensity=density, s_x=s_x, s_y=s_y)


__all__ = ["DirectionCosineSpectrum", "direction_cosine_spectrum"]
