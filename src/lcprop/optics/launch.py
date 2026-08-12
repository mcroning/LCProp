"""Build multichannel launch fields from beam specifications.

This module converts human-facing ``BeamStack`` objects into optical
channel stacks consumed by the algorithms.

It does not propagate fields, build FFT kernels, solve material state, or
manage products.

Array convention
----------------
Launch fields have shape ``(Nch, Nx, Ny)``.
A single beam is still a one-channel stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from lcprop.core.beams import BeamStack
from lcprop.core.grid import RuntimeGrid

Array = Any


@dataclass(frozen=True)
class LaunchResult:
    """Prepared multichannel optical launch."""

    A0: Array
    physical_powers_mW: Array
    power_fractions: Array
    physical_total_power_mW: float
    wavelengths_um: Array
    coherence: str
    coherence_groups: tuple[str, ...]

    def summary(self) -> dict:
        return {
            "Nch": int(self.A0.shape[0]),
            "coherence": self.coherence,
            "coherence_groups": list(self.coherence_groups),
            "physical_channel_powers_mW": [float(x) for x in np.asarray(_to_numpy(self.physical_powers_mW)).ravel()],
            "physical_total_power_mW": float(self.physical_total_power_mW),
            "power_fractions": [float(x) for x in np.asarray(_to_numpy(self.power_fractions)).ravel()],
            "field_normalization": "sum_channel_integrals_equals_one",
            "wavelengths_um": [float(x) for x in np.asarray(_to_numpy(self.wavelengths_um)).ravel()],
        }


def _to_numpy(a: Any) -> np.ndarray:
    try:
        import cupy as cp  # type: ignore

        if isinstance(a, cp.ndarray):
            return cp.asnumpy(a)
    except Exception:
        pass
    return np.asarray(a)


def gaussian_channel(
    ch,
    grid: RuntimeGrid,
    *,
    power_fraction: float,
    complex_dtype: Any,
) -> Array:
    """Return a Gaussian whose intensity integral is ``power_fraction``.

    ``BeamChannel.power_mW`` remains physical request metadata. It is converted
    to a normalized channel fraction by :func:`build_launch`; physical power is
    not embedded in the optical field amplitude.

    Channel tilts are transverse phase gradients in rad/um, so the phase
    factor is exactly ``exp(1j * (kx*x + ky*y + phase))``.
    """

    xp = grid.xp
    X = grid.x_um[:, None]
    Y = grid.y_um[None, :]

    amp = xp.exp(
        -(((X - float(ch.x0_um)) / float(ch.waist_x_um)) ** 2)
        -(((Y - float(ch.y0_um)) / float(ch.waist_y_um)) ** 2)
    )

    phase = float(ch.phase_rad)
    if ch.tilt_x_rad_per_um or ch.tilt_y_rad_per_um or phase:
        amp = amp * xp.exp(
            1j
            * (
                float(ch.tilt_x_rad_per_um) * X
                + float(ch.tilt_y_rad_per_um) * Y
                + phase
            )
        )

    amp = amp.astype(complex_dtype, copy=False)

    # Normalize the transverse shape density in 1/um^2. A zero-power channel is
    # permitted by the beam model and becomes a deterministic zero field.
    dxdy = float(grid.dx_um) * float(grid.dy_um)
    p0 = xp.sum(xp.abs(amp) ** 2) * dxdy
    if float(power_fraction) == 0.0:
        amp = xp.zeros_like(amp)
    else:
        amp = amp * xp.sqrt(float(power_fraction) / p0)

    return amp.astype(complex_dtype, copy=False)


def build_launch(
    beams: BeamStack,
    grid: RuntimeGrid,
    *,
    complex_dtype: Any = np.complex64,
) -> LaunchResult:
    """Build ``A0`` channel stack from a ``BeamStack``."""

    beams.validate()
    xp = grid.xp

    physical_powers_mW = xp.asarray(
        [float(ch.power_mW) for ch in beams.channels],
        dtype=grid.real_dtype,
    )
    physical_total_power_mW = float(sum(float(ch.power_mW) for ch in beams.channels))
    if physical_total_power_mW <= 0.0:
        raise ValueError("beam stack total physical power must be positive")
    power_fractions = physical_powers_mW / physical_total_power_mW

    fields = [
        gaussian_channel(
            ch,
            grid,
            power_fraction=float(power_fractions[index]),
            complex_dtype=complex_dtype,
        )
        for index, ch in enumerate(beams.channels)
    ]

    A0 = xp.stack(fields, axis=0).astype(complex_dtype, copy=False)

    wavelengths_um = xp.asarray(
        [float(ch.wavelength_um) for ch in beams.channels],
        dtype=grid.real_dtype,
    )

    return LaunchResult(
        A0=A0,
        physical_powers_mW=physical_powers_mW,
        power_fractions=power_fractions,
        physical_total_power_mW=physical_total_power_mW,
        wavelengths_um=wavelengths_um,
        coherence=beams.coherence,
        coherence_groups=beams.coherence_groups,
    )


def normalized_power(A0: Array, grid: RuntimeGrid) -> float:
    """Return the normalized channel integral ``sum_c integral |A_c|^2``."""
    xp = grid.xp
    p = xp.sum(xp.abs(A0) ** 2) * float(grid.dx_um) * float(grid.dy_um)
    return float(_to_numpy(p))


def channel_power_integrals(A0: Array, grid: RuntimeGrid) -> np.ndarray:
    """Return normalized per-channel field integrals, independent of coherence."""
    xp = grid.xp
    values = xp.sum(xp.abs(A0) ** 2, axis=(-2, -1)) * float(grid.dx_um) * float(grid.dy_um)
    return np.asarray(_to_numpy(values), dtype=float)


def reconstructed_physical_powers_mW(
    A0: Array,
    grid: RuntimeGrid,
    launch: LaunchResult,
) -> np.ndarray:
    """Reconstruct per-channel physical powers from normalized field integrals."""
    return channel_power_integrals(A0, grid) * float(launch.physical_total_power_mW)


def total_power(A0: Array, grid: RuntimeGrid) -> float:
    """Compatibility alias for :func:`normalized_power`.

    This value is a dimensionless normalized field integral, not milliwatts.
    """
    return normalized_power(A0, grid)


__all__ = [
    "Array",
    "LaunchResult",
    "gaussian_channel",
    "build_launch",
    "channel_power_integrals",
    "normalized_power",
    "reconstructed_physical_powers_mW",
    "total_power",
]
