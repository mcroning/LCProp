"""Experimental PRProp-style correlated transverse phase screens.

The implementation intentionally generates and filters one CPU NumPy layer at
a time. If the supplied optical field is a CuPy array, only that layer is
transferred to the GPU. No full three-dimensional noise volume is allocated.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter


@dataclass(frozen=True)
class CorrelatedPhaseNoise3D:
    """Generate deterministic independent PRProp-style phase layers."""

    nx: int
    ny: int
    dx_um: float
    dy_um: float
    z_length_um: float
    dz_um: float
    eps: float = 0.002
    sigma_um: float = 0.4
    seed: int = 12345
    mode: str = "independent"

    def __post_init__(self) -> None:
        if self.nx < 1 or self.ny < 1:
            raise ValueError("nx and ny must be positive")
        if self.dx_um <= 0.0 or self.dy_um <= 0.0:
            raise ValueError("transverse spacings must be positive")
        if self.z_length_um <= 0.0 or self.dz_um <= 0.0:
            raise ValueError("longitudinal dimensions must be positive")
        if self.eps < 0.0:
            raise ValueError("eps must be nonnegative")
        if self.sigma_um < 0.0:
            raise ValueError("sigma_um must be nonnegative")
        if self.mode not in ("off", "independent"):
            raise ValueError("mode must be 'off' or 'independent'")

    @property
    def n_steps(self) -> int:
        return max(1, int(round(self.z_length_um / self.dz_um)))

    @property
    def sigma_x_px(self) -> float:
        return self.sigma_um / self.dx_um

    @property
    def sigma_y_px(self) -> float:
        return self.sigma_um / self.dy_um

    @property
    def raw_std(self) -> float:
        return math.sqrt(self.eps * 4.0 * math.pi / self.n_steps)

    def _rng(self, step_index: int) -> np.random.Generator:
        if step_index < 0:
            raise ValueError("step_index must be nonnegative")
        # A child stream is derived solely from the run seed and physical step
        # index, so execution order cannot affect the generated layer.
        sequence = np.random.SeedSequence([int(self.seed), int(step_index)])
        return np.random.default_rng(sequence)

    def raw_layer(self, step_index: int) -> np.ndarray:
        """Return one deterministic raw white-noise layer."""

        if self.mode == "off" or self.eps == 0.0:
            return np.zeros((self.nx, self.ny), dtype=np.float64)
        return self._rng(step_index).normal(
            loc=0.0,
            scale=self.raw_std,
            size=(self.nx, self.ny),
        )

    def phase_layer(self, step_index: int) -> np.ndarray:
        """Return one correlated dimensionless phase layer in radians."""

        raw = self.raw_layer(step_index)
        if self.mode == "off" or self.eps == 0.0:
            return raw
        phase = gaussian_filter(
            raw,
            sigma=(self.sigma_x_px, self.sigma_y_px),
        )
        return phase * math.sqrt(self.sigma_x_px * self.sigma_y_px)

    def __call__(self, field: Any, step_index: int):
        """Apply ``field *= exp(1j * phase_layer(step_index))`` in place."""

        if self.mode == "off" or self.eps == 0.0:
            # Preserve exact noiseless identity, including bit patterns.
            return field

        phase_np = self.phase_layer(step_index)
        if type(field).__module__.split(".")[0] == "cupy":
            import cupy as cp  # type: ignore

            phase = cp.asarray(phase_np)
            xp = cp
        else:
            phase = phase_np
            xp = np

        screen = xp.exp(1j * phase)
        if field.ndim == 3:
            field[...] = field * screen[None, :, :]
        elif field.ndim == 2:
            field[...] = field * screen
        else:
            raise ValueError("field must have shape (Nch, Nx, Ny) or (Nx, Ny)")
        return field


def normalized_spectrum(array: np.ndarray) -> np.ndarray:
    """Return centered spectral power normalized to a discrete sum of one."""

    power = np.abs(np.fft.fftshift(np.fft.fft2(array))) ** 2
    total = float(np.sum(power))
    if total == 0.0:
        return np.zeros_like(power)
    return power / total


def grid_scale_spectral_fraction(array: np.ndarray) -> float:
    """Return power near either transverse Nyquist edge.

    The grid-scale region begins at 40% of the Nyquist magnitude on either
    axis. This fixed diagnostic is used only to show suppression by filtering.
    """

    spectrum = normalized_spectrum(array)
    fx = np.fft.fftshift(np.fft.fftfreq(array.shape[0]))
    fy = np.fft.fftshift(np.fft.fftfreq(array.shape[1]))
    mask = (np.abs(fx[:, None]) >= 0.4) | (np.abs(fy[None, :]) >= 0.4)
    return float(np.sum(spectrum[mask]))


__all__ = [
    "CorrelatedPhaseNoise3D",
    "grid_scale_spectral_fraction",
    "normalized_spectrum",
]
