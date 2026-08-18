"""Spectral carrier transport and electrostatic closure for Profile v1."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class PRTransverseState:
    """Potential and fields derived from one accepted potential volume."""

    psi: np.ndarray
    carrier_density: np.ndarray
    E_x: np.ndarray
    E_y: np.ndarray


def _real_array(value, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError(f"{name} must have dtype float32 or float64")
    if array.ndim not in (2, 3) or min(array.shape[-2:]) < 3:
        raise ValueError(f"{name} must end in (Nx, Ny), each at least 3")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def spectral_wavevectors(
    shape: tuple[int, int], *, dx_normalized: float, dy_normalized: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return the reference-consistent periodic first-derivative symbols."""

    if min(shape) < 3:
        raise ValueError("both transverse dimensions must be at least 3")
    if dx_normalized <= 0.0 or dy_normalized <= 0.0:
        raise ValueError("normalized spacings must be positive")
    kx = 2.0 * math.pi * np.fft.fftfreq(shape[0], d=dx_normalized)[:, None]
    ky = 2.0 * math.pi * np.fft.fftfreq(shape[1], d=dy_normalized)[None, :]
    if shape[0] % 2 == 0:
        kx[shape[0] // 2, 0] = 0.0
    if shape[1] % 2 == 0:
        ky[0, shape[1] // 2] = 0.0
    return kx, ky


def spectral_derivatives(
    field: np.ndarray, *, kx: np.ndarray, ky: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    # Preserve the analytically exact derivative of a spatially uniform plane.
    # FFT roundoff in a non-binary constant can otherwise seed a nonzero rate
    # in the required uniform-equilibrium regression.
    if np.all(field == field[..., :1, :1]):
        zeros = np.zeros_like(field)
        return zeros, zeros.copy()
    transformed = np.fft.fft2(field, axes=(-2, -1))
    dx = np.fft.ifft2(1j * kx * transformed, axes=(-2, -1)).real
    dy = np.fft.ifft2(1j * ky * transformed, axes=(-2, -1)).real
    return dx.astype(field.dtype, copy=False), dy.astype(field.dtype, copy=False)


def state_from_potential(
    psi,
    *,
    dx_normalized: float,
    dy_normalized: float,
    h_y: float = 1.0,
    applied_field_x: float = 0.0,
) -> PRTransverseState:
    """Reconstruct ``P``, ``E_x`` and ``E_y`` from zero-mean periodic ψ."""

    potential = _real_array(psi, name="psi")
    kx, ky = spectral_wavevectors(
        potential.shape[-2:],
        dx_normalized=dx_normalized,
        dy_normalized=dy_normalized,
    )
    denominator = kx * kx + float(h_y) * ky * ky
    potential_hat = np.fft.fft2(potential, axes=(-2, -1))
    potential_hat = np.where(denominator == 0.0, 0.0, potential_hat)
    resolved = np.fft.ifft2(potential_hat, axes=(-2, -1)).real.astype(
        potential.dtype, copy=False
    )
    psi_x, psi_y = spectral_derivatives(resolved, kx=kx, ky=ky)
    carrier = 1.0 + np.fft.ifft2(
        denominator * potential_hat, axes=(-2, -1)
    ).real
    return PRTransverseState(
        psi=resolved,
        carrier_density=carrier.astype(potential.dtype, copy=False),
        E_x=(float(applied_field_x) - psi_x).astype(potential.dtype, copy=False),
        E_y=(-psi_y).astype(potential.dtype, copy=False),
    )


def potential_rhs(
    psi,
    intensity,
    *,
    dx_normalized: float,
    dy_normalized: float,
    m_y: float = 1.0,
    h_y: float = 1.0,
    applied_field_x: float = 0.0,
) -> np.ndarray:
    """Evaluate the frozen full-transverse potential-rate equation."""

    potential = _real_array(psi, name="psi")
    driving = _real_array(intensity, name="intensity")
    if driving.shape != potential.shape:
        raise ValueError("psi and intensity must have identical shapes")
    if np.any(driving < 0.0):
        raise ValueError("intensity must be nonnegative")
    state = state_from_potential(
        potential,
        dx_normalized=dx_normalized,
        dy_normalized=dy_normalized,
        h_y=h_y,
        applied_field_x=applied_field_x,
    )
    kx, ky = spectral_wavevectors(
        potential.shape[-2:],
        dx_normalized=dx_normalized,
        dy_normalized=dy_normalized,
    )
    carrier_intensity = state.carrier_density * driving
    grad_x, grad_y = spectral_derivatives(carrier_intensity, kx=kx, ky=ky)
    flux_x = grad_x - carrier_intensity * state.E_x
    flux_y = float(m_y) * (grad_y - carrier_intensity * state.E_y)
    carrier_rhs_hat = (
        1j * kx * np.fft.fft2(flux_x, axes=(-2, -1))
        + 1j * ky * np.fft.fft2(flux_y, axes=(-2, -1))
    )
    denominator = kx * kx + float(h_y) * ky * ky
    psi_rhs_hat = np.divide(
        carrier_rhs_hat,
        denominator,
        out=np.zeros_like(carrier_rhs_hat),
        where=denominator > 0.0,
    )
    return np.fft.ifft2(psi_rhs_hat, axes=(-2, -1)).real.astype(
        potential.dtype, copy=False
    )


def explicit_euler_step(
    psi,
    intensity,
    *,
    dt_normalized: float,
    dx_normalized: float,
    dy_normalized: float,
    m_y: float = 1.0,
    h_y: float = 1.0,
    applied_field_x: float = 0.0,
) -> np.ndarray:
    """Advance one transparent reference step and restore the ψ gauge."""

    potential = _real_array(psi, name="psi")
    candidate = potential + float(dt_normalized) * potential_rhs(
        potential,
        intensity,
        dx_normalized=dx_normalized,
        dy_normalized=dy_normalized,
        m_y=m_y,
        h_y=h_y,
        applied_field_x=applied_field_x,
    )
    candidate = candidate - np.mean(candidate, axis=(-2, -1), keepdims=True)
    candidate = candidate.astype(potential.dtype, copy=False)
    if not np.all(np.isfinite(candidate)):
        raise FloatingPointError("nonfinite potential during transverse evolution")
    return candidate


def imex_euler_step(
    psi,
    intensity,
    *,
    dt_normalized: float,
    dx_normalized: float,
    dy_normalized: float,
    m_y: float = 1.0,
    h_y: float = 1.0,
    applied_field_x: float = 0.0,
) -> np.ndarray:
    """Advance one first-order spectral IMEX Euler material step.

    For every longitudinal plane, ``Istar`` is the maximum of the supplied
    production transport intensity. The frozen split is

    ``A psi = -Istar * (1 + L) psi``, with ``L = -laplacian``, and
    ``N = F - A psi``. The supplied intensity is used unchanged; callers must
    not add dark or uniform background terms again.
    """

    potential = _real_array(psi, name="psi")
    driving = _real_array(intensity, name="intensity")
    if driving.shape != potential.shape:
        raise ValueError("psi and intensity must have identical shapes")
    if np.any(driving < 0.0):
        raise ValueError("intensity must be nonnegative")
    dt = float(dt_normalized)
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt_normalized must be finite and positive")

    full_rhs = potential_rhs(
        potential,
        driving,
        dx_normalized=dx_normalized,
        dy_normalized=dy_normalized,
        m_y=m_y,
        h_y=h_y,
        applied_field_x=applied_field_x,
    )
    kx, ky = spectral_wavevectors(
        potential.shape[-2:],
        dx_normalized=dx_normalized,
        dy_normalized=dy_normalized,
    )
    k_squared = kx * kx + ky * ky
    potential_hat = np.fft.fft2(potential, axes=(-2, -1))
    if potential.ndim == 2:
        Istar = np.max(driving).reshape(1, 1)
    else:
        Istar = np.max(driving, axis=(-2, -1), keepdims=True)
    implicit_rate = Istar * (1.0 + k_squared)
    A_psi_hat = -implicit_rate * potential_hat
    nonlinear_hat = (
        np.fft.fft2(full_rhs, axes=(-2, -1)) - A_psi_hat
    )
    candidate_hat = (
        potential_hat + dt * nonlinear_hat
    ) / (1.0 + dt * implicit_rate)
    candidate_hat = np.where(k_squared == 0.0, 0.0, candidate_hat)
    candidate = np.fft.ifft2(candidate_hat, axes=(-2, -1)).real
    candidate = candidate.astype(potential.dtype, copy=False)
    if not np.all(np.isfinite(candidate)):
        raise FloatingPointError("nonfinite potential during transverse IMEX evolution")
    return candidate


__all__ = [
    "PRTransverseState",
    "explicit_euler_step",
    "imex_euler_step",
    "potential_rhs",
    "spectral_derivatives",
    "spectral_wavevectors",
    "state_from_potential",
]
