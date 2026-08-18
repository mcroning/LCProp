"""Spectral carrier transport and electrostatic closure for Profile v1."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PRTransverseState:
    """Potential and fields derived from one accepted potential volume."""

    psi: Any
    carrier_density: Any
    E_x: Any
    E_y: Any


def _real_array(value, *, name: str, xp=np):
    array = xp.asarray(value)
    if array.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError(f"{name} must have dtype float32 or float64")
    if array.ndim not in (2, 3) or min(array.shape[-2:]) < 3:
        raise ValueError(f"{name} must end in (Nx, Ny), each at least 3")
    # NumPy validation preserves the trusted reference behavior. CuPy callers
    # validate at workflow boundaries so the material-step hot path has no
    # device-to-host scalar synchronization.
    if xp is np and not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def spectral_wavevectors(
    shape: tuple[int, int], *, dx_normalized: float, dy_normalized: float, xp=np
):
    """Return the reference-consistent periodic first-derivative symbols."""

    if min(shape) < 3:
        raise ValueError("both transverse dimensions must be at least 3")
    if dx_normalized <= 0.0 or dy_normalized <= 0.0:
        raise ValueError("normalized spacings must be positive")
    kx = 2.0 * math.pi * xp.fft.fftfreq(shape[0], d=dx_normalized)[:, None]
    ky = 2.0 * math.pi * xp.fft.fftfreq(shape[1], d=dy_normalized)[None, :]
    if shape[0] % 2 == 0:
        kx[shape[0] // 2, 0] = 0.0
    if shape[1] % 2 == 0:
        ky[0, shape[1] // 2] = 0.0
    return kx, ky


def spectral_derivatives(
    field, *, kx, ky, xp=np
):
    # Preserve the analytically exact derivative of a spatially uniform plane.
    # FFT roundoff in a non-binary constant can otherwise seed a nonzero rate
    # in the required uniform-equilibrium regression.
    if xp is np and np.all(field == field[..., :1, :1]):
        zeros = xp.zeros_like(field)
        return zeros, zeros.copy()
    transformed = xp.fft.fft2(field, axes=(-2, -1))
    dx = xp.fft.ifft2(1j * kx * transformed, axes=(-2, -1)).real
    dy = xp.fft.ifft2(1j * ky * transformed, axes=(-2, -1)).real
    return dx.astype(field.dtype, copy=False), dy.astype(field.dtype, copy=False)


def state_from_potential(
    psi,
    *,
    dx_normalized: float,
    dy_normalized: float,
    h_y: float = 1.0,
    applied_field_x: float = 0.0,
    xp=np,
) -> PRTransverseState:
    """Reconstruct ``P``, ``E_x`` and ``E_y`` from zero-mean periodic ψ."""

    potential = _real_array(psi, name="psi", xp=xp)
    kx, ky = spectral_wavevectors(
        potential.shape[-2:],
        dx_normalized=dx_normalized,
        dy_normalized=dy_normalized,
        xp=xp,
    )
    denominator = kx * kx + float(h_y) * ky * ky
    potential_hat = xp.fft.fft2(potential, axes=(-2, -1))
    potential_hat = xp.where(denominator == 0.0, 0.0, potential_hat)
    resolved = xp.fft.ifft2(potential_hat, axes=(-2, -1)).real.astype(
        potential.dtype, copy=False
    )
    psi_x, psi_y = spectral_derivatives(resolved, kx=kx, ky=ky, xp=xp)
    carrier = 1.0 + xp.fft.ifft2(
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
    xp=np,
):
    """Evaluate the frozen full-transverse potential-rate equation."""

    potential = _real_array(psi, name="psi", xp=xp)
    driving = _real_array(intensity, name="intensity", xp=xp)
    if driving.shape != potential.shape:
        raise ValueError("psi and intensity must have identical shapes")
    if xp is np and np.any(driving < 0.0):
        raise ValueError("intensity must be nonnegative")
    state = state_from_potential(
        potential,
        dx_normalized=dx_normalized,
        dy_normalized=dy_normalized,
        h_y=h_y,
        applied_field_x=applied_field_x,
        xp=xp,
    )
    kx, ky = spectral_wavevectors(
        potential.shape[-2:],
        dx_normalized=dx_normalized,
        dy_normalized=dy_normalized,
        xp=xp,
    )
    carrier_intensity = state.carrier_density * driving
    grad_x, grad_y = spectral_derivatives(
        carrier_intensity, kx=kx, ky=ky, xp=xp
    )
    flux_x = grad_x - carrier_intensity * state.E_x
    flux_y = float(m_y) * (grad_y - carrier_intensity * state.E_y)
    carrier_rhs_hat = (
        1j * kx * xp.fft.fft2(flux_x, axes=(-2, -1))
        + 1j * ky * xp.fft.fft2(flux_y, axes=(-2, -1))
    )
    denominator = kx * kx + float(h_y) * ky * ky
    if xp is np:
        psi_rhs_hat = np.divide(
            carrier_rhs_hat,
            denominator,
            out=np.zeros_like(carrier_rhs_hat),
            where=denominator > 0.0,
        )
    else:
        # CuPy does not support NumPy's combined ``out``/``where`` ufunc
        # keywords. The safe denominator prevents an eager zero-mode divide;
        # the following mask restores the same zero-gauge result.
        safe_denominator = xp.where(denominator > 0.0, denominator, 1.0)
        psi_rhs_hat = xp.where(
            denominator > 0.0,
            carrier_rhs_hat / safe_denominator,
            0.0,
        )
    return xp.fft.ifft2(psi_rhs_hat, axes=(-2, -1)).real.astype(
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
    xp=np,
):
    """Advance one transparent reference step and restore the ψ gauge."""

    potential = _real_array(psi, name="psi", xp=xp)
    candidate = potential + float(dt_normalized) * potential_rhs(
        potential,
        intensity,
        dx_normalized=dx_normalized,
        dy_normalized=dy_normalized,
        m_y=m_y,
        h_y=h_y,
        applied_field_x=applied_field_x,
        xp=xp,
    )
    candidate = candidate - xp.mean(candidate, axis=(-2, -1), keepdims=True)
    candidate = candidate.astype(potential.dtype, copy=False)
    if xp is np and not np.all(np.isfinite(candidate)):
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
    xp=np,
):
    """Advance one first-order spectral IMEX Euler material step.

    For every longitudinal plane, ``Istar`` is the maximum of the supplied
    production transport intensity. The frozen split is

    ``A psi = -Istar * (1 + L) psi``, with ``L = -laplacian``, and
    ``N = F - A psi``. The supplied intensity is used unchanged; callers must
    not add dark or uniform background terms again.
    """

    potential = _real_array(psi, name="psi", xp=xp)
    driving = _real_array(intensity, name="intensity", xp=xp)
    if driving.shape != potential.shape:
        raise ValueError("psi and intensity must have identical shapes")
    if xp is np and np.any(driving < 0.0):
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
        xp=xp,
    )
    kx, ky = spectral_wavevectors(
        potential.shape[-2:],
        dx_normalized=dx_normalized,
        dy_normalized=dy_normalized,
        xp=xp,
    )
    k_squared = kx * kx + ky * ky
    potential_hat = xp.fft.fft2(potential, axes=(-2, -1))
    if potential.ndim == 2:
        Istar = xp.max(driving).reshape(1, 1)
    else:
        Istar = xp.max(driving, axis=(-2, -1), keepdims=True)
    implicit_rate = Istar * (1.0 + k_squared)
    A_psi_hat = -implicit_rate * potential_hat
    nonlinear_hat = (
        xp.fft.fft2(full_rhs, axes=(-2, -1)) - A_psi_hat
    )
    candidate_hat = (
        potential_hat + dt * nonlinear_hat
    ) / (1.0 + dt * implicit_rate)
    candidate_hat = xp.where(k_squared == 0.0, 0.0, candidate_hat)
    candidate = xp.fft.ifft2(candidate_hat, axes=(-2, -1)).real
    candidate = candidate.astype(potential.dtype, copy=False)
    if xp is np and not np.all(np.isfinite(candidate)):
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
