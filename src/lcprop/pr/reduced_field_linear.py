"""Paper Eq. (5), with local total intensity and production centered differences.

E + E_app D_x E - D_xx E = (E_app I_b + D_x I) / I.
The prescribed I includes normalized optical, dark and uniform intensity.
This is a static approximation, not the arbitrary-I field-only Taylor tangent.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from lcprop.core.backend import BackendSpec, get_backend
from lcprop.pr.evolution import periodic_derivatives_x, periodic_first_derivative_x
from lcprop.pr.reduced_linearized import centered_derivative_symbols

PR_REDUCED_FIELD_LINEAR_RESPONSE_V1 = "pr_reduced_local_intensity_field_linear_v1"
_DEFAULT_BACKEND = BackendSpec(backend="numpy", precision="float64", verbose=False)


@dataclass(frozen=True)
class PRReducedFieldLinearSpec:
    applied_field: float
    background_intensity: float
    dx_normalized: float

    def validate(self):
        if not math.isfinite(float(self.applied_field)):
            raise ValueError("applied_field must be finite")
        if not math.isfinite(float(self.background_intensity)) or self.background_intensity < 0:
            raise ValueError("background_intensity must be finite and nonnegative")
        if not math.isfinite(float(self.dx_normalized)) or self.dx_normalized <= 0:
            raise ValueError("dx_normalized must be finite and positive")


@dataclass(frozen=True)
class PRReducedFieldLinearResult:
    E: Any
    residual: Any
    denominator: Any


def _validate_intensity(intensity, *, xp):
    if intensity.ndim not in (2, 3) or intensity.shape[-2] < 3:
        raise ValueError("intensity must have shape (Nx, Ny) or (Nbatch, Nx, Ny), Nx >= 3")
    # Only bounded scalar reductions cross the device boundary. Never floor I.
    if bool(xp.any(~xp.isfinite(intensity) | (intensity <= 0)).item()):
        raise ValueError("local total transport intensity must be finite and strictly positive")


def reduced_field_linear_residual(E, intensity, *, spec, xp):
    """Matching paper residual; diagnostic only, with no host plane transfer."""
    spec.validate()
    _validate_intensity(intensity, xp=xp)
    if E.shape != intensity.shape:
        raise ValueError("E and intensity must have identical shapes")
    ix = periodic_first_derivative_x(intensity, dx_normalized=spec.dx_normalized, xp=xp)
    ex, exx = periodic_derivatives_x(E, dx_normalized=spec.dx_normalized, xp=xp)
    return E + spec.applied_field * ex - exx - (
        spec.applied_field * spec.background_intensity + ix
    ) / intensity


def solve_pr_reduced_field_linear_intensity(intensity, *, spec, backend=_DEFAULT_BACKEND):
    """Direct backend-native periodic solve, independent along each y row."""
    spec.validate()
    resolved = get_backend(backend if isinstance(backend, BackendSpec) else
                           BackendSpec(backend=backend, precision="float64", verbose=False))
    xp = resolved.xp
    driving = xp.asarray(intensity)
    _validate_intensity(driving, xp=xp)
    if driving.dtype != xp.dtype(resolved.real_dtype):
        raise TypeError(f"intensity must have dtype {resolved.real_dtype}")
    ix = periodic_first_derivative_x(driving, dx_normalized=spec.dx_normalized, xp=xp)
    q = (ix + spec.applied_field * spec.background_intensity) / driving
    k1, k2_squared = centered_derivative_symbols(
        driving.shape[-2], dx_normalized=spec.dx_normalized, backend=backend
    )
    denominator = (1 + k2_squared + 1j * spec.applied_field * k1).astype(resolved.complex_dtype)
    shape = [1] * driving.ndim
    shape[-2] = driving.shape[-2]
    q_hat = xp.fft.fft(q, axis=-2).astype(resolved.complex_dtype, copy=False)
    E = xp.fft.ifft(q_hat / denominator.reshape(shape), axis=-2).real.astype(
        resolved.real_dtype, copy=False
    )
    residual = reduced_field_linear_residual(E, driving, spec=spec, xp=xp)
    return PRReducedFieldLinearResult(E, residual, denominator)
