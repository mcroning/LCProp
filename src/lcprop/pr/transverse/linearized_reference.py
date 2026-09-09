"""NumPy reference for periodic biased full-transverse PR linearization.

This module is intentionally isolated from workflow registration.  It solves
the frozen-intensity material tangent problem for the fixed-mean-field,
current-carrying periodic bulk profile documented in
``docs/research/pr_periodic_biased_current_carrying_profile.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from lcprop.pr.transverse.transport import spectral_wavevectors


PR_PERIODIC_BIASED_LINEARIZED_REFERENCE_V1 = (
    "pr_full_transverse_periodic_biased_linearized_reference_v1"
)
PR_PERIODIC_BIASED_CURRENT_PROFILE_V1 = (
    "pr_full_transverse_periodic_biased_current_v1"
)
FIXED_MEAN_FIELD_ENSEMBLE = "fixed_harmonic_mean_field"


@dataclass(frozen=True)
class PRBiasedLinearizedReferenceSpec:
    """Parameters for one frozen-intensity periodic reference solve."""

    reference_intensity: float
    applied_field: float
    dx_normalized: float
    dy_normalized: float
    m_y: float = 1.0
    h_y: float = 1.0

    def validate(self) -> None:
        for name in (
            "reference_intensity",
            "dx_normalized",
            "dy_normalized",
            "m_y",
            "h_y",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(float(self.applied_field)):
            raise ValueError("applied_field must be finite")


@dataclass(frozen=True)
class PRBiasedLinearizedFourierSymbol:
    """Discrete spectral coordinates and the accepted response symbol."""

    kx: np.ndarray
    ky: np.ndarray
    a_M: np.ndarray
    a_H: np.ndarray
    denominator: np.ndarray
    response_kernel: np.ndarray
    resolved_mask: np.ndarray


@dataclass(frozen=True)
class PRBiasedLinearizedReferenceResult:
    """Material perturbations and compact scientific provenance."""

    delta_psi: np.ndarray
    delta_E_x: np.ndarray
    delta_E_y: np.ndarray
    delta_P: np.ndarray
    delta_mean_current: np.ndarray
    mean_intensity_perturbation: np.ndarray
    response_kernel: np.ndarray
    denominator: np.ndarray
    reference_intensity: float
    applied_field: float
    dx_normalized: float
    dy_normalized: float
    m_y: float
    h_y: float
    model_id: str = PR_PERIODIC_BIASED_LINEARIZED_REFERENCE_V1
    profile_id: str = PR_PERIODIC_BIASED_CURRENT_PROFILE_V1
    electrical_ensemble: str = FIXED_MEAN_FIELD_ENSEMBLE
    forward_fft_count: int = 1
    inverse_fft_count: int = 4


def _validated_intensity(intensity) -> np.ndarray:
    array = np.asarray(intensity)
    if array.dtype != np.dtype(np.float64):
        raise TypeError("NumPy linearized reference requires float64 intensity")
    if array.ndim not in (2, 3) or min(array.shape[-2:]) < 3:
        raise ValueError("intensity must have shape (Nx, Ny) or (Nz, Nx, Ny)")
    if not np.all(np.isfinite(array)):
        raise ValueError("intensity must contain only finite values")
    if np.any(array <= 0.0):
        raise ValueError("physical total intensity must be strictly positive")
    return array


def biased_linearized_fourier_symbol(
    shape: tuple[int, int],
    *,
    spec: PRBiasedLinearizedReferenceSpec,
) -> PRBiasedLinearizedFourierSymbol:
    """Return the repository-consistent Fourier grid and response kernel.

    The canonical transverse derivative convention zeros each even-grid
    Nyquist first-derivative symbol.  Consequently ``resolved_mask`` excludes
    every joint derivative-null mode, including the constant gauge mode and,
    on even grids, the corresponding Nyquist null combinations.
    """

    spec.validate()
    if len(shape) != 2 or min(shape) < 3:
        raise ValueError("shape must contain transverse sizes (Nx, Ny), each at least 3")
    kx, ky = spectral_wavevectors(
        shape,
        dx_normalized=float(spec.dx_normalized),
        dy_normalized=float(spec.dy_normalized),
        xp=np,
    )
    a_M = kx * kx + float(spec.m_y) * ky * ky
    a_H = kx * kx + float(spec.h_y) * ky * ky
    bias_kx = float(spec.applied_field) * kx
    denominator = float(spec.reference_intensity) * (
        a_M * (1.0 + a_H) + 1j * bias_kx * a_H
    )
    numerator = -(a_M + 1j * bias_kx)
    resolved_mask = a_H > 0.0
    response_kernel = np.zeros(shape, dtype=np.complex128)
    np.divide(
        numerator,
        denominator,
        out=response_kernel,
        where=resolved_mask,
    )
    return PRBiasedLinearizedFourierSymbol(
        kx=kx,
        ky=ky,
        a_M=a_M,
        a_H=a_H,
        denominator=denominator,
        response_kernel=response_kernel,
        resolved_mask=resolved_mask,
    )


def solve_pr_biased_linearized_reference(
    intensity,
    *,
    spec: PRBiasedLinearizedReferenceSpec,
) -> PRBiasedLinearizedReferenceResult:
    """Solve the frozen-intensity biased material tangent problem.

    ``intensity`` is the complete normalized physical transport intensity and
    ``delta_I = intensity - reference_intensity``.  A nonzero mean
    ``delta_I`` is allowed: its Fourier zero mode produces no perturbation
    potential in the fixed-reference solve, but changes the first-order mean
    current by ``-applied_field * mean(delta_I)``.

    The solve performs one forward two-dimensional FFT and four inverse FFTs
    (potential, two field components, and carrier perturbation).
    """

    spec.validate()
    driving = _validated_intensity(intensity)
    symbol = biased_linearized_fourier_symbol(driving.shape[-2:], spec=spec)
    delta_intensity = driving - float(spec.reference_intensity)
    mean_intensity_perturbation = np.mean(
        delta_intensity,
        axis=(-2, -1),
    )
    zero_mean_delta_intensity = delta_intensity - np.expand_dims(
        mean_intensity_perturbation,
        axis=(-2, -1),
    )
    delta_intensity_hat = np.fft.fft2(
        zero_mean_delta_intensity,
        axes=(-2, -1),
    )
    delta_psi_hat = delta_intensity_hat * symbol.response_kernel

    delta_psi = np.fft.ifft2(delta_psi_hat, axes=(-2, -1)).real
    delta_E_x = np.fft.ifft2(
        -1j * symbol.kx * delta_psi_hat,
        axes=(-2, -1),
    ).real
    delta_E_y = np.fft.ifft2(
        -1j * symbol.ky * delta_psi_hat,
        axes=(-2, -1),
    ).real
    delta_P = np.fft.ifft2(
        symbol.a_H * delta_psi_hat,
        axes=(-2, -1),
    ).real

    delta_mean_current = np.stack(
        (
            -float(spec.applied_field) * mean_intensity_perturbation,
            np.zeros_like(mean_intensity_perturbation),
        ),
        axis=-1,
    )

    return PRBiasedLinearizedReferenceResult(
        delta_psi=delta_psi.astype(np.float64, copy=False),
        delta_E_x=delta_E_x.astype(np.float64, copy=False),
        delta_E_y=delta_E_y.astype(np.float64, copy=False),
        delta_P=delta_P.astype(np.float64, copy=False),
        delta_mean_current=delta_mean_current.astype(np.float64, copy=False),
        mean_intensity_perturbation=np.asarray(
            mean_intensity_perturbation, dtype=np.float64
        ),
        response_kernel=symbol.response_kernel,
        denominator=symbol.denominator,
        reference_intensity=float(spec.reference_intensity),
        applied_field=float(spec.applied_field),
        dx_normalized=float(spec.dx_normalized),
        dy_normalized=float(spec.dy_normalized),
        m_y=float(spec.m_y),
        h_y=float(spec.h_y),
    )


def total_fields_from_perturbation(
    result: PRBiasedLinearizedReferenceResult,
) -> tuple[np.ndarray, np.ndarray]:
    """Return total fields without obscuring perturbation-result semantics."""

    return result.applied_field + result.delta_E_x, result.delta_E_y.copy()


__all__ = [
    "FIXED_MEAN_FIELD_ENSEMBLE",
    "PR_PERIODIC_BIASED_CURRENT_PROFILE_V1",
    "PR_PERIODIC_BIASED_LINEARIZED_REFERENCE_V1",
    "PRBiasedLinearizedFourierSymbol",
    "PRBiasedLinearizedReferenceResult",
    "PRBiasedLinearizedReferenceSpec",
    "biased_linearized_fourier_symbol",
    "solve_pr_biased_linearized_reference",
    "total_fields_from_perturbation",
]
