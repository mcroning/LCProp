"""Backend-aware reference for periodic biased full-transverse PR linearization.

This module is intentionally isolated from workflow registration.  It solves
the frozen-intensity material tangent problem for the fixed-mean-field,
current-carrying periodic bulk profile documented in
``docs/research/pr_periodic_biased_current_carrying_profile.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from lcprop.core.backend import Backend, BackendSpec, asnumpy, get_backend
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

    kx: Any
    ky: Any
    a_M: Any
    a_H: Any
    denominator: Any
    response_kernel: Any
    resolved_mask: Any


@dataclass(frozen=True)
class PRBiasedLinearizedReferenceResult:
    """Material perturbations and compact scientific provenance."""

    delta_psi: Any
    delta_E_x: Any
    delta_E_y: Any
    delta_P: Any
    delta_mean_current: Any
    mean_intensity_perturbation: Any
    response_kernel: Any
    denominator: Any
    reference_intensity: float
    applied_field: float
    dx_normalized: float
    dy_normalized: float
    m_y: float
    h_y: float
    requested_backend: str
    resolved_backend: str
    real_dtype: str
    complex_dtype: str
    is_gpu: bool
    device_identity: str | None
    model_id: str = PR_PERIODIC_BIASED_LINEARIZED_REFERENCE_V1
    profile_id: str = PR_PERIODIC_BIASED_CURRENT_PROFILE_V1
    electrical_ensemble: str = FIXED_MEAN_FIELD_ENSEMBLE
    forward_fft_count: int = 1
    inverse_fft_count: int = 4


_DEFAULT_BACKEND = BackendSpec(
    backend="numpy",
    precision="float64",
    verbose=False,
)


def _resolve_backend(spec: BackendSpec | str) -> tuple[Backend, str]:
    if isinstance(spec, str):
        requested = spec
        backend = get_backend(spec, precision="float64")
    else:
        requested = spec.backend
        backend = get_backend(spec)
    return backend, requested


def _validated_intensity(intensity, *, backend: Backend):
    xp = backend.xp
    if xp is np:
        array = np.asarray(asnumpy(intensity))
    else:
        array = xp.asarray(intensity)
    expected_dtype = np.dtype(backend.real_dtype)
    if array.dtype != expected_dtype:
        raise TypeError(
            f"{backend.name} linearized reference requires "
            f"{expected_dtype.name} intensity"
        )
    if array.ndim not in (2, 3) or min(array.shape[-2:]) < 3:
        raise ValueError(
            "intensity must have shape (Nx, Ny) or (Nbatch, Nx, Ny); "
            "Nbatch contains independent frozen-intensity planes, not "
            "longitudinal evolution or retained z history"
        )
    valid = xp.all(xp.isfinite(array) & (array > 0.0))
    # This is the sole CuPy synchronization in the solver: validation occurs
    # once at the API boundary before the device-native computational path.
    if not bool(asnumpy(valid)):
        raise ValueError(
            "physical total intensity must be finite and strictly positive"
        )
    return array


def _device_identity(backend: Backend) -> str | None:
    if not backend.is_gpu:
        return None
    return f"cuda:{backend.xp.cuda.Device().id}"


def biased_linearized_fourier_symbol(
    shape: tuple[int, int],
    *,
    spec: PRBiasedLinearizedReferenceSpec,
    backend: BackendSpec | str = _DEFAULT_BACKEND,
) -> PRBiasedLinearizedFourierSymbol:
    """Return the repository-consistent Fourier grid and response kernel.

    The canonical transverse derivative convention zeros each even-grid
    Nyquist first-derivative symbol.  Consequently ``resolved_mask`` excludes
    every joint derivative-null mode, including the constant gauge mode and,
    on even grids, the corresponding Nyquist null combinations.
    """

    resolved_backend, _ = _resolve_backend(backend)
    return _biased_linearized_fourier_symbol(
        shape,
        spec=spec,
        backend=resolved_backend,
    )


def _biased_linearized_fourier_symbol(
    shape: tuple[int, int],
    *,
    spec: PRBiasedLinearizedReferenceSpec,
    backend: Backend,
) -> PRBiasedLinearizedFourierSymbol:
    spec.validate()
    if len(shape) != 2 or min(shape) < 3:
        raise ValueError("shape must contain transverse sizes (Nx, Ny), each at least 3")
    resolved_backend = backend
    xp = resolved_backend.xp
    kx, ky = spectral_wavevectors(
        shape,
        dx_normalized=float(spec.dx_normalized),
        dy_normalized=float(spec.dy_normalized),
        xp=xp,
    )
    kx = kx.astype(resolved_backend.real_dtype, copy=False)
    ky = ky.astype(resolved_backend.real_dtype, copy=False)
    a_M = kx * kx + float(spec.m_y) * ky * ky
    a_H = kx * kx + float(spec.h_y) * ky * ky
    bias_kx = float(spec.applied_field) * kx
    denominator = float(spec.reference_intensity) * (
        a_M * (1.0 + a_H) + 1j * bias_kx * a_H
    )
    numerator = -(a_M + 1j * bias_kx)
    resolved_mask = a_H > 0.0
    denominator = denominator.astype(resolved_backend.complex_dtype, copy=False)
    numerator = numerator.astype(resolved_backend.complex_dtype, copy=False)
    if xp is np:
        response_kernel = np.zeros(shape, dtype=resolved_backend.complex_dtype)
        np.divide(
            numerator,
            denominator,
            out=response_kernel,
            where=resolved_mask,
        )
    else:
        # CuPy does not support NumPy's combined out/where ufunc form.
        # Avoid evaluating a division at the joint derivative-null modes.
        safe_denominator = xp.where(resolved_mask, denominator, 1.0)
        response_kernel = xp.where(
            resolved_mask,
            numerator / safe_denominator,
            0.0,
        ).astype(resolved_backend.complex_dtype, copy=False)
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
    backend: BackendSpec | str = _DEFAULT_BACKEND,
) -> PRBiasedLinearizedReferenceResult:
    """Solve the frozen-intensity biased material tangent problem.

    ``intensity`` is the complete normalized physical transport intensity and
    ``delta_I = intensity - reference_intensity``.  A nonzero mean
    ``delta_I`` is allowed: its Fourier zero mode produces no perturbation
    potential in the fixed-reference solve, but changes the first-order mean
    current by ``-applied_field * mean(delta_I)``.

    ``backend`` uses the repository's :class:`BackendSpec` convention. The
    default preserves the original NumPy/float64 API. Result arrays belong to
    the resolved backend; no implicit host conversion occurs on return.

    The solve performs one forward two-dimensional FFT and four inverse FFTs
    (potential, two field components, and carrier perturbation). For 3-D
    input, the leading axis is an independent batch of frozen transverse
    planes, not a retained longitudinal volume.
    """

    spec.validate()
    resolved_backend, requested_backend = _resolve_backend(backend)
    xp = resolved_backend.xp
    driving = _validated_intensity(intensity, backend=resolved_backend)
    symbol = _biased_linearized_fourier_symbol(
        driving.shape[-2:],
        spec=spec,
        backend=resolved_backend,
    )
    delta_intensity = driving - float(spec.reference_intensity)
    mean_intensity_perturbation = xp.mean(
        delta_intensity,
        axis=(-2, -1),
    )
    zero_mean_delta_intensity = delta_intensity - xp.expand_dims(
        mean_intensity_perturbation,
        axis=(-2, -1),
    )
    delta_intensity_hat = xp.fft.fft2(
        zero_mean_delta_intensity,
        axes=(-2, -1),
    )
    delta_psi_hat = delta_intensity_hat * symbol.response_kernel

    delta_psi = xp.fft.ifft2(delta_psi_hat, axes=(-2, -1)).real
    delta_E_x = xp.fft.ifft2(
        -1j * symbol.kx * delta_psi_hat,
        axes=(-2, -1),
    ).real
    delta_E_y = xp.fft.ifft2(
        -1j * symbol.ky * delta_psi_hat,
        axes=(-2, -1),
    ).real
    delta_P = xp.fft.ifft2(
        symbol.a_H * delta_psi_hat,
        axes=(-2, -1),
    ).real

    delta_mean_current = xp.stack(
        (
            -float(spec.applied_field) * mean_intensity_perturbation,
            xp.zeros_like(mean_intensity_perturbation),
        ),
        axis=-1,
    )

    return PRBiasedLinearizedReferenceResult(
        delta_psi=delta_psi.astype(resolved_backend.real_dtype, copy=False),
        delta_E_x=delta_E_x.astype(resolved_backend.real_dtype, copy=False),
        delta_E_y=delta_E_y.astype(resolved_backend.real_dtype, copy=False),
        delta_P=delta_P.astype(resolved_backend.real_dtype, copy=False),
        delta_mean_current=delta_mean_current.astype(
            resolved_backend.real_dtype, copy=False
        ),
        mean_intensity_perturbation=xp.asarray(
            mean_intensity_perturbation, dtype=resolved_backend.real_dtype
        ),
        response_kernel=symbol.response_kernel,
        denominator=symbol.denominator,
        reference_intensity=float(spec.reference_intensity),
        applied_field=float(spec.applied_field),
        dx_normalized=float(spec.dx_normalized),
        dy_normalized=float(spec.dy_normalized),
        m_y=float(spec.m_y),
        h_y=float(spec.h_y),
        requested_backend=requested_backend,
        resolved_backend=resolved_backend.name,
        real_dtype=np.dtype(resolved_backend.real_dtype).name,
        complex_dtype=np.dtype(resolved_backend.complex_dtype).name,
        is_gpu=resolved_backend.is_gpu,
        device_identity=_device_identity(resolved_backend),
    )


def total_fields_from_perturbation(
    result: PRBiasedLinearizedReferenceResult,
) -> tuple[Any, Any]:
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
