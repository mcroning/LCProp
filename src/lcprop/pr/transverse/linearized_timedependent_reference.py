"""Exact modal reference for periodic biased full-transverse PR dynamics.

This module is intentionally isolated from production workflow registration.
It evolves the tangent potential for a prescribed, time-independent intensity
perturbation and reuses the accepted static response kernel as its equilibrium.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from lcprop.core.backend import Backend, BackendSpec, asnumpy, get_backend
from lcprop.pr.transverse.linearized_reference import (
    PRBiasedLinearizedReferenceSpec,
    biased_linearized_fourier_symbol,
)


PR_PERIODIC_BIASED_LINEARIZED_TIMEDEPENDENT_REFERENCE_V1 = (
    "pr_full_transverse_periodic_biased_linearized_timedependent_reference_v1"
)


@dataclass(frozen=True)
class PRBiasedLinearizedTimedependentFourierSymbol:
    """Modal decay, source, and accepted equilibrium coefficients."""

    kx: Any
    ky: Any
    a_M: Any
    a_H: Any
    decay_rate: Any
    source_kernel: Any
    equilibrium_kernel: Any
    resolved_mask: Any


@dataclass(frozen=True)
class PRBiasedLinearizedTimedependentReferenceResult:
    """Exact transient material perturbation and compact provenance."""

    delta_psi_initial: Any
    delta_psi: Any
    delta_psi_equilibrium: Any
    delta_E_x: Any
    delta_E_y: Any
    delta_P: Any
    decay_rate: Any
    source_kernel: Any
    equilibrium_kernel: Any
    resolved_mask: Any
    time_normalized: float
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
    model_id: str = PR_PERIODIC_BIASED_LINEARIZED_TIMEDEPENDENT_REFERENCE_V1


_DEFAULT_BACKEND = BackendSpec(
    backend="numpy",
    precision="float64",
    verbose=False,
)


def _resolve_backend(spec: BackendSpec | str) -> tuple[Backend, str]:
    if isinstance(spec, str):
        return get_backend(spec, precision="float64"), spec
    return get_backend(spec), spec.backend


def _device_identity(backend: Backend) -> str | None:
    if not backend.is_gpu:
        return None
    return f"cuda:{backend.xp.cuda.Device().id}"


def _validated_plane_batch(value, *, name: str, backend: Backend):
    xp = backend.xp
    if xp is np:
        array = np.asarray(asnumpy(value))
    else:
        array = xp.asarray(value)
    expected_dtype = np.dtype(backend.real_dtype)
    if array.dtype != expected_dtype:
        raise TypeError(
            f"{backend.name} linearized TD reference requires "
            f"{expected_dtype.name} {name}"
        )
    if array.ndim not in (2, 3) or min(array.shape[-2:]) < 3:
        raise ValueError(
            f"{name} must have shape (Nx, Ny) or (Nbatch, Nx, Ny); "
            "Nbatch contains independent material planes, not time history"
        )
    finite = xp.all(xp.isfinite(array))
    if not bool(asnumpy(finite)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def biased_linearized_timedependent_fourier_symbol(
    shape: tuple[int, int],
    *,
    spec: PRBiasedLinearizedReferenceSpec,
    backend: BackendSpec | str = _DEFAULT_BACKEND,
) -> PRBiasedLinearizedTimedependentFourierSymbol:
    """Return ``Lambda``, ``S``, and the accepted static response kernel.

    On each resolved mode the repository-normalized modal equation is

    ``delta_psi_hat_tau = -Lambda*delta_psi_hat + S*delta_I_hat``.

    Joint first-derivative null modes have all three coefficients set to zero.
    """

    spec.validate()
    resolved_backend, _ = _resolve_backend(backend)
    xp = resolved_backend.xp
    static_symbol = biased_linearized_fourier_symbol(
        shape,
        spec=spec,
        backend=backend,
    )
    safe_a_H = xp.where(static_symbol.resolved_mask, static_symbol.a_H, 1.0)
    decay_rate = xp.where(
        static_symbol.resolved_mask,
        static_symbol.denominator / safe_a_H,
        0.0,
    ).astype(resolved_backend.complex_dtype, copy=False)
    source_kernel = xp.where(
        static_symbol.resolved_mask,
        -(
            static_symbol.a_M
            + 1j * float(spec.applied_field) * static_symbol.kx
        )
        / safe_a_H,
        0.0,
    ).astype(resolved_backend.complex_dtype, copy=False)
    return PRBiasedLinearizedTimedependentFourierSymbol(
        kx=static_symbol.kx,
        ky=static_symbol.ky,
        a_M=static_symbol.a_M,
        a_H=static_symbol.a_H,
        decay_rate=decay_rate,
        source_kernel=source_kernel,
        equilibrium_kernel=static_symbol.response_kernel,
        resolved_mask=static_symbol.resolved_mask,
    )


def _zero_mean_resolved_hat(field, *, symbol, xp):
    zero_mean = field - xp.mean(field, axis=(-2, -1), keepdims=True)
    transformed = xp.fft.fft2(zero_mean, axes=(-2, -1))
    return xp.where(symbol.resolved_mask, transformed, 0.0)


def linearized_timedependent_rhs(
    delta_psi,
    intensity,
    *,
    spec: PRBiasedLinearizedReferenceSpec,
    backend: BackendSpec | str = _DEFAULT_BACKEND,
):
    """Evaluate the independently derived linearized potential-rate RHS."""

    spec.validate()
    resolved_backend, _ = _resolve_backend(backend)
    xp = resolved_backend.xp
    potential = _validated_plane_batch(
        delta_psi, name="delta_psi", backend=resolved_backend
    )
    driving = _validated_plane_batch(
        intensity, name="intensity", backend=resolved_backend
    )
    if potential.shape != driving.shape:
        raise ValueError("delta_psi and intensity must have identical shapes")
    positive = xp.all(driving > 0.0)
    if not bool(asnumpy(positive)):
        raise ValueError("physical total intensity must be strictly positive")
    symbol = biased_linearized_timedependent_fourier_symbol(
        driving.shape[-2:], spec=spec, backend=backend
    )
    delta_intensity_hat = _zero_mean_resolved_hat(
        driving - float(spec.reference_intensity), symbol=symbol, xp=xp
    )
    delta_psi_hat = _zero_mean_resolved_hat(potential, symbol=symbol, xp=xp)
    rhs_hat = (
        -symbol.decay_rate * delta_psi_hat
        + symbol.source_kernel * delta_intensity_hat
    )
    rhs = xp.fft.ifft2(rhs_hat, axes=(-2, -1)).real
    return rhs.astype(resolved_backend.real_dtype, copy=False)


def solve_pr_biased_linearized_timedependent_reference(
    intensity,
    *,
    time_normalized: float,
    spec: PRBiasedLinearizedReferenceSpec,
    initial_delta_psi=None,
    backend: BackendSpec | str = _DEFAULT_BACKEND,
) -> PRBiasedLinearizedTimedependentReferenceResult:
    """Evolve the prescribed-intensity tangent problem exactly in Fourier space.

    ``time_normalized`` is the repository material time ``tau = t/t0``.  The
    default initial condition is zero perturbation.  Supplied initial data are
    projected onto the same gauge-compatible derivative-resolved subspace as
    the accepted static reference.  A leading batch dimension denotes
    independent material planes, never retained time or longitudinal history.
    """

    spec.validate()
    time_value = float(time_normalized)
    if not math.isfinite(time_value) or time_value < 0.0:
        raise ValueError("time_normalized must be finite and nonnegative")
    resolved_backend, requested_backend = _resolve_backend(backend)
    xp = resolved_backend.xp
    driving = _validated_plane_batch(
        intensity, name="intensity", backend=resolved_backend
    )
    positive = xp.all(driving > 0.0)
    if not bool(asnumpy(positive)):
        raise ValueError("physical total intensity must be strictly positive")
    if initial_delta_psi is None:
        initial = xp.zeros_like(driving)
    else:
        initial = _validated_plane_batch(
            initial_delta_psi,
            name="initial_delta_psi",
            backend=resolved_backend,
        )
        if initial.shape != driving.shape:
            raise ValueError(
                "initial_delta_psi and intensity must have identical shapes"
            )

    symbol = biased_linearized_timedependent_fourier_symbol(
        driving.shape[-2:], spec=spec, backend=backend
    )
    delta_intensity_hat = _zero_mean_resolved_hat(
        driving - float(spec.reference_intensity), symbol=symbol, xp=xp
    )
    initial_hat = _zero_mean_resolved_hat(initial, symbol=symbol, xp=xp)
    equilibrium_hat = delta_intensity_hat * symbol.equilibrium_kernel
    transient = xp.exp(-symbol.decay_rate * time_value)
    final_hat = equilibrium_hat + (initial_hat - equilibrium_hat) * transient
    final_hat = xp.where(symbol.resolved_mask, final_hat, 0.0)

    initial_projected = xp.fft.ifft2(initial_hat, axes=(-2, -1)).real
    equilibrium = xp.fft.ifft2(equilibrium_hat, axes=(-2, -1)).real
    delta_psi = xp.fft.ifft2(final_hat, axes=(-2, -1)).real
    delta_E_x = xp.fft.ifft2(
        -1j * symbol.kx * final_hat, axes=(-2, -1)
    ).real
    delta_E_y = xp.fft.ifft2(
        -1j * symbol.ky * final_hat, axes=(-2, -1)
    ).real
    delta_P = xp.fft.ifft2(
        symbol.a_H * final_hat, axes=(-2, -1)
    ).real

    return PRBiasedLinearizedTimedependentReferenceResult(
        delta_psi_initial=initial_projected.astype(
            resolved_backend.real_dtype, copy=False
        ),
        delta_psi=delta_psi.astype(resolved_backend.real_dtype, copy=False),
        delta_psi_equilibrium=equilibrium.astype(
            resolved_backend.real_dtype, copy=False
        ),
        delta_E_x=delta_E_x.astype(resolved_backend.real_dtype, copy=False),
        delta_E_y=delta_E_y.astype(resolved_backend.real_dtype, copy=False),
        delta_P=delta_P.astype(resolved_backend.real_dtype, copy=False),
        decay_rate=symbol.decay_rate,
        source_kernel=symbol.source_kernel,
        equilibrium_kernel=symbol.equilibrium_kernel,
        resolved_mask=symbol.resolved_mask,
        time_normalized=time_value,
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


__all__ = [
    "PR_PERIODIC_BIASED_LINEARIZED_TIMEDEPENDENT_REFERENCE_V1",
    "PRBiasedLinearizedTimedependentFourierSymbol",
    "PRBiasedLinearizedTimedependentReferenceResult",
    "biased_linearized_timedependent_fourier_symbol",
    "linearized_timedependent_rhs",
    "solve_pr_biased_linearized_timedependent_reference",
]
