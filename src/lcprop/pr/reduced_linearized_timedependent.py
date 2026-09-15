"""Exact transient linearization of the reduced x-only PR equation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from lcprop.core.backend import BackendSpec, get_backend
from lcprop.pr.reduced_linearized import (
    PRReducedLinearizedSpec,
    reduced_linearized_residual,
    reduced_linearized_symbol,
)


PR_REDUCED_LINEARIZED_TIMEDEPENDENT_V1 = (
    "pr_reduced_x_linearized_timedependent_v1"
)
_DEFAULT_BACKEND = BackendSpec(
    backend="numpy", precision="float64", verbose=False
)


@dataclass(frozen=True)
class PRReducedLinearizedTimeDependentResult:
    """One exact frozen-source update of the authoritative total E field."""

    E: Any
    delta_E: Any
    equilibrium_E: Any
    rhs: Any
    decay_rate: Any
    source_coefficient: Any
    equilibrium_field: float
    time_step_normalized: float
    requested_backend: str
    resolved_backend: str
    real_dtype: str
    complex_dtype: str
    is_gpu: bool


def reduced_linearized_timedependent_coefficients(
    nx: int,
    *,
    spec: PRReducedLinearizedSpec,
    backend: BackendSpec = _DEFAULT_BACKEND,
):
    """Return discrete ``(Lambda, S)`` for ``dE_hat/dtau=-Lambda E_hat+S I_hat``."""

    resolved = get_backend(backend)
    symbol = reduced_linearized_symbol(nx, spec=spec, backend=backend)
    xp = resolved.xp
    reference = xp.asarray(spec.reference_intensity, dtype=resolved.real_dtype)
    decay_rate = (reference * symbol.denominator).astype(
        resolved.complex_dtype, copy=False
    )
    source = (
        -xp.asarray(spec.equilibrium_field, dtype=resolved.real_dtype)
        + 1j * symbol.k1
    ).astype(resolved.complex_dtype, copy=False)
    return decay_rate, source


def reduced_linearized_timedependent_rhs(
    E,
    intensity,
    *,
    spec: PRReducedLinearizedSpec,
    xp: Any = np,
):
    """Evaluate the independently derived first-order reduced transient RHS."""

    return reduced_linearized_residual(E, intensity, spec=spec, xp=xp)


def solve_pr_reduced_linearized_timedependent(
    E,
    intensity,
    *,
    dt_normalized: float,
    spec: PRReducedLinearizedSpec,
    backend: BackendSpec = _DEFAULT_BACKEND,
) -> PRReducedLinearizedTimeDependentResult:
    """Advance the reduced linearized state exactly for one frozen source."""

    spec.validate()
    dt = float(dt_normalized)
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt_normalized must be finite and positive")
    resolved = get_backend(backend)
    xp = resolved.xp
    state = xp.asarray(E)
    driving = xp.asarray(intensity)
    if state.shape != driving.shape:
        raise ValueError("E and intensity must have identical shapes")
    if state.ndim not in (2, 3):
        raise ValueError(
            "E and intensity must have shape (Nx, Ny) or (Nbatch, Nx, Ny)"
        )
    expected_dtype = xp.dtype(resolved.real_dtype)
    if state.dtype != expected_dtype or driving.dtype != expected_dtype:
        raise TypeError(
            f"{resolved.name} E and intensity must have dtype "
            f"{np.dtype(resolved.real_dtype).name}"
        )
    if bool(xp.any(~xp.isfinite(state)).item()) or bool(
        xp.any(~xp.isfinite(driving)).item()
    ):
        raise ValueError("E and intensity must contain only finite values")
    if bool(xp.any(driving <= 0.0).item()):
        raise ValueError("intensity must be strictly positive")

    nx = int(state.shape[-2])
    symbol = reduced_linearized_symbol(nx, spec=spec, backend=backend)
    decay_rate, source = reduced_linearized_timedependent_coefficients(
        nx, spec=spec, backend=backend
    )
    shape = [1] * state.ndim
    shape[-2] = nx
    response = symbol.response_kernel.reshape(shape)
    decay = decay_rate.reshape(shape)

    delta_I_hat = xp.fft.fft(
        driving - xp.asarray(spec.reference_intensity, dtype=resolved.real_dtype),
        axis=-2,
    ).astype(resolved.complex_dtype, copy=False)
    equilibrium_hat = (delta_I_hat * response).astype(
        resolved.complex_dtype, copy=False
    )
    delta_E_hat = xp.fft.fft(
        state - xp.asarray(spec.equilibrium_field, dtype=resolved.real_dtype),
        axis=-2,
    ).astype(resolved.complex_dtype, copy=False)
    dt_value = xp.asarray(dt, dtype=resolved.real_dtype)
    propagated_hat = equilibrium_hat + (delta_E_hat - equilibrium_hat) * xp.exp(
        -decay * dt_value
    )
    delta_E = xp.fft.ifft(propagated_hat, axis=-2).real.astype(
        resolved.real_dtype, copy=False
    )
    total = (
        delta_E + xp.asarray(spec.equilibrium_field, dtype=resolved.real_dtype)
    ).astype(resolved.real_dtype, copy=False)
    equilibrium_E = xp.fft.ifft(equilibrium_hat, axis=-2).real.astype(
        resolved.real_dtype, copy=False
    )
    equilibrium_E = (
        equilibrium_E
        + xp.asarray(spec.equilibrium_field, dtype=resolved.real_dtype)
    ).astype(resolved.real_dtype, copy=False)
    rhs = reduced_linearized_timedependent_rhs(
        total, driving, spec=spec, xp=xp
    ).astype(resolved.real_dtype, copy=False)
    return PRReducedLinearizedTimeDependentResult(
        E=total,
        delta_E=delta_E,
        equilibrium_E=equilibrium_E,
        rhs=rhs,
        decay_rate=decay_rate,
        source_coefficient=source,
        equilibrium_field=float(spec.equilibrium_field),
        time_step_normalized=dt,
        requested_backend=backend.backend,
        resolved_backend=resolved.name,
        real_dtype=np.dtype(resolved.real_dtype).name,
        complex_dtype=np.dtype(resolved.complex_dtype).name,
        is_gpu=resolved.is_gpu,
    )


__all__ = [
    "PR_REDUCED_LINEARIZED_TIMEDEPENDENT_V1",
    "PRReducedLinearizedTimeDependentResult",
    "reduced_linearized_timedependent_coefficients",
    "reduced_linearized_timedependent_rhs",
    "solve_pr_reduced_linearized_timedependent",
]
