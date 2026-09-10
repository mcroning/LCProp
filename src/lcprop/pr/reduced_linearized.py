"""Direct linearized material response for reduced x-only PR transport."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from lcprop.core.backend import Backend, BackendSpec, get_backend
from lcprop.pr.evolution import periodic_derivatives_x


PR_REDUCED_LINEARIZED_RESPONSE_V1 = "pr_reduced_x_linearized_response_v1"
_DEFAULT_BACKEND = BackendSpec(
    backend="numpy", precision="float64", verbose=False
)


@dataclass(frozen=True)
class PRReducedLinearizedSpec:
    """Uniform reference state for the canonical reduced hopping equation."""

    reference_intensity: float
    applied_field: float
    background_intensity: float
    dx_normalized: float

    @property
    def equilibrium_field(self) -> float:
        return (
            float(self.applied_field)
            * float(self.background_intensity)
            / float(self.reference_intensity)
        )

    def validate(self) -> None:
        if (
            not math.isfinite(float(self.reference_intensity))
            or float(self.reference_intensity) <= 0.0
        ):
            raise ValueError("reference_intensity must be finite and positive")
        if not math.isfinite(float(self.applied_field)):
            raise ValueError("applied_field must be finite")
        if (
            not math.isfinite(float(self.background_intensity))
            or float(self.background_intensity) < 0.0
        ):
            raise ValueError(
                "background_intensity must be finite and nonnegative"
            )
        if (
            not math.isfinite(float(self.dx_normalized))
            or float(self.dx_normalized) <= 0.0
        ):
            raise ValueError("dx_normalized must be finite and positive")


@dataclass(frozen=True)
class PRReducedLinearizedSymbol:
    """Centered-difference symbols and response on the periodic x grid."""

    k1: Any
    k2_squared: Any
    denominator: Any
    response_kernel: Any


@dataclass(frozen=True)
class PRReducedLinearizedResult:
    """Total and perturbation fields from one frozen-intensity solve."""

    E: Any
    delta_E: Any
    residual: Any
    response_kernel: Any
    denominator: Any
    equilibrium_field: float
    reference_intensity: float
    applied_field: float
    background_intensity: float
    dx_normalized: float
    requested_backend: str
    resolved_backend: str
    real_dtype: str
    complex_dtype: str
    is_gpu: bool


def _resolve_backend(spec: BackendSpec | str) -> tuple[Backend, str]:
    if isinstance(spec, BackendSpec):
        requested = spec.backend
        resolved = get_backend(spec)
    else:
        requested = str(spec)
        resolved = get_backend(
            BackendSpec(backend=requested, precision="float64", verbose=False)
        )
    return resolved, requested


def reduced_linearized_symbol(
    nx: int,
    *,
    spec: PRReducedLinearizedSpec,
    backend: BackendSpec | str = _DEFAULT_BACKEND,
) -> PRReducedLinearizedSymbol:
    """Return the exact centered-difference response of the reduced model."""

    spec.validate()
    if int(nx) < 3:
        raise ValueError("nx must be at least 3")
    resolved, _ = _resolve_backend(backend)
    xp = resolved.xp
    phase = (
        2.0
        * math.pi
        * xp.fft.fftfreq(int(nx))
    ).astype(resolved.real_dtype, copy=False)
    dx = float(spec.dx_normalized)
    k1 = (xp.sin(phase) / dx).astype(resolved.real_dtype, copy=False)
    if int(nx) % 2 == 0:
        k1[int(nx) // 2] = 0.0
    k2_squared = (
        4.0 * xp.sin(0.5 * phase) ** 2 / (dx * dx)
    ).astype(resolved.real_dtype, copy=False)
    equilibrium = float(spec.equilibrium_field)
    denominator = (
        1.0 + k2_squared + 1j * equilibrium * k1
    ).astype(resolved.complex_dtype, copy=False)
    numerator = (
        (1j * k1 - equilibrium) / float(spec.reference_intensity)
    ).astype(resolved.complex_dtype, copy=False)
    response = (numerator / denominator).astype(
        resolved.complex_dtype, copy=False
    )
    return PRReducedLinearizedSymbol(
        k1=k1,
        k2_squared=k2_squared,
        denominator=denominator,
        response_kernel=response,
    )


def reduced_linearized_residual(
    E,
    intensity,
    *,
    spec: PRReducedLinearizedSpec,
    xp: Any = np,
):
    """Evaluate the first-order static RHS of the reduced hopping equation."""

    spec.validate()
    if E.shape != intensity.shape:
        raise ValueError("E and intensity must have identical shapes")
    delta_E = E - float(spec.equilibrium_field)
    delta_I = intensity - float(spec.reference_intensity)
    delta_E_x, delta_E_xx = periodic_derivatives_x(
        delta_E, dx_normalized=spec.dx_normalized, xp=xp
    )
    delta_I_x, _ = periodic_derivatives_x(
        delta_I, dx_normalized=spec.dx_normalized, xp=xp
    )
    reference = float(spec.reference_intensity)
    equilibrium = float(spec.equilibrium_field)
    return (
        -equilibrium * delta_I
        - reference * delta_E
        + delta_I_x
        - equilibrium * reference * delta_E_x
        + reference * delta_E_xx
    )


def solve_pr_reduced_linearized_intensity(
    intensity,
    *,
    spec: PRReducedLinearizedSpec,
    backend: BackendSpec | str = _DEFAULT_BACKEND,
) -> PRReducedLinearizedResult:
    """Solve the reduced first-order static equation along independent y rows."""

    spec.validate()
    resolved, requested = _resolve_backend(backend)
    xp = resolved.xp
    driving = xp.asarray(intensity)
    if driving.ndim not in (2, 3):
        raise ValueError(
            "intensity must have shape (Nx, Ny) or (Nbatch, Nx, Ny)"
        )
    if driving.dtype != xp.dtype(resolved.real_dtype):
        raise TypeError(
            f"{resolved.name} intensity must have dtype "
            f"{np.dtype(resolved.real_dtype).name}"
        )
    if bool(xp.any(~xp.isfinite(driving)).item()):
        raise ValueError("intensity must contain only finite values")
    if bool(xp.any(driving <= 0.0).item()):
        raise ValueError("intensity must be strictly positive")

    symbol = reduced_linearized_symbol(
        driving.shape[-2], spec=spec, backend=backend
    )
    delta_I = driving - float(spec.reference_intensity)
    delta_I_hat = xp.fft.fft(delta_I, axis=-2).astype(
        resolved.complex_dtype, copy=False
    )
    shape = [1] * driving.ndim
    shape[-2] = driving.shape[-2]
    response = symbol.response_kernel.reshape(shape)
    delta_E_hat = delta_I_hat * response
    delta_E = xp.fft.ifft(delta_E_hat, axis=-2).real.astype(
        resolved.real_dtype, copy=False
    )
    total_E = (delta_E + float(spec.equilibrium_field)).astype(
        resolved.real_dtype, copy=False
    )
    residual = reduced_linearized_residual(
        total_E, driving, spec=spec, xp=xp
    ).astype(resolved.real_dtype, copy=False)
    return PRReducedLinearizedResult(
        E=total_E,
        delta_E=delta_E,
        residual=residual,
        response_kernel=symbol.response_kernel,
        denominator=symbol.denominator,
        equilibrium_field=float(spec.equilibrium_field),
        reference_intensity=float(spec.reference_intensity),
        applied_field=float(spec.applied_field),
        background_intensity=float(spec.background_intensity),
        dx_normalized=float(spec.dx_normalized),
        requested_backend=requested,
        resolved_backend=resolved.name,
        real_dtype=np.dtype(resolved.real_dtype).name,
        complex_dtype=np.dtype(resolved.complex_dtype).name,
        is_gpu=resolved.is_gpu,
    )


__all__ = [
    "PR_REDUCED_LINEARIZED_RESPONSE_V1",
    "PRReducedLinearizedResult",
    "PRReducedLinearizedSpec",
    "PRReducedLinearizedSymbol",
    "reduced_linearized_residual",
    "reduced_linearized_symbol",
    "solve_pr_reduced_linearized_intensity",
]
