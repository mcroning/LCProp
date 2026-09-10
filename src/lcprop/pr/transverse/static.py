"""Backend-native solver for frozen full-transverse PR equilibrium."""

from __future__ import annotations

from dataclasses import dataclass
import math
from time import perf_counter
from typing import Any, Callable

import numpy as np

from lcprop.core.backend import asnumpy, scalar_float
from lcprop.pr.transverse.transport import (
    potential_rhs,
    spectral_derivatives,
    spectral_wavevectors,
    state_from_potential,
)


CancellationCheck = Callable[[], bool]


class _StaticCancellationRequested(Exception):
    """Abort a discardable nonlinear material candidate at a safe checkpoint."""

    def __init__(self, stage: str) -> None:
        super().__init__(stage)
        self.stage = stage


def _raise_if_cancelled(
    cancellation_check: CancellationCheck | None,
    *,
    stage: str,
) -> None:
    if cancellation_check is not None and cancellation_check():
        raise _StaticCancellationRequested(stage)


@dataclass(frozen=True)
class PRTransverseStaticMaterialSolverOptions:
    """Controls for one or more independent frozen-intensity planes.

    ``td_rhs_*`` tolerances are retained for the explicitly invoked
    finite-grid TD fixed-point corrector.  They do not gate the canonical
    zero-flux solve.
    """

    max_newton_iterations: int = 30
    max_pcg_iterations: int = 200
    pcg_relative_tolerance: float = 1.0e-8
    pcg_absolute_tolerance: float = 1.0e-12
    pcg_near_gate_relative_tolerance: float = 1.0e-7
    pcg_near_gate_absolute_tolerance: float = 1.0e-9
    equilibrium_rms_tolerance: float = 1.0e-9
    equilibrium_max_tolerance: float = 1.0e-8
    td_rhs_rms_tolerance: float = 1.0e-8
    td_rhs_max_tolerance: float = 1.0e-7
    max_backtracks: int = 16
    minimum_step_scale: float = 2.0**-16
    armijo_fraction: float = 1.0e-4

    def validate(self) -> None:
        if int(self.max_newton_iterations) < 1:
            raise ValueError("max_newton_iterations must be at least one")
        if int(self.max_pcg_iterations) < 1:
            raise ValueError("max_pcg_iterations must be at least one")
        if int(self.max_backtracks) < 0:
            raise ValueError("max_backtracks must be nonnegative")
        for name in (
            "pcg_relative_tolerance",
            "pcg_absolute_tolerance",
            "pcg_near_gate_relative_tolerance",
            "pcg_near_gate_absolute_tolerance",
            "equilibrium_rms_tolerance",
            "equilibrium_max_tolerance",
            "td_rhs_rms_tolerance",
            "td_rhs_max_tolerance",
            "minimum_step_scale",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        fraction = float(self.armijo_fraction)
        if not math.isfinite(fraction) or not 0.0 < fraction < 1.0:
            raise ValueError("armijo_fraction must be between zero and one")


@dataclass(frozen=True)
class PRTransverseStaticNewtonRecord:
    """One accepted or rejected Newton correction for one z plane."""

    plane_index: int
    newton_iteration: int
    equilibrium_rms_before: float
    equilibrium_max_before: float
    equilibrium_rms_after: float
    equilibrium_max_after: float
    pcg_iterations: int
    pcg_converged: bool
    pcg_effective_tolerance: float
    pcg_final_residual_norm: float
    pcg_forcing_tightened: bool
    step_scale: float
    backtracks: int
    accepted: bool


@dataclass(frozen=True)
class PRTransverseStaticPlaneSummary:
    """Final frozen-intensity convergence evidence for one plane."""

    plane_index: int
    converged: bool
    status: str
    newton_iterations: int
    pcg_iterations: int
    backtracks: int
    equilibrium_rms: float
    equilibrium_max: float
    td_rhs_rms: float
    td_rhs_max: float
    null_residual_rms: float
    null_residual_max: float
    carrier_mean: float
    carrier_minimum: float
    potential_mean: float
    physical_state_valid: bool


@dataclass(frozen=True)
class PRTransverseStaticMaterialResult:
    """Frozen-intensity potential and complete convergence diagnostics."""

    psi: Any
    equilibrium_residual: Any
    td_rhs_residual: Any
    converged: bool
    status: str
    plane_summaries: tuple[PRTransverseStaticPlaneSummary, ...]
    iteration_records: tuple[PRTransverseStaticNewtonRecord, ...]
    elapsed_seconds: float


@dataclass(frozen=True)
class PRTransverseDiscreteStaticCorrectorOptions:
    """Controls for the production-discrete frozen-intensity corrector."""

    max_newton_iterations: int = 8
    max_gmres_iterations: int = 120
    gmres_restart: int = 30
    gmres_relative_tolerance_float64: float = 1.0e-8
    gmres_relative_tolerance_float32: float = 2.0e-4
    max_backtracks: int = 16
    minimum_step_scale: float = 2.0**-16
    armijo_fraction: float = 1.0e-4

    def validate(self) -> None:
        if int(self.max_newton_iterations) < 1:
            raise ValueError("max_newton_iterations must be at least one")
        if int(self.max_gmres_iterations) < 1:
            raise ValueError("max_gmres_iterations must be at least one")
        if int(self.gmres_restart) < 1:
            raise ValueError("gmres_restart must be at least one")
        if int(self.max_backtracks) < 0:
            raise ValueError("max_backtracks must be nonnegative")
        for name in (
            "gmres_relative_tolerance_float64",
            "gmres_relative_tolerance_float32",
            "minimum_step_scale",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        fraction = float(self.armijo_fraction)
        if not math.isfinite(fraction) or not 0.0 < fraction < 1.0:
            raise ValueError("armijo_fraction must be between zero and one")


@dataclass(frozen=True)
class PRTransverseDiscreteStaticNewtonRecord:
    """One discrete Newton/GMRES correction for one frozen z plane."""

    plane_index: int
    newton_iteration: int
    residual_rms_before: float
    residual_max_before: float
    residual_rms_after: float
    residual_max_after: float
    gmres_iterations: int
    gmres_restarts: int
    gmres_relative_residual: float
    gmres_converged: bool
    gmres_status: str
    step_scale: float
    backtracks: int
    accepted: bool


@dataclass(frozen=True)
class PRTransverseDiscreteStaticPlaneSummary:
    """Final production-discrete convergence evidence for one z plane."""

    plane_index: int
    converged: bool
    status: str
    initial_residual_rms: float
    initial_residual_max: float
    final_residual_rms: float
    final_residual_max: float
    newton_iterations: int
    gmres_iterations: int
    gmres_restarts: int
    backtracks: int
    carrier_mean: float
    carrier_minimum: float
    potential_mean: float


@dataclass(frozen=True)
class PRTransverseDiscreteStaticMaterialResult:
    """Two-stage continuum-initialized production-discrete material solve."""

    psi: Any
    equilibrium_residual: Any
    td_rhs_residual: Any
    converged: bool
    status: str
    continuum_result: PRTransverseStaticMaterialResult
    plane_summaries: tuple[PRTransverseDiscreteStaticPlaneSummary, ...]
    iteration_records: tuple[PRTransverseDiscreteStaticNewtonRecord, ...]
    continuum_elapsed_seconds: float
    discrete_elapsed_seconds: float
    elapsed_seconds: float


def _validate_real_array(value: Any, *, name: str, xp: Any) -> Any:
    array = xp.asarray(value)
    if array.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError(f"{name} must have dtype float32 or float64")
    if array.ndim not in (2, 3) or min(array.shape[-2:]) < 3:
        raise ValueError(f"{name} must end in (Nx, Ny), each at least 3")
    if not bool(asnumpy(xp.all(xp.isfinite(array)))):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _symbols(
    shape,
    *,
    dx_normalized: float,
    dy_normalized: float,
    h_y: float,
    xp: Any = np,
):
    kx, ky = spectral_wavevectors(
        shape,
        dx_normalized=dx_normalized,
        dy_normalized=dy_normalized,
        xp=xp,
    )
    denominator = kx * kx + float(h_y) * ky * ky
    return denominator, denominator == 0.0


def project_production_resolved_modes(
    value,
    *,
    dx_normalized: float,
    dy_normalized: float,
    h_y: float = 1.0,
    xp: Any = np,
) -> Any:
    """Project onto the modes resolved by production transverse derivatives."""

    array = _validate_real_array(value, name="value", xp=xp)
    _, null_mask = _symbols(
        array.shape[-2:],
        dx_normalized=dx_normalized,
        dy_normalized=dy_normalized,
        h_y=h_y,
        xp=xp,
    )
    return _project_resolved_modes(array, null_mask=null_mask, xp=xp)


def _project_resolved_modes(array, *, null_mask, xp: Any):
    transformed = xp.fft.fft2(array, axes=(-2, -1))
    transformed[..., null_mask] = 0.0
    projected = xp.fft.ifft2(transformed, axes=(-2, -1)).real
    return projected.astype(array.dtype, copy=False)


def _normalized_equilibrium_carrier(psi, intensity, *, xp: Any = np):
    accumulator_dtype = xp.float64 if psi.dtype == np.dtype(np.float64) else xp.float32
    log_weight = -psi.astype(accumulator_dtype, copy=False) - xp.log(
        intensity.astype(accumulator_dtype, copy=False)
    )
    log_weight -= xp.max(log_weight, axis=(-2, -1), keepdims=True)
    weight = xp.exp(log_weight)
    weight /= xp.mean(weight, axis=(-2, -1), keepdims=True)
    return weight.astype(psi.dtype, copy=False)


def static_equilibrium_residual(
    psi,
    intensity,
    *,
    dx_normalized: float,
    dy_normalized: float,
    h_y: float = 1.0,
    xp: Any = np,
) -> Any:
    """Return ``P - (exp(-psi)/I)/mean(exp(-psi)/I)``.

    The complete transport intensity must already include optical, dark, and
    optional uniform-background contributions.
    """

    potential = _validate_real_array(psi, name="psi", xp=xp)
    driving = _validate_real_array(intensity, name="intensity", xp=xp)
    if xp is np and driving.dtype != np.dtype(np.float64):
        raise TypeError("NumPy static reference solver requires float64 intensity")
    if potential.shape != driving.shape:
        raise ValueError("psi and intensity must have identical shapes")
    if bool(asnumpy(xp.any(driving <= 0.0))):
        raise ValueError("static transport intensity must be strictly positive")
    _, null_mask = _symbols(
        potential.shape[-2:],
        dx_normalized=dx_normalized,
        dy_normalized=dy_normalized,
        h_y=h_y,
        xp=xp,
    )
    return _static_equilibrium_residual_arrays(
        potential,
        driving,
        null_mask=null_mask,
        dx_normalized=dx_normalized,
        dy_normalized=dy_normalized,
        h_y=h_y,
        xp=xp,
    )


def _static_equilibrium_residual_arrays(
    potential,
    driving,
    *,
    null_mask,
    dx_normalized: float,
    dy_normalized: float,
    h_y: float,
    xp: Any,
):
    """Evaluate validated backend arrays without repeating host-side checks."""

    resolved = _project_resolved_modes(potential, null_mask=null_mask, xp=xp)
    state = state_from_potential(
        resolved,
        dx_normalized=dx_normalized,
        dy_normalized=dy_normalized,
        h_y=h_y,
        xp=xp,
    )
    equilibrium_carrier = _normalized_equilibrium_carrier(resolved, driving, xp=xp)
    return (state.carrier_density - equilibrium_carrier).astype(
        potential.dtype, copy=False
    )


def _metrics(value, *, xp: Any = np) -> tuple[float, float]:
    accumulator_dtype = xp.float64 if value.dtype == np.dtype(np.float64) else xp.float32
    work = value.astype(accumulator_dtype, copy=False)
    return (
        scalar_float(xp.sqrt(xp.mean(work * work))),
        scalar_float(xp.max(xp.abs(work))),
    )


def _physical_state_validity(state, *, xp: Any = np) -> tuple[bool, float, float, float]:
    """Return physical static validity and its scalar normalization evidence."""

    carrier_mean = scalar_float(xp.mean(state.carrier_density))
    carrier_minimum = scalar_float(xp.min(state.carrier_density))
    potential_mean = scalar_float(xp.mean(state.psi))
    dtype_epsilon = float(np.finfo(state.psi.dtype).eps)
    potential_scale = max(1.0, scalar_float(xp.max(xp.abs(state.psi))))
    valid = (
        bool(asnumpy(xp.all(xp.isfinite(state.psi))))
        and bool(asnumpy(xp.all(xp.isfinite(state.carrier_density))))
        and carrier_minimum > 0.0
        and abs(carrier_mean - 1.0) <= 64.0 * dtype_epsilon
        and abs(potential_mean) <= 64.0 * dtype_epsilon * potential_scale
    )
    return valid, carrier_mean, carrier_minimum, potential_mean


def derivative_null_residual(
    residual,
    *,
    dx_normalized: float,
    dy_normalized: float,
    h_y: float = 1.0,
    xp: Any = np,
) -> Any:
    _, null_mask = _symbols(
        residual.shape[-2:],
        dx_normalized=dx_normalized,
        dy_normalized=dy_normalized,
        h_y=h_y,
        xp=xp,
    )
    return _derivative_null_residual_array(residual, null_mask=null_mask, xp=xp)


def _derivative_null_residual_array(residual, *, null_mask, xp: Any):
    transformed = xp.fft.fft2(residual, axes=(-2, -1))
    transformed[..., ~null_mask] = 0.0
    return xp.fft.ifft2(transformed, axes=(-2, -1)).real.astype(
        residual.dtype, copy=False
    )


def production_steady_residual(
    psi,
    intensity,
    *,
    dx_normalized: float,
    dy_normalized: float,
    m_y: float = 1.0,
    h_y: float = 1.0,
    applied_field_x: float = 0.0,
    xp: Any = np,
) -> Any:
    """Return the exact production-TD potential rate for frozen intensity.

    ``potential_rhs`` already applies the production elliptic pseudoinverse
    and therefore already lies in the dynamically resolved potential
    subspace.  Keeping this helper as a direct call prevents a second,
    potentially inconsistent residual definition.
    """

    return potential_rhs(
        psi,
        intensity,
        dx_normalized=dx_normalized,
        dy_normalized=dy_normalized,
        m_y=m_y,
        h_y=h_y,
        applied_field_x=applied_field_x,
        xp=xp,
    )


def _production_steady_jvp_from_state(
    vector,
    *,
    intensity,
    state,
    kx,
    ky,
    denominator,
    null_mask,
    m_y: float,
    xp: Any,
) -> Any:
    transformed = xp.fft.fft2(vector, axes=(-2, -1))
    transformed = xp.where(null_mask, 0.0, transformed)
    resolved = xp.fft.ifft2(transformed, axes=(-2, -1)).real.astype(
        vector.dtype, copy=False
    )
    delta_carrier = xp.fft.ifft2(
        denominator * transformed, axes=(-2, -1)
    ).real.astype(vector.dtype, copy=False)
    derivative_x, derivative_y = spectral_derivatives(
        resolved, kx=kx, ky=ky, xp=xp
    )
    carrier_intensity = state.carrier_density * intensity
    delta_carrier_intensity = (intensity * delta_carrier).astype(
        vector.dtype, copy=False
    )
    delta_gradient_x, delta_gradient_y = spectral_derivatives(
        delta_carrier_intensity, kx=kx, ky=ky, xp=xp
    )
    delta_flux_x = (
        delta_gradient_x
        - delta_carrier_intensity * state.E_x
        + carrier_intensity * derivative_x
    )
    delta_flux_y = float(m_y) * (
        delta_gradient_y
        - delta_carrier_intensity * state.E_y
        + carrier_intensity * derivative_y
    )
    carrier_rhs_hat = (
        1j * kx * xp.fft.fft2(delta_flux_x, axes=(-2, -1))
        + 1j * ky * xp.fft.fft2(delta_flux_y, axes=(-2, -1))
    )
    safe_denominator = xp.where(denominator > 0.0, denominator, 1.0)
    potential_rhs_hat = xp.where(
        denominator > 0.0,
        carrier_rhs_hat / safe_denominator,
        0.0,
    )
    return xp.fft.ifft2(
        potential_rhs_hat, axes=(-2, -1)
    ).real.astype(vector.dtype, copy=False)


def production_steady_jvp(
    psi,
    intensity,
    vector,
    *,
    dx_normalized: float,
    dy_normalized: float,
    m_y: float = 1.0,
    h_y: float = 1.0,
    applied_field_x: float = 0.0,
    xp: Any = np,
) -> Any:
    """Apply the exact analytic Jacobian of ``production_steady_residual``."""

    potential = _validate_real_array(psi, name="psi", xp=xp)
    driving = _validate_real_array(intensity, name="intensity", xp=xp)
    direction = _validate_real_array(vector, name="vector", xp=xp)
    if potential.shape != driving.shape or potential.shape != direction.shape:
        raise ValueError("psi, intensity, and vector must have identical shapes")
    if potential.dtype != driving.dtype or potential.dtype != direction.dtype:
        raise TypeError("psi, intensity, and vector must have identical dtypes")
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
    denominator = kx * kx + float(h_y) * ky * ky
    return _production_steady_jvp_from_state(
        direction,
        intensity=driving,
        state=state,
        kx=kx,
        ky=ky,
        denominator=denominator,
        null_mask=denominator == 0.0,
        m_y=m_y,
        xp=xp,
    )


def _discrete_precondition(value, *, denominator, null_mask, xp: Any):
    transformed = xp.fft.fft2(value)
    transformed = -transformed / (denominator + 1.0)
    transformed = xp.where(null_mask, 0.0, transformed)
    return xp.fft.ifft2(transformed).real.astype(value.dtype, copy=False)


def _right_preconditioned_gmres(
    rhs,
    *,
    apply_operator,
    apply_preconditioner,
    relative_tolerance: float,
    absolute_tolerance: float,
    restart: int,
    max_iterations: int,
    xp: Any,
) -> tuple[Any, int, int, float, bool, str]:
    """Solve ``A x = rhs`` by restarted right-preconditioned GMRES.

    The effective linear tolerance is capped at half the initial residual
    norm.  This simple inexact-Newton forcing bound prevents an unconverged
    nonlinear state from producing a successful zero-iteration linear solve
    when the caller's absolute tolerance happens to exceed ``||rhs||``.
    """

    solution = xp.zeros_like(rhs)
    norm_rhs = scalar_float(xp.linalg.norm(rhs.ravel()))
    requested_tolerance = max(
        float(absolute_tolerance), float(relative_tolerance) * norm_rhs
    )
    tolerance = min(requested_tolerance, 0.5 * norm_rhs)
    if norm_rhs <= tolerance:
        return solution, 0, 0, 0.0, True, "converged"
    residual = rhs.copy()
    total_iterations = 0
    restart_count = 0
    relative_residual = 1.0
    while total_iterations < int(max_iterations):
        beta = scalar_float(xp.linalg.norm(residual.ravel()))
        if not math.isfinite(beta):
            return (
                solution,
                total_iterations,
                restart_count,
                math.inf,
                False,
                "nonfinite_residual",
            )
        if beta <= tolerance:
            return (
                solution,
                total_iterations,
                restart_count,
                beta / norm_rhs,
                True,
                "converged",
            )
        cycle_size = min(int(restart), int(max_iterations) - total_iterations)
        basis = [residual / beta]
        hessenberg = np.zeros((cycle_size + 1, cycle_size), dtype=np.float64)
        rotations_c = np.zeros(cycle_size, dtype=np.float64)
        rotations_s = np.zeros(cycle_size, dtype=np.float64)
        transformed_rhs = np.zeros(cycle_size + 1, dtype=np.float64)
        transformed_rhs[0] = beta
        used = 0
        for column in range(cycle_size):
            work = apply_operator(apply_preconditioner(basis[column]))
            # Modified Gram-Schmidt with one reorthogonalization pass keeps the
            # basis stable while all full-plane vectors remain backend-native.
            for _ in range(2):
                for row in range(column + 1):
                    coefficient = scalar_float(
                        xp.vdot(basis[row].ravel(), work.ravel()).real
                    )
                    hessenberg[row, column] += coefficient
                    work -= coefficient * basis[row]
            hessenberg[column + 1, column] = scalar_float(
                xp.linalg.norm(work.ravel())
            )
            if hessenberg[column + 1, column] > np.finfo(float).eps:
                basis.append(work / hessenberg[column + 1, column])
            for row in range(column):
                upper = hessenberg[row, column]
                lower = hessenberg[row + 1, column]
                hessenberg[row, column] = (
                    rotations_c[row] * upper + rotations_s[row] * lower
                )
                hessenberg[row + 1, column] = (
                    -rotations_s[row] * upper + rotations_c[row] * lower
                )
            diagonal = hessenberg[column, column]
            subdiagonal = hessenberg[column + 1, column]
            magnitude = math.hypot(diagonal, subdiagonal)
            if magnitude == 0.0:
                rotations_c[column], rotations_s[column] = 1.0, 0.0
            else:
                rotations_c[column] = diagonal / magnitude
                rotations_s[column] = subdiagonal / magnitude
            hessenberg[column, column] = magnitude
            hessenberg[column + 1, column] = 0.0
            upper_rhs = transformed_rhs[column]
            transformed_rhs[column] = rotations_c[column] * upper_rhs
            transformed_rhs[column + 1] = -rotations_s[column] * upper_rhs
            total_iterations += 1
            used = column + 1
            relative_residual = abs(transformed_rhs[used]) / norm_rhs
            if abs(transformed_rhs[used]) <= tolerance:
                break
            if len(basis) <= column + 1:
                break
        if used == 0:
            return (
                solution,
                total_iterations,
                restart_count,
                relative_residual,
                False,
                "arnoldi_breakdown",
            )
        try:
            coefficients = np.linalg.solve(
                hessenberg[:used, :used], transformed_rhs[:used]
            )
        except np.linalg.LinAlgError:
            return (
                solution,
                total_iterations,
                restart_count,
                relative_residual,
                False,
                "singular_hessenberg",
            )
        combination = xp.zeros_like(rhs)
        for index, coefficient in enumerate(coefficients):
            combination += float(coefficient) * basis[index]
        solution += apply_preconditioner(combination)
        residual = rhs - apply_operator(solution)
        restart_count += 1
    final_norm = scalar_float(xp.linalg.norm(residual.ravel()))
    return (
        solution,
        total_iterations,
        restart_count,
        final_norm / norm_rhs,
        final_norm <= tolerance,
        "converged" if final_norm <= tolerance else "maximum_iterations",
    )


def _elliptic_apply(value, denominator, *, xp: Any = np):
    transformed = xp.fft.fft2(value)
    return xp.fft.ifft2(denominator * transformed).real.astype(
        value.dtype, copy=False
    )


def _jacobian_action(
    value,
    *,
    weight,
    denominator,
    null_mask,
    xp: Any = np,
):
    projected = xp.fft.fft2(value)
    projected[null_mask] = 0.0
    projected = xp.fft.ifft2(projected).real.astype(value.dtype, copy=False)
    accumulator_dtype = xp.float64 if value.dtype == np.dtype(np.float64) else xp.float32
    weighted_mean = xp.mean(
        weight.astype(accumulator_dtype, copy=False)
        * projected.astype(accumulator_dtype, copy=False)
    )
    result = (
        _elliptic_apply(projected, denominator, xp=xp)
        + weight * projected
        - weight * weighted_mean
    )
    transformed = xp.fft.fft2(result)
    transformed[null_mask] = 0.0
    return xp.fft.ifft2(transformed).real.astype(value.dtype, copy=False)


def _precondition(
    residual,
    *,
    denominator,
    null_mask,
    xp: Any = np,
):
    transformed = xp.fft.fft2(residual)
    transformed /= denominator + 1.0
    transformed[null_mask] = 0.0
    return xp.fft.ifft2(transformed).real.astype(residual.dtype, copy=False)


def _pcg(
    rhs,
    *,
    weight,
    denominator,
    null_mask,
    options: PRTransverseStaticMaterialSolverOptions,
    relative_tolerance: float,
    absolute_tolerance: float,
    cancellation_check: CancellationCheck | None = None,
    xp: Any = np,
) -> tuple[Any, int, bool, str, float, float]:
    """Solve the zero-flux Newton system with a true-residual PCG recurrence.

    Recomputing ``rhs - J(solution)`` after every solution update prevents a
    recursively accumulated float32 residual from drifting below the actual
    linear residual near the spectral/projection noise floor.  The directly
    evaluated residual controls convergence and every subsequent recurrence
    quantity on both NumPy and CuPy.
    """

    _raise_if_cancelled(cancellation_check, stage="material_pcg_iteration")
    solution = xp.zeros_like(rhs)
    residual = rhs.copy()
    norm_rhs = scalar_float(xp.linalg.norm(rhs.ravel()))
    tolerance = max(
        float(absolute_tolerance),
        float(relative_tolerance) * norm_rhs,
    )
    if norm_rhs <= tolerance:
        return solution, 0, True, "converged", tolerance, norm_rhs
    preconditioned = _precondition(
        residual, denominator=denominator, null_mask=null_mask, xp=xp
    )
    direction = preconditioned.copy()
    rz = scalar_float(xp.vdot(residual.ravel(), preconditioned.ravel()).real)
    if not math.isfinite(rz) or rz <= 0.0:
        return (
            solution,
            0,
            False,
            "nonpositive_preconditioned_residual",
            tolerance,
            norm_rhs,
        )
    for iteration in range(1, int(options.max_pcg_iterations) + 1):
        _raise_if_cancelled(cancellation_check, stage="material_pcg_iteration")
        image = _jacobian_action(
            direction,
            weight=weight,
            denominator=denominator,
            null_mask=null_mask,
            xp=xp,
        )
        _raise_if_cancelled(cancellation_check, stage="material_pcg_iteration")
        curvature = scalar_float(xp.vdot(direction.ravel(), image.ravel()).real)
        if not math.isfinite(curvature) or curvature <= 0.0:
            return (
                solution,
                iteration - 1,
                False,
                "nonpositive_curvature",
                tolerance,
                scalar_float(xp.linalg.norm(residual.ravel())),
            )
        alpha = rz / curvature
        solution += alpha * direction
        residual = rhs - _jacobian_action(
            solution,
            weight=weight,
            denominator=denominator,
            null_mask=null_mask,
            xp=xp,
        )
        residual_norm = scalar_float(xp.linalg.norm(residual.ravel()))
        if residual_norm <= tolerance:
            return (
                solution,
                iteration,
                True,
                "converged",
                tolerance,
                residual_norm,
            )
        next_preconditioned = _precondition(
            residual, denominator=denominator, null_mask=null_mask, xp=xp
        )
        next_rz = scalar_float(
            xp.vdot(residual.ravel(), next_preconditioned.ravel()).real
        )
        if not math.isfinite(next_rz) or next_rz <= 0.0:
            return (
                solution,
                iteration,
                False,
                "nonpositive_preconditioned_residual",
                tolerance,
                residual_norm,
            )
        direction = next_preconditioned + (next_rz / rz) * direction
        preconditioned = next_preconditioned
        rz = next_rz
    return (
        solution,
        int(options.max_pcg_iterations),
        False,
        "maximum_iterations",
        tolerance,
        scalar_float(xp.linalg.norm(residual.ravel())),
    )


def _resolved_pcg_forcing(
    equilibrium_rms: float,
    equilibrium_max: float,
    options: PRTransverseStaticMaterialSolverOptions,
) -> tuple[float, float, bool]:
    """Resolve the linear tolerance without oversolving early Newton steps.

    Once the zero-flux RMS gate is met but its pointwise maximum remains
    unresolved, the ordinary forcing can be too loose relative to the
    float32 spectral/projection floor.  Tightening is expressed only through
    configured tolerances and never changes the nonlinear physical gates.
    Float64 defaults are already tighter than the near-gate caps and are
    therefore unchanged.
    """

    relative = float(options.pcg_relative_tolerance)
    absolute = float(options.pcg_absolute_tolerance)
    tighten = (
        equilibrium_rms <= float(options.equilibrium_rms_tolerance)
        and equilibrium_max > float(options.equilibrium_max_tolerance)
    )
    if tighten:
        relative = min(
            relative, float(options.pcg_near_gate_relative_tolerance)
        )
        absolute = min(
            absolute, float(options.pcg_near_gate_absolute_tolerance)
        )
    return relative, absolute, tighten


def _criteria_met(
    equilibrium_rms: float,
    equilibrium_max: float,
    options: PRTransverseStaticMaterialSolverOptions,
) -> bool:
    return (
        equilibrium_rms <= float(options.equilibrium_rms_tolerance)
        and equilibrium_max <= float(options.equilibrium_max_tolerance)
    )


def _solve_plane(
    intensity,
    initial_psi,
    *,
    plane_index: int,
    dx_normalized: float,
    dy_normalized: float,
    h_y: float,
    options: PRTransverseStaticMaterialSolverOptions,
    spectral_operators=None,
    cancellation_check: CancellationCheck | None = None,
    xp: Any = np,
) -> tuple[
    Any,
    Any,
    Any,
    PRTransverseStaticPlaneSummary,
    list[PRTransverseStaticNewtonRecord],
]:
    _raise_if_cancelled(cancellation_check, stage="material_plane_boundary")
    if spectral_operators is None:
        spectral_operators = _symbols(
            intensity.shape,
            dx_normalized=dx_normalized,
            dy_normalized=dy_normalized,
            h_y=h_y,
            xp=xp,
        )
    denominator, null_mask = spectral_operators
    psi = _project_resolved_modes(
        initial_psi, null_mask=null_mask, xp=xp
    ).copy()
    initial_state = state_from_potential(
        psi,
        dx_normalized=dx_normalized,
        dy_normalized=dy_normalized,
        h_y=h_y,
        xp=xp,
    )
    if bool(asnumpy(xp.any(initial_state.carrier_density <= 0.0))):
        raise ValueError("initial_psi reconstructs nonpositive carrier density")
    records: list[PRTransverseStaticNewtonRecord] = []
    total_pcg = 0
    total_backtracks = 0
    status = "maximum_newton_iterations"

    for newton_iteration in range(int(options.max_newton_iterations) + 1):
        _raise_if_cancelled(
            cancellation_check, stage="material_newton_iteration"
        )
        residual = _static_equilibrium_residual_arrays(
            psi,
            intensity,
            null_mask=null_mask,
            dx_normalized=dx_normalized,
            dy_normalized=dy_normalized,
            h_y=h_y,
            xp=xp,
        )
        _raise_if_cancelled(
            cancellation_check, stage="material_residual_construction"
        )
        td_residual = potential_rhs(
            psi,
            intensity,
            dx_normalized=dx_normalized,
            dy_normalized=dy_normalized,
            h_y=h_y,
            applied_field_x=0.0,
            xp=xp,
        )
        resolved_residual = _project_resolved_modes(
            residual, null_mask=null_mask, xp=xp
        )
        equilibrium_rms, equilibrium_max = _metrics(resolved_residual, xp=xp)
        td_rhs_rms, td_rhs_max = _metrics(td_residual, xp=xp)
        if _criteria_met(
            equilibrium_rms,
            equilibrium_max,
            options,
        ):
            status = "residual_tolerance"
            break
        if newton_iteration == int(options.max_newton_iterations):
            break
        weight = _normalized_equilibrium_carrier(psi, intensity, xp=xp)
        rhs_hat = xp.fft.fft2(-resolved_residual)
        rhs_hat[null_mask] = 0.0
        rhs = xp.fft.ifft2(rhs_hat).real.astype(psi.dtype, copy=False)
        pcg_relative, pcg_absolute, pcg_forcing_tightened = (
            _resolved_pcg_forcing(equilibrium_rms, equilibrium_max, options)
        )
        (
            direction,
            pcg_iterations,
            pcg_converged,
            pcg_status,
            pcg_effective_tolerance,
            pcg_final_residual_norm,
        ) = _pcg(
            rhs,
            weight=weight,
            denominator=denominator,
            null_mask=null_mask,
            options=options,
            relative_tolerance=pcg_relative,
            absolute_tolerance=pcg_absolute,
            cancellation_check=cancellation_check,
            xp=xp,
        )
        total_pcg += pcg_iterations
        if not pcg_converged:
            status = f"pcg_{pcg_status}"
            records.append(PRTransverseStaticNewtonRecord(
                plane_index=plane_index,
                newton_iteration=newton_iteration + 1,
                equilibrium_rms_before=equilibrium_rms,
                equilibrium_max_before=equilibrium_max,
                equilibrium_rms_after=equilibrium_rms,
                equilibrium_max_after=equilibrium_max,
                pcg_iterations=pcg_iterations,
                pcg_converged=False,
                pcg_effective_tolerance=pcg_effective_tolerance,
                pcg_final_residual_norm=pcg_final_residual_norm,
                pcg_forcing_tightened=pcg_forcing_tightened,
                step_scale=0.0,
                backtracks=0,
                accepted=False,
            ))
            break
        step_scale = 1.0
        accepted = False
        after_rms = equilibrium_rms
        after_max = equilibrium_max
        backtracks = 0
        for backtracks in range(int(options.max_backtracks) + 1):
            _raise_if_cancelled(
                cancellation_check, stage="material_line_search"
            )
            trial = _project_resolved_modes(
                psi + step_scale * direction,
                null_mask=null_mask,
                xp=xp,
            )
            trial_state = state_from_potential(
                trial,
                dx_normalized=dx_normalized,
                dy_normalized=dy_normalized,
                h_y=h_y,
                xp=xp,
            )
            valid_trial = xp.all(
                xp.isfinite(trial) & (trial_state.carrier_density > 0.0)
            )
            if bool(asnumpy(valid_trial)):
                trial_residual = _static_equilibrium_residual_arrays(
                    trial,
                    intensity,
                    null_mask=null_mask,
                    dx_normalized=dx_normalized,
                    dy_normalized=dy_normalized,
                    h_y=h_y,
                    xp=xp,
                )
                _raise_if_cancelled(
                    cancellation_check, stage="material_line_search"
                )
                trial_resolved_residual = _project_resolved_modes(
                    trial_residual,
                    null_mask=null_mask,
                    xp=xp,
                )
                after_rms, after_max = _metrics(trial_resolved_residual, xp=xp)
                if (
                    math.isfinite(after_rms)
                    and after_rms
                    <= (1.0 - float(options.armijo_fraction) * step_scale)
                    * equilibrium_rms
                    and after_max <= equilibrium_max
                ):
                    accepted = True
                    psi = trial
                    break
            step_scale *= 0.5
            if step_scale < float(options.minimum_step_scale):
                break
        total_backtracks += backtracks
        records.append(PRTransverseStaticNewtonRecord(
            plane_index=plane_index,
            newton_iteration=newton_iteration + 1,
            equilibrium_rms_before=equilibrium_rms,
            equilibrium_max_before=equilibrium_max,
            equilibrium_rms_after=after_rms,
            equilibrium_max_after=after_max,
            pcg_iterations=pcg_iterations,
            pcg_converged=True,
            pcg_effective_tolerance=pcg_effective_tolerance,
            pcg_final_residual_norm=pcg_final_residual_norm,
            pcg_forcing_tightened=pcg_forcing_tightened,
            step_scale=step_scale if accepted else 0.0,
            backtracks=backtracks,
            accepted=accepted,
        ))
        if not accepted:
            status = "line_search_failed"
            break

    _raise_if_cancelled(cancellation_check, stage="material_final_residual")
    residual = _static_equilibrium_residual_arrays(
        psi,
        intensity,
        null_mask=null_mask,
        dx_normalized=dx_normalized,
        dy_normalized=dy_normalized,
        h_y=h_y,
        xp=xp,
    )
    td_residual = potential_rhs(
        psi,
        intensity,
        dx_normalized=dx_normalized,
        dy_normalized=dy_normalized,
        h_y=h_y,
        applied_field_x=0.0,
        xp=xp,
    )
    resolved_residual = _project_resolved_modes(
        residual, null_mask=null_mask, xp=xp
    )
    equilibrium_rms, equilibrium_max = _metrics(resolved_residual, xp=xp)
    td_rhs_rms, td_rhs_max = _metrics(td_residual, xp=xp)
    null_rms, null_max = _metrics(
        _derivative_null_residual_array(
            residual, null_mask=null_mask, xp=xp
        ),
        xp=xp,
    )
    state = state_from_potential(
        psi,
        dx_normalized=dx_normalized,
        dy_normalized=dy_normalized,
        h_y=h_y,
        xp=xp,
    )
    (
        physical_state_valid,
        carrier_mean,
        carrier_minimum,
        potential_mean,
    ) = _physical_state_validity(
        state, xp=xp
    )
    converged = _criteria_met(
        equilibrium_rms,
        equilibrium_max,
        options,
    ) and physical_state_valid
    if converged:
        status = "residual_tolerance"
    elif _criteria_met(equilibrium_rms, equilibrium_max, options):
        status = "physical_state_invalid"
    summary = PRTransverseStaticPlaneSummary(
        plane_index=plane_index,
        converged=converged,
        status=status,
        newton_iterations=sum(record.accepted for record in records),
        pcg_iterations=total_pcg,
        backtracks=total_backtracks,
        equilibrium_rms=equilibrium_rms,
        equilibrium_max=equilibrium_max,
        td_rhs_rms=td_rhs_rms,
        td_rhs_max=td_rhs_max,
        null_residual_rms=null_rms,
        null_residual_max=null_max,
        carrier_mean=carrier_mean,
        carrier_minimum=carrier_minimum,
        potential_mean=potential_mean,
        physical_state_valid=physical_state_valid,
    )
    return psi, residual, td_residual, summary, records


def _solve_continuum_static_intensity(
    intensity,
    *,
    dx_normalized: float,
    dy_normalized: float,
    initial_psi=None,
    h_y: float = 1.0,
    options: PRTransverseStaticMaterialSolverOptions | None = None,
    xp: Any = np,
) -> PRTransverseStaticMaterialResult:
    started = perf_counter()
    driving = _validate_real_array(intensity, name="intensity", xp=xp)
    if xp is np and driving.dtype != np.dtype(np.float64):
        raise TypeError("NumPy static reference solver requires float64 intensity")
    if bool(asnumpy(xp.any(driving <= 0.0))):
        raise ValueError("static transport intensity must be strictly positive")
    if not math.isfinite(float(dx_normalized)) or dx_normalized <= 0.0:
        raise ValueError("dx_normalized must be finite and positive")
    if not math.isfinite(float(dy_normalized)) or dy_normalized <= 0.0:
        raise ValueError("dy_normalized must be finite and positive")
    if not math.isfinite(float(h_y)) or float(h_y) != 1.0:
        raise ValueError("Profile v1 static reference requires h_y=1")
    resolved_options = options or PRTransverseStaticMaterialSolverOptions()
    resolved_options.validate()
    if initial_psi is None:
        initial = xp.zeros_like(driving)
    else:
        initial = _validate_real_array(initial_psi, name="initial_psi", xp=xp)
        if initial.shape != driving.shape:
            raise ValueError("initial_psi and intensity must have identical shapes")
        initial = initial.astype(driving.dtype, copy=True)

    was_plane = driving.ndim == 2
    driving_volume = driving[None, ...] if was_plane else driving
    initial_volume = initial[None, ...] if was_plane else initial
    psi_volume = xp.empty_like(driving_volume)
    residual_volume = xp.empty_like(driving_volume)
    td_volume = xp.empty_like(driving_volume)
    summaries: list[PRTransverseStaticPlaneSummary] = []
    records: list[PRTransverseStaticNewtonRecord] = []
    spectral_operators = _symbols(
        driving_volume.shape[-2:],
        dx_normalized=dx_normalized,
        dy_normalized=dy_normalized,
        h_y=h_y,
        xp=xp,
    )
    for plane_index in range(driving_volume.shape[0]):
        psi_plane, residual_plane, td_plane, summary, plane_records = _solve_plane(
            driving_volume[plane_index],
            initial_volume[plane_index],
            plane_index=plane_index,
            dx_normalized=dx_normalized,
            dy_normalized=dy_normalized,
            h_y=h_y,
            options=resolved_options,
            spectral_operators=spectral_operators,
            xp=xp,
        )
        psi_volume[plane_index] = psi_plane
        residual_volume[plane_index] = residual_plane
        td_volume[plane_index] = td_plane
        summaries.append(summary)
        records.extend(plane_records)
    converged = all(summary.converged for summary in summaries)
    status = "converged" if converged else next(
        summary.status for summary in summaries if not summary.converged
    )
    return PRTransverseStaticMaterialResult(
        psi=psi_volume[0] if was_plane else psi_volume,
        equilibrium_residual=(
            residual_volume[0] if was_plane else residual_volume
        ),
        td_rhs_residual=td_volume[0] if was_plane else td_volume,
        converged=converged,
        status=status,
        plane_summaries=tuple(summaries),
        iteration_records=tuple(records),
        elapsed_seconds=perf_counter() - started,
    )


def solve_pr_transverse_static_intensity(
    intensity,
    *,
    dx_normalized: float,
    dy_normalized: float,
    initial_psi=None,
    h_y: float = 1.0,
    options: PRTransverseStaticMaterialSolverOptions | None = None,
    xp: Any = np,
) -> PRTransverseStaticMaterialResult:
    """Solve authoritative periodic zero-flux frozen-intensity equilibria.

    The production-TD residual is returned as a discretization diagnostic but
    is not an acceptance condition for the physical static solution.
    """

    return _solve_continuum_static_intensity(
        intensity,
        dx_normalized=dx_normalized,
        dy_normalized=dy_normalized,
        initial_psi=initial_psi,
        h_y=h_y,
        options=options,
        xp=xp,
    )


def _solve_pr_transverse_static_intensity_host_volume(
    intensity,
    *,
    dx_normalized: float,
    dy_normalized: float,
    initial_psi,
    h_y: float = 1.0,
    options: PRTransverseStaticMaterialSolverOptions | None = None,
    cancellation_check: CancellationCheck | None = None,
    xp: Any = np,
) -> PRTransverseStaticMaterialResult:
    """Solve a retained host volume with only one backend plane live.

    The plane solve and its numerical ordering are exactly those used by
    :func:`solve_pr_transverse_static_intensity`.  This storage adapter exists
    for the coupled static workflow, whose longitudinal planes are independent
    for a frozen optical source.  It prevents the retained ``Nz`` volume from
    becoming a backend FFT batch while preserving per-plane diagnostics and
    indices.
    """

    started = perf_counter()
    driving_volume = np.asarray(intensity)
    initial_volume = np.asarray(initial_psi)
    if driving_volume.ndim != 3 or min(driving_volume.shape[-2:]) < 3:
        raise ValueError("intensity must have shape (Nz, Nx, Ny)")
    if initial_volume.shape != driving_volume.shape:
        raise ValueError("initial_psi and intensity must have identical shapes")
    if driving_volume.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError("intensity must have dtype float32 or float64")
    if initial_volume.dtype != driving_volume.dtype:
        raise TypeError("initial_psi and intensity must have identical dtypes")
    for driving_plane, initial_plane in zip(driving_volume, initial_volume):
        if not np.all(np.isfinite(driving_plane)):
            raise ValueError("intensity must contain only finite values")
        if not np.all(np.isfinite(initial_plane)):
            raise ValueError("initial_psi must contain only finite values")
        if np.any(driving_plane <= 0.0):
            raise ValueError("static transport intensity must be strictly positive")
    if xp is np and driving_volume.dtype != np.dtype(np.float64):
        raise TypeError("NumPy static reference solver requires float64 intensity")
    if not math.isfinite(float(dx_normalized)) or dx_normalized <= 0.0:
        raise ValueError("dx_normalized must be finite and positive")
    if not math.isfinite(float(dy_normalized)) or dy_normalized <= 0.0:
        raise ValueError("dy_normalized must be finite and positive")
    if not math.isfinite(float(h_y)) or float(h_y) != 1.0:
        raise ValueError("Profile v1 static reference requires h_y=1")

    resolved_options = options or PRTransverseStaticMaterialSolverOptions()
    resolved_options.validate()
    psi_volume = np.empty_like(driving_volume)
    residual_volume = np.empty_like(driving_volume)
    td_volume = np.empty_like(driving_volume)
    summaries: list[PRTransverseStaticPlaneSummary] = []
    records: list[PRTransverseStaticNewtonRecord] = []
    spectral_operators = _symbols(
        driving_volume.shape[-2:],
        dx_normalized=dx_normalized,
        dy_normalized=dy_normalized,
        h_y=h_y,
        xp=xp,
    )
    for plane_index in range(driving_volume.shape[0]):
        _raise_if_cancelled(cancellation_check, stage="material_plane_boundary")
        driving_plane = xp.asarray(driving_volume[plane_index])
        initial_plane = xp.asarray(initial_volume[plane_index])
        psi_plane, residual_plane, td_plane, summary, plane_records = _solve_plane(
            driving_plane,
            initial_plane,
            plane_index=plane_index,
            dx_normalized=dx_normalized,
            dy_normalized=dy_normalized,
            h_y=h_y,
            options=resolved_options,
            spectral_operators=spectral_operators,
            cancellation_check=cancellation_check,
            xp=xp,
        )
        psi_volume[plane_index] = np.asarray(asnumpy(psi_plane))
        residual_volume[plane_index] = np.asarray(asnumpy(residual_plane))
        td_volume[plane_index] = np.asarray(asnumpy(td_plane))
        summaries.append(summary)
        records.extend(plane_records)
    converged = all(summary.converged for summary in summaries)
    status = "converged" if converged else next(
        summary.status for summary in summaries if not summary.converged
    )
    return PRTransverseStaticMaterialResult(
        psi=psi_volume,
        equilibrium_residual=residual_volume,
        td_rhs_residual=td_volume,
        converged=converged,
        status=status,
        plane_summaries=tuple(summaries),
        iteration_records=tuple(records),
        elapsed_seconds=perf_counter() - started,
    )


def _discrete_criteria_met(
    residual_rms: float,
    residual_max: float,
    options: PRTransverseStaticMaterialSolverOptions,
) -> bool:
    return (
        residual_rms <= float(options.td_rhs_rms_tolerance)
        and residual_max <= float(options.td_rhs_max_tolerance)
    )


def _solve_discrete_plane(
    intensity,
    initial_psi,
    *,
    plane_index: int,
    dx_normalized: float,
    dy_normalized: float,
    h_y: float,
    material_options: PRTransverseStaticMaterialSolverOptions,
    corrector_options: PRTransverseDiscreteStaticCorrectorOptions,
    xp: Any,
) -> tuple[
    Any,
    Any,
    PRTransverseDiscreteStaticPlaneSummary,
    list[PRTransverseDiscreteStaticNewtonRecord],
]:
    kx, ky = spectral_wavevectors(
        intensity.shape,
        dx_normalized=dx_normalized,
        dy_normalized=dy_normalized,
        xp=xp,
    )
    denominator = kx * kx + float(h_y) * ky * ky
    null_mask = denominator == 0.0
    psi = project_production_resolved_modes(
        initial_psi,
        dx_normalized=dx_normalized,
        dy_normalized=dy_normalized,
        h_y=h_y,
        xp=xp,
    ).copy()
    residual = production_steady_residual(
        psi,
        intensity,
        dx_normalized=dx_normalized,
        dy_normalized=dy_normalized,
        h_y=h_y,
        xp=xp,
    )
    initial_rms, initial_max = _metrics(residual, xp=xp)
    residual_rms, residual_max = initial_rms, initial_max
    records: list[PRTransverseDiscreteStaticNewtonRecord] = []
    total_gmres = 0
    total_restarts = 0
    total_backtracks = 0
    status = "maximum_discrete_newton_iterations"
    relative_tolerance = (
        float(corrector_options.gmres_relative_tolerance_float64)
        if psi.dtype == np.dtype(np.float64)
        else float(corrector_options.gmres_relative_tolerance_float32)
    )
    absolute_tolerance = (
        0.1
        * math.sqrt(float(psi.size))
        * float(material_options.td_rhs_rms_tolerance)
    )

    for newton_iteration in range(
        1, int(corrector_options.max_newton_iterations) + 1
    ):
        if _discrete_criteria_met(residual_rms, residual_max, material_options):
            status = "residual_tolerance"
            break
        before_rms, before_max = residual_rms, residual_max
        state = state_from_potential(
            psi,
            dx_normalized=dx_normalized,
            dy_normalized=dy_normalized,
            h_y=h_y,
            applied_field_x=0.0,
            xp=xp,
        )

        def apply_operator(value):
            return _production_steady_jvp_from_state(
                value,
                intensity=intensity,
                state=state,
                kx=kx,
                ky=ky,
                denominator=denominator,
                null_mask=null_mask,
                m_y=1.0,
                xp=xp,
            )

        def apply_preconditioner(value):
            return _discrete_precondition(
                value, denominator=denominator, null_mask=null_mask, xp=xp
            )

        (
            direction,
            gmres_iterations,
            gmres_restarts,
            gmres_relative_residual,
            gmres_converged,
            gmres_status,
        ) = _right_preconditioned_gmres(
            -residual,
            apply_operator=apply_operator,
            apply_preconditioner=apply_preconditioner,
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
            restart=int(corrector_options.gmres_restart),
            max_iterations=int(corrector_options.max_gmres_iterations),
            xp=xp,
        )
        total_gmres += gmres_iterations
        total_restarts += gmres_restarts
        if not gmres_converged:
            status = "discrete_gmres_failed"
            records.append(PRTransverseDiscreteStaticNewtonRecord(
                plane_index=plane_index,
                newton_iteration=newton_iteration,
                residual_rms_before=residual_rms,
                residual_max_before=residual_max,
                residual_rms_after=residual_rms,
                residual_max_after=residual_max,
                gmres_iterations=gmres_iterations,
                gmres_restarts=gmres_restarts,
                gmres_relative_residual=gmres_relative_residual,
                gmres_converged=False,
                gmres_status=gmres_status,
                step_scale=0.0,
                backtracks=0,
                accepted=False,
            ))
            break
        direction = project_production_resolved_modes(
            direction,
            dx_normalized=dx_normalized,
            dy_normalized=dy_normalized,
            h_y=h_y,
            xp=xp,
        )
        step_scale = 1.0
        accepted = False
        after_rms, after_max = residual_rms, residual_max
        backtracks = 0
        for backtracks in range(int(corrector_options.max_backtracks) + 1):
            trial = project_production_resolved_modes(
                psi + step_scale * direction,
                dx_normalized=dx_normalized,
                dy_normalized=dy_normalized,
                h_y=h_y,
                xp=xp,
            )
            trial_state = state_from_potential(
                trial,
                dx_normalized=dx_normalized,
                dy_normalized=dy_normalized,
                h_y=h_y,
                xp=xp,
            )
            valid = xp.all(
                xp.isfinite(trial) & (trial_state.carrier_density > 0.0)
            )
            if bool(asnumpy(valid)):
                trial_residual = production_steady_residual(
                    trial,
                    intensity,
                    dx_normalized=dx_normalized,
                    dy_normalized=dy_normalized,
                    h_y=h_y,
                    xp=xp,
                )
                after_rms, after_max = _metrics(trial_residual, xp=xp)
                if (
                    math.isfinite(after_rms)
                    and math.isfinite(after_max)
                    and after_rms
                    <= (
                        1.0
                        - float(corrector_options.armijo_fraction) * step_scale
                    )
                    * residual_rms
                    and after_max <= residual_max
                ):
                    psi = trial
                    residual = trial_residual
                    residual_rms, residual_max = after_rms, after_max
                    accepted = True
                    break
            step_scale *= 0.5
            if step_scale < float(corrector_options.minimum_step_scale):
                break
        total_backtracks += backtracks
        records.append(PRTransverseDiscreteStaticNewtonRecord(
            plane_index=plane_index,
            newton_iteration=newton_iteration,
            residual_rms_before=before_rms,
            residual_max_before=before_max,
            residual_rms_after=after_rms,
            residual_max_after=after_max,
            gmres_iterations=gmres_iterations,
            gmres_restarts=gmres_restarts,
            gmres_relative_residual=gmres_relative_residual,
            gmres_converged=True,
            gmres_status=gmres_status,
            step_scale=step_scale if accepted else 0.0,
            backtracks=backtracks,
            accepted=accepted,
        ))
        if not accepted:
            status = "discrete_line_search_failed"
            break

    converged = _discrete_criteria_met(
        residual_rms, residual_max, material_options
    )
    if converged:
        status = "residual_tolerance"
    state = state_from_potential(
        psi,
        dx_normalized=dx_normalized,
        dy_normalized=dy_normalized,
        h_y=h_y,
        xp=xp,
    )
    summary = PRTransverseDiscreteStaticPlaneSummary(
        plane_index=plane_index,
        converged=converged,
        status=status,
        initial_residual_rms=initial_rms,
        initial_residual_max=initial_max,
        final_residual_rms=residual_rms,
        final_residual_max=residual_max,
        newton_iterations=sum(record.accepted for record in records),
        gmres_iterations=total_gmres,
        gmres_restarts=total_restarts,
        backtracks=total_backtracks,
        carrier_mean=scalar_float(xp.mean(state.carrier_density)),
        carrier_minimum=scalar_float(xp.min(state.carrier_density)),
        potential_mean=scalar_float(xp.mean(state.psi)),
    )
    return psi, residual, summary, records


def solve_pr_transverse_discrete_static_intensity(
    intensity,
    *,
    dx_normalized: float,
    dy_normalized: float,
    initial_psi=None,
    h_y: float = 1.0,
    options: PRTransverseStaticMaterialSolverOptions | None = None,
    corrector_options: PRTransverseDiscreteStaticCorrectorOptions | None = None,
    xp: Any = np,
) -> PRTransverseDiscreteStaticMaterialResult:
    """Experimentally solve continuum initialization then TD-discrete balance.

    This explicit entry point is retained for finite-grid TD/static
    discretization studies.  It is not the canonical static material solver.
    """

    started = perf_counter()
    material_options = options or PRTransverseStaticMaterialSolverOptions()
    material_options.validate()
    resolved_corrector = (
        corrector_options or PRTransverseDiscreteStaticCorrectorOptions()
    )
    resolved_corrector.validate()
    continuum = _solve_continuum_static_intensity(
        intensity,
        dx_normalized=dx_normalized,
        dy_normalized=dy_normalized,
        initial_psi=initial_psi,
        h_y=h_y,
        options=material_options,
        xp=xp,
    )
    continuum_elapsed = continuum.elapsed_seconds
    if not continuum.converged:
        return PRTransverseDiscreteStaticMaterialResult(
            psi=continuum.psi,
            equilibrium_residual=continuum.equilibrium_residual,
            td_rhs_residual=continuum.td_rhs_residual,
            converged=False,
            status="continuum_initializer_failed",
            continuum_result=continuum,
            plane_summaries=(),
            iteration_records=(),
            continuum_elapsed_seconds=continuum_elapsed,
            discrete_elapsed_seconds=0.0,
            elapsed_seconds=perf_counter() - started,
        )
    driving = _validate_real_array(intensity, name="intensity", xp=xp)
    was_plane = driving.ndim == 2
    driving_volume = driving[None, ...] if was_plane else driving
    continuum_volume = continuum.psi[None, ...] if was_plane else continuum.psi
    psi_volume = xp.empty_like(continuum_volume)
    residual_volume = xp.empty_like(continuum_volume)
    summaries: list[PRTransverseDiscreteStaticPlaneSummary] = []
    records: list[PRTransverseDiscreteStaticNewtonRecord] = []
    discrete_started = perf_counter()
    for plane_index in range(driving_volume.shape[0]):
        psi_plane, residual_plane, summary, plane_records = _solve_discrete_plane(
            driving_volume[plane_index],
            continuum_volume[plane_index],
            plane_index=plane_index,
            dx_normalized=dx_normalized,
            dy_normalized=dy_normalized,
            h_y=h_y,
            material_options=material_options,
            corrector_options=resolved_corrector,
            xp=xp,
        )
        psi_volume[plane_index] = psi_plane
        residual_volume[plane_index] = residual_plane
        summaries.append(summary)
        records.extend(plane_records)
    discrete_elapsed = perf_counter() - discrete_started
    psi_result = psi_volume[0] if was_plane else psi_volume
    td_result = residual_volume[0] if was_plane else residual_volume
    equilibrium_result = static_equilibrium_residual(
        psi_result,
        driving,
        dx_normalized=dx_normalized,
        dy_normalized=dy_normalized,
        h_y=h_y,
        xp=xp,
    )
    converged = all(summary.converged for summary in summaries)
    status = "converged" if converged else next(
        summary.status for summary in summaries if not summary.converged
    )
    return PRTransverseDiscreteStaticMaterialResult(
        psi=psi_result,
        equilibrium_residual=equilibrium_result,
        td_rhs_residual=td_result,
        converged=converged,
        status=status,
        continuum_result=continuum,
        plane_summaries=tuple(summaries),
        iteration_records=tuple(records),
        continuum_elapsed_seconds=continuum_elapsed,
        discrete_elapsed_seconds=discrete_elapsed,
        elapsed_seconds=perf_counter() - started,
    )


__all__ = [
    "PRTransverseDiscreteStaticCorrectorOptions",
    "PRTransverseDiscreteStaticMaterialResult",
    "PRTransverseDiscreteStaticNewtonRecord",
    "PRTransverseDiscreteStaticPlaneSummary",
    "PRTransverseStaticMaterialResult",
    "PRTransverseStaticMaterialSolverOptions",
    "PRTransverseStaticNewtonRecord",
    "PRTransverseStaticPlaneSummary",
    "derivative_null_residual",
    "project_production_resolved_modes",
    "production_steady_jvp",
    "production_steady_residual",
    "solve_pr_transverse_discrete_static_intensity",
    "solve_pr_transverse_static_intensity",
    "static_equilibrium_residual",
]
