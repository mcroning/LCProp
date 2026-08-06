"""Strict fixed-intensity photorefractive material solve."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable

import numpy as np

from lcprop.pr.cyclic import solve_cyclic_tridiagonal_rows
from lcprop.pr.evolution import hopping_rhs, periodic_derivatives_x


@dataclass(frozen=True)
class PRStaticSolverOptions:
    """Controls for the CPU reference damped-Newton solve."""

    max_iterations: int = 40
    residual_rms_tolerance: float = 1e-10
    residual_max_tolerance: float = 1e-9
    max_backtracks: int = 20
    minimum_step_scale: float = 2.0**-20
    armijo_fraction: float = 1e-4

    def validate(self) -> None:
        if int(self.max_iterations) < 0:
            raise ValueError("max_iterations must be nonnegative")
        if int(self.max_backtracks) < 0:
            raise ValueError("max_backtracks must be nonnegative")
        for name in (
            "residual_rms_tolerance",
            "residual_max_tolerance",
            "minimum_step_scale",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        fraction = float(self.armijo_fraction)
        if not math.isfinite(fraction) or not 0.0 < fraction < 1.0:
            raise ValueError("armijo_fraction must be finite and between zero and one")


@dataclass(frozen=True)
class PRStaticIterationRecord:
    """One accepted residual observation in the Newton solve."""

    iteration: int
    residual_rms: float
    residual_max: float
    step_scale: float


@dataclass(frozen=True)
class PRStaticResult:
    """Strict fixed-intensity root and convergence evidence."""

    E: Any
    converged: bool
    status: str
    iterations: int
    residual_rms: float
    residual_max: float
    records: tuple[PRStaticIterationRecord, ...]
    message: str = ""


def _validated_inputs(
    intensity,
    initial_E,
    *,
    applied_field: float,
    background_intensity: float,
    dx_normalized: float,
    options: PRStaticSolverOptions,
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    options.validate()
    supplied_intensity = np.asarray(intensity)
    intensity_array = supplied_intensity.astype(np.float64, copy=True)
    if intensity_array.ndim not in (2, 3):
        raise ValueError(
            "intensity must have shape (Nx, Ny) or (Nz, Nx, Ny)"
        )
    if int(intensity_array.shape[-2]) < 3:
        raise ValueError("fixed-intensity PR solve requires Nx >= 3")
    if supplied_intensity.dtype.kind != "f":
        raise TypeError("intensity must have a real floating-point dtype")
    if not np.all(np.isfinite(intensity_array)):
        raise ValueError("intensity must contain only finite values")
    if np.any(intensity_array < 0.0):
        raise ValueError("intensity must be nonnegative")

    if initial_E is None:
        state = np.zeros_like(intensity_array)
    else:
        supplied = np.asarray(initial_E)
        if supplied.shape != intensity_array.shape:
            raise ValueError("initial_E must have the same shape as intensity")
        if supplied.dtype.kind != "f":
            raise TypeError("initial_E must have a real floating-point dtype")
        if not np.all(np.isfinite(supplied)):
            raise ValueError("initial_E must contain only finite values")
        state = supplied.astype(np.float64, copy=True)

    applied = float(applied_field)
    background = float(background_intensity)
    dx = float(dx_normalized)
    if not math.isfinite(applied):
        raise ValueError("applied_field must be finite")
    if not math.isfinite(background) or background < 0.0:
        raise ValueError("background_intensity must be finite and nonnegative")
    if not math.isfinite(dx) or dx <= 0.0:
        raise ValueError("dx_normalized must be finite and positive")
    return intensity_array, state, applied, background, dx


def _array_has_true(value, *, xp: Any) -> bool:
    reduced = xp.any(value)
    return bool(reduced.item() if hasattr(reduced, "item") else reduced)


def _validated_batched_inputs(
    intensity,
    initial_E,
    *,
    applied_field: float,
    background_intensity: float,
    dx_normalized: float,
    options: PRStaticSolverOptions,
    xp: Any,
):
    options.validate()
    intensity_array = xp.asarray(intensity)
    if intensity_array.ndim not in (2, 3):
        raise ValueError(
            "intensity must have shape (Nx, Ny) or (Nz, Nx, Ny)"
        )
    if int(intensity_array.shape[-2]) < 3:
        raise ValueError("fixed-intensity PR solve requires Nx >= 3")
    if getattr(intensity_array.dtype, "kind", None) != "f":
        raise TypeError("intensity must have a real floating-point dtype")
    if _array_has_true(~xp.isfinite(intensity_array), xp=xp):
        raise ValueError("intensity must contain only finite values")
    if _array_has_true(intensity_array < 0.0, xp=xp):
        raise ValueError("intensity must be nonnegative")
    intensity_array = intensity_array.copy()

    if initial_E is None:
        state = xp.zeros_like(intensity_array)
    else:
        state = xp.asarray(initial_E)
        if state.shape != intensity_array.shape:
            raise ValueError("initial_E must have the same shape as intensity")
        if getattr(state.dtype, "kind", None) != "f":
            raise TypeError("initial_E must have a real floating-point dtype")
        if state.dtype != intensity_array.dtype:
            raise TypeError("initial_E and intensity must have the same dtype")
        if _array_has_true(~xp.isfinite(state), xp=xp):
            raise ValueError("initial_E must contain only finite values")
        state = state.copy()

    applied = float(applied_field)
    background = float(background_intensity)
    dx = float(dx_normalized)
    if not math.isfinite(applied):
        raise ValueError("applied_field must be finite")
    if not math.isfinite(background) or background < 0.0:
        raise ValueError("background_intensity must be finite and nonnegative")
    if not math.isfinite(dx) or dx <= 0.0:
        raise ValueError("dx_normalized must be finite and positive")
    return intensity_array, state, applied, background, dx


def _fixed_intensity_jacobian_rows_backend(
    state,
    fixed_intensity,
    *,
    dx_normalized: float,
    xp: Any,
):
    E_x, _ = periodic_derivatives_x(
        state,
        dx_normalized=dx_normalized,
        xp=xp,
    )
    I_x, _ = periodic_derivatives_x(
        fixed_intensity,
        dx_normalized=dx_normalized,
        xp=xp,
    )
    charge_flux = state * fixed_intensity - I_x
    inverse_dx_squared = 1.0 / (dx_normalized * dx_normalized)
    lower = (
        fixed_intensity * inverse_dx_squared
        + charge_flux / (2.0 * dx_normalized)
    )
    diagonal = (
        -fixed_intensity * (1.0 + E_x)
        - 2.0 * fixed_intensity * inverse_dx_squared
    )
    upper = (
        fixed_intensity * inverse_dx_squared
        - charge_flux / (2.0 * dx_normalized)
    )
    return lower, diagonal, upper


def fixed_intensity_jacobian_rows(
    E,
    intensity,
    *,
    dx_normalized: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return periodic ``(lower, diagonal, upper)`` rows of ``dR/dE``.

    The returned arrays have the same shape as ``E``. At x index ``j``, the
    entries multiply perturbations at ``j-1``, ``j``, and ``j+1`` with
    periodic indexing. ``intensity`` is prescribed and is not differentiated
    with respect to ``E``.
    """

    state = np.asarray(E)
    fixed_intensity = np.asarray(intensity)
    if state.shape != fixed_intensity.shape:
        raise ValueError("E and intensity must have identical shapes")
    if state.ndim not in (2, 3):
        raise ValueError("E and intensity must include x and y axes")
    dx = float(dx_normalized)
    if not math.isfinite(dx) or dx <= 0.0:
        raise ValueError("dx_normalized must be finite and positive")

    return _fixed_intensity_jacobian_rows_backend(
        state,
        fixed_intensity,
        dx_normalized=dx,
        xp=np,
    )


def _newton_direction(
    residual,
    lower,
    diagonal,
    upper,
) -> np.ndarray:
    """Solve the independent periodic Jacobian systems by dense reference."""

    Nx = int(residual.shape[-2])
    residual_rows = np.moveaxis(residual, -2, -1).reshape((-1, Nx))
    lower_rows = np.moveaxis(lower, -2, -1).reshape((-1, Nx))
    diagonal_rows = np.moveaxis(diagonal, -2, -1).reshape((-1, Nx))
    upper_rows = np.moveaxis(upper, -2, -1).reshape((-1, Nx))
    direction_rows = np.empty_like(residual_rows)

    indices = np.arange(Nx)
    for row_index in range(residual_rows.shape[0]):
        jacobian = np.zeros((Nx, Nx), dtype=residual.dtype)
        jacobian[indices, indices] = diagonal_rows[row_index]
        jacobian[indices, (indices - 1) % Nx] = lower_rows[row_index]
        jacobian[indices, (indices + 1) % Nx] = upper_rows[row_index]
        direction_rows[row_index] = np.linalg.solve(
            jacobian,
            -residual_rows[row_index],
        )

    moved_shape = np.moveaxis(residual, -2, -1).shape
    return np.moveaxis(direction_rows.reshape(moved_shape), -1, -2)


def batched_newton_direction(
    residual,
    lower,
    diagonal,
    upper,
    *,
    xp: Any,
):
    """Solve the periodic PR Newton systems with the batched O(Nx) kernel."""

    return solve_cyclic_tridiagonal_rows(
        lower,
        diagonal,
        upper,
        -residual,
        xp=xp,
    )


def _backend_scalar(value) -> float:
    return float(value.item() if hasattr(value, "item") else value)


def _residual_metrics(residual, *, xp: Any) -> tuple[float, float]:
    return (
        _backend_scalar(xp.sqrt(xp.mean(residual * residual))),
        _backend_scalar(xp.max(xp.abs(residual))),
    )


def _solve_pr_static_intensity_backend(
    intensity,
    *,
    applied_field: float,
    background_intensity: float,
    dx_normalized: float,
    initial_E: Any | None = None,
    options: PRStaticSolverOptions = PRStaticSolverOptions(),
    direction_solver: Callable[[Any, Any, Any, Any], Any],
    input_validator: Callable[..., Any],
    jacobian_builder: Callable[[Any, Any], Any],
    xp: Any,
) -> PRStaticResult:
    """Run the shared Newton loop with selected backend linear algebra."""

    intensity_array, state, applied, background, dx = input_validator(
        intensity,
        initial_E,
        applied_field=applied_field,
        background_intensity=background_intensity,
        dx_normalized=dx_normalized,
        options=options,
    )
    records: list[PRStaticIterationRecord] = []

    def residual_at(candidate):
        return hopping_rhs(
            candidate,
            intensity_array,
            applied_field=applied,
            background_intensity=background,
            dx_normalized=dx,
            xp=xp,
        )

    residual = residual_at(state)
    residual_rms, residual_max = _residual_metrics(residual, xp=xp)
    records.append(
        PRStaticIterationRecord(
            iteration=0,
            residual_rms=residual_rms,
            residual_max=residual_max,
            step_scale=0.0,
        )
    )

    for iteration in range(int(options.max_iterations) + 1):
        if (
            residual_rms <= float(options.residual_rms_tolerance)
            and residual_max <= float(options.residual_max_tolerance)
        ):
            return PRStaticResult(
                E=state.copy(),
                converged=True,
                status="converged",
                iterations=iteration,
                residual_rms=residual_rms,
                residual_max=residual_max,
                records=tuple(records),
                message="full discrete PR residual converged",
            )
        if iteration == int(options.max_iterations):
            break

        lower, diagonal, upper = jacobian_builder(
            state,
            intensity_array,
        )
        try:
            direction = direction_solver(
                residual,
                lower,
                diagonal,
                upper,
            )
        except np.linalg.LinAlgError:
            return PRStaticResult(
                E=state.copy(),
                converged=False,
                status="singular_jacobian",
                iterations=iteration,
                residual_rms=residual_rms,
                residual_max=residual_max,
                records=tuple(records),
                message="fixed-intensity PR Jacobian is singular",
            )

        step_scale = 1.0
        accepted = False
        for _ in range(int(options.max_backtracks) + 1):
            candidate = state + step_scale * direction
            candidate_residual = residual_at(candidate)
            candidate_rms, candidate_max = _residual_metrics(
                candidate_residual,
                xp=xp,
            )
            required_rms = (
                1.0 - float(options.armijo_fraction) * step_scale
            ) * residual_rms
            if (
                not _array_has_true(~xp.isfinite(candidate), xp=xp)
                and math.isfinite(candidate_rms)
                and candidate_rms <= required_rms
            ):
                accepted = True
                break
            step_scale *= 0.5
            if step_scale < float(options.minimum_step_scale):
                break

        if not accepted:
            return PRStaticResult(
                E=state.copy(),
                converged=False,
                status="line_search_failed",
                iterations=iteration,
                residual_rms=residual_rms,
                residual_max=residual_max,
                records=tuple(records),
                message="damped Newton step did not reduce residual RMS",
            )

        state = candidate
        residual = candidate_residual
        residual_rms = candidate_rms
        residual_max = candidate_max
        records.append(
            PRStaticIterationRecord(
                iteration=iteration + 1,
                residual_rms=residual_rms,
                residual_max=residual_max,
                step_scale=step_scale,
            )
        )

    return PRStaticResult(
        E=state.copy(),
        converged=False,
        status="max_iterations",
        iterations=int(options.max_iterations),
        residual_rms=residual_rms,
        residual_max=residual_max,
        records=tuple(records),
        message="maximum Newton iterations reached before residual convergence",
    )


def solve_pr_static_intensity(
    intensity,
    *,
    applied_field: float,
    background_intensity: float,
    dx_normalized: float,
    initial_E: Any | None = None,
    options: PRStaticSolverOptions = PRStaticSolverOptions(),
) -> PRStaticResult:
    """Solve ``R(E; I)=0`` with the dense CPU NumPy reference direction.

    The complete centered-difference residual, damping, convergence criteria,
    and result semantics are shared with the structured implementation. This
    dense path remains the small-system validation oracle.
    """

    def dense_jacobian(state, fixed_intensity):
        return fixed_intensity_jacobian_rows(
            state,
            fixed_intensity,
            dx_normalized=dx_normalized,
        )

    return _solve_pr_static_intensity_backend(
        intensity,
        applied_field=applied_field,
        background_intensity=background_intensity,
        dx_normalized=dx_normalized,
        initial_E=initial_E,
        options=options,
        direction_solver=_newton_direction,
        input_validator=_validated_inputs,
        jacobian_builder=dense_jacobian,
        xp=np,
    )


def solve_pr_static_intensity_batched(
    intensity,
    *,
    applied_field: float,
    background_intensity: float,
    dx_normalized: float,
    initial_E: Any | None = None,
    options: PRStaticSolverOptions = PRStaticSolverOptions(),
    xp: Any = np,
) -> PRStaticResult:
    """Solve ``R(E; I)=0`` with the O(Nx) batched cyclic direction.

    Arrays and nonlinear iteration remain on the explicitly supplied NumPy- or
    CuPy-like backend. The dense oracle uses the same residual, damping, and
    acceptance contract; only the linear-system implementation differs.
    """

    def structured_direction(residual, lower, diagonal, upper):
        return batched_newton_direction(
            residual,
            lower,
            diagonal,
            upper,
            xp=xp,
        )

    def validate_backend_inputs(*args, **kwargs):
        return _validated_batched_inputs(*args, **kwargs, xp=xp)

    def structured_jacobian(state, fixed_intensity):
        return _fixed_intensity_jacobian_rows_backend(
            state,
            fixed_intensity,
            dx_normalized=dx_normalized,
            xp=xp,
        )

    return _solve_pr_static_intensity_backend(
        intensity,
        applied_field=applied_field,
        background_intensity=background_intensity,
        dx_normalized=dx_normalized,
        initial_E=initial_E,
        options=options,
        direction_solver=structured_direction,
        input_validator=validate_backend_inputs,
        jacobian_builder=structured_jacobian,
        xp=xp,
    )


__all__ = [
    "PRStaticIterationRecord",
    "PRStaticResult",
    "PRStaticSolverOptions",
    "batched_newton_direction",
    "fixed_intensity_jacobian_rows",
    "solve_pr_static_intensity",
    "solve_pr_static_intensity_batched",
]
