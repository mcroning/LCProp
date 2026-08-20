"""NumPy reference solver for frozen full-transverse PR equilibrium."""

from __future__ import annotations

from dataclasses import dataclass
import math
from time import perf_counter
from typing import Any

import numpy as np

from lcprop.pr.transverse.transport import (
    potential_rhs,
    spectral_wavevectors,
    state_from_potential,
)


@dataclass(frozen=True)
class PRTransverseStaticMaterialSolverOptions:
    """Controls for one or more independent frozen-intensity planes."""

    max_newton_iterations: int = 30
    max_pcg_iterations: int = 200
    pcg_relative_tolerance: float = 1.0e-8
    pcg_absolute_tolerance: float = 1.0e-12
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
    potential_mean: float


@dataclass(frozen=True)
class PRTransverseStaticMaterialResult:
    """Frozen-intensity potential and complete convergence diagnostics."""

    psi: np.ndarray
    equilibrium_residual: np.ndarray
    td_rhs_residual: np.ndarray
    converged: bool
    status: str
    plane_summaries: tuple[PRTransverseStaticPlaneSummary, ...]
    iteration_records: tuple[PRTransverseStaticNewtonRecord, ...]
    elapsed_seconds: float


def _validate_numpy_real_array(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError(f"{name} must have dtype float32 or float64")
    if array.ndim not in (2, 3) or min(array.shape[-2:]) < 3:
        raise ValueError(f"{name} must end in (Nx, Ny), each at least 3")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _symbols(shape, *, dx_normalized: float, dy_normalized: float, h_y: float):
    kx, ky = spectral_wavevectors(
        shape,
        dx_normalized=dx_normalized,
        dy_normalized=dy_normalized,
        xp=np,
    )
    denominator = kx * kx + float(h_y) * ky * ky
    return denominator, denominator == 0.0


def project_production_resolved_modes(
    value,
    *,
    dx_normalized: float,
    dy_normalized: float,
    h_y: float = 1.0,
) -> np.ndarray:
    """Project onto the modes resolved by production transverse derivatives."""

    array = _validate_numpy_real_array(value, name="value")
    denominator, null_mask = _symbols(
        array.shape[-2:],
        dx_normalized=dx_normalized,
        dy_normalized=dy_normalized,
        h_y=h_y,
    )
    transformed = np.fft.fft2(array, axes=(-2, -1))
    transformed[..., null_mask] = 0.0
    projected = np.fft.ifft2(transformed, axes=(-2, -1)).real
    return projected.astype(array.dtype, copy=False)


def _normalized_equilibrium_carrier(psi: np.ndarray, intensity: np.ndarray):
    log_weight = -psi.astype(np.float64, copy=False) - np.log(
        intensity.astype(np.float64, copy=False)
    )
    log_weight -= np.max(log_weight, axis=(-2, -1), keepdims=True)
    weight = np.exp(log_weight)
    weight /= np.mean(weight, axis=(-2, -1), keepdims=True)
    return weight.astype(psi.dtype, copy=False)


def static_equilibrium_residual(
    psi,
    intensity,
    *,
    dx_normalized: float,
    dy_normalized: float,
    h_y: float = 1.0,
) -> np.ndarray:
    """Return ``P - (exp(-psi)/I)/mean(exp(-psi)/I)``.

    The complete transport intensity must already include optical, dark, and
    optional uniform-background contributions.
    """

    potential = _validate_numpy_real_array(psi, name="psi")
    driving = _validate_numpy_real_array(intensity, name="intensity")
    if driving.dtype != np.dtype(np.float64):
        raise TypeError("NumPy static reference solver requires float64 intensity")
    if potential.shape != driving.shape:
        raise ValueError("psi and intensity must have identical shapes")
    if np.any(driving <= 0.0):
        raise ValueError("static transport intensity must be strictly positive")
    resolved = project_production_resolved_modes(
        potential,
        dx_normalized=dx_normalized,
        dy_normalized=dy_normalized,
        h_y=h_y,
    )
    state = state_from_potential(
        resolved,
        dx_normalized=dx_normalized,
        dy_normalized=dy_normalized,
        h_y=h_y,
        xp=np,
    )
    equilibrium_carrier = _normalized_equilibrium_carrier(resolved, driving)
    return (state.carrier_density - equilibrium_carrier).astype(
        potential.dtype, copy=False
    )


def _metrics(value: np.ndarray) -> tuple[float, float]:
    work = np.asarray(value, dtype=np.float64)
    return float(np.sqrt(np.mean(work * work))), float(np.max(np.abs(work)))


def derivative_null_residual(
    residual: np.ndarray,
    *,
    dx_normalized: float,
    dy_normalized: float,
    h_y: float = 1.0,
) -> np.ndarray:
    _, null_mask = _symbols(
        residual.shape[-2:],
        dx_normalized=dx_normalized,
        dy_normalized=dy_normalized,
        h_y=h_y,
    )
    transformed = np.fft.fft2(residual, axes=(-2, -1))
    transformed[..., ~null_mask] = 0.0
    return np.fft.ifft2(transformed, axes=(-2, -1)).real.astype(
        residual.dtype, copy=False
    )


def _elliptic_apply(value: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    transformed = np.fft.fft2(value)
    return np.fft.ifft2(denominator * transformed).real.astype(
        value.dtype, copy=False
    )


def _jacobian_action(
    value: np.ndarray,
    *,
    weight: np.ndarray,
    denominator: np.ndarray,
    null_mask: np.ndarray,
) -> np.ndarray:
    projected = np.fft.fft2(value)
    projected[null_mask] = 0.0
    projected = np.fft.ifft2(projected).real.astype(value.dtype, copy=False)
    weighted_mean = np.mean(
        weight.astype(np.float64, copy=False)
        * projected.astype(np.float64, copy=False)
    )
    result = (
        _elliptic_apply(projected, denominator)
        + weight * projected
        - weight * weighted_mean
    )
    transformed = np.fft.fft2(result)
    transformed[null_mask] = 0.0
    return np.fft.ifft2(transformed).real.astype(value.dtype, copy=False)


def _precondition(
    residual: np.ndarray,
    *,
    denominator: np.ndarray,
    null_mask: np.ndarray,
) -> np.ndarray:
    transformed = np.fft.fft2(residual)
    transformed /= denominator + 1.0
    transformed[null_mask] = 0.0
    return np.fft.ifft2(transformed).real.astype(residual.dtype, copy=False)


def _pcg(
    rhs: np.ndarray,
    *,
    weight: np.ndarray,
    denominator: np.ndarray,
    null_mask: np.ndarray,
    options: PRTransverseStaticMaterialSolverOptions,
) -> tuple[np.ndarray, int, bool, str]:
    solution = np.zeros_like(rhs)
    residual = rhs.copy()
    norm_rhs = float(np.linalg.norm(rhs.ravel()))
    tolerance = max(
        float(options.pcg_absolute_tolerance),
        float(options.pcg_relative_tolerance) * norm_rhs,
    )
    if norm_rhs <= tolerance:
        return solution, 0, True, "converged"
    preconditioned = _precondition(
        residual, denominator=denominator, null_mask=null_mask
    )
    direction = preconditioned.copy()
    rz = float(np.vdot(residual.ravel(), preconditioned.ravel()).real)
    if not math.isfinite(rz) or rz <= 0.0:
        return solution, 0, False, "nonpositive_preconditioned_residual"
    for iteration in range(1, int(options.max_pcg_iterations) + 1):
        image = _jacobian_action(
            direction,
            weight=weight,
            denominator=denominator,
            null_mask=null_mask,
        )
        curvature = float(np.vdot(direction.ravel(), image.ravel()).real)
        if not math.isfinite(curvature) or curvature <= 0.0:
            return solution, iteration - 1, False, "nonpositive_curvature"
        alpha = rz / curvature
        solution += alpha * direction
        residual -= alpha * image
        if float(np.linalg.norm(residual.ravel())) <= tolerance:
            return solution, iteration, True, "converged"
        next_preconditioned = _precondition(
            residual, denominator=denominator, null_mask=null_mask
        )
        next_rz = float(
            np.vdot(residual.ravel(), next_preconditioned.ravel()).real
        )
        if not math.isfinite(next_rz) or next_rz <= 0.0:
            return solution, iteration, False, "nonpositive_preconditioned_residual"
        direction = next_preconditioned + (next_rz / rz) * direction
        preconditioned = next_preconditioned
        rz = next_rz
    return solution, int(options.max_pcg_iterations), False, "maximum_iterations"


def _criteria_met(
    equilibrium_rms: float,
    equilibrium_max: float,
    td_rhs_rms: float,
    td_rhs_max: float,
    options: PRTransverseStaticMaterialSolverOptions,
) -> bool:
    return (
        equilibrium_rms <= float(options.equilibrium_rms_tolerance)
        and equilibrium_max <= float(options.equilibrium_max_tolerance)
        and td_rhs_rms <= float(options.td_rhs_rms_tolerance)
        and td_rhs_max <= float(options.td_rhs_max_tolerance)
    )


def _solve_plane(
    intensity: np.ndarray,
    initial_psi: np.ndarray,
    *,
    plane_index: int,
    dx_normalized: float,
    dy_normalized: float,
    h_y: float,
    options: PRTransverseStaticMaterialSolverOptions,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    PRTransverseStaticPlaneSummary,
    list[PRTransverseStaticNewtonRecord],
]:
    denominator, null_mask = _symbols(
        intensity.shape,
        dx_normalized=dx_normalized,
        dy_normalized=dy_normalized,
        h_y=h_y,
    )
    psi = project_production_resolved_modes(
        initial_psi,
        dx_normalized=dx_normalized,
        dy_normalized=dy_normalized,
        h_y=h_y,
    ).copy()
    initial_state = state_from_potential(
        psi,
        dx_normalized=dx_normalized,
        dy_normalized=dy_normalized,
        h_y=h_y,
        xp=np,
    )
    if np.any(initial_state.carrier_density <= 0.0):
        raise ValueError("initial_psi reconstructs nonpositive carrier density")
    records: list[PRTransverseStaticNewtonRecord] = []
    total_pcg = 0
    total_backtracks = 0
    status = "maximum_newton_iterations"

    for newton_iteration in range(int(options.max_newton_iterations) + 1):
        residual = static_equilibrium_residual(
            psi,
            intensity,
            dx_normalized=dx_normalized,
            dy_normalized=dy_normalized,
            h_y=h_y,
        )
        td_residual = potential_rhs(
            psi,
            intensity,
            dx_normalized=dx_normalized,
            dy_normalized=dy_normalized,
            h_y=h_y,
            applied_field_x=0.0,
            xp=np,
        )
        resolved_residual = project_production_resolved_modes(
            residual,
            dx_normalized=dx_normalized,
            dy_normalized=dy_normalized,
            h_y=h_y,
        )
        equilibrium_rms, equilibrium_max = _metrics(resolved_residual)
        td_rhs_rms, td_rhs_max = _metrics(td_residual)
        if _criteria_met(
            equilibrium_rms,
            equilibrium_max,
            td_rhs_rms,
            td_rhs_max,
            options,
        ):
            status = "residual_tolerance"
            break
        if newton_iteration == int(options.max_newton_iterations):
            break
        weight = _normalized_equilibrium_carrier(psi, intensity)
        rhs_hat = np.fft.fft2(-resolved_residual)
        rhs_hat[null_mask] = 0.0
        rhs = np.fft.ifft2(rhs_hat).real.astype(psi.dtype, copy=False)
        direction, pcg_iterations, pcg_converged, pcg_status = _pcg(
            rhs,
            weight=weight,
            denominator=denominator,
            null_mask=null_mask,
            options=options,
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
                step_scale=0.0,
                backtracks=0,
                accepted=False,
            ))
            break
        if (
            pcg_iterations == 0
            and not np.any(direction)
            and equilibrium_rms <= float(options.equilibrium_rms_tolerance)
            and equilibrium_max <= float(options.equilibrium_max_tolerance)
        ):
            status = "td_rhs_tolerance_not_met"
            break

        step_scale = 1.0
        accepted = False
        after_rms = equilibrium_rms
        after_max = equilibrium_max
        backtracks = 0
        for backtracks in range(int(options.max_backtracks) + 1):
            trial = project_production_resolved_modes(
                psi + step_scale * direction,
                dx_normalized=dx_normalized,
                dy_normalized=dy_normalized,
                h_y=h_y,
            )
            trial_state = state_from_potential(
                trial,
                dx_normalized=dx_normalized,
                dy_normalized=dy_normalized,
                h_y=h_y,
                xp=np,
            )
            if np.all(np.isfinite(trial)) and np.all(trial_state.carrier_density > 0.0):
                trial_residual = static_equilibrium_residual(
                    trial,
                    intensity,
                    dx_normalized=dx_normalized,
                    dy_normalized=dy_normalized,
                    h_y=h_y,
                )
                trial_resolved_residual = project_production_resolved_modes(
                    trial_residual,
                    dx_normalized=dx_normalized,
                    dy_normalized=dy_normalized,
                    h_y=h_y,
                )
                after_rms, after_max = _metrics(trial_resolved_residual)
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
            step_scale=step_scale if accepted else 0.0,
            backtracks=backtracks,
            accepted=accepted,
        ))
        if not accepted:
            status = "line_search_failed"
            break

    residual = static_equilibrium_residual(
        psi,
        intensity,
        dx_normalized=dx_normalized,
        dy_normalized=dy_normalized,
        h_y=h_y,
    )
    td_residual = potential_rhs(
        psi,
        intensity,
        dx_normalized=dx_normalized,
        dy_normalized=dy_normalized,
        h_y=h_y,
        applied_field_x=0.0,
        xp=np,
    )
    resolved_residual = project_production_resolved_modes(
        residual,
        dx_normalized=dx_normalized,
        dy_normalized=dy_normalized,
        h_y=h_y,
    )
    equilibrium_rms, equilibrium_max = _metrics(resolved_residual)
    td_rhs_rms, td_rhs_max = _metrics(td_residual)
    null_rms, null_max = _metrics(derivative_null_residual(
        residual,
        dx_normalized=dx_normalized,
        dy_normalized=dy_normalized,
        h_y=h_y,
    ))
    state = state_from_potential(
        psi,
        dx_normalized=dx_normalized,
        dy_normalized=dy_normalized,
        h_y=h_y,
        xp=np,
    )
    converged = _criteria_met(
        equilibrium_rms,
        equilibrium_max,
        td_rhs_rms,
        td_rhs_max,
        options,
    )
    if converged:
        status = "residual_tolerance"
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
        carrier_mean=float(np.mean(state.carrier_density, dtype=np.float64)),
        potential_mean=float(np.mean(state.psi, dtype=np.float64)),
    )
    return psi, residual, td_residual, summary, records


def solve_pr_transverse_static_intensity(
    intensity,
    *,
    dx_normalized: float,
    dy_normalized: float,
    initial_psi=None,
    h_y: float = 1.0,
    options: PRTransverseStaticMaterialSolverOptions | None = None,
) -> PRTransverseStaticMaterialResult:
    """Solve independent periodic frozen-intensity equilibrium planes on NumPy."""

    started = perf_counter()
    driving = _validate_numpy_real_array(intensity, name="intensity")
    if np.any(driving <= 0.0):
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
        initial = np.zeros_like(driving)
    else:
        initial = _validate_numpy_real_array(initial_psi, name="initial_psi")
        if initial.shape != driving.shape:
            raise ValueError("initial_psi and intensity must have identical shapes")
        initial = initial.astype(driving.dtype, copy=True)

    was_plane = driving.ndim == 2
    driving_volume = driving[None, ...] if was_plane else driving
    initial_volume = initial[None, ...] if was_plane else initial
    psi_volume = np.empty_like(driving_volume)
    residual_volume = np.empty_like(driving_volume)
    td_volume = np.empty_like(driving_volume)
    summaries: list[PRTransverseStaticPlaneSummary] = []
    records: list[PRTransverseStaticNewtonRecord] = []
    for plane_index in range(driving_volume.shape[0]):
        psi_plane, residual_plane, td_plane, summary, plane_records = _solve_plane(
            driving_volume[plane_index],
            initial_volume[plane_index],
            plane_index=plane_index,
            dx_normalized=dx_normalized,
            dy_normalized=dy_normalized,
            h_y=h_y,
            options=resolved_options,
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


__all__ = [
    "PRTransverseStaticMaterialResult",
    "PRTransverseStaticMaterialSolverOptions",
    "PRTransverseStaticNewtonRecord",
    "PRTransverseStaticPlaneSummary",
    "derivative_null_residual",
    "project_production_resolved_modes",
    "solve_pr_transverse_static_intensity",
    "static_equilibrium_residual",
]
