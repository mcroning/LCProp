"""Self-consistent NumPy workflow for full-transverse PR equilibrium."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from time import perf_counter
from typing import Any, Callable

import numpy as np

from lcprop.core.backend import BackendSpec, get_backend
from lcprop.core.beams import BeamStack
from lcprop.core.context import GridSpec
from lcprop.core.execution import CancellationToken, RunProgress
from lcprop.core.grid import make_grid
from lcprop.optics.launch import build_launch, normalized_power
from lcprop.optics.splitstep import linear_kernel
from lcprop.pr.scattering import (
    PRCanonicalScatteringSpec,
    canonical_scattering_provenance,
)
from lcprop.pr.source import channel_peak_intensity_reference
from lcprop.pr.specs import PRMaterialSpec
from lcprop.pr.transverse.diagnostics import state_diagnostics
from lcprop.pr.transverse.specs import (
    PR_FULL_TRANSVERSE_PROFILE_V1,
    PRTransverseBoundaryProfile,
    PRTransverseDielectricProfile,
    PRTransverseProjectionProfile,
    PRTransverseTransportProfile,
)
from lcprop.pr.transverse.static import (
    PRTransverseStaticMaterialSolverOptions,
    PRTransverseStaticNewtonRecord,
    derivative_null_residual,
    project_production_resolved_modes,
    solve_pr_transverse_static_intensity,
    static_equilibrium_residual,
)
from lcprop.pr.transverse.transport import potential_rhs, state_from_potential
from lcprop.pr.transverse.workflow import (
    _initial_fields,
    _optical_pass,
)
from lcprop.pr.workflow import _validate_canonical_scattering_for_grid


PR_TRANSVERSE_STATIC_WORKFLOW = "pr_transverse_static"
ProgressCallback = Callable[[RunProgress], None]


@dataclass(frozen=True)
class PRTransverseStaticWorkflowOptions:
    """Coupled full-volume static controls, distinct from transient controls."""

    material_solver: PRTransverseStaticMaterialSolverOptions = field(
        default_factory=PRTransverseStaticMaterialSolverOptions
    )
    max_coupled_iterations: int = 20
    equilibrium_rms_tolerance: float = 1.0e-8
    equilibrium_max_tolerance: float = 1.0e-7
    td_rhs_rms_tolerance: float = 1.0e-8
    td_rhs_max_tolerance: float = 1.0e-7
    max_backtracks: int = 12
    minimum_step_scale: float = 2.0**-12
    armijo_fraction: float = 1.0e-4
    optical_substeps: int = 1
    replay_rtol: float = 1.0e-11
    replay_atol: float = 1.0e-12
    record_iteration_history: bool = True

    def validate(self) -> None:
        self.material_solver.validate()
        if int(self.max_coupled_iterations) < 1:
            raise ValueError("max_coupled_iterations must be at least one")
        if int(self.max_backtracks) < 0:
            raise ValueError("max_backtracks must be nonnegative")
        if int(self.optical_substeps) < 1:
            raise ValueError("optical_substeps must be at least one")
        for name in (
            "equilibrium_rms_tolerance",
            "equilibrium_max_tolerance",
            "td_rhs_rms_tolerance",
            "td_rhs_max_tolerance",
            "minimum_step_scale",
            "replay_rtol",
            "replay_atol",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        fraction = float(self.armijo_fraction)
        if not math.isfinite(fraction) or not 0.0 < fraction < 1.0:
            raise ValueError("armijo_fraction must be between zero and one")


@dataclass(frozen=True)
class PRTransverseStaticRunRequest:
    """Headless request for the separate full-transverse static workflow."""

    grid: GridSpec
    beams: BeamStack
    material: PRMaterialSpec = PRMaterialSpec()
    transport: PRTransverseTransportProfile = PRTransverseTransportProfile()
    dielectric: PRTransverseDielectricProfile = PRTransverseDielectricProfile()
    boundary: PRTransverseBoundaryProfile = PRTransverseBoundaryProfile()
    projection: PRTransverseProjectionProfile = PRTransverseProjectionProfile()
    solver: PRTransverseStaticWorkflowOptions = field(
        default_factory=PRTransverseStaticWorkflowOptions
    )
    backend: BackendSpec = BackendSpec(
        backend="numpy", precision="float64", verbose=False
    )
    initial_A: Any | None = None
    initial_psi: Any | None = None
    scattering: PRCanonicalScatteringSpec | None = None


@dataclass(frozen=True)
class PRTransverseStaticCoupledRecord:
    """One accepted or rejected refreshed-source outer correction."""

    coupled_iteration: int
    equilibrium_rms_before: float
    equilibrium_max_before: float
    td_rhs_rms_before: float
    td_rhs_max_before: float
    equilibrium_rms_after: float
    equilibrium_max_after: float
    td_rhs_rms_after: float
    td_rhs_max_after: float
    psi_change_rms: float
    psi_change_max: float
    source_change_rms: float
    optical_field_change_rms: float
    step_scale: float
    backtracks: int
    material_newton_iterations: int
    material_pcg_iterations: int
    accepted: bool
    termination_reason: str


@dataclass(frozen=True)
class PRTransverseStaticMaterialIterationRecord:
    """One per-plane Newton record associated with an outer iteration."""

    coupled_iteration: int
    newton_record: PRTransverseStaticNewtonRecord


@dataclass(frozen=True)
class PRTransverseStaticRunResult:
    """Accepted static potential, optical fields, and replay evidence."""

    A_initial: np.ndarray
    A_final: np.ndarray
    psi_initial: np.ndarray
    psi_final: np.ndarray
    source_intensity_stack: np.ndarray
    equilibrium_residual_stack: np.ndarray
    td_rhs_residual_stack: np.ndarray
    power_initial: float
    power_final: float
    converged: bool
    completed_coupled_iterations: int
    iteration_records: tuple[PRTransverseStaticCoupledRecord, ...]
    material_iteration_records: tuple[
        PRTransverseStaticMaterialIterationRecord, ...
    ]
    grid_summary: dict[str, Any]
    launch_summary: dict[str, Any]
    backend_summary: dict[str, Any]
    resolved_profile: dict[str, Any]
    replay_diagnostics: dict[str, Any]
    diagnostics: dict[str, Any]
    timing: dict[str, float]
    status: str


def _validate_request(request: PRTransverseStaticRunRequest) -> None:
    request.grid.validate()
    request.beams.validate()
    request.material.validate()
    request.transport.validate()
    request.dielectric.validate()
    request.boundary.validate()
    request.projection.validate()
    request.solver.validate()
    request.backend.validate()
    if request.scattering is not None:
        request.scattering.validate()
    if request.backend.backend != "numpy":
        raise ValueError("full-transverse static reference requires backend='numpy'")
    if request.backend.precision != "float64":
        raise ValueError(
            "full-transverse static NumPy reference requires precision='float64'"
        )
    if float(request.material.applied_field) != 0.0:
        raise ValueError("Profile v1 requires material.applied_field=0")


def _metrics(value) -> tuple[float, float]:
    work = np.asarray(value, dtype=np.float64)
    return float(np.sqrt(np.mean(work * work))), float(np.max(np.abs(work)))


def _difference_rms(left, right) -> float:
    difference = np.asarray(left) - np.asarray(right)
    return float(np.sqrt(np.mean(np.abs(difference) ** 2, dtype=np.float64)))


def _residuals(
    psi,
    source,
    *,
    dx_normalized: float,
    dy_normalized: float,
    h_y: float,
):
    equilibrium = static_equilibrium_residual(
        psi,
        source,
        dx_normalized=dx_normalized,
        dy_normalized=dy_normalized,
        h_y=h_y,
    )
    td = potential_rhs(
        psi,
        source,
        dx_normalized=dx_normalized,
        dy_normalized=dy_normalized,
        h_y=h_y,
        applied_field_x=0.0,
        xp=np,
    )
    resolved_equilibrium = project_production_resolved_modes(
        equilibrium,
        dx_normalized=dx_normalized,
        dy_normalized=dy_normalized,
        h_y=h_y,
    )
    return equilibrium, td, (*_metrics(resolved_equilibrium), *_metrics(td))


def _criteria_met(metrics, options: PRTransverseStaticWorkflowOptions) -> bool:
    equilibrium_rms, equilibrium_max, td_rms, td_max = metrics
    return (
        equilibrium_rms <= float(options.equilibrium_rms_tolerance)
        and equilibrium_max <= float(options.equilibrium_max_tolerance)
        and td_rms <= float(options.td_rhs_rms_tolerance)
        and td_max <= float(options.td_rhs_max_tolerance)
    )


def _scaled_merit(metrics, options: PRTransverseStaticWorkflowOptions) -> float:
    tolerances = (
        options.equilibrium_rms_tolerance,
        options.equilibrium_max_tolerance,
        options.td_rhs_rms_tolerance,
        options.td_rhs_max_tolerance,
    )
    return max(float(value) / float(tolerance) for value, tolerance in zip(metrics, tolerances))


def run_pr_transverse_static(
    request: PRTransverseStaticRunRequest,
    *,
    cancellation_token: CancellationToken | None = None,
    progress_callback: ProgressCallback | None = None,
) -> PRTransverseStaticRunResult:
    """Solve the self-consistent Profile-v1 static optical/material problem."""

    started = perf_counter()
    _validate_request(request)
    backend = get_backend(request.backend)
    grid = make_grid(request.grid, xp=np, real_dtype=backend.real_dtype)
    _validate_canonical_scattering_for_grid(
        request.scattering, grid=grid, z_length_um=request.grid.z_length_um
    )
    launch = build_launch(request.beams, grid, complex_dtype=backend.complex_dtype)
    A0, psi = _initial_fields(
        request,
        launch=launch,
        grid=grid,
        complex_dtype=backend.complex_dtype,
        real_dtype=backend.real_dtype,
    )
    psi = project_production_resolved_modes(
        np.asarray(psi),
        dx_normalized=request.material.characteristic_wavenumber_per_um * grid.dx_um,
        dy_normalized=request.material.characteristic_wavenumber_per_um * grid.dy_um,
        h_y=request.dielectric.h_y,
    )
    psi_initial = psi.copy()
    initial_state = state_from_potential(
        psi,
        dx_normalized=request.material.characteristic_wavenumber_per_um * grid.dx_um,
        dy_normalized=request.material.characteristic_wavenumber_per_um * grid.dy_um,
        h_y=request.dielectric.h_y,
        xp=np,
    )
    if np.any(initial_state.carrier_density <= 0.0):
        raise ValueError("initial_psi reconstructs nonpositive carrier density")
    wavelengths = tuple(float(channel.wavelength_um) for channel in request.beams.channels)
    if any(value != wavelengths[0] for value in wavelengths[1:]):
        raise ValueError("transverse PR workflow requires one shared wavelength")
    wavelength_um = wavelengths[0]
    kernel = linear_kernel(
        grid.fxy2_um,
        dz=grid.dz_um / int(request.solver.optical_substeps),
        wavelength=wavelength_um,
        n_ref=request.material.refractive_index,
        xp=np,
    )
    peak_reference = channel_peak_intensity_reference(A0, xp=np)
    k0 = request.material.characteristic_wavenumber_per_um
    dx_normalized = k0 * grid.dx_um
    dy_normalized = k0 * grid.dy_um

    optical_seconds = 0.0
    material_seconds = 0.0

    def optical_pass(state):
        nonlocal optical_seconds
        pass_started = perf_counter()
        result = _optical_pass(
            A0,
            state,
            request=request,
            grid=grid,
            kernel=kernel,
            peak_reference=peak_reference,
            wavelength_um=wavelength_um,
            dx_normalized=dx_normalized,
            dy_normalized=dy_normalized,
            cancellation_token=None,
        )
        optical_seconds += perf_counter() - pass_started
        return result

    A_accepted, source_accepted = optical_pass(psi)
    equilibrium, td_residual, metrics = _residuals(
        psi,
        source_accepted,
        dx_normalized=dx_normalized,
        dy_normalized=dy_normalized,
        h_y=request.dielectric.h_y,
    )
    records: list[PRTransverseStaticCoupledRecord] = []
    material_records: list[PRTransverseStaticMaterialIterationRecord] = []
    completed_iterations = 0
    converged = _criteria_met(metrics, request.solver)
    cancelled = False
    termination_reason = "residual_tolerance" if converged else "maximum_coupled_iterations"

    for coupled_iteration in range(1, int(request.solver.max_coupled_iterations) + 1):
        if cancellation_token is not None and cancellation_token.is_cancelled():
            cancelled = True
            termination_reason = "cancelled_at_accepted_boundary"
            break
        if converged:
            break
        material_started = perf_counter()
        material = solve_pr_transverse_static_intensity(
            source_accepted,
            dx_normalized=dx_normalized,
            dy_normalized=dy_normalized,
            initial_psi=psi,
            h_y=request.dielectric.h_y,
            options=request.solver.material_solver,
        )
        material_seconds += perf_counter() - material_started
        material_newton = sum(
            summary.newton_iterations for summary in material.plane_summaries
        )
        material_pcg = sum(
            summary.pcg_iterations for summary in material.plane_summaries
        )
        if request.solver.record_iteration_history:
            material_records.extend(
                PRTransverseStaticMaterialIterationRecord(
                    coupled_iteration=coupled_iteration,
                    newton_record=record,
                )
                for record in material.iteration_records
            )
        if cancellation_token is not None and cancellation_token.is_cancelled():
            cancelled = True
            termination_reason = "cancelled_at_accepted_boundary"
            break
        if not material.converged:
            termination_reason = f"material_{material.status}"
            if request.solver.record_iteration_history:
                records.append(PRTransverseStaticCoupledRecord(
                    coupled_iteration=coupled_iteration,
                    equilibrium_rms_before=metrics[0],
                    equilibrium_max_before=metrics[1],
                    td_rhs_rms_before=metrics[2],
                    td_rhs_max_before=metrics[3],
                    equilibrium_rms_after=metrics[0],
                    equilibrium_max_after=metrics[1],
                    td_rhs_rms_after=metrics[2],
                    td_rhs_max_after=metrics[3],
                    psi_change_rms=0.0,
                    psi_change_max=0.0,
                    source_change_rms=0.0,
                    optical_field_change_rms=0.0,
                    step_scale=0.0,
                    backtracks=0,
                    material_newton_iterations=material_newton,
                    material_pcg_iterations=material_pcg,
                    accepted=False,
                    termination_reason=termination_reason,
                ))
            break

        direction = material.psi - psi
        merit = _scaled_merit(metrics, request.solver)
        step_scale = 1.0
        accepted = False
        backtracks = 0
        candidate_metrics = metrics
        candidate_psi = psi
        candidate_A = A_accepted
        candidate_source = source_accepted
        candidate_equilibrium = equilibrium
        candidate_td = td_residual
        for backtracks in range(int(request.solver.max_backtracks) + 1):
            trial_psi = project_production_resolved_modes(
                psi + step_scale * direction,
                dx_normalized=dx_normalized,
                dy_normalized=dy_normalized,
                h_y=request.dielectric.h_y,
            )
            trial_state = state_from_potential(
                trial_psi,
                dx_normalized=dx_normalized,
                dy_normalized=dy_normalized,
                h_y=request.dielectric.h_y,
                xp=np,
            )
            if np.all(np.isfinite(trial_psi)) and np.all(trial_state.carrier_density > 0.0):
                trial_A, trial_source = optical_pass(trial_psi)
                trial_equilibrium, trial_td, trial_metrics = _residuals(
                    trial_psi,
                    trial_source,
                    dx_normalized=dx_normalized,
                    dy_normalized=dy_normalized,
                    h_y=request.dielectric.h_y,
                )
                trial_merit = _scaled_merit(trial_metrics, request.solver)
                if (
                    math.isfinite(trial_merit)
                    and trial_merit
                    <= (1.0 - float(request.solver.armijo_fraction) * step_scale)
                    * merit
                    and trial_metrics[1] <= metrics[1]
                    and trial_metrics[3] <= metrics[3]
                ):
                    accepted = True
                    candidate_psi = trial_psi
                    candidate_A = trial_A
                    candidate_source = trial_source
                    candidate_equilibrium = trial_equilibrium
                    candidate_td = trial_td
                    candidate_metrics = trial_metrics
                    break
            step_scale *= 0.5
            if step_scale < float(request.solver.minimum_step_scale):
                break

        if (
            accepted
            and cancellation_token is not None
            and cancellation_token.is_cancelled()
        ):
            accepted = False
            cancelled = True
            termination_reason = "cancelled_during_trial"
            candidate_metrics = metrics
            candidate_psi = psi
            candidate_A = A_accepted
            candidate_source = source_accepted
            candidate_equilibrium = equilibrium
            candidate_td = td_residual

        delta = candidate_psi - psi
        delta_rms, delta_max = _metrics(delta)
        if request.solver.record_iteration_history:
            records.append(PRTransverseStaticCoupledRecord(
                coupled_iteration=coupled_iteration,
                equilibrium_rms_before=metrics[0],
                equilibrium_max_before=metrics[1],
                td_rhs_rms_before=metrics[2],
                td_rhs_max_before=metrics[3],
                equilibrium_rms_after=candidate_metrics[0],
                equilibrium_max_after=candidate_metrics[1],
                td_rhs_rms_after=candidate_metrics[2],
                td_rhs_max_after=candidate_metrics[3],
                psi_change_rms=delta_rms,
                psi_change_max=delta_max,
                source_change_rms=_difference_rms(candidate_source, source_accepted),
                optical_field_change_rms=_difference_rms(candidate_A, A_accepted),
                step_scale=step_scale if accepted else 0.0,
                backtracks=backtracks,
                material_newton_iterations=material_newton,
                material_pcg_iterations=material_pcg,
                accepted=accepted,
                termination_reason=(
                    "accepted"
                    if accepted
                    else (
                        "cancelled_during_trial"
                        if cancelled
                        else "coupled_line_search_failed"
                    )
                ),
            ))
        if not accepted:
            if not cancelled:
                termination_reason = "coupled_line_search_failed"
            break
        if cancellation_token is not None and cancellation_token.is_cancelled():
            cancelled = True
            termination_reason = "cancelled_at_accepted_boundary"
            break
        psi = candidate_psi
        A_accepted = candidate_A
        source_accepted = candidate_source
        equilibrium = candidate_equilibrium
        td_residual = candidate_td
        metrics = candidate_metrics
        completed_iterations = coupled_iteration
        converged = _criteria_met(metrics, request.solver)
        termination_reason = (
            "residual_tolerance" if converged else "maximum_coupled_iterations"
        )
        if progress_callback is not None:
            progress_callback(RunProgress(
                workflow=PR_TRANSVERSE_STATIC_WORKFLOW,
                status="running",
                completed_units=completed_iterations,
                total_units=int(request.solver.max_coupled_iterations),
                current_coordinate=float(completed_iterations),
                coordinate_name="coupled_iteration",
                coordinate_unit="1",
                elapsed_wall_time=perf_counter() - started,
                latest_field_state={"psi_current": psi.copy()},
                diagnostics={
                    "equilibrium_rms": metrics[0],
                    "equilibrium_max": metrics[1],
                    "td_rhs_rms": metrics[2],
                    "td_rhs_max": metrics[3],
                },
                message="transverse PR static outer iteration accepted",
            ))

    replay_A, replay_source = optical_pass(psi)
    replay_field_match = np.allclose(
        replay_A,
        A_accepted,
        rtol=float(request.solver.replay_rtol),
        atol=float(request.solver.replay_atol),
    )
    replay_source_match = np.allclose(
        replay_source,
        source_accepted,
        rtol=float(request.solver.replay_rtol),
        atol=float(request.solver.replay_atol),
    )
    replay_diagnostics = {
        "field_match": bool(replay_field_match),
        "source_match": bool(replay_source_match),
        "field_relative_l2": _difference_rms(replay_A, A_accepted)
        / max(_difference_rms(replay_A, np.zeros_like(replay_A)), np.finfo(float).tiny),
        "source_relative_l2": _difference_rms(replay_source, source_accepted)
        / max(_difference_rms(replay_source, np.zeros_like(replay_source)), np.finfo(float).tiny),
        "complete_independent_replay": True,
    }
    if converged and not (replay_field_match and replay_source_match):
        converged = False
        termination_reason = "replay_mismatch"

    final_state = state_from_potential(
        psi,
        dx_normalized=dx_normalized,
        dy_normalized=dy_normalized,
        h_y=request.dielectric.h_y,
        xp=np,
    )
    diagnostics = state_diagnostics(
        final_state,
        dx_normalized=dx_normalized,
        dy_normalized=dy_normalized,
        h_y=request.dielectric.h_y,
        xp=np,
    )
    null = derivative_null_residual(
        equilibrium,
        dx_normalized=dx_normalized,
        dy_normalized=dy_normalized,
        h_y=request.dielectric.h_y,
    )
    null_rms, null_max = _metrics(null)
    power_initial = normalized_power(A0, grid)
    power_final = normalized_power(replay_A, grid)
    diagnostics.update({
        "equilibrium_residual_rms": metrics[0],
        "equilibrium_residual_max": metrics[1],
        "td_rhs_residual_rms": metrics[2],
        "td_rhs_residual_max": metrics[3],
        "derivative_null_residual_rms": null_rms,
        "derivative_null_residual_max": null_max,
        "optical_power_relative_drift": (power_final - power_initial) / power_initial,
        "termination_reason": termination_reason,
        "cancelled": cancelled,
    })
    scattering_provenance = None
    if request.scattering is not None:
        scattering_provenance = canonical_scattering_provenance(
            request.scattering,
            z_length_um=request.grid.z_length_um,
            Nx=grid.Nx,
            Ny=grid.Ny,
            x_aperture_um=request.grid.x_aperture_um,
            y_aperture_um=request.grid.y_aperture_um,
            real_dtype=grid.real_dtype,
            xp=np,
        )
        diagnostics["canonical_scattering"] = scattering_provenance
    resolved_profile = {
        "physics_profile_id": PR_FULL_TRANSVERSE_PROFILE_V1,
        "workflow": PR_TRANSVERSE_STATIC_WORKFLOW,
        "grid_request": asdict(request.grid),
        "beam_request": asdict(request.beams),
        "material": asdict(request.material),
        "transport": asdict(request.transport),
        "dielectric": asdict(request.dielectric),
        "boundary": asdict(request.boundary),
        "projection": asdict(request.projection),
        "solver": asdict(request.solver),
        "characteristic_wavenumber_per_um": k0,
        "dx_normalized": dx_normalized,
        "dy_normalized": dy_normalized,
    }
    if scattering_provenance is not None:
        resolved_profile["canonical_scattering"] = scattering_provenance
    timing = {
        "total_seconds": perf_counter() - started,
        "optical_pass_seconds": optical_seconds,
        "material_solve_seconds": material_seconds,
    }
    status = "cancelled" if cancelled else ("converged" if converged else "not_converged")
    return PRTransverseStaticRunResult(
        A_initial=np.asarray(A0).copy(),
        A_final=np.asarray(replay_A).copy(),
        psi_initial=np.asarray(psi_initial).copy(),
        psi_final=np.asarray(psi).copy(),
        source_intensity_stack=np.asarray(replay_source).copy(),
        equilibrium_residual_stack=np.asarray(equilibrium).copy(),
        td_rhs_residual_stack=np.asarray(td_residual).copy(),
        power_initial=power_initial,
        power_final=power_final,
        converged=converged,
        completed_coupled_iterations=completed_iterations,
        iteration_records=tuple(records),
        material_iteration_records=tuple(material_records),
        grid_summary=grid.summary(),
        launch_summary=launch.summary(),
        backend_summary=backend.summary(),
        resolved_profile=resolved_profile,
        replay_diagnostics=replay_diagnostics,
        diagnostics=diagnostics,
        timing=timing,
        status=status,
    )


__all__ = [
    "PR_TRANSVERSE_STATIC_WORKFLOW",
    "PRTransverseStaticCoupledRecord",
    "PRTransverseStaticMaterialIterationRecord",
    "PRTransverseStaticRunRequest",
    "PRTransverseStaticRunResult",
    "PRTransverseStaticWorkflowOptions",
    "run_pr_transverse_static",
]
