"""NumPy reference for causal zero-flux transverse PR static propagation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from time import perf_counter
from typing import Any

import numpy as np

from lcprop.core.backend import BackendSpec
from lcprop.core.beams import BeamStack
from lcprop.core.context import GridSpec
from lcprop.core.grid import make_grid
from lcprop.optics.launch import build_launch, normalized_power
from lcprop.optics.splitstep import linear_kernel
from lcprop.pr.scattering import (
    PRCanonicalScatteringSpec,
    canonical_scattering_provenance,
)
from lcprop.pr.source import channel_peak_intensity_reference, pr_driving_intensity
from lcprop.pr.specs import PRMaterialSpec
from lcprop.pr.transverse.projection import project_active_field
from lcprop.pr.transverse.specs import (
    PR_FULL_TRANSVERSE_PROFILE_V1,
    PRTransverseBoundaryProfile,
    PRTransverseDielectricProfile,
    PRTransverseProjectionProfile,
    PRTransverseTransportProfile,
)
from lcprop.pr.transverse.static import (
    PRTransverseStaticMaterialSolverOptions,
    _physical_state_validity,
    project_production_resolved_modes,
    solve_pr_transverse_static_intensity,
    static_equilibrium_residual,
)
from lcprop.pr.transverse.transport import state_from_potential
from lcprop.pr.workflow import (
    _apply_canonical_scattering_after_slice,
    _validate_canonical_scattering_for_grid,
    advance_pr_slice_with_midpoint_source,
)


PR_TRANSVERSE_MARCHING_STATIC_WORKFLOW = "pr_transverse_marching_static_reference"


@dataclass(frozen=True)
class PRTransverseMarchingStaticOptions:
    """Reference-only controls for damped local Picard iteration."""

    material_solver: PRTransverseStaticMaterialSolverOptions = field(
        default_factory=PRTransverseStaticMaterialSolverOptions
    )
    max_local_iterations: int = 20
    equilibrium_rms_tolerance: float = 1.0e-8
    equilibrium_max_tolerance: float = 1.0e-7
    max_backtracks: int = 12
    minimum_step_scale: float = 2.0**-12
    armijo_fraction: float = 1.0e-4
    optical_substeps: int = 1
    record_iteration_history: bool = True

    def validate(self) -> None:
        self.material_solver.validate()
        if int(self.max_local_iterations) < 1:
            raise ValueError("max_local_iterations must be at least one")
        if int(self.max_backtracks) < 0:
            raise ValueError("max_backtracks must be nonnegative")
        if int(self.optical_substeps) < 1:
            raise ValueError("optical_substeps must be at least one")
        for name in (
            "equilibrium_rms_tolerance",
            "equilibrium_max_tolerance",
            "minimum_step_scale",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        fraction = float(self.armijo_fraction)
        if not math.isfinite(fraction) or not 0.0 < fraction < 1.0:
            raise ValueError("armijo_fraction must be between zero and one")


@dataclass(frozen=True)
class PRTransverseMarchingStaticRunRequest:
    """Request for the additive NumPy causal-marching reference.

    ``initial_psi`` is an optional two-dimensional guess for the first
    interval only.  Every later interval is warm-started from the preceding
    accepted plane; no longitudinal interpolation is performed.
    """

    grid: GridSpec
    beams: BeamStack
    material: PRMaterialSpec = PRMaterialSpec()
    transport: PRTransverseTransportProfile = PRTransverseTransportProfile()
    dielectric: PRTransverseDielectricProfile = PRTransverseDielectricProfile()
    boundary: PRTransverseBoundaryProfile = PRTransverseBoundaryProfile()
    projection: PRTransverseProjectionProfile = PRTransverseProjectionProfile()
    solver: PRTransverseMarchingStaticOptions = field(
        default_factory=PRTransverseMarchingStaticOptions
    )
    backend: BackendSpec = BackendSpec(
        backend="numpy", precision="float64", verbose=False
    )
    initial_A: Any | None = None
    initial_psi: Any | None = None
    scattering: PRCanonicalScatteringSpec | None = None


@dataclass(frozen=True)
class PRTransverseMarchingPicardRecord:
    """One accepted or rejected refreshed-source local Picard correction."""

    interval_index: int
    local_iteration: int
    equilibrium_rms_before: float
    equilibrium_max_before: float
    equilibrium_rms_after: float
    equilibrium_max_after: float
    potential_update_rms: float
    potential_update_max: float
    source_update_rms: float
    source_update_max: float
    step_scale: float
    backtracks: int
    material_newton_iterations: int
    material_pcg_iterations: int
    accepted: bool
    termination_reason: str


@dataclass(frozen=True)
class PRTransverseMarchingIntervalSummary:
    """Final convergence and physical-state evidence for one interval."""

    interval_index: int
    z_midpoint_um: float
    converged: bool
    local_iterations: int
    backtracks: int
    accepted_damping: float
    equilibrium_rms: float
    equilibrium_max: float
    carrier_minimum: float
    carrier_mean_error: float
    potential_gauge_error: float
    potential_update_rms: float
    potential_update_max: float
    source_update_rms: float
    source_update_max: float
    material_newton_iterations: int
    material_pcg_iterations: int
    termination_reason: str


@dataclass(frozen=True)
class PRTransverseMarchingStaticRunResult:
    """Accepted prefix and audit evidence from the causal reference march."""

    A_initial: np.ndarray
    A_final: np.ndarray
    incoming_field_stack: np.ndarray
    pre_scattering_exit_stack: np.ndarray
    psi_accepted: np.ndarray
    source_intensity_stack: np.ndarray
    equilibrium_residual_stack: np.ndarray
    converged: bool
    completed_intervals: int
    failed_interval: int | None
    interval_summaries: tuple[PRTransverseMarchingIntervalSummary, ...]
    iteration_records: tuple[PRTransverseMarchingPicardRecord, ...]
    power_initial: float
    power_final: float
    grid_summary: dict[str, Any]
    launch_summary: dict[str, Any]
    backend_summary: dict[str, Any]
    resolved_profile: dict[str, Any]
    diagnostics: dict[str, Any]
    timing: dict[str, float]
    status: str


@dataclass(frozen=True)
class _LocalEvaluation:
    psi: np.ndarray
    A_pre_scattering: np.ndarray
    source: np.ndarray
    residual: np.ndarray
    equilibrium_rms: float
    equilibrium_max: float
    physical_state_valid: bool
    carrier_mean: float
    carrier_minimum: float
    potential_mean: float


def _metrics(value) -> tuple[float, float]:
    work = np.asarray(value, dtype=np.float64)
    return float(np.sqrt(np.mean(work * work))), float(np.max(np.abs(work)))


def _difference_metrics(left, right) -> tuple[float, float]:
    difference = np.asarray(left) - np.asarray(right)
    return (
        float(np.sqrt(np.mean(np.abs(difference) ** 2))),
        float(np.max(np.abs(difference))),
    )


def _criteria_met(evaluation, options) -> bool:
    return (
        evaluation.physical_state_valid
        and evaluation.equilibrium_rms
        <= float(options.equilibrium_rms_tolerance)
        and evaluation.equilibrium_max
        <= float(options.equilibrium_max_tolerance)
    )


def _validate_request(request: PRTransverseMarchingStaticRunRequest) -> None:
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
    if request.backend.backend != "numpy" or request.backend.precision != "float64":
        raise ValueError(
            "marching static reference requires backend='numpy', precision='float64'"
        )
    if float(request.material.applied_field) != 0.0:
        raise ValueError("Profile v1 requires material.applied_field=0")


def _evaluate_local_state(
    A_in,
    psi,
    *,
    request,
    grid,
    kernel,
    peak_reference,
    intensity_before,
    wavelength_um,
    dx_normalized,
    dy_normalized,
) -> _LocalEvaluation:
    resolved = project_production_resolved_modes(
        psi,
        dx_normalized=dx_normalized,
        dy_normalized=dy_normalized,
        h_y=request.dielectric.h_y,
        xp=np,
    )
    state = state_from_potential(
        resolved,
        dx_normalized=dx_normalized,
        dy_normalized=dy_normalized,
        h_y=request.dielectric.h_y,
        applied_field_x=request.boundary.applied_field_x,
        xp=np,
    )
    active = project_active_field(
        state.E_x, state.E_y, profile=request.projection, xp=np
    )
    A_pre, source = advance_pr_slice_with_midpoint_source(
        A_in,
        active,
        kernel=kernel,
        optical_substeps=request.solver.optical_substeps,
        dz_um=grid.dz_um,
        wavelength_um=wavelength_um,
        interaction_length_um=request.grid.z_length_um,
        gain_length_product=request.material.gain_length_product,
        peak_intensity_reference=peak_reference,
        background_intensity=request.material.background_intensity,
        coherence_groups=request.beams.coherence_groups,
        xp=np,
        _intensity_before=intensity_before,
    )
    residual = static_equilibrium_residual(
        resolved,
        source,
        dx_normalized=dx_normalized,
        dy_normalized=dy_normalized,
        h_y=request.dielectric.h_y,
        xp=np,
    )
    resolved_residual = project_production_resolved_modes(
        residual,
        dx_normalized=dx_normalized,
        dy_normalized=dy_normalized,
        h_y=request.dielectric.h_y,
        xp=np,
    )
    residual_rms, residual_max = _metrics(resolved_residual)
    valid, carrier_mean, carrier_minimum, potential_mean = (
        _physical_state_validity(state, xp=np)
    )
    return _LocalEvaluation(
        psi=np.asarray(resolved),
        A_pre_scattering=np.asarray(A_pre),
        source=np.asarray(source),
        residual=np.asarray(residual),
        equilibrium_rms=residual_rms,
        equilibrium_max=residual_max,
        physical_state_valid=valid,
        carrier_mean=carrier_mean,
        carrier_minimum=carrier_minimum,
        potential_mean=potential_mean,
    )


def run_pr_transverse_static_marching(
    request: PRTransverseMarchingStaticRunRequest,
) -> PRTransverseMarchingStaticRunResult:
    """March causally through locally implicit zero-flux PR intervals."""

    started = perf_counter()
    _validate_request(request)
    grid = make_grid(request.grid, xp=np, real_dtype=np.float64)
    _validate_canonical_scattering_for_grid(
        request.scattering, grid=grid, z_length_um=request.grid.z_length_um
    )
    launch = build_launch(request.beams, grid, complex_dtype=np.complex128)
    if request.initial_A is None:
        A0 = np.asarray(launch.A0).copy()
    else:
        A0 = np.asarray(request.initial_A, dtype=np.complex128).copy()
        if A0.shape != launch.A0.shape or not np.all(np.isfinite(A0)):
            raise ValueError("initial_A must be finite and match the launch shape")
    if request.initial_psi is None:
        first_guess = np.zeros((grid.Nx, grid.Ny), dtype=np.float64)
    else:
        first_guess = np.asarray(request.initial_psi, dtype=np.float64).copy()
        if first_guess.shape != (grid.Nx, grid.Ny):
            raise ValueError("initial_psi must have shape (Nx, Ny)")
        if not np.all(np.isfinite(first_guess)):
            raise ValueError("initial_psi must contain only finite values")

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
    intensity_before = pr_driving_intensity(
        A0,
        peak_intensity_reference=peak_reference,
        background_intensity=request.material.background_intensity,
        coherence_groups=request.beams.coherence_groups,
        xp=np,
    )
    k0 = request.material.characteristic_wavenumber_per_um
    dx_normalized = k0 * grid.dx_um
    dy_normalized = k0 * grid.dy_um

    A_in = A0.copy()
    incoming_fields = [A_in.copy()]
    pre_scattering_outputs: list[np.ndarray] = []
    accepted_psi: list[np.ndarray] = []
    accepted_sources: list[np.ndarray] = []
    accepted_residuals: list[np.ndarray] = []
    summaries: list[PRTransverseMarchingIntervalSummary] = []
    records: list[PRTransverseMarchingPicardRecord] = []
    failed_interval: int | None = None
    optical_seconds = 0.0
    material_seconds = 0.0

    for interval_index in range(grid.Nz):
        psi = first_guess.copy() if interval_index == 0 else accepted_psi[-1].copy()
        evaluation = None
        last_update = (0.0, 0.0)
        last_source_update = (0.0, 0.0)
        total_backtracks = 0
        accepted_damping = 1.0
        material_newton = 0
        material_pcg = 0
        local_iterations = 0
        termination_reason = "maximum_local_iterations"

        for local_iteration in range(1, int(request.solver.max_local_iterations) + 1):
            local_iterations = local_iteration - 1
            optical_started = perf_counter()
            evaluation = _evaluate_local_state(
                A_in,
                psi,
                request=request,
                grid=grid,
                kernel=kernel,
                peak_reference=peak_reference,
                intensity_before=intensity_before,
                wavelength_um=wavelength_um,
                dx_normalized=dx_normalized,
                dy_normalized=dy_normalized,
            )
            optical_seconds += perf_counter() - optical_started
            psi = evaluation.psi
            if _criteria_met(evaluation, request.solver):
                termination_reason = "residual_tolerance"
                break

            material_started = perf_counter()
            material = solve_pr_transverse_static_intensity(
                evaluation.source,
                dx_normalized=dx_normalized,
                dy_normalized=dy_normalized,
                initial_psi=psi,
                h_y=request.dielectric.h_y,
                options=request.solver.material_solver,
                xp=np,
            )
            material_seconds += perf_counter() - material_started
            summary = material.plane_summaries[0]
            material_newton += summary.newton_iterations
            material_pcg += summary.pcg_iterations
            if not material.converged:
                termination_reason = f"material_{material.status}"
                break

            direction = np.asarray(material.psi) - psi
            step_scale = 1.0
            trial_accepted = False
            candidate = evaluation
            backtracks = 0
            for backtracks in range(int(request.solver.max_backtracks) + 1):
                trial_psi = psi + step_scale * direction
                optical_started = perf_counter()
                trial = _evaluate_local_state(
                    A_in,
                    trial_psi,
                    request=request,
                    grid=grid,
                    kernel=kernel,
                    peak_reference=peak_reference,
                    intensity_before=intensity_before,
                    wavelength_um=wavelength_um,
                    dx_normalized=dx_normalized,
                    dy_normalized=dy_normalized,
                )
                optical_seconds += perf_counter() - optical_started
                target_rms = (
                    1.0 - float(request.solver.armijo_fraction) * step_scale
                ) * evaluation.equilibrium_rms
                if trial.physical_state_valid and (
                    _criteria_met(trial, request.solver)
                    or (
                        math.isfinite(trial.equilibrium_rms)
                        and math.isfinite(trial.equilibrium_max)
                        and trial.equilibrium_rms <= target_rms
                        and trial.equilibrium_max <= evaluation.equilibrium_max
                    )
                ):
                    trial_accepted = True
                    candidate = trial
                    break
                step_scale *= 0.5
                if step_scale < float(request.solver.minimum_step_scale):
                    break

            total_backtracks += backtracks
            potential_update = _difference_metrics(candidate.psi, psi)
            source_update = _difference_metrics(candidate.source, evaluation.source)
            if request.solver.record_iteration_history:
                records.append(PRTransverseMarchingPicardRecord(
                    interval_index=interval_index,
                    local_iteration=local_iteration,
                    equilibrium_rms_before=evaluation.equilibrium_rms,
                    equilibrium_max_before=evaluation.equilibrium_max,
                    equilibrium_rms_after=candidate.equilibrium_rms,
                    equilibrium_max_after=candidate.equilibrium_max,
                    potential_update_rms=potential_update[0],
                    potential_update_max=potential_update[1],
                    source_update_rms=source_update[0],
                    source_update_max=source_update[1],
                    step_scale=step_scale if trial_accepted else 0.0,
                    backtracks=backtracks,
                    material_newton_iterations=summary.newton_iterations,
                    material_pcg_iterations=summary.pcg_iterations,
                    accepted=trial_accepted,
                    termination_reason=(
                        "accepted" if trial_accepted else "local_line_search_failed"
                    ),
                ))
            if not trial_accepted:
                termination_reason = "local_line_search_failed"
                break
            psi = candidate.psi
            evaluation = candidate
            last_update = potential_update
            last_source_update = source_update
            accepted_damping = step_scale
            local_iterations = local_iteration
            if _criteria_met(evaluation, request.solver):
                termination_reason = "residual_tolerance"
                break

        assert evaluation is not None
        converged_interval = (
            termination_reason == "residual_tolerance"
            and _criteria_met(evaluation, request.solver)
        )
        summaries.append(PRTransverseMarchingIntervalSummary(
            interval_index=interval_index,
            z_midpoint_um=(interval_index + 0.5) * grid.dz_um,
            converged=converged_interval,
            local_iterations=local_iterations,
            backtracks=total_backtracks,
            accepted_damping=accepted_damping,
            equilibrium_rms=evaluation.equilibrium_rms,
            equilibrium_max=evaluation.equilibrium_max,
            carrier_minimum=evaluation.carrier_minimum,
            carrier_mean_error=abs(evaluation.carrier_mean - 1.0),
            potential_gauge_error=abs(evaluation.potential_mean),
            potential_update_rms=last_update[0],
            potential_update_max=last_update[1],
            source_update_rms=last_source_update[0],
            source_update_max=last_source_update[1],
            material_newton_iterations=material_newton,
            material_pcg_iterations=material_pcg,
            termination_reason=termination_reason,
        ))
        if not converged_interval:
            failed_interval = interval_index
            break

        accepted_psi.append(evaluation.psi.copy())
        accepted_sources.append(evaluation.source.copy())
        accepted_residuals.append(evaluation.residual.copy())
        A_pre = evaluation.A_pre_scattering.copy()
        pre_scattering_outputs.append(A_pre.copy())
        A_in = A_pre.copy()
        _apply_canonical_scattering_after_slice(
            A_in,
            scattering=request.scattering,
            z_index=interval_index,
            grid=grid,
            z_length_um=request.grid.z_length_um,
            xp=np,
        )
        incoming_fields.append(A_in.copy())
        intensity_before = pr_driving_intensity(
            A_in,
            peak_intensity_reference=peak_reference,
            background_intensity=request.material.background_intensity,
            coherence_groups=request.beams.coherence_groups,
            xp=np,
        )

    completed = len(accepted_psi)
    converged = completed == grid.Nz
    status = "converged" if converged else "not_converged"
    empty_real = np.empty((0, grid.Nx, grid.Ny), dtype=np.float64)
    empty_complex = np.empty(
        (0, len(request.beams.channels), grid.Nx, grid.Ny), dtype=np.complex128
    )
    psi_stack = np.stack(accepted_psi) if accepted_psi else empty_real.copy()
    source_stack = np.stack(accepted_sources) if accepted_sources else empty_real.copy()
    residual_stack = np.stack(accepted_residuals) if accepted_residuals else empty_real.copy()
    pre_stack = (
        np.stack(pre_scattering_outputs)
        if pre_scattering_outputs
        else empty_complex.copy()
    )
    incoming_stack = np.stack(incoming_fields)
    power_initial = normalized_power(A0, grid)
    power_final = normalized_power(A_in, grid)
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
    resolved_profile = {
        "physics_profile_id": PR_FULL_TRANSVERSE_PROFILE_V1,
        "workflow": PR_TRANSVERSE_MARCHING_STATIC_WORKFLOW,
        "grid_request": asdict(request.grid),
        "beam_request": asdict(request.beams),
        "material": asdict(request.material),
        "transport": asdict(request.transport),
        "dielectric": asdict(request.dielectric),
        "boundary": asdict(request.boundary),
        "projection": asdict(request.projection),
        "solver": asdict(request.solver),
        "state_location": "interval_midpoint_piecewise_constant",
        "characteristic_wavenumber_per_um": k0,
        "dx_normalized": dx_normalized,
        "dy_normalized": dy_normalized,
    }
    if scattering_provenance is not None:
        resolved_profile["canonical_scattering"] = scattering_provenance
    diagnostics = {
        "termination_reason": (
            "all_intervals_converged"
            if converged
            else summaries[-1].termination_reason
        ),
        "failed_interval": failed_interval,
        "accepted_upstream_intervals_are_final": True,
        "authoritative_static_residual": "zero_flux_equilibrium",
        "midpoint_source": "arithmetic_entrance_pre_scattering_exit",
        "scattering_placement": "after_local_acceptance",
        "optical_power_relative_drift": (
            (power_final - power_initial) / power_initial
        ),
    }
    if scattering_provenance is not None:
        diagnostics["canonical_scattering"] = scattering_provenance
    elapsed = perf_counter() - started
    return PRTransverseMarchingStaticRunResult(
        A_initial=A0.copy(),
        A_final=A_in.copy(),
        incoming_field_stack=incoming_stack,
        pre_scattering_exit_stack=pre_stack,
        psi_accepted=psi_stack,
        source_intensity_stack=source_stack,
        equilibrium_residual_stack=residual_stack,
        converged=converged,
        completed_intervals=completed,
        failed_interval=failed_interval,
        interval_summaries=tuple(summaries),
        iteration_records=tuple(records),
        power_initial=power_initial,
        power_final=power_final,
        grid_summary=grid.summary(),
        launch_summary=launch.summary(),
        backend_summary={
            "backend": "numpy",
            "precision": "float64",
            "real_dtype": "float64",
            "complex_dtype": "complex128",
        },
        resolved_profile=resolved_profile,
        diagnostics=diagnostics,
        timing={
            "total_seconds": elapsed,
            "optical_slice_seconds": optical_seconds,
            "material_solve_seconds": material_seconds,
        },
        status=status,
    )


__all__ = [
    "PR_TRANSVERSE_MARCHING_STATIC_WORKFLOW",
    "PRTransverseMarchingIntervalSummary",
    "PRTransverseMarchingPicardRecord",
    "PRTransverseMarchingStaticOptions",
    "PRTransverseMarchingStaticRunRequest",
    "PRTransverseMarchingStaticRunResult",
    "run_pr_transverse_static_marching",
]
