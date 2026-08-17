"""Self-consistent static photorefractive workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from time import perf_counter
from typing import Any, Callable

import numpy as np

from lcprop.core.backend import BackendSpec, asnumpy, get_backend
from lcprop.core.beams import BeamStack
from lcprop.core.context import GridSpec
from lcprop.core.execution import CancellationToken, RunProgress
from lcprop.core.grid import make_grid
from lcprop.optics.launch import build_launch, normalized_power
from lcprop.optics.splitstep import linear_kernel
from lcprop.pr.evolution import hopping_rhs
from lcprop.pr.source import channel_peak_intensity_reference
from lcprop.pr.specs import PRMaterialSpec
from lcprop.pr.static import (
    PRStaticSolverOptions,
    solve_pr_static_intensity_batched,
)
from lcprop.pr.workflow import advance_pr_slice_with_midpoint_source


PR_STATIC_WORKFLOW = "pr_static"


ProgressCallback = Callable[[RunProgress], None]


@dataclass(frozen=True)
class PRStaticWorkflowOptions:
    """Controls for the coupled static workflow.

    ``None`` selects documented precision-aware defaults for the material,
    coupled-residual, and replay tolerances. Explicit values are preserved
    exactly.
    """

    material_solver: PRStaticSolverOptions | None = None
    max_coupled_passes: int = 20
    residual_rms_tolerance: float | None = None
    residual_max_tolerance: float | None = None
    max_backtracks: int = 16
    minimum_step_scale: float = 2.0**-16
    armijo_fraction: float = 1e-4
    optical_substeps: int = 1
    replay_rtol: float | None = None
    replay_atol: float | None = None
    record_iteration_history: bool = True

    def validate(self) -> None:
        if self.material_solver is not None:
            self.material_solver.validate()
        if int(self.max_coupled_passes) < 1:
            raise ValueError("max_coupled_passes must be at least one")
        if int(self.max_backtracks) < 0:
            raise ValueError("max_backtracks must be nonnegative")
        if int(self.optical_substeps) < 1:
            raise ValueError("optical_substeps must be at least one")
        for name in (
            "residual_rms_tolerance",
            "residual_max_tolerance",
            "minimum_step_scale",
            "replay_rtol",
            "replay_atol",
        ):
            supplied = getattr(self, name)
            if supplied is None:
                continue
            value = float(supplied)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        fraction = float(self.armijo_fraction)
        if not math.isfinite(fraction) or not 0.0 < fraction < 1.0:
            raise ValueError("armijo_fraction must be between zero and one")


@dataclass(frozen=True)
class PRStaticRunRequest:
    """Request for a headless self-consistent static PR calculation."""

    grid: GridSpec
    beams: BeamStack
    material: PRMaterialSpec = PRMaterialSpec()
    solver: PRStaticWorkflowOptions = field(
        default_factory=PRStaticWorkflowOptions
    )
    backend: BackendSpec = BackendSpec(
        backend="numpy",
        precision="float64",
        verbose=False,
    )
    initial_A: Any | None = None
    initial_E: Any | None = None


@dataclass(frozen=True)
class PRCoupledStaticIterationRecord:
    """One coupled material/optical correction at a longitudinal slice."""

    z_index: int
    coupled_pass: int
    residual_before_rms: float
    residual_before_max: float
    residual_after_rms: float
    residual_after_max: float
    delta_E_rms: float
    delta_E_max: float
    step_scale: float
    material_status: str
    accepted: bool


@dataclass(frozen=True)
class PRCoupledStaticSliceSummary:
    """Final convergence evidence for one longitudinal slice."""

    z_index: int
    z_um: float
    coupled_passes: int
    final_residual_rms: float
    final_residual_max: float
    final_delta_E_rms: float
    final_delta_E_max: float
    converged: bool
    termination_reason: str


@dataclass(frozen=True)
class PRStaticRunResult:
    """Static PR state, optical fields, and strict replay diagnostics."""

    A_initial: np.ndarray
    A_final: np.ndarray
    E_initial: np.ndarray
    E_final: np.ndarray
    source_intensity_stack: np.ndarray
    residual_stack: np.ndarray
    power_initial: float
    power_final: float
    converged: bool
    completed_slices: int
    iteration_records: tuple[PRCoupledStaticIterationRecord, ...]
    slice_summaries: tuple[PRCoupledStaticSliceSummary, ...]
    grid_summary: dict[str, Any]
    launch_summary: dict[str, Any]
    backend_summary: dict[str, Any]
    tolerance_provenance: dict[str, Any]
    replay_diagnostics: dict[str, Any]
    status: str


def _backend_scalar(value) -> float:
    return float(value.item() if hasattr(value, "item") else value)


@dataclass(frozen=True)
class _ResolvedStaticTolerances:
    precision: str
    material_solver: PRStaticSolverOptions
    material_source: str
    residual_rms_tolerance: float
    residual_rms_source: str
    residual_max_tolerance: float
    residual_max_source: str
    replay_rtol: float
    replay_rtol_source: str
    replay_atol: float
    replay_atol_source: str

    def provenance(self) -> dict[str, Any]:
        return {
            "precision": self.precision,
            "material_solver": {
                "source": self.material_source,
                "residual_rms_tolerance": float(
                    self.material_solver.residual_rms_tolerance
                ),
                "residual_max_tolerance": float(
                    self.material_solver.residual_max_tolerance
                ),
            },
            "coupled_residual_rms_tolerance": {
                "value": self.residual_rms_tolerance,
                "source": self.residual_rms_source,
            },
            "coupled_residual_max_tolerance": {
                "value": self.residual_max_tolerance,
                "source": self.residual_max_source,
            },
            "replay_rtol": {
                "value": self.replay_rtol,
                "source": self.replay_rtol_source,
            },
            "replay_atol": {
                "value": self.replay_atol,
                "source": self.replay_atol_source,
            },
        }


def _resolved_value(value, default: float) -> tuple[float, str]:
    if value is None:
        return float(default), "precision_default"
    return float(value), "explicit_override"


def _resolve_static_tolerances(
    options: PRStaticWorkflowOptions,
    *,
    real_dtype: Any,
) -> _ResolvedStaticTolerances:
    precision = str(np.dtype(real_dtype))
    if precision == "float64":
        material_rms = 1e-10
        material_max = 1e-9
        coupled_rms = 1e-8
        coupled_max = 1e-7
        replay_rtol = 1e-11
        replay_atol = 1e-12
    elif precision == "float32":
        material_rms = 2e-6
        material_max = 1e-5
        coupled_rms = 2e-6
        coupled_max = 1e-5
        replay_rtol = 2e-6
        replay_atol = 2e-7
    else:  # pragma: no cover - backend precision validation owns this case.
        raise ValueError(f"unsupported PR static precision {precision!r}")

    if options.material_solver is None:
        material_solver = PRStaticSolverOptions(
            residual_rms_tolerance=material_rms,
            residual_max_tolerance=material_max,
        )
        material_source = "precision_default"
    else:
        material_solver = options.material_solver
        material_source = "explicit_PRStaticSolverOptions"

    resolved_rms, resolved_rms_source = _resolved_value(
        options.residual_rms_tolerance,
        coupled_rms,
    )
    resolved_max, resolved_max_source = _resolved_value(
        options.residual_max_tolerance,
        coupled_max,
    )
    resolved_replay_rtol, replay_rtol_source = _resolved_value(
        options.replay_rtol,
        replay_rtol,
    )
    resolved_replay_atol, replay_atol_source = _resolved_value(
        options.replay_atol,
        replay_atol,
    )
    return _ResolvedStaticTolerances(
        precision=precision,
        material_solver=material_solver,
        material_source=material_source,
        residual_rms_tolerance=resolved_rms,
        residual_rms_source=resolved_rms_source,
        residual_max_tolerance=resolved_max,
        residual_max_source=resolved_max_source,
        replay_rtol=resolved_replay_rtol,
        replay_rtol_source=replay_rtol_source,
        replay_atol=resolved_replay_atol,
        replay_atol_source=replay_atol_source,
    )


def _array_has_true(value, *, xp: Any) -> bool:
    reduced = xp.any(value)
    return bool(reduced.item() if hasattr(reduced, "item") else reduced)


def _residual_metrics(residual, *, xp: Any) -> tuple[float, float]:
    return (
        _backend_scalar(xp.sqrt(xp.mean(residual * residual))),
        _backend_scalar(xp.max(xp.abs(residual))),
    )


def _update_metrics(new_state, old_state, *, xp: Any) -> tuple[float, float]:
    delta = new_state - old_state
    return (
        _backend_scalar(xp.sqrt(xp.mean(delta * delta))),
        _backend_scalar(xp.max(xp.abs(delta))),
    )


def _criteria_met(
    residual_rms: float,
    residual_max: float,
    tolerances: _ResolvedStaticTolerances,
) -> bool:
    return (
        residual_rms <= tolerances.residual_rms_tolerance
        and residual_max <= tolerances.residual_max_tolerance
    )


def run_pr_static(
    request: PRStaticRunRequest,
    *,
    cancellation_token: CancellationToken | None = None,
    progress_callback: ProgressCallback | None = None,
) -> PRStaticRunResult:
    """Solve the self-consistent static PR problem by a local coupled z-march.

    At each slice, the accepted incoming optical field remains fixed while a
    prescribed-intensity Newton solve supplies a block material correction.
    Every trial correction is checked against a freshly propagated midpoint
    intensity. A final independent replay verifies the assembled state.
    """

    started_at = perf_counter()
    request.grid.validate()
    request.beams.validate()
    request.material.validate()
    request.solver.validate()
    request.backend.validate()
    wavelengths = tuple(
        float(channel.wavelength_um) for channel in request.beams.channels
    )
    if any(value != wavelengths[0] for value in wavelengths[1:]):
        raise ValueError("minimal PR workflow requires one shared wavelength")

    backend = get_backend(request.backend)
    xp = backend.xp
    tolerances = _resolve_static_tolerances(
        request.solver,
        real_dtype=backend.real_dtype,
    )
    grid = make_grid(request.grid, xp=backend.xp, real_dtype=backend.real_dtype)
    launch = build_launch(
        request.beams,
        grid,
        complex_dtype=backend.complex_dtype,
    )
    if request.initial_A is None:
        A0 = launch.A0.copy()
    else:
        A0 = xp.asarray(request.initial_A, dtype=backend.complex_dtype).copy()
        if A0.shape != launch.A0.shape:
            raise ValueError(
                f"initial_A shape {A0.shape} does not match {launch.A0.shape}"
            )

    E_shape = (grid.Nz, grid.Nx, grid.Ny)
    if request.initial_E is None:
        E_initial = xp.zeros(E_shape, dtype=backend.real_dtype)
    else:
        E_initial = xp.asarray(
            request.initial_E,
            dtype=backend.real_dtype,
        ).copy()
        if E_initial.shape != E_shape:
            raise ValueError(
                f"initial_E shape {E_initial.shape} does not match {E_shape}"
            )
        if _array_has_true(~xp.isfinite(E_initial), xp=xp):
            raise ValueError("initial_E must contain only finite values")

    wavelength_um = wavelengths[0]
    kernel = linear_kernel(
        grid.fxy2_um,
        dz=grid.dz_um / int(request.solver.optical_substeps),
        wavelength=wavelength_um,
        n_ref=request.material.refractive_index,
        xp=xp,
    )
    peak_reference = channel_peak_intensity_reference(A0, xp=xp)
    dx_normalized = (
        request.material.characteristic_wavenumber_per_um * grid.dx_um
    )
    groups = request.beams.coherence_groups
    E_stack = xp.empty(E_shape, dtype=backend.real_dtype)
    source_stack = xp.empty(E_shape, dtype=backend.real_dtype)
    residual_stack = xp.empty(E_shape, dtype=backend.real_dtype)
    A = A0.copy()
    records: list[PRCoupledStaticIterationRecord] = []
    summaries: list[PRCoupledStaticSliceSummary] = []
    completed_slices = 0
    cancelled = False

    def advance_slice(A_in, state):
        return advance_pr_slice_with_midpoint_source(
            A_in,
            state,
            kernel=kernel,
            optical_substeps=request.solver.optical_substeps,
            dz_um=grid.dz_um,
            wavelength_um=wavelength_um,
            interaction_length_um=request.grid.z_length_um,
            gain_length_product=request.material.gain_length_product,
            peak_intensity_reference=peak_reference,
            background_intensity=request.material.background_intensity,
            coherence_groups=groups,
            xp=xp,
        )

    def residual_at(state, intensity):
        return hopping_rhs(
            state,
            intensity,
            applied_field=request.material.applied_field,
            background_intensity=request.material.background_intensity,
            dx_normalized=dx_normalized,
            xp=xp,
        )

    def report_completion_phase(
        phase: str,
        *,
        completed_units: int,
        total_units: int,
        message: str,
    ) -> None:
        if progress_callback is None:
            return
        progress_callback(
            RunProgress(
                workflow=PR_STATIC_WORKFLOW,
                status="running",
                completed_units=int(completed_units),
                total_units=int(total_units),
                current_coordinate=float(completed_units),
                coordinate_name="replay_slice",
                coordinate_unit="1",
                elapsed_wall_time=perf_counter() - started_at,
                message=message,
                diagnostics={"phase": phase},
            )
        )

    for k in range(grid.Nz):
        if cancellation_token is not None and cancellation_token.is_cancelled():
            cancelled = True
            break

        A_slice_in = A.copy()
        if request.initial_E is not None:
            state = E_initial[k].copy()
        elif k == 0:
            state = E_initial[0].copy()
        else:
            state = E_stack[k - 1].copy()

        A_trial, intensity = advance_slice(A_slice_in, state)
        residual = residual_at(state, intensity)
        residual_rms, residual_max = _residual_metrics(residual, xp=xp)
        converged = _criteria_met(residual_rms, residual_max, tolerances)
        termination_reason = (
            "residual_tolerance" if converged else "maximum_coupled_passes"
        )
        coupled_passes = 0
        final_delta_rms = 0.0
        final_delta_max = 0.0

        for coupled_pass in range(1, int(request.solver.max_coupled_passes) + 1):
            if converged:
                break
            coupled_passes = coupled_pass
            material_result = solve_pr_static_intensity_batched(
                intensity,
                applied_field=request.material.applied_field,
                background_intensity=request.material.background_intensity,
                dx_normalized=dx_normalized,
                initial_E=state,
                options=tolerances.material_solver,
                xp=xp,
            )
            if not material_result.converged:
                termination_reason = f"material_{material_result.status}"
                if request.solver.record_iteration_history:
                    records.append(
                        PRCoupledStaticIterationRecord(
                            z_index=k,
                            coupled_pass=coupled_pass,
                            residual_before_rms=residual_rms,
                            residual_before_max=residual_max,
                            residual_after_rms=residual_rms,
                            residual_after_max=residual_max,
                            delta_E_rms=0.0,
                            delta_E_max=0.0,
                            step_scale=0.0,
                            material_status=material_result.status,
                            accepted=False,
                        )
                    )
                break

            direction = material_result.E - state
            merit = 0.5 * residual_rms * residual_rms
            step_scale = 1.0
            accepted = False
            candidate = state
            candidate_A = A_trial
            candidate_intensity = intensity
            candidate_residual = residual
            candidate_rms = residual_rms
            candidate_max = residual_max
            for _ in range(int(request.solver.max_backtracks) + 1):
                trial_state = state + step_scale * direction
                trial_A, trial_intensity = advance_slice(
                    A_slice_in, trial_state
                )
                trial_residual = residual_at(trial_state, trial_intensity)
                trial_rms, trial_max = _residual_metrics(
                    trial_residual,
                    xp=xp,
                )
                trial_merit = 0.5 * trial_rms * trial_rms
                if (
                    not _array_has_true(~xp.isfinite(trial_state), xp=xp)
                    and not _array_has_true(~xp.isfinite(trial_A), xp=xp)
                    and math.isfinite(trial_merit)
                    and trial_merit
                    <= (1.0 - request.solver.armijo_fraction * step_scale)
                    * merit
                ):
                    accepted = True
                    candidate = trial_state
                    candidate_A = trial_A
                    candidate_intensity = trial_intensity
                    candidate_residual = trial_residual
                    candidate_rms = trial_rms
                    candidate_max = trial_max
                    break
                step_scale *= 0.5
                if step_scale < request.solver.minimum_step_scale:
                    break

            if not accepted:
                termination_reason = "coupled_line_search_failed"
                if request.solver.record_iteration_history:
                    records.append(
                        PRCoupledStaticIterationRecord(
                            z_index=k,
                            coupled_pass=coupled_pass,
                            residual_before_rms=residual_rms,
                            residual_before_max=residual_max,
                            residual_after_rms=residual_rms,
                            residual_after_max=residual_max,
                            delta_E_rms=0.0,
                            delta_E_max=0.0,
                            step_scale=0.0,
                            material_status=material_result.status,
                            accepted=False,
                        )
                    )
                break

            final_delta_rms, final_delta_max = _update_metrics(
                candidate,
                state,
                xp=xp,
            )
            if request.solver.record_iteration_history:
                records.append(
                    PRCoupledStaticIterationRecord(
                        z_index=k,
                        coupled_pass=coupled_pass,
                        residual_before_rms=residual_rms,
                        residual_before_max=residual_max,
                        residual_after_rms=candidate_rms,
                        residual_after_max=candidate_max,
                        delta_E_rms=final_delta_rms,
                        delta_E_max=final_delta_max,
                        step_scale=step_scale,
                        material_status=material_result.status,
                        accepted=True,
                    )
                )
            state = candidate
            A_trial = candidate_A
            intensity = candidate_intensity
            residual = candidate_residual
            residual_rms = candidate_rms
            residual_max = candidate_max
            converged = _criteria_met(
                residual_rms, residual_max, tolerances
            )
            if converged:
                termination_reason = "residual_tolerance"
                break

        E_stack[k] = state
        source_stack[k] = intensity
        residual_stack[k] = residual
        A = A_trial
        summaries.append(
            PRCoupledStaticSliceSummary(
                z_index=k,
                z_um=float(k * grid.dz_um),
                coupled_passes=coupled_passes,
                final_residual_rms=residual_rms,
                final_residual_max=residual_max,
                final_delta_E_rms=final_delta_rms,
                final_delta_E_max=final_delta_max,
                converged=converged,
                termination_reason=termination_reason,
            )
        )
        completed_slices = k + 1
        if progress_callback is not None:
            progress_callback(
                RunProgress(
                    workflow=PR_STATIC_WORKFLOW,
                    status="running",
                    completed_units=completed_slices,
                    total_units=grid.Nz,
                    current_coordinate=float(completed_slices * grid.dz_um),
                    coordinate_name="z",
                    coordinate_unit="um",
                    elapsed_wall_time=perf_counter() - started_at,
                    latest_field_state={
                        "A_initial": np.asarray(asnumpy(A0)).copy(),
                        "A_current": np.asarray(asnumpy(A)).copy(),
                        "E_current": np.asarray(asnumpy(state)).copy(),
                        "source_intensity_current": np.asarray(
                            asnumpy(intensity)
                        ).copy(),
                        "residual_current": np.asarray(
                            asnumpy(residual)
                        ).copy(),
                        "completed_slices": completed_slices,
                        "grid_summary": grid.summary(),
                        "launch_summary": launch.summary(),
                    },
                    checkpoint_available=False,
                    message="PR static z slice completed",
                    diagnostics={
                        "phase": "solve",
                        "converged": converged,
                        "termination_reason": termination_reason,
                        "residual_rms": residual_rms,
                        "residual_max": residual_max,
                    },
                )
            )

    sequential_A = A.copy()
    replay_A = A0.copy()
    completed_E = E_stack[:completed_slices]
    completed_source = source_stack[:completed_slices]
    completed_residual = residual_stack[:completed_slices]
    replay_source = xp.empty_like(completed_source)
    replay_residual = xp.empty_like(completed_residual)
    report_completion_phase(
        "replay",
        completed_units=0,
        total_units=completed_slices,
        message=(
            "Validating static solution: independent replay "
            f"0/{completed_slices}"
        ),
    )
    replay_progress_cadence = max(1, math.ceil(completed_slices / 100))
    for k in range(completed_slices):
        replay_A, replay_source[k] = advance_slice(replay_A, E_stack[k])
        replay_residual[k] = residual_at(E_stack[k], replay_source[k])
        replay_completed = k + 1
        if (
            replay_completed % replay_progress_cadence == 0
            or replay_completed == completed_slices
        ):
            report_completion_phase(
                "replay",
                completed_units=replay_completed,
                total_units=completed_slices,
                message=(
                    "Validating static solution: independent replay "
                    f"{replay_completed}/{completed_slices}"
                ),
            )

    report_completion_phase(
        "replay_validation",
        completed_units=completed_slices,
        total_units=completed_slices,
        message="Validating replay consistency...",
    )
    if completed_slices:
        replay_rms, replay_max = _residual_metrics(replay_residual, xp=xp)
    else:
        replay_rms = 0.0
        replay_max = 0.0
    field_max_abs = _backend_scalar(xp.max(xp.abs(replay_A - sequential_A)))
    if completed_slices:
        source_max_abs = _backend_scalar(
            xp.max(xp.abs(replay_source - completed_source))
        )
        residual_max_abs = _backend_scalar(
            xp.max(xp.abs(replay_residual - completed_residual))
        )
    else:
        source_max_abs = 0.0
        residual_max_abs = 0.0
    field_consistent = not _array_has_true(
        ~xp.isclose(
            replay_A,
            sequential_A,
            rtol=tolerances.replay_rtol,
            atol=tolerances.replay_atol,
        ),
        xp=xp,
    )
    source_consistent = not _array_has_true(
        ~xp.isclose(
            replay_source,
            completed_source,
            rtol=tolerances.replay_rtol,
            atol=tolerances.replay_atol,
        ),
        xp=xp,
    )
    residual_consistent = not _array_has_true(
        ~xp.isclose(
            replay_residual,
            completed_residual,
            rtol=tolerances.replay_rtol,
            atol=tolerances.replay_atol,
        ),
        xp=xp,
    )
    replay_converged = bool(
        completed_slices == grid.Nz
        and _criteria_met(replay_rms, replay_max, tolerances)
    )
    converged = bool(
        not cancelled
        and completed_slices == grid.Nz
        and all(summary.converged for summary in summaries)
        and replay_converged
        and field_consistent
        and source_consistent
        and residual_consistent
    )

    report_completion_phase(
        "scientific_result",
        completed_units=completed_slices,
        total_units=completed_slices,
        message="Preparing scientific results...",
    )

    return PRStaticRunResult(
        A_initial=np.asarray(asnumpy(A0)).copy(),
        A_final=np.asarray(asnumpy(replay_A)).copy(),
        E_initial=np.asarray(asnumpy(E_initial[:completed_slices])).copy(),
        E_final=np.asarray(asnumpy(completed_E)).copy(),
        source_intensity_stack=np.asarray(asnumpy(replay_source)).copy(),
        residual_stack=np.asarray(asnumpy(replay_residual)).copy(),
        power_initial=normalized_power(A0, grid),
        power_final=normalized_power(replay_A, grid),
        converged=converged,
        completed_slices=completed_slices,
        iteration_records=tuple(records),
        slice_summaries=tuple(summaries),
        grid_summary=grid.summary(),
        launch_summary=launch.summary(),
        backend_summary=backend.summary(),
        tolerance_provenance=tolerances.provenance(),
        replay_diagnostics={
            "performed_slices": completed_slices,
            "requested_slices": grid.Nz,
            "residual_rms": replay_rms,
            "residual_max": replay_max,
            "field_max_abs_difference": field_max_abs,
            "source_max_abs_difference": source_max_abs,
            "residual_max_abs_difference": residual_max_abs,
            "field_consistent": field_consistent,
            "source_consistent": source_consistent,
            "residual_consistent": residual_consistent,
            "residual_converged": replay_converged,
        },
        status=(
            "cancelled"
            if cancelled
            else ("converged" if converged else "not_converged")
        ),
    )


__all__ = [
    "PRCoupledStaticIterationRecord",
    "PRCoupledStaticSliceSummary",
    "PRStaticRunRequest",
    "PRStaticRunResult",
    "PRStaticWorkflowOptions",
    "PR_STATIC_WORKFLOW",
    "run_pr_static",
]
