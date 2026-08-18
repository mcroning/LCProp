"""NumPy time-dependent workflow for full-transverse PR Profile v1."""

from __future__ import annotations

from dataclasses import asdict
from time import perf_counter
from typing import Callable

import numpy as np

from lcprop.core.execution import CancellationToken, RunProgress
from lcprop.core.grid import make_grid
from lcprop.optics.launch import build_launch, normalized_power
from lcprop.optics.splitstep import linear_kernel
from lcprop.pr.source import channel_peak_intensity_reference
from lcprop.pr.transverse.diagnostics import state_diagnostics
from lcprop.pr.transverse.projection import project_active_field
from lcprop.pr.transverse.specs import (
    PR_FULL_TRANSVERSE_PROFILE_V1,
    PR_TRANSVERSE_EXPLICIT_EULER_REFERENCE,
    PR_TRANSVERSE_IMEX_EULER,
    PR_TRANSVERSE_TIMEDEPENDENT_WORKFLOW,
    PRTransverseRunRequest,
    PRTransverseRunResult,
)
from lcprop.pr.transverse.transport import (
    explicit_euler_step,
    imex_euler_step,
    state_from_potential,
)
from lcprop.pr.workflow import advance_pr_slice_with_midpoint_source


ProgressCallback = Callable[[RunProgress], None]


class _CancellationRequested(Exception):
    def __init__(self, stage: str):
        super().__init__(stage)
        self.stage = stage


def _check_cancel(token: CancellationToken | None, stage: str) -> None:
    if token is not None and token.is_cancelled():
        raise _CancellationRequested(stage)


def _validate_request(request: PRTransverseRunRequest) -> None:
    request.grid.validate()
    request.beams.validate()
    request.material.validate()
    request.transport.validate()
    request.dielectric.validate()
    request.boundary.validate()
    request.projection.validate()
    request.solver.validate()
    request.backend.validate()
    if request.backend.backend != "numpy":
        raise ValueError("Profile v1 production reference currently requires backend='numpy'")
    if float(request.material.applied_field) != 0.0:
        raise ValueError("Profile v1 requires material.applied_field=0")


def _initial_fields(request, *, launch, grid, complex_dtype, real_dtype):
    if request.initial_A is None:
        A0 = launch.A0.copy()
    else:
        A0 = np.asarray(request.initial_A, dtype=complex_dtype).copy()
        if A0.shape != launch.A0.shape:
            raise ValueError(f"initial_A shape {A0.shape} does not match {launch.A0.shape}")
        if not np.all(np.isfinite(A0)):
            raise ValueError("initial_A must contain only finite values")
    expected = (grid.Nz, grid.Nx, grid.Ny)
    if request.initial_psi is None:
        psi = np.zeros(expected, dtype=real_dtype)
    else:
        psi = np.asarray(request.initial_psi, dtype=real_dtype).copy()
        if psi.shape != expected:
            raise ValueError(f"initial_psi shape {psi.shape} does not match {expected}")
        if not np.all(np.isfinite(psi)):
            raise ValueError("initial_psi must contain only finite values")
    psi -= np.mean(psi, axis=(-2, -1), keepdims=True)
    return A0, psi


def _optical_pass(
    A0,
    psi,
    *,
    request,
    grid,
    kernel,
    peak_reference,
    wavelength_um,
    dx_normalized,
    dy_normalized,
    cancellation_token=None,
):
    A = A0.copy()
    source = np.empty(psi.shape, dtype=grid.real_dtype)
    intensity_before = None
    state = state_from_potential(
        psi,
        dx_normalized=dx_normalized,
        dy_normalized=dy_normalized,
        h_y=request.dielectric.h_y,
        applied_field_x=request.boundary.applied_field_x,
    )
    active = project_active_field(
        state.E_x, state.E_y, profile=request.projection
    )
    for z_index in range(grid.Nz):
        _check_cancel(cancellation_token, "material_source_optical_z_march")
        A, source[z_index], intensity_before = advance_pr_slice_with_midpoint_source(
            A,
            active[z_index],
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
            _return_exit_intensity=True,
        )
    return A, source


def run_pr_transverse_timedependent(
    request: PRTransverseRunRequest,
    *,
    cancellation_token: CancellationToken | None = None,
    progress_callback: ProgressCallback | None = None,
) -> PRTransverseRunResult:
    """Run the first NumPy full-transverse PR reference workflow.

    A material candidate becomes physical only after its complete source pass
    and finite Euler update. Cancellation during either discardable stage
    returns the preceding accepted ψ state, followed by a consistent replay.
    """

    started = perf_counter()
    _validate_request(request)
    real_dtype = np.float32 if request.backend.precision == "float32" else np.float64
    complex_dtype = np.complex64 if real_dtype is np.float32 else np.complex128
    grid = make_grid(request.grid, xp=np, real_dtype=real_dtype)
    launch = build_launch(request.beams, grid, complex_dtype=complex_dtype)
    A0, psi = _initial_fields(
        request,
        launch=launch,
        grid=grid,
        complex_dtype=complex_dtype,
        real_dtype=real_dtype,
    )
    psi_initial = psi.copy()
    wavelengths = tuple(float(ch.wavelength_um) for ch in request.beams.channels)
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
    initial_state = state_from_potential(
        psi,
        dx_normalized=dx_normalized,
        dy_normalized=dy_normalized,
        h_y=request.dielectric.h_y,
    )
    initial_carrier = np.sum(initial_state.carrier_density, axis=(-2, -1))
    max_carrier_drift = 0.0
    completed_steps = 0
    cancelled = False
    cancellation_stage = None

    for step_index in range(int(request.solver.Nt)):
        try:
            _check_cancel(cancellation_token, "material_step_boundary")
            _, source = _optical_pass(
                A0,
                psi,
                request=request,
                grid=grid,
                kernel=kernel,
                peak_reference=peak_reference,
                wavelength_um=wavelength_um,
                dx_normalized=dx_normalized,
                dy_normalized=dy_normalized,
                cancellation_token=cancellation_token,
            )
            if request.solver.integrator == PR_TRANSVERSE_IMEX_EULER:
                step_function = imex_euler_step
            elif request.solver.integrator == PR_TRANSVERSE_EXPLICIT_EULER_REFERENCE:
                step_function = explicit_euler_step
            else:  # guarded by PRTransverseSolverOptions.validate()
                raise ValueError(
                    f"unknown transverse PR integrator: {request.solver.integrator}"
                )
            candidate = step_function(
                psi,
                source,
                dt_normalized=request.solver.dt_normalized,
                dx_normalized=dx_normalized,
                dy_normalized=dy_normalized,
                m_y=request.transport.m_y,
                h_y=request.dielectric.h_y,
                applied_field_x=request.boundary.applied_field_x,
            )
            _check_cancel(cancellation_token, "after_material_candidate")
        except _CancellationRequested as exc:
            cancelled = True
            cancellation_stage = exc.stage
            break

        psi = candidate
        completed_steps = step_index + 1
        accepted_state = state_from_potential(
            psi,
            dx_normalized=dx_normalized,
            dy_normalized=dy_normalized,
            h_y=request.dielectric.h_y,
        )
        current_carrier = np.sum(accepted_state.carrier_density, axis=(-2, -1))
        max_carrier_drift = max(
            max_carrier_drift,
            float(np.max(np.abs(current_carrier - initial_carrier) / np.abs(initial_carrier))),
        )
        if progress_callback is not None:
            material_time = completed_steps * float(request.solver.dt_normalized)
            progress_callback(
                RunProgress(
                    workflow=PR_TRANSVERSE_TIMEDEPENDENT_WORKFLOW,
                    status="running",
                    completed_units=completed_steps,
                    total_units=int(request.solver.Nt),
                    current_coordinate=material_time,
                    coordinate_name="material_time",
                    coordinate_unit="normalized",
                    elapsed_wall_time=perf_counter() - started,
                    latest_field_state={"psi_current": psi.copy()},
                    message="transverse PR material-time step accepted",
                    completed_step=completed_steps,
                    total_steps=int(request.solver.Nt),
                    current_time=material_time,
                    segment_completed_steps=completed_steps,
                    segment_total_steps=int(request.solver.Nt),
                    cumulative_completed_steps=completed_steps,
                    segment_elapsed_time=material_time,
                    cumulative_time=material_time,
                )
            )

    # A complete replay through the last accepted state keeps products physical.
    A_final, _ = _optical_pass(
        A0,
        psi,
        request=request,
        grid=grid,
        kernel=kernel,
        peak_reference=peak_reference,
        wavelength_um=wavelength_um,
        dx_normalized=dx_normalized,
        dy_normalized=dy_normalized,
    )
    final_state = state_from_potential(
        psi,
        dx_normalized=dx_normalized,
        dy_normalized=dy_normalized,
        h_y=request.dielectric.h_y,
    )
    diagnostics = state_diagnostics(
        final_state,
        dx_normalized=dx_normalized,
        dy_normalized=dy_normalized,
        h_y=request.dielectric.h_y,
    )
    power_initial = normalized_power(A0, grid)
    power_final = normalized_power(A_final, grid)
    diagnostics.update(
        {
            "carrier_relative_drift_max": max_carrier_drift,
            "optical_power_relative_drift": (power_final - power_initial) / power_initial,
            "finite_optical_state": bool(np.all(np.isfinite(A_final))),
            "cancellation_observed_stage": cancellation_stage,
            "integrator_policy": (
                "production_first_order_spectral_imex"
                if request.solver.integrator == PR_TRANSVERSE_IMEX_EULER
                else "transparent_reference_not_production_default"
            ),
            "complete_final_optical_replay": True,
        }
    )
    resolved_profile = {
        "physics_profile_id": PR_FULL_TRANSVERSE_PROFILE_V1,
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
    return PRTransverseRunResult(
        A_initial=np.asarray(A0).copy(),
        A_final=np.asarray(A_final).copy(),
        psi_initial=np.asarray(psi_initial).copy(),
        psi_final=np.asarray(psi).copy(),
        power_initial=power_initial,
        power_final=power_final,
        completed_steps=completed_steps,
        time_normalized=completed_steps * float(request.solver.dt_normalized),
        grid_summary=grid.summary(),
        launch_summary=launch.summary(),
        backend_summary={
            "backend": "numpy",
            "real_dtype": str(np.dtype(real_dtype)),
            "complex_dtype": str(np.dtype(complex_dtype)),
            "is_gpu": False,
        },
        resolved_profile=resolved_profile,
        status="cancelled" if cancelled else "completed",
        requested_steps=int(request.solver.Nt),
        diagnostics=diagnostics,
    )


__all__ = ["run_pr_transverse_timedependent"]
