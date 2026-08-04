"""Minimal headless time-dependent photorefractive propagation workflow."""

from __future__ import annotations

from time import perf_counter
from typing import Any, Callable

import numpy as np

from lcprop.core.backend import asnumpy, get_backend
from lcprop.core.execution import CancellationToken, RunProgress
from lcprop.core.grid import make_grid
from lcprop.optics.launch import build_launch, normalized_power
from lcprop.optics.splitstep import advance_prepared_response, linear_kernel
from lcprop.pr.evolution import (
    euler_step,
    legacy_conservative_timestep_limit,
    paper_conservative_timestep_limit,
    validate_timestep,
)
from lcprop.pr.optical_response import half_step_response_from_E
from lcprop.pr.source import (
    channel_peak_intensity_reference,
    pr_driving_intensity,
)
from lcprop.pr.specs import (
    PRRunRequest,
    PRRunResult,
    PR_TIMEDEPENDENT_WORKFLOW,
)


ProgressCallback = Callable[[RunProgress], None]


def _initial_fields(request: PRRunRequest, *, launch, grid, complex_dtype, real_dtype):
    xp = grid.xp
    if request.initial_A is None:
        A0 = launch.A0.copy()
    else:
        A0 = xp.asarray(request.initial_A, dtype=complex_dtype).copy()
        if A0.shape != launch.A0.shape:
            raise ValueError(
                f"initial_A shape {A0.shape} does not match {launch.A0.shape}"
            )

    E_shape = (grid.Nz, grid.Nx, grid.Ny)
    if request.initial_E is None:
        E0 = xp.zeros(E_shape, dtype=real_dtype)
    else:
        E0 = xp.asarray(request.initial_E, dtype=real_dtype).copy()
        if E0.shape != E_shape:
            raise ValueError(f"initial_E shape {E0.shape} does not match {E_shape}")
    return A0, E0


def _optical_pass(
    A0,
    E,
    *,
    request: PRRunRequest,
    grid,
    kernel,
    peak_reference: float,
    wavelength_um: float,
):
    xp = grid.xp
    A = A0.copy()
    source_stack = xp.empty(E.shape, dtype=grid.real_dtype)
    groups = request.beams.coherence_groups
    dz_substep = grid.dz_um / int(request.solver.optical_substeps)

    for k in range(grid.Nz):
        I_before = pr_driving_intensity(
            A,
            peak_intensity_reference=peak_reference,
            background_intensity=request.material.background_intensity,
            coherence_groups=groups,
            xp=xp,
        )
        half_response = half_step_response_from_E(
            E[k],
            dz_substep_um=dz_substep,
            wavelength_um=wavelength_um,
            interaction_length_um=request.grid.z_length_um,
            gain_length_product=request.material.gain_length_product,
            xp=xp,
        )
        advance_prepared_response(
            A,
            kernel=kernel,
            half_step_response=half_response,
            Nsub=request.solver.optical_substeps,
            xp=xp,
        )
        I_after = pr_driving_intensity(
            A,
            peak_intensity_reference=peak_reference,
            background_intensity=request.material.background_intensity,
            coherence_groups=groups,
            xp=xp,
        )
        source_stack[k] = 0.5 * (I_before + I_after)
    return A, source_stack


def run_pr_timedependent(
    request: PRRunRequest,
    *,
    cancellation_token: CancellationToken | None = None,
    progress_callback: ProgressCallback | None = None,
) -> PRRunResult:
    """Run the frozen-state PR workflow with optional execution controls.

    Cancellation is observed between complete material-time updates. Progress
    snapshots contain the latest accepted ``E`` state and a detached optical
    replay through that state; neither observation changes subsequent physics.
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
    grid = make_grid(
        request.grid,
        xp=backend.xp,
        real_dtype=backend.real_dtype,
    )
    launch = build_launch(
        request.beams,
        grid,
        complex_dtype=backend.complex_dtype,
    )
    A0, E = _initial_fields(
        request,
        launch=launch,
        grid=grid,
        complex_dtype=backend.complex_dtype,
        real_dtype=backend.real_dtype,
    )
    E_initial = E.copy()
    peak_reference = channel_peak_intensity_reference(A0, xp=grid.xp)
    timestep_limit = validate_timestep(
        request.solver.dt_normalized,
        grid,
        request.material,
    )

    wavelength_um = wavelengths[0]
    dz_substep = grid.dz_um / int(request.solver.optical_substeps)
    kernel = linear_kernel(
        grid.fxy2_um,
        dz=dz_substep,
        wavelength=wavelength_um,
        n_ref=request.material.refractive_index,
        xp=grid.xp,
    )

    requested_steps = int(request.solver.Nt)
    completed_steps = 0
    cancelled = False
    source_stack = grid.xp.empty(E.shape, dtype=grid.real_dtype)
    for step_index in range(requested_steps):
        if cancellation_token is not None and cancellation_token.is_cancelled():
            cancelled = True
            break

        _, source_stack = _optical_pass(
            A0,
            E,
            request=request,
            grid=grid,
            kernel=kernel,
            peak_reference=peak_reference,
            wavelength_um=wavelength_um,
        )
        E = euler_step(
            E,
            source_stack,
            dt_normalized=request.solver.dt_normalized,
            applied_field=request.material.applied_field,
            background_intensity=request.material.background_intensity,
            dx_normalized=(
                request.material.characteristic_wavenumber_per_um
                * grid.dx_um
            ),
            xp=grid.xp,
        )
        completed_steps = step_index + 1

        if progress_callback is not None:
            A_display, display_source_stack = _optical_pass(
                A0,
                E,
                request=request,
                grid=grid,
                kernel=kernel,
                peak_reference=peak_reference,
                wavelength_um=wavelength_um,
            )
            material_time = (
                completed_steps * float(request.solver.dt_normalized)
            )
            progress_callback(
                RunProgress(
                    workflow=PR_TIMEDEPENDENT_WORKFLOW,
                    status="running",
                    completed_units=completed_steps,
                    total_units=requested_steps,
                    current_coordinate=material_time,
                    coordinate_name="material_time",
                    coordinate_unit="normalized",
                    elapsed_wall_time=perf_counter() - started_at,
                    latest_field_state={
                        "A_initial": np.asarray(asnumpy(A0)).copy(),
                        "A_current": np.asarray(asnumpy(A_display)).copy(),
                        "E_initial": np.asarray(asnumpy(E_initial)).copy(),
                        "E_current": np.asarray(asnumpy(E)).copy(),
                        "source_intensity_stack": np.asarray(
                            asnumpy(display_source_stack)
                        ).copy(),
                        "grid_summary": grid.summary(),
                        "launch_summary": launch.summary(),
                        "material_time_normalized": material_time,
                    },
                    checkpoint_available=False,
                    message="PR material-time step completed",
                    completed_step=completed_steps,
                    total_steps=requested_steps,
                    current_time=material_time,
                    prior_completed_steps=0,
                    segment_completed_steps=completed_steps,
                    segment_total_steps=requested_steps,
                    cumulative_completed_steps=completed_steps,
                    segment_start_time=0.0,
                    segment_elapsed_time=material_time,
                    cumulative_time=material_time,
                )
            )

    A_final, source_stack = _optical_pass(
        A0,
        E,
        request=request,
        grid=grid,
        kernel=kernel,
        peak_reference=peak_reference,
        wavelength_um=wavelength_um,
    )

    status = "cancelled" if cancelled else "completed"
    return PRRunResult(
        A_initial=np.asarray(asnumpy(A0)).copy(),
        A_final=np.asarray(asnumpy(A_final)).copy(),
        E_initial=np.asarray(asnumpy(E_initial)).copy(),
        E_final=np.asarray(asnumpy(E)).copy(),
        source_intensity_stack=np.asarray(asnumpy(source_stack)).copy(),
        power_initial=normalized_power(A0, grid),
        power_final=normalized_power(A_final, grid),
        completed_steps=completed_steps,
        time_normalized=(
            completed_steps * float(request.solver.dt_normalized)
        ),
        grid_summary=grid.summary(),
        launch_summary=launch.summary(),
        status=status,
        requested_steps=requested_steps,
        diagnostics={
            "backend": backend.summary(),
            "characteristic_wavenumber_per_um": (
                request.material.characteristic_wavenumber_per_um
            ),
            "peak_intensity_reference": peak_reference,
            "conservative_dt_limit": timestep_limit,
            "paper_equation_15_dt_limit": paper_conservative_timestep_limit(
                grid,
                request.material,
            ),
            "legacy_prprop3d_dt_limit": legacy_conservative_timestep_limit(
                grid,
                request.material,
            ),
            "stepping_order": "frozen-E optical Strang pass, then synchronous E Euler update",
        },
    )


__all__ = ["run_pr_timedependent"]
