"""Minimal headless time-dependent photorefractive propagation workflow."""

from __future__ import annotations

from dataclasses import replace
from time import perf_counter
from typing import Any, Callable

import numpy as np

from lcprop.core.backend import asnumpy, get_backend
from lcprop.core.execution import CancellationToken, RunProgress
from lcprop.core.grid import make_grid
from lcprop.optics.launch import build_launch, normalized_power
from lcprop.optics.splitstep import advance_prepared_response, linear_kernel
from lcprop.pr.checkpoint import (
    PRTimeDependentCheckpoint,
    validate_pr_checkpoint,
    validate_pr_continuation,
)
from lcprop.pr.evolution import (
    euler_step,
    legacy_conservative_timestep_limit,
    paper_conservative_timestep_limit,
    semi_implicit_trapezoidal_step,
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
    PR_EULER_INTEGRATOR,
    PR_SEMI_IMPLICIT_INTEGRATOR,
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


def advance_pr_slice_with_midpoint_source(
    A_in,
    E_slice,
    *,
    kernel,
    optical_substeps: int,
    dz_um: float,
    wavelength_um: float,
    interaction_length_um: float,
    gain_length_product: float,
    peak_intensity_reference: float,
    background_intensity: float,
    coherence_groups,
    xp,
):
    """Advance one frozen-E PR slice and return its midpoint source.

    The input field is not modified. The returned source is the arithmetic
    mean of the normalized PR-driving intensities immediately before and
    after the prepared-response Strang slice, matching the time-dependent
    workflow's established material-source definition.
    """

    resolved_substeps = int(optical_substeps)
    if resolved_substeps < 1:
        raise ValueError("optical_substeps must be at least one")
    A_out = A_in.copy()
    I_before = pr_driving_intensity(
        A_out,
        peak_intensity_reference=peak_intensity_reference,
        background_intensity=background_intensity,
        coherence_groups=coherence_groups,
        xp=xp,
    )
    half_response = half_step_response_from_E(
        E_slice,
        dz_substep_um=float(dz_um) / resolved_substeps,
        wavelength_um=wavelength_um,
        interaction_length_um=interaction_length_um,
        gain_length_product=gain_length_product,
        xp=xp,
    )
    advance_prepared_response(
        A_out,
        kernel=kernel,
        half_step_response=half_response,
        Nsub=resolved_substeps,
        xp=xp,
    )
    I_after = pr_driving_intensity(
        A_out,
        peak_intensity_reference=peak_intensity_reference,
        background_intensity=background_intensity,
        coherence_groups=coherence_groups,
        xp=xp,
    )
    return A_out, 0.5 * (I_before + I_after)


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

    for k in range(grid.Nz):
        A, source_stack[k] = advance_pr_slice_with_midpoint_source(
            A,
            E[k],
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
    return A, source_stack


def run_pr_timedependent(
    request: PRRunRequest,
    *,
    cancellation_token: CancellationToken | None = None,
    progress_callback: ProgressCallback | None = None,
    _completed_steps_offset: int = 0,
    _requested_steps_total: int | None = None,
    _cumulative_start_time: float = 0.0,
    _checkpoint_request: PRRunRequest | None = None,
    _origin_E: Any | None = None,
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
    if _origin_E is None:
        E_initial = E.copy()
    else:
        E_initial = grid.xp.asarray(
            _origin_E,
            dtype=grid.real_dtype,
        ).copy()
        if E_initial.shape != E.shape:
            raise ValueError(
                "origin E shape does not match continuation state shape"
            )
    peak_reference = channel_peak_intensity_reference(A0, xp=grid.xp)
    timestep_limit = validate_timestep(
        request.solver.dt_normalized,
        grid,
        request.material,
        integrator=request.solver.integrator,
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

    segment_total_steps = int(request.solver.Nt)
    requested_steps = (
        int(_completed_steps_offset) + segment_total_steps
        if _requested_steps_total is None
        else int(_requested_steps_total)
    )
    completed_steps = int(_completed_steps_offset)
    checkpoint_request = _checkpoint_request or replace(
        request,
        initial_A=None,
        initial_E=None,
    )
    cancelled = False
    source_stack = grid.xp.empty(E.shape, dtype=grid.real_dtype)

    def source_intensity_for_state(candidate_E):
        _, candidate_source = _optical_pass(
            A0,
            candidate_E,
            request=request,
            grid=grid,
            kernel=kernel,
            peak_reference=peak_reference,
            wavelength_um=wavelength_um,
        )
        return candidate_source

    for step_index in range(segment_total_steps):
        if cancellation_token is not None and cancellation_token.is_cancelled():
            cancelled = True
            break

        if request.solver.integrator == PR_EULER_INTEGRATOR:
            source_stack = source_intensity_for_state(E)
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
        elif request.solver.integrator == PR_SEMI_IMPLICIT_INTEGRATOR:
            E = semi_implicit_trapezoidal_step(
                E,
                source_intensity_for_state,
                dt_normalized=request.solver.dt_normalized,
                applied_field=request.material.applied_field,
                background_intensity=request.material.background_intensity,
                dx_normalized=(
                    request.material.characteristic_wavenumber_per_um
                    * grid.dx_um
                ),
                xp=grid.xp,
            )
        else:  # guarded by PRSolverOptions.validate()
            raise ValueError(
                f"unknown PR integrator: {request.solver.integrator}"
            )
        segment_completed_steps = step_index + 1
        completed_steps = (
            int(_completed_steps_offset) + segment_completed_steps
        )

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
            segment_elapsed_time = (
                segment_completed_steps
                * float(request.solver.dt_normalized)
            )
            material_time = (
                float(_cumulative_start_time) + segment_elapsed_time
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
                    checkpoint_available=True,
                    message="PR material-time step completed",
                    completed_step=completed_steps,
                    total_steps=requested_steps,
                    current_time=material_time,
                    prior_completed_steps=int(_completed_steps_offset),
                    segment_completed_steps=segment_completed_steps,
                    segment_total_steps=segment_total_steps,
                    cumulative_completed_steps=completed_steps,
                    segment_start_time=float(_cumulative_start_time),
                    segment_elapsed_time=segment_elapsed_time,
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
    A0_host = np.asarray(asnumpy(A0)).copy()
    E_initial_host = np.asarray(asnumpy(E_initial)).copy()
    E_final_host = np.asarray(asnumpy(E)).copy()
    current_time = (
        float(_cumulative_start_time)
        + (completed_steps - int(_completed_steps_offset))
        * float(request.solver.dt_normalized)
    )
    checkpoint = PRTimeDependentCheckpoint(
        request=checkpoint_request,
        E_initial=E_initial_host.copy(),
        E_current=E_final_host.copy(),
        A0=A0_host.copy(),
        completed_steps=completed_steps,
        requested_steps=requested_steps,
        time_normalized=current_time,
        grid_summary=grid.summary(),
        E_dtype=str(E_final_host.dtype),
        A0_dtype=str(A0_host.dtype),
        status=status,
    )
    validate_pr_checkpoint(checkpoint)

    return PRRunResult(
        A_initial=A0_host,
        A_final=np.asarray(asnumpy(A_final)).copy(),
        E_initial=E_initial_host,
        E_final=E_final_host,
        source_intensity_stack=np.asarray(asnumpy(source_stack)).copy(),
        power_initial=normalized_power(A0, grid),
        power_final=normalized_power(A_final, grid),
        completed_steps=completed_steps,
        time_normalized=current_time,
        grid_summary=grid.summary(),
        launch_summary=launch.summary(),
        status=status,
        requested_steps=requested_steps,
        checkpoint=checkpoint,
        diagnostics={
            "backend": backend.summary(),
            "characteristic_wavenumber_per_um": (
                request.material.characteristic_wavenumber_per_um
            ),
            "peak_intensity_reference": peak_reference,
            "integrator": request.solver.integrator,
            "conservative_dt_limit": timestep_limit,
            "paper_equation_15_dt_limit": paper_conservative_timestep_limit(
                grid,
                request.material,
            ),
            "legacy_prprop3d_dt_limit": legacy_conservative_timestep_limit(
                grid,
                request.material,
            ),
            "stepping_order": (
                "frozen-E optical Strang pass, then synchronous E Euler update"
                if request.solver.integrator == PR_EULER_INTEGRATOR
                else (
                    "accepted/predicted frozen-E optical Strang passes with "
                    "synchronous linearly implicit trapezoidal E update"
                )
            ),
        },
    )


def continue_pr_timedependent(
    request: PRRunRequest,
    checkpoint: PRTimeDependentCheckpoint,
    additional_steps: int,
    *,
    cancellation_token: CancellationToken | None = None,
    progress_callback: ProgressCallback | None = None,
) -> PRRunResult:
    """Continue a compatible accepted PR state for additional material steps."""

    resolved_additional_steps = int(additional_steps)
    if (
        isinstance(additional_steps, bool)
        or resolved_additional_steps != additional_steps
        or resolved_additional_steps < 0
    ):
        raise ValueError("additional_steps must be a nonnegative integer")
    validate_pr_continuation(request, checkpoint)
    continuation_request = replace(
        request,
        solver=replace(request.solver, Nt=resolved_additional_steps),
        initial_A=np.asarray(checkpoint.A0).copy(),
        initial_E=np.asarray(checkpoint.E_current).copy(),
    )
    return run_pr_timedependent(
        continuation_request,
        cancellation_token=cancellation_token,
        progress_callback=progress_callback,
        _completed_steps_offset=int(checkpoint.completed_steps),
        _requested_steps_total=(
            int(checkpoint.completed_steps) + resolved_additional_steps
        ),
        _cumulative_start_time=float(checkpoint.time_normalized),
        _checkpoint_request=checkpoint.request,
        _origin_E=checkpoint.E_initial,
    )


__all__ = ["continue_pr_timedependent", "run_pr_timedependent"]
