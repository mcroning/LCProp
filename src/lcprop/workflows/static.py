"""Static LCProp workflow."""

from __future__ import annotations

from dataclasses import replace
from time import perf_counter
from typing import Callable

import numpy as np

from lcprop.core.requests import StaticRunRequest
from lcprop.core.backend import asnumpy
from lcprop.core.execution import CancellationToken, RunProgress
from lcprop.core.results import (
    StaticIterationRecord,
    StaticRunResult,
    StaticSliceSummary,
)
from lcprop.core.grid import make_grid
from lcprop.core.derived import resolved_b
from lcprop.lc.coupling import resolved_bi
from lcprop.lc.bias import build_bias
from lcprop.optics.launch import (
    build_launch,
    normalized_power,
    reconstructed_physical_powers_mW,
)
from lcprop.optics.splitstep import (
    advance_slice,
    advance_slice_with_midintensity,
    total_intensity,
)
from lcprop.optics.substeps import build_optical_substep_kernel
from lcprop.algorithms.theta_cn import prepare_cn_operator
from lcprop.algorithms.theta_cn import static_director_residual_metrics
from lcprop.algorithms.theta_picard import cn_trapezoid_picard_step
from lcprop.persistence.static import (
    StaticCheckpoint,
    static_request_fingerprint,
    validate_static_checkpoint,
)


ProgressCallback = Callable[[RunProgress], None]


def run_static(
    request: StaticRunRequest,
    *,
    cancellation_token: CancellationToken | None = None,
    progress_callback: ProgressCallback | None = None,
    _checkpoint: StaticCheckpoint | None = None,
    _phase_screen_fn=None,
) -> StaticRunResult:
    """Run a static propagation workflow."""

    started_at = perf_counter()
    request.grid.validate()
    request.material.validate()
    request.bias.validate()
    request.beams.validate()
    request.runtime.validate()
    if request.solver.workflow.strategy == "local_self_consistent":
        if request.solver.static_max_relax_iterations < 1:
            raise ValueError("static_max_relax_iterations must be >= 1")
        if request.solver.resolved_static_max_coupled_passes < 1:
            raise ValueError("static_max_coupled_passes must be >= 1")

    grid = make_grid(request.grid, real_dtype=np.float64 if request.runtime.precision == "float64" else np.float32)
    bias = build_bias(request.bias, grid, request.material)
    launch = build_launch(request.beams, grid, complex_dtype=np.complex128 if request.runtime.precision == "float64" else np.complex64)

    A0 = launch.A0.copy() if request.initial_A is None else grid.xp.asarray(request.initial_A, dtype=launch.A0.dtype).copy()
    if A0.shape != launch.A0.shape:
        raise ValueError(f"initial_A shape {A0.shape} does not match launch field shape {launch.A0.shape}")

    power_initial = normalized_power(A0, grid)
    physical_power_initial_mW = float(
        reconstructed_physical_powers_mW(A0, grid, launch).sum()
    )

    wavelength_um = float(request.beams.channels[0].wavelength_um)
    n_ref = float(request.material.no)

    optical_substeps, kernel = build_optical_substep_kernel(
        grid.fxy2_um,
        dz_um=grid.dz_um,
        propagation_wavelength_um=wavelength_um,
        active_wavelengths_um=(
            channel.wavelength_um for channel in request.beams.channels
        ),
        n_ref=n_ref,
        enabled=request.runtime.optical_substeps_enabled,
        dn_max_est=request.runtime.optical_dn_max_est,
        max_phase_per_substep_rad=(
            request.runtime.optical_max_phase_per_substep_rad
        ),
        max_substeps=request.runtime.optical_max_substeps,
        xp=grid.xp,
    )

    if request.solver.workflow.strategy == "local_self_consistent":
        live_input_theta = None

        def emit_static_progress(
            completed, A_next, theta_first, theta_latest, intensity_latest
        ) -> None:
            nonlocal live_input_theta
            if progress_callback is None:
                return
            if live_input_theta is None:
                live_input_theta = np.asarray(asnumpy(theta_first)).copy()
            progress_callback(
                RunProgress(
                    workflow="static",
                    status="running",
                    completed_units=completed,
                    total_units=grid.Nz,
                    current_coordinate=float(completed * grid.dz_um),
                    coordinate_name="z",
                    coordinate_unit="um",
                    elapsed_wall_time=perf_counter() - started_at,
                    latest_field_state={
                        "A_initial": np.asarray(asnumpy(A0)),
                        "A_current": np.asarray(asnumpy(A_next)).copy(),
                        "theta_current": np.asarray(asnumpy(theta_latest)).copy(),
                        "theta_input": live_input_theta,
                        "theta_bias": np.asarray(asnumpy(bias.theta_2d)),
                        "grid_summary": grid.summary(),
                        "launch_summary": launch.summary(),
                        "optical_diagnostics": optical_substeps.diagnostics(),
                        "completed_slices": completed,
                    },
                    checkpoint_available=True,
                    message="Static z slice completed",
                )
            )

        result = _run_local_self_consistent_zmarch(
            request=request,
            grid=grid,
            bias=bias,
            launch=launch,
            A0=A0,
            kernel=kernel,
            wavelength_um=wavelength_um,
            n_ref=n_ref,
            optical_Nsub=optical_substeps.Nsub,
            checkpoint=_checkpoint,
            should_cancel=(
                None
                if cancellation_token is None
                else cancellation_token.is_cancelled
            ),
            phase_screen_fn=_phase_screen_fn,
            step_observer=(
                emit_static_progress if progress_callback is not None else None
            ),
        )
        A = result.A
        theta = result.theta_stack
        intensity_stack = result.intensity_stack
        theta_intensity_stack = result.theta_intensity_stack
        iteration_records = result.iteration_records
        slice_summaries = result.slice_summaries
        warnings = ()
        n_steps = result.relax_steps
        completed_slices = result.completed_slices
        cancelled = result.cancelled
        theta_seed = result.theta_seed

    else:
        if _checkpoint is None:
            A = A0.copy()
            start_slice = 0
        else:
            validate_static_checkpoint(_checkpoint)
            if static_request_fingerprint(request) != _checkpoint.request_fingerprint:
                raise ValueError("cannot continue static checkpoint with incompatible request")
            A = grid.xp.asarray(_checkpoint.A_next, dtype=A0.dtype).copy()
            start_slice = _checkpoint.next_slice_index
        intensity_stack = grid.xp.empty(
            (grid.Nz, grid.Nx, grid.Ny),
            dtype=grid.real_dtype,
        )
        theta = (
            grid.xp.asarray(_checkpoint.theta_seed, dtype=bias.theta_2d.dtype).copy()
            if _checkpoint is not None
            else bias.theta_2d.copy() if request.initial_theta is None else grid.xp.asarray(request.initial_theta, dtype=bias.theta_2d.dtype).copy()
        )
        if theta.shape != bias.theta_2d.shape:
            raise ValueError(f"initial_theta shape {theta.shape} does not match bias field shape {bias.theta_2d.shape}")

        if _checkpoint is not None:
            intensity_stack[:start_slice] = grid.xp.asarray(
                _checkpoint.intensity_stack
            )
        completed_slices = start_slice
        cancelled = False
        for k in range(start_slice, grid.Nz):
            if cancellation_token is not None and cancellation_token.is_cancelled():
                cancelled = True
                break
            intensity_before = total_intensity(
                A,
                coherence_groups=launch.coherence_groups,
                xp=grid.xp,
            )
            advance_slice(
                A,
                theta,
                kernel=kernel,
                dz=grid.dz_um,
                wavelength=wavelength_um,
                n_ref=n_ref,
                ne=request.material.ne,
                no=request.material.no,
                Nsub=optical_substeps.Nsub,
                xp=grid.xp,
            )
            intensity_after = total_intensity(
                A,
                coherence_groups=launch.coherence_groups,
                xp=grid.xp,
            )
            intensity_stack[k] = 0.5 * (intensity_before + intensity_after)
            completed_slices = k + 1
            if progress_callback is not None:
                progress_callback(
                    RunProgress(
                        workflow="static",
                        status="running",
                        completed_units=completed_slices,
                        total_units=grid.Nz,
                        current_coordinate=float(completed_slices * grid.dz_um),
                        coordinate_name="z",
                        coordinate_unit="um",
                        elapsed_wall_time=perf_counter() - started_at,
                        latest_field_state={
                            "A_initial": np.asarray(asnumpy(A0)),
                            "A_current": np.asarray(asnumpy(A)).copy(),
                            "theta_current": np.asarray(asnumpy(theta)),
                            "theta_input": np.asarray(asnumpy(theta)),
                            "theta_bias": np.asarray(asnumpy(bias.theta_2d)),
                            "grid_summary": grid.summary(),
                            "launch_summary": launch.summary(),
                            "optical_diagnostics": (
                                optical_substeps.diagnostics()
                            ),
                            "completed_slices": completed_slices,
                        },
                        checkpoint_available=True,
                        message="Static z slice completed",
                    )
                )

        intensity_stack = intensity_stack[:completed_slices]

        warnings = (
            "static workflow currently uses fixed prepared theta; self-consistent static solve not enabled for this method",
        )
        n_steps = completed_slices
        iteration_records = ()
        slice_summaries = ()
        theta_intensity_stack = None
        theta_seed = theta

    if slice_summaries:
        final_residual_rms = np.asarray(
            [item.final_residual_rms for item in slice_summaries],
            dtype=float,
        )
        final_residual_max = np.asarray(
            [item.final_residual_max for item in slice_summaries],
            dtype=float,
        )
        safe_residual_rms = np.where(
            np.isfinite(final_residual_rms),
            final_residual_rms,
            np.inf,
        )
        safe_residual_max = np.where(
            np.isfinite(final_residual_max),
            final_residual_max,
            np.inf,
        )
        worst_slice_index = int(np.argmax(safe_residual_rms))
        all_slices_converged = all(item.converged for item in slice_summaries)
        max_final_residual_rms = float(np.max(safe_residual_rms))
        median_final_residual_rms = float(np.median(safe_residual_rms))
        rms_over_z_final_residual = float(
            np.sqrt(np.mean(safe_residual_rms * safe_residual_rms))
        )
        max_final_residual_max = float(np.max(safe_residual_max))
    else:
        all_slices_converged = None
        max_final_residual_rms = None
        median_final_residual_rms = None
        rms_over_z_final_residual = None
        max_final_residual_max = None
        worst_slice_index = None

    if optical_substeps.cap_reached:
        warnings = (
            *warnings,
            "optical substep cap reached",
        )

    status = "stopped" if cancelled else "completed"
    checkpoint_request = replace(request, initial_A=None, initial_theta=None)
    theta_checkpoint_stack = (
        np.asarray(asnumpy(theta)).copy()
        if np.ndim(theta) == 3
        else np.repeat(
            np.asarray(asnumpy(theta))[None, :, :], completed_slices, axis=0
        )
    )
    checkpoint = StaticCheckpoint(
        request=checkpoint_request,
        next_slice_index=completed_slices,
        completed_slices=completed_slices,
        z_reached_um=float(completed_slices * grid.dz_um),
        A_next=np.asarray(asnumpy(A)).copy(),
        theta_seed=np.asarray(asnumpy(theta_seed)).copy(),
        theta_stack=theta_checkpoint_stack,
        intensity_stack=np.asarray(asnumpy(intensity_stack)).copy(),
        theta_intensity_stack=None
        if theta_intensity_stack is None
        else np.asarray(asnumpy(theta_intensity_stack)).copy(),
        slice_summaries=tuple(slice_summaries),
        iteration_records=tuple(iteration_records),
        relax_steps=n_steps,
        grid_summary=grid.summary(),
        launch_summary=launch.summary(),
        normalized_power_initial=float(power_initial),
        physical_power_initial_mW=float(physical_power_initial_mW),
        A_dtype=str(A.dtype),
        theta_dtype=str(theta_seed.dtype),
        status=status,
        request_fingerprint=static_request_fingerprint(checkpoint_request),
    )

    power_final = normalized_power(A, grid)
    physical_power_final_mW = float(
        reconstructed_physical_powers_mW(A, grid, launch).sum()
    )

    return StaticRunResult(
        A_final=A,
        theta_final=(
            theta_checkpoint_stack
            if cancelled and np.ndim(theta) == 2
            else theta
        ),
        power_initial=power_initial,
        power_final=power_final,
        grid_summary=grid.summary(),
        launch_summary=launch.summary(),
        bias_summary=bias.summary(),
        n_steps=n_steps,
        method=request.solver.workflow.strategy,
        physical_power_initial_mW=physical_power_initial_mW,
        physical_power_final_mW=physical_power_final_mW,
        coupling_summary={
            "bi_um2": float(resolved_bi(request.grid, request.material, request.beams)),
            **optical_substeps.diagnostics(),
        },
        A_initial=A0,
        intensity_stack=intensity_stack,
        theta_intensity_stack=theta_intensity_stack,
        iteration_records=iteration_records,
        slice_summaries=slice_summaries,
        all_slices_converged=all_slices_converged,
        max_final_residual_rms=max_final_residual_rms,
        median_final_residual_rms=median_final_residual_rms,
        rms_over_z_final_residual=rms_over_z_final_residual,
        max_final_residual_max=max_final_residual_max,
        worst_slice_index=worst_slice_index,
        warnings=warnings,
        theta_bias=bias.theta_2d,
        status=status,
        completed_slices=completed_slices,
        total_slices=grid.Nz,
        z_reached_um=float(completed_slices * grid.dz_um),
        checkpoint=checkpoint,
        request=checkpoint_request,
        provenance=optical_substeps.diagnostics(),
    )


class _StaticZMarchResult:
    """Internal result for the slice-local static z march."""

    def __init__(
        self,
        *,
        A,
        theta_stack,
        intensity_stack,
        theta_intensity_stack,
        relax_steps: int,
        iteration_records,
        slice_summaries,
        completed_slices: int,
        cancelled: bool,
        theta_seed,
    ):
        self.A = A
        self.theta_stack = theta_stack
        self.intensity_stack = intensity_stack
        self.theta_intensity_stack = theta_intensity_stack
        self.relax_steps = int(relax_steps)
        self.iteration_records = tuple(iteration_records)
        self.slice_summaries = tuple(slice_summaries)
        self.completed_slices = int(completed_slices)
        self.cancelled = bool(cancelled)
        self.theta_seed = theta_seed


def _run_local_self_consistent_zmarch(
    *,
    request: StaticRunRequest,
    grid,
    bias,
    launch,
    A0,
    kernel,
    wavelength_um: float,
    n_ref: float,
    optical_Nsub: int,
    checkpoint: StaticCheckpoint | None = None,
    should_cancel=None,
    phase_screen_fn=None,
    step_observer=None,
):
    """Relax theta locally and carry the optical field forward through z."""

    xp = grid.xp
    if checkpoint is None:
        theta0 = bias.theta_2d.copy() if request.initial_theta is None else xp.asarray(request.initial_theta, dtype=bias.theta_2d.dtype).copy()
        if theta0.shape == (grid.Nz, grid.Nx, grid.Ny):
            theta_seed = theta0[0].copy()
        elif theta0.shape == bias.theta_2d.shape:
            theta_seed = theta0.copy()
        else:
            raise ValueError(
                f"initial_theta shape {theta0.shape} does not match "
                f"{bias.theta_2d.shape} or {(grid.Nz, grid.Nx, grid.Ny)}"
            )
        start_slice = 0
        A = A0.copy()
        iteration_records: list[StaticIterationRecord] = []
        slice_summaries: list[StaticSliceSummary] = []
        relax_steps = 0
    else:
        validate_static_checkpoint(checkpoint)
        if static_request_fingerprint(request) != checkpoint.request_fingerprint:
            raise ValueError("cannot continue static checkpoint with incompatible request")
        start_slice = checkpoint.next_slice_index
        if start_slice > grid.Nz:
            raise ValueError("static checkpoint extends beyond requested grid")
        A = xp.asarray(checkpoint.A_next, dtype=A0.dtype).copy()
        theta_seed = xp.asarray(
            checkpoint.theta_seed, dtype=bias.theta_2d.dtype
        ).copy()
        iteration_records = list(checkpoint.iteration_records)
        slice_summaries = list(checkpoint.slice_summaries)
        relax_steps = checkpoint.relax_steps

    b = resolved_b(request.material, request.bias)
    bi = resolved_bi(request.grid, request.material, request.beams)

    s, off, diag, _ = prepare_cn_operator(
        dt=0.01,
        mobility=1.0,
        dx=grid.du,
        dy=grid.dv,
        Ny=grid.Ny,
        xp=xp,
        dtype=grid.real_dtype,
    )

    residual_rms_tol = request.solver.static_residual_rms_tol
    residual_max_tol = request.solver.static_residual_max_tol
    delta_rms_tol = request.solver.resolved_delta_theta_rms_tol
    delta_max_tol = request.solver.resolved_delta_theta_max_tol

    def scalar(value) -> float:
        if hasattr(xp, "asnumpy"):
            value = xp.asnumpy(value)
        return float(value)

    def update_metrics(theta_new, theta_old) -> tuple[float, float]:
        delta = theta_new - theta_old
        return (
            scalar(xp.sqrt(xp.mean(delta * delta))),
            scalar(xp.max(xp.abs(delta))),
        )

    def criteria_met(
        rms_value: float,
        max_value: float,
        rms_tol,
        max_tol,
    ) -> bool:
        checks = []
        if rms_tol is not None:
            checks.append(rms_value <= float(rms_tol))
        if max_tol is not None:
            checks.append(max_value <= float(max_tol))
        return bool(checks) and all(checks)

    theta_stack = xp.empty(
        (grid.Nz, grid.Nx, grid.Ny),
        dtype=bias.theta_2d.dtype,
    )
    intensity_stack = xp.empty(
        (grid.Nz, grid.Nx, grid.Ny),
        dtype=grid.real_dtype,
    )
    theta_intensity_stack = xp.empty(
        (grid.Nz, grid.Nx, grid.Ny),
        dtype=grid.real_dtype,
    )
    if checkpoint is not None:
        theta_stack[:start_slice] = xp.asarray(checkpoint.theta_stack)
        intensity_stack[:start_slice] = xp.asarray(checkpoint.intensity_stack)
        if checkpoint.theta_intensity_stack is not None:
            theta_intensity_stack[:start_slice] = xp.asarray(
                checkpoint.theta_intensity_stack
            )

    cancelled = False
    completed_slices = start_slice
    for k in range(start_slice, grid.Nz):
        if should_cancel is not None and should_cancel():
            cancelled = True
            break
        A_slice_in = A.copy()

        def optical_midpoint(theta):
            A_trial, _, _, I_mid = advance_slice_with_midintensity(
                A_slice_in.copy(),
                theta,
                kernel=kernel,
                dz=grid.dz_um,
                wavelength=wavelength_um,
                n_ref=n_ref,
                ne=request.material.ne,
                no=request.material.no,
                coherence_groups=launch.coherence_groups,
                theta_weights=launch.theta_weights,
                Nsub=optical_Nsub,
                xp=xp,
            )
            return A_trial, I_mid

        theta = theta_seed.copy()
        A_trial, midpoint_intensity = optical_midpoint(theta)
        total_relax_iterations = 0
        coupled_passes = 0
        converged = False
        termination_reason = "maximum_coupled_passes"
        final_delta_rms = 0.0
        final_delta_max = 0.0
        final_residual = static_director_residual_metrics(
            theta,
            midpoint_intensity,
            b=b,
            bi=bi,
            dx=grid.du,
            dy=grid.dv,
            xp=xp,
        )

        for coupled_pass in range(
            1,
            request.solver.resolved_static_max_coupled_passes + 1,
        ):
            coupled_passes = coupled_pass
            last_record_index = None
            finite = True

            for relax_iteration in range(
                1,
                int(request.solver.static_max_relax_iterations) + 1,
            ):
                theta_previous = theta.copy()
                theta = cn_trapezoid_picard_step(
                    theta,
                    midpoint_intensity,
                    midpoint_intensity,
                    b=b,
                    bi=bi,
                    dt=0.01,
                    mobility=1.0,
                    dx=grid.du,
                    dy=grid.dv,
                    s=s,
                    off=off,
                    diag=diag,
                    max_iter=4,
                    tol_update=1e-6,
                    clamp=(request.bias.theta_min, request.bias.theta_max),
                    xp=xp,
                )
                total_relax_iterations += 1
                final_delta_rms, final_delta_max = update_metrics(
                    theta,
                    theta_previous,
                )
                residual_before = static_director_residual_metrics(
                    theta,
                    midpoint_intensity,
                    b=b,
                    bi=bi,
                    dx=grid.du,
                    dy=grid.dv,
                    xp=xp,
                )
                final_residual = residual_before
                theta_min = scalar(xp.min(theta))
                theta_max = scalar(xp.max(theta))
                intensity_peak = scalar(xp.max(midpoint_intensity))
                intensity_integral = (
                    scalar(xp.sum(midpoint_intensity))
                    * grid.dx_um
                    * grid.dy_um
                )
                finite = all(
                    np.isfinite(value)
                    for value in (
                        residual_before["residual_rms"],
                        residual_before["residual_max"],
                        final_delta_rms,
                        final_delta_max,
                        theta_min,
                        theta_max,
                        intensity_peak,
                        intensity_integral,
                    )
                )
                if request.solver.record_iteration_history:
                    iteration_records.append(
                        StaticIterationRecord(
                            z_index=k,
                            z_um=float(k * grid.dz_um),
                            optical_pass=coupled_pass,
                            coupled_pass=coupled_pass,
                            relax_iteration=relax_iteration,
                            residual_rms=float(residual_before["residual_rms"]),
                            residual_max=float(residual_before["residual_max"]),
                            delta_theta_rms=final_delta_rms,
                            delta_theta_max=final_delta_max,
                            theta_min=theta_min,
                            theta_max=theta_max,
                            intensity_peak=intensity_peak,
                            normalized_intensity_integral=float(intensity_integral),
                            converged=False,
                            residual_before_refresh_rms=float(
                                residual_before["residual_rms"]
                            ),
                            residual_before_refresh_max=float(
                                residual_before["residual_max"]
                            ),
                        )
                    )
                    last_record_index = len(iteration_records) - 1

                if not finite:
                    termination_reason = "nonfinite_value"
                    break
                if criteria_met(
                    residual_before["residual_rms"],
                    residual_before["residual_max"],
                    residual_rms_tol,
                    residual_max_tol,
                ):
                    break
                if criteria_met(
                    final_delta_rms,
                    final_delta_max,
                    delta_rms_tol,
                    delta_max_tol,
                ):
                    break

            if not finite:
                break

            # Acceptance always uses an optical midpoint recomputed from the
            # newly relaxed theta. A pre-refresh residual can end the inner
            # solve, but can never accept the coupled pair.
            A_trial, midpoint_intensity = optical_midpoint(theta)
            final_residual = static_director_residual_metrics(
                theta,
                midpoint_intensity,
                b=b,
                bi=bi,
                dx=grid.du,
                dy=grid.dv,
                xp=xp,
            )
            finite = all(
                np.isfinite(value)
                for value in (
                    final_residual["residual_rms"],
                    final_residual["residual_max"],
                    scalar(xp.min(theta)),
                    scalar(xp.max(theta)),
                )
            )
            converged = finite and criteria_met(
                final_residual["residual_rms"],
                final_residual["residual_max"],
                residual_rms_tol,
                residual_max_tol,
            )
            if last_record_index is not None:
                iteration_records[last_record_index] = replace(
                    iteration_records[last_record_index],
                    converged=converged,
                    residual_after_refresh_rms=float(
                        final_residual["residual_rms"]
                    ),
                    residual_after_refresh_max=float(
                        final_residual["residual_max"]
                    ),
                )
            if not finite:
                termination_reason = "nonfinite_value"
                break
            if converged:
                termination_reason = "residual_tolerance"
                break

        theta_min = scalar(xp.min(theta))
        theta_max = scalar(xp.max(theta))
        slice_summaries.append(
            StaticSliceSummary(
                z_index=k,
                z_um=float(k * grid.dz_um),
                optical_passes=coupled_passes,
                relaxation_iterations=total_relax_iterations,
                final_residual_rms=float(final_residual["residual_rms"]),
                final_residual_max=float(final_residual["residual_max"]),
                final_delta_theta_rms=final_delta_rms,
                final_delta_theta_max=final_delta_max,
                theta_min=theta_min,
                theta_max=theta_max,
                converged=converged,
                termination_reason=termination_reason,
            )
        )

        theta_stack[k] = theta
        theta_intensity_stack[k] = midpoint_intensity
        intensity_stack[k] = 0.5 * (
            total_intensity(
                A_slice_in,
                coherence_groups=launch.coherence_groups,
                xp=xp,
            )
            + total_intensity(
                A_trial,
                coherence_groups=launch.coherence_groups,
                xp=xp,
            )
        )
        theta_seed = theta.copy()
        A = A_trial
        if phase_screen_fn is not None:
            # Experimental private hook: apply one screen only after accepting
            # the self-consistent optical/director slice. Applying it inside
            # optical_midpoint() would incorrectly draw a new layer for every
            # coupled trial within the same physical z step.
            A = phase_screen_fn(A, k)
        relax_steps += coupled_passes
        completed_slices = k + 1
        if step_observer is not None:
            step_observer(
                completed_slices,
                A,
                theta_stack[0],
                theta,
                intensity_stack[k],
            )

    theta_stack = theta_stack[:completed_slices]
    intensity_stack = intensity_stack[:completed_slices]
    theta_intensity_stack = theta_intensity_stack[:completed_slices]

    return _StaticZMarchResult(
        A=A,
        theta_stack=theta_stack,
        intensity_stack=intensity_stack,
        theta_intensity_stack=theta_intensity_stack,
        relax_steps=relax_steps,
        iteration_records=iteration_records,
        slice_summaries=slice_summaries,
        completed_slices=completed_slices,
        cancelled=cancelled,
        theta_seed=theta_seed,
    )


def validate_static_continuation(
    request: StaticRunRequest, checkpoint: StaticCheckpoint
) -> None:
    """Raise when a static checkpoint cannot resume ``request`` exactly."""

    validate_static_checkpoint(checkpoint)
    if static_request_fingerprint(request) != checkpoint.request_fingerprint:
        raise ValueError("cannot continue static checkpoint with incompatible request")


def continue_static(
    request: StaticRunRequest,
    checkpoint: StaticCheckpoint,
    *,
    cancellation_token: CancellationToken | None = None,
    progress_callback: ProgressCallback | None = None,
) -> StaticRunResult:
    """Continue from the next uncomputed static z slice."""

    validate_static_continuation(request, checkpoint)
    return run_static(
        request,
        cancellation_token=cancellation_token,
        progress_callback=progress_callback,
        _checkpoint=checkpoint,
    )


__all__ = [
    "ProgressCallback",
    "continue_static",
    "run_static",
    "validate_static_continuation",
]
