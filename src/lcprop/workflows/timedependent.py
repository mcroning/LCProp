"""Time-dependent LCProp workflow."""

from __future__ import annotations

from dataclasses import replace
from time import perf_counter
from typing import Callable

import numpy as np

from lcprop.core.backend import asnumpy
from lcprop.core.execution import CancellationToken, RunProgress
from lcprop.core.requests import TimeDependentRunRequest, StaticRunRequest, StaticSolverOptions
from lcprop.core.results import TimeDependentRunResult
from lcprop.persistence.timedependent import TimeDependentCheckpoint
from lcprop.optics.launch import normalized_power, reconstructed_physical_powers_mW
from lcprop.optics.splitstep import total_intensity
from lcprop.algorithms.td_zmarch import TDZMarchControls, run_td_zmarch
from lcprop.workflows.runtime import (
    build_runtime_components,
    initial_A_field,
    make_td_optics_step,
    make_zcoupled_theta_step,
)


ProgressCallback = Callable[[RunProgress], None]


def _initial_theta_stack(request, runtime):
    initial_theta = request.initial_theta
    if initial_theta is None:
        theta_2d = runtime.bias.theta_2d.copy()
        return runtime.grid.xp.repeat(
            theta_2d[None, :, :],
            runtime.grid.Nz,
            axis=0,
        )

    theta = runtime.grid.xp.asarray(
        initial_theta,
        dtype=runtime.bias.theta_2d.dtype,
    ).copy()
    if theta.shape == runtime.bias.theta_2d.shape:
        return runtime.grid.xp.repeat(
            theta[None, :, :],
            runtime.grid.Nz,
            axis=0,
        )
    expected = (runtime.grid.Nz, runtime.grid.Nx, runtime.grid.Ny)
    if theta.shape != expected:
        raise ValueError(
            f"initial_theta shape {theta.shape} does not match "
            f"{runtime.bias.theta_2d.shape} or {expected}"
        )
    return theta


def run_timedependent(
    request: TimeDependentRunRequest,
    *,
    cancellation_token: CancellationToken | None = None,
    progress_callback: ProgressCallback | None = None,
    _completed_steps_offset: int = 0,
    _requested_steps_total: int | None = None,
    _cumulative_start_time: float = 0.0,
    _checkpoint_request: TimeDependentRunRequest | None = None,
) -> TimeDependentRunResult:
    """Run a time-dependent LC propagation workflow."""

    started_at = perf_counter()

    static_like_request = StaticRunRequest(
        grid=request.grid,
        material=request.material,
        bias=request.bias,
        beams=request.beams,
        solver=StaticSolverOptions(workflow=request.solver.workflow),
        output=request.output,
        runtime=request.runtime,
        initial_A=request.initial_A,
        initial_theta=request.initial_theta,
    )

    runtime = build_runtime_components(
        static_like_request,
        theta_dt=request.solver.dt,
        mobility=1.0,
    )

    A0 = initial_A_field(runtime)
    theta0 = _initial_theta_stack(request, runtime)

    power_initial = normalized_power(A0, runtime.grid)
    physical_power_initial_mW = float(
        reconstructed_physical_powers_mW(A0, runtime.grid, runtime.launch).sum()
    )

    optics_step = make_td_optics_step(runtime)

    def recorded_optical_state(theta_stack):
        A_probe = A0.copy()
        intensity_stack = runtime.grid.xp.empty_like(theta_stack)
        source_intensity_stack = runtime.grid.xp.empty_like(theta_stack)
        for k in range(runtime.grid.Nz):
            intensity_before = total_intensity(
                A_probe,
                coherence_groups=runtime.coherence_groups,
                xp=runtime.grid.xp,
            )
            A_probe, I_mid = optics_step(A_probe, theta_stack[k], k)
            intensity_after = total_intensity(
                A_probe,
                coherence_groups=runtime.coherence_groups,
                xp=runtime.grid.xp,
            )
            intensity_stack[k] = 0.5 * (
                intensity_before + intensity_after
            )
            source_intensity_stack[k] = I_mid
        return A_probe, intensity_stack, source_intensity_stack

    (
        A_initial_output,
        initial_intensity_stack,
        initial_source_intensity_stack,
    ) = recorded_optical_state(theta0)
    initial_output_plane_intensity = total_intensity(
        A_initial_output,
        coherence_groups=runtime.coherence_groups,
        xp=runtime.grid.xp,
    )

    theta_step = make_zcoupled_theta_step(
        runtime,
        theta_dt=request.solver.dt,
        mobility=1.0,
        gamma_z=request.solver.gamma_z,
        max_iter=request.solver.max_picard_iter,
        tol_update=request.solver.tolerance_update,
    )

    controls = TDZMarchControls(
        Nt=request.solver.Nt,
        observer_stride_z=1,
        observer_stride_t=1,
    )

    def observed_optics_step(A, theta, k):
        return optics_step(A, theta, k)

    requested_steps_total = (
        _completed_steps_offset + int(request.solver.Nt)
        if _requested_steps_total is None
        else int(_requested_steps_total)
    )
    latest_theta = theta0.copy()

    def observe_td_state(state) -> None:
        latest_theta[state["k"]] = state["theta_k"]

    def completed_step(step: int) -> None:
        if progress_callback is None:
            return
        # Reuse the same observational optical reconstruction as final-result
        # construction. It does not feed back into the accepted TD state.
        A_display, current_intensity_stack, _ = recorded_optical_state(
            latest_theta
        )
        output_plane_intensity = total_intensity(
            A_display,
            coherence_groups=runtime.coherence_groups,
            xp=runtime.grid.xp,
        )
        cumulative_step = _completed_steps_offset + step
        segment_elapsed_time = step * float(request.solver.dt)
        cumulative_time = _cumulative_start_time + segment_elapsed_time
        progress_callback(
            RunProgress(
                workflow="timedependent",
                status="running",
                completed_units=cumulative_step,
                total_units=requested_steps_total,
                current_coordinate=cumulative_time,
                coordinate_name="t",
                coordinate_unit="",
                latest_field_state={
                    "A_initial": np.asarray(asnumpy(A0)),
                    "A_current": np.asarray(asnumpy(A_display)).copy(),
                    "output_plane_intensity": np.asarray(
                        asnumpy(output_plane_intensity)
                    ).copy(),
                    "initial_output_plane_intensity": np.asarray(
                        asnumpy(initial_output_plane_intensity)
                    ),
                    "initial_intensity_stack": np.asarray(
                        asnumpy(initial_intensity_stack)
                    ),
                    "current_intensity_stack": np.asarray(
                        asnumpy(current_intensity_stack)
                    ).copy(),
                    "theta_current": np.asarray(asnumpy(latest_theta)).copy(),
                    "theta_initial": np.asarray(asnumpy(theta0)),
                    "theta_bias": np.asarray(asnumpy(runtime.bias.theta_2d)),
                    "grid_summary": runtime.grid.summary(),
                    "launch_summary": runtime.launch.summary(),
                    "current_time": cumulative_time,
                },
                checkpoint_available=True,
                message="TD step completed",
                completed_step=cumulative_step,
                total_steps=requested_steps_total,
                current_time=cumulative_time,
                elapsed_wall_time=perf_counter() - started_at,
                prior_completed_steps=_completed_steps_offset,
                segment_completed_steps=step,
                segment_total_steps=int(request.solver.Nt),
                cumulative_completed_steps=cumulative_step,
                segment_start_time=_cumulative_start_time,
                segment_elapsed_time=segment_elapsed_time,
                cumulative_time=cumulative_time,
            )
        )

    td = run_td_zmarch(
        theta0,
        A0,
        optics_step=observed_optics_step,
        theta_step=theta_step,
        controls=controls,
        observer=observe_td_state,
        should_cancel=(
            None
            if cancellation_token is None
            else cancellation_token.is_cancelled
        ),
        step_observer=completed_step,
    )

    (
        A_final,
        final_intensity_stack,
        final_source_intensity_stack,
    ) = recorded_optical_state(td.theta)

    power_final = normalized_power(A_final, runtime.grid)
    physical_power_final_mW = float(
        reconstructed_physical_powers_mW(
            A_final,
            runtime.grid,
            runtime.launch,
        ).sum()
    )

    completed_steps = _completed_steps_offset + td.steps
    segment_elapsed_time = td.steps * float(request.solver.dt)
    current_time = _cumulative_start_time + segment_elapsed_time
    status = "cancelled" if td.cancelled else "completed"
    checkpoint_request = _checkpoint_request or request
    checkpoint_request = replace(
        checkpoint_request,
        initial_A=None,
        initial_theta=None,
    )
    checkpoint = TimeDependentCheckpoint(
        request=checkpoint_request,
        theta=np.asarray(asnumpy(td.theta)).copy(),
        A0=np.asarray(asnumpy(A0)).copy(),
        completed_steps=completed_steps,
        requested_steps=requested_steps_total,
        current_time=current_time,
        grid_summary=runtime.grid.summary(),
        theta_dtype=str(td.theta.dtype),
        A0_dtype=str(A0.dtype),
        status=status,
    )

    return TimeDependentRunResult(
        A_final=A_final,
        theta_final=td.theta,
        A_initial=A0,
        theta_initial=theta0,
        initial_intensity_stack=initial_intensity_stack,
        final_intensity_stack=final_intensity_stack,
        initial_output_plane_intensity=initial_output_plane_intensity,
        initial_source_intensity_stack=initial_source_intensity_stack,
        final_source_intensity_stack=final_source_intensity_stack,
        theta_bias=runtime.bias.theta_2d,
        power_initial=power_initial,
        power_final=power_final,
        grid_summary=runtime.grid.summary(),
        launch_summary=runtime.launch.summary(),
        bias_summary=runtime.bias.summary(),
        Nt=td.steps,
        method=request.solver.workflow.strategy,
        physical_power_initial_mW=physical_power_initial_mW,
        physical_power_final_mW=physical_power_final_mW,
        status=status,
        completed_steps=completed_steps,
        requested_steps=requested_steps_total,
        current_time=current_time,
        prior_completed_steps=_completed_steps_offset,
        segment_completed_steps=td.steps,
        cumulative_completed_steps=completed_steps,
        segment_start_time=_cumulative_start_time,
        segment_elapsed_time=segment_elapsed_time,
        cumulative_time=current_time,
        checkpoint=checkpoint,
        warnings=("time-dependent run cancelled at a completed-step boundary",)
        if td.cancelled
        else (),
    )


def _validate_continuation_request(
    request: TimeDependentRunRequest,
    checkpoint: TimeDependentCheckpoint,
) -> None:
    original = checkpoint.request
    scalar_pairs = (
        (request.grid, original.grid, "grid"),
        (request.material, original.material, "material"),
        (request.bias, original.bias, "bias"),
        (request.beams, original.beams, "beams"),
        (request.runtime, original.runtime, "runtime"),
        (
            replace(request.solver, Nt=0),
            replace(original.solver, Nt=0),
            "TD solver",
        ),
    )
    for current, saved, label in scalar_pairs:
        if current != saved:
            raise ValueError(
                f"cannot continue TD checkpoint with incompatible {label}"
            )

    expected_shape = (
        int(checkpoint.grid_summary["Nz"]),
        int(checkpoint.grid_summary["Nx"]),
        int(checkpoint.grid_summary["Ny"]),
    )
    if checkpoint.theta.shape != expected_shape:
        raise ValueError("checkpoint theta shape does not match saved grid")
    if checkpoint.A0.shape[1:] != expected_shape[1:]:
        raise ValueError("checkpoint A0 shape does not match saved grid")
    expected_time = checkpoint.completed_steps * float(request.solver.dt)
    if not np.isclose(checkpoint.current_time, expected_time, rtol=1e-12, atol=1e-15):
        raise ValueError("checkpoint time is inconsistent with completed steps")


def validate_timedependent_continuation(
    request: TimeDependentRunRequest,
    checkpoint: TimeDependentCheckpoint,
) -> None:
    """Raise ``ValueError`` when ``request`` cannot resume ``checkpoint``."""
    _validate_continuation_request(request, checkpoint)


def continue_timedependent(
    request: TimeDependentRunRequest,
    checkpoint: TimeDependentCheckpoint,
    additional_steps: int,
    *,
    cancellation_token: CancellationToken | None = None,
    progress_callback: ProgressCallback | None = None,
) -> TimeDependentRunResult:
    """Continue a compatible checkpoint for ``additional_steps`` TD steps."""

    if int(additional_steps) < 0:
        raise ValueError("additional_steps must be nonnegative")
    _validate_continuation_request(request, checkpoint)
    continuation_request = replace(
        request,
        solver=replace(request.solver, Nt=int(additional_steps)),
        initial_A=checkpoint.A0,
        initial_theta=checkpoint.theta,
    )
    return run_timedependent(
        continuation_request,
        cancellation_token=cancellation_token,
        progress_callback=progress_callback,
        _completed_steps_offset=checkpoint.completed_steps,
        _requested_steps_total=checkpoint.completed_steps + int(additional_steps),
        _cumulative_start_time=checkpoint.current_time,
        _checkpoint_request=checkpoint.request,
    )


def _first_dataclass_mismatch(current, source, prefix: str) -> str | None:
    """Return the first exact configuration field that differs."""

    from dataclasses import fields, is_dataclass

    if is_dataclass(current) and is_dataclass(source):
        for item in fields(current):
            name = f"{prefix}.{item.name}" if prefix else item.name
            mismatch = _first_dataclass_mismatch(
                getattr(current, item.name), getattr(source, item.name), name
            )
            if mismatch is not None:
                return mismatch
        return None
    if isinstance(current, tuple) and isinstance(source, tuple):
        if len(current) != len(source):
            return f"{prefix}.length"
        for index, (left, right) in enumerate(zip(current, source)):
            mismatch = _first_dataclass_mismatch(
                left, right, f"{prefix}[{index}]"
            )
            if mismatch is not None:
                return mismatch
        return None
    return None if current == source else prefix


def timedependent_state_from_static_result(
    static_result,
    td_request: TimeDependentRunRequest,
) -> TimeDependentRunRequest:
    """Return a TD request initialized from a compatible completed static run.

    The current TD algorithm reconstructs the complete optical propagation from
    the launch field on every TD step. Consequently only the accepted static
    theta volume is transferred; an output-plane optical field is never treated
    as a TD optical state.
    """

    if getattr(static_result, "status", "completed") != "completed":
        raise ValueError(
            "TD initialization requires a fully completed static result; "
            "partial static results are not supported"
        )
    source_request = getattr(static_result, "request", None)
    if source_request is None:
        checkpoint = getattr(static_result, "checkpoint", None)
        source_request = None if checkpoint is None else checkpoint.request
    if source_request is None:
        raise ValueError("static result does not retain its source request")

    for field_name in ("grid", "material", "bias", "beams", "runtime"):
        mismatch = _first_dataclass_mismatch(
            getattr(td_request, field_name),
            getattr(source_request, field_name),
            field_name,
        )
        if mismatch is not None:
            raise ValueError(
                "cannot initialize TD from static result: incompatible "
                f"{mismatch}"
            )

    source_checkpoint = getattr(static_result, "checkpoint", None)
    accepted_theta = (
        source_checkpoint.theta_stack
        if source_checkpoint is not None
        else static_result.theta_final
    )
    theta = np.asarray(asnumpy(accepted_theta))
    expected = (
        int(static_result.grid_summary["Nz"]),
        int(static_result.grid_summary["Nx"]),
        int(static_result.grid_summary["Ny"]),
    )
    if theta.shape != expected:
        raise ValueError(
            "TD initialization requires a fully completed static theta volume; "
            f"expected {expected}, got {theta.shape}"
        )
    return replace(
        td_request,
        initial_A=None,
        initial_theta=theta.copy(),
    )


__all__ = [
    "ProgressCallback",
    "continue_timedependent",
    "run_timedependent",
    "timedependent_state_from_static_result",
    "validate_timedependent_continuation",
]
