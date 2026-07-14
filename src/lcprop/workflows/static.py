"""Static LCProp workflow."""

from __future__ import annotations

import numpy as np

from lcprop.core.requests import StaticRunRequest
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
    linear_kernel,
    advance_slice,
    advance_slice_with_midintensity,
    total_intensity,
    weighted_theta_intensity,
)
from lcprop.algorithms.static_relax import (
    StaticRelaxControls,
    run_static_relax,
)
from lcprop.algorithms.theta_cn import prepare_cn_operator
from lcprop.algorithms.theta_cn import static_director_residual_metrics
from lcprop.algorithms.theta_picard import cn_trapezoid_picard_step


def run_static(request: StaticRunRequest) -> StaticRunResult:
    """Run a static propagation workflow."""

    request.grid.validate()
    request.material.validate()
    request.bias.validate()
    request.beams.validate()

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

    kernel = linear_kernel(
        grid.fxy2_um,
        dz=grid.dz_um,
        wavelength=wavelength_um,
        n_ref=n_ref,
        xp=grid.xp,
    )

    if request.solver.workflow.strategy == "local_self_consistent":
        result = _run_local_self_consistent_zmarch(
            request=request,
            grid=grid,
            bias=bias,
            launch=launch,
            A0=A0,
            kernel=kernel,
            wavelength_um=wavelength_um,
            n_ref=n_ref,
        )
        A = result.A
        theta = result.theta_stack
        intensity_stack = result.intensity_stack
        iteration_records = result.iteration_records
        slice_summaries = result.slice_summaries
        warnings = ()
        n_steps = result.relax_steps

    else:
        A = A0.copy()
        intensity_stack = grid.xp.empty(
            (grid.Nz, grid.Nx, grid.Ny),
            dtype=grid.real_dtype,
        )
        theta = bias.theta_2d.copy() if request.initial_theta is None else grid.xp.asarray(request.initial_theta, dtype=bias.theta_2d.dtype).copy()
        if theta.shape != bias.theta_2d.shape:
            raise ValueError(f"initial_theta shape {theta.shape} does not match bias field shape {bias.theta_2d.shape}")

        for k in range(grid.Nz):
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
                xp=grid.xp,
            )
            intensity_after = total_intensity(
                A,
                coherence_groups=launch.coherence_groups,
                xp=grid.xp,
            )
            intensity_stack[k] = 0.5 * (intensity_before + intensity_after)

        warnings = (
            "static workflow currently uses fixed prepared theta; self-consistent static solve not enabled for this method",
        )
        n_steps = grid.Nz
        iteration_records = ()
        slice_summaries = ()

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

    power_final = normalized_power(A, grid)
    physical_power_final_mW = float(
        reconstructed_physical_powers_mW(A, grid, launch).sum()
    )

    return StaticRunResult(
        A_final=A,
        theta_final=theta,
        power_initial=power_initial,
        power_final=power_final,
        grid_summary=grid.summary(),
        launch_summary=launch.summary(),
        bias_summary=bias.summary(),
        n_steps=n_steps,
        method=request.solver.workflow.strategy,
        physical_power_initial_mW=physical_power_initial_mW,
        physical_power_final_mW=physical_power_final_mW,
        A_initial=A0,
        intensity_stack=intensity_stack,
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
    )


class _StaticZMarchResult:
    """Internal result for the slice-local static z march."""

    def __init__(
        self,
        *,
        A,
        theta_stack,
        intensity_stack,
        relax_steps: int,
        iteration_records,
        slice_summaries,
    ):
        self.A = A
        self.theta_stack = theta_stack
        self.intensity_stack = intensity_stack
        self.relax_steps = int(relax_steps)
        self.iteration_records = tuple(iteration_records)
        self.slice_summaries = tuple(slice_summaries)


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
):
    """Relax theta locally and carry the optical field forward through z."""

    xp = grid.xp
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

    iteration_records: list[StaticIterationRecord] = []
    slice_summaries: list[StaticSliceSummary] = []
    relaxation_iterations = [0 for _ in range(grid.Nz)]

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

    def theta_relax(theta, intensity, outer):
        z_index = current_z_index

        def record_picard_iteration(iteration, theta_previous, theta_new):
            relaxation_iterations[z_index] += 1
            residual = static_director_residual_metrics(
                theta_new,
                intensity,
                b=b,
                bi=bi,
                dx=grid.du,
                dy=grid.dv,
                xp=xp,
            )
            delta_rms, delta_max = update_metrics(theta_new, theta_previous)
            theta_min = scalar(xp.min(theta_new))
            theta_max = scalar(xp.max(theta_new))
            intensity_peak = scalar(xp.max(intensity))
            intensity_integral = scalar(xp.sum(intensity)) * grid.dx_um * grid.dy_um
            finite = all(
                np.isfinite(value)
                for value in (
                    residual["residual_rms"],
                    residual["residual_max"],
                    delta_rms,
                    delta_max,
                    theta_min,
                    theta_max,
                    intensity_peak,
                    intensity_integral,
                )
            )
            converged = finite and (
                criteria_met(
                    residual["residual_rms"],
                    residual["residual_max"],
                    residual_rms_tol,
                    residual_max_tol,
                )
                or criteria_met(
                    delta_rms,
                    delta_max,
                    delta_rms_tol,
                    delta_max_tol,
                )
            )
            if request.solver.record_iteration_history:
                iteration_records.append(
                    StaticIterationRecord(
                        z_index=z_index,
                        z_um=float(z_index * grid.dz_um),
                        optical_pass=int(outer),
                        relax_iteration=int(iteration),
                        residual_rms=float(residual["residual_rms"]),
                        residual_max=float(residual["residual_max"]),
                        delta_theta_rms=delta_rms,
                        delta_theta_max=delta_max,
                        theta_min=theta_min,
                        theta_max=theta_max,
                        intensity_peak=intensity_peak,
                        normalized_intensity_integral=float(intensity_integral),
                        converged=bool(converged),
                    )
                )

        return cn_trapezoid_picard_step(
            theta,
            intensity,
            intensity,
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
            iteration_observer=record_picard_iteration,
            xp=xp,
        )

    controls = StaticRelaxControls(
        max_outer=request.solver.resolved_static_max_iterations,
        observer_stride=1,
    )

    theta_stack = xp.empty(
        (grid.Nz, grid.Nx, grid.Ny),
        dtype=bias.theta_2d.dtype,
    )
    intensity_stack = xp.empty(
        (grid.Nz, grid.Nx, grid.Ny),
        dtype=grid.real_dtype,
    )
    A = A0.copy()
    relax_steps = 0
    current_z_index = 0

    for k in range(grid.Nz):
        current_z_index = k
        A_slice_in = A.copy()
        intensity_in = weighted_theta_intensity(
            A_slice_in,
            launch.theta_weights,
            coherence_groups=launch.coherence_groups,
            xp=xp,
        )

        def propagate_slice(A_trial_unused, theta, outer):
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
                xp=xp,
            )
            return A_trial, I_mid

        def convergence(theta, theta_previous, info):
            residual = static_director_residual_metrics(
                theta,
                info["intensity"],
                b=b,
                bi=bi,
                dx=grid.du,
                dy=grid.dv,
                xp=xp,
            )
            delta_rms, delta_max = update_metrics(theta, theta_previous)
            theta_min = scalar(xp.min(theta))
            theta_max = scalar(xp.max(theta))
            values = (
                residual["residual_rms"],
                residual["residual_max"],
                delta_rms,
                delta_max,
                theta_min,
                theta_max,
            )
            info.update(
                residual_rms=float(residual["residual_rms"]),
                residual_max=float(residual["residual_max"]),
                delta_theta_rms=delta_rms,
                delta_theta_max=delta_max,
                theta_min=theta_min,
                theta_max=theta_max,
            )
            if not all(np.isfinite(value) for value in values):
                info["termination_reason"] = "nonfinite_value"
                info["stop"] = True
                return False
            if criteria_met(
                residual["residual_rms"],
                residual["residual_max"],
                residual_rms_tol,
                residual_max_tol,
            ):
                info["termination_reason"] = "residual_tolerance"
                return True
            if criteria_met(
                delta_rms,
                delta_max,
                delta_rms_tol,
                delta_max_tol,
            ):
                info["termination_reason"] = "update_tolerance"
                return True
            return False

        slice_result = run_static_relax(
            theta_seed,
            A_slice_in,
            intensity_in,
            theta_relax=theta_relax,
            optics_update=propagate_slice,
            controls=controls,
            convergence=convergence,
        )

        if slice_result.history:
            final_info = slice_result.history[-1]
        else:
            residual = static_director_residual_metrics(
                slice_result.theta,
                slice_result.intensity,
                b=b,
                bi=bi,
                dx=grid.du,
                dy=grid.dv,
                xp=xp,
            )
            final_info = {
                **residual,
                "delta_theta_rms": 0.0,
                "delta_theta_max": 0.0,
                "theta_min": scalar(xp.min(slice_result.theta)),
                "theta_max": scalar(xp.max(slice_result.theta)),
            }
        termination_reason = final_info.get("termination_reason")
        if termination_reason is None:
            termination_reason = "maximum_iterations"
        slice_summaries.append(
            StaticSliceSummary(
                z_index=k,
                z_um=float(k * grid.dz_um),
                optical_passes=int(slice_result.outer_steps),
                relaxation_iterations=int(relaxation_iterations[k]),
                final_residual_rms=float(final_info["residual_rms"]),
                final_residual_max=float(final_info["residual_max"]),
                final_delta_theta_rms=float(final_info["delta_theta_rms"]),
                final_delta_theta_max=float(final_info["delta_theta_max"]),
                theta_min=float(final_info["theta_min"]),
                theta_max=float(final_info["theta_max"]),
                converged=bool(slice_result.converged),
                termination_reason=str(termination_reason),
            )
        )

        theta_stack[k] = slice_result.theta
        intensity_stack[k] = 0.5 * (
            total_intensity(
                A_slice_in,
                coherence_groups=launch.coherence_groups,
                xp=xp,
            )
            + total_intensity(
                slice_result.A,
                coherence_groups=launch.coherence_groups,
                xp=xp,
            )
        )
        theta_seed = slice_result.theta.copy()
        A = slice_result.A
        relax_steps += slice_result.outer_steps

    return _StaticZMarchResult(
        A=A,
        theta_stack=theta_stack,
        intensity_stack=intensity_stack,
        relax_steps=relax_steps,
        iteration_records=iteration_records,
        slice_summaries=slice_summaries,
    )
