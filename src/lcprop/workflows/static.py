"""Static LCProp workflow."""

from __future__ import annotations

import numpy as np

from lcprop.core.requests import StaticRunRequest
from lcprop.core.results import StaticRunResult
from lcprop.core.grid import make_grid
from lcprop.core.derived import resolved_b
from lcprop.lc.coupling import resolved_bi
from lcprop.lc.bias import build_bias
from lcprop.optics.launch import build_launch, total_power
from lcprop.optics.splitstep import (
    linear_kernel,
    advance_slice,
    advance_slice_with_midintensity,
    weighted_theta_intensity,
)
from lcprop.algorithms.static_relax import (
    StaticRelaxControls,
    run_static_relax,
)
from lcprop.algorithms.theta_cn import prepare_cn_operator
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

    power_initial = total_power(A0, grid)

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
        warnings = ()
        n_steps = result.relax_steps

    else:
        A = A0.copy()
        theta = bias.theta_2d.copy() if request.initial_theta is None else grid.xp.asarray(request.initial_theta, dtype=bias.theta_2d.dtype).copy()
        if theta.shape != bias.theta_2d.shape:
            raise ValueError(f"initial_theta shape {theta.shape} does not match bias field shape {bias.theta_2d.shape}")

        for _ in range(grid.Nz):
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

        warnings = (
            "static workflow currently uses fixed prepared theta; self-consistent static solve not enabled for this method",
        )
        n_steps = grid.Nz

    power_final = total_power(A, grid)

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
        warnings=warnings,
        theta_bias=bias.theta_2d,
    )


class _StaticZMarchResult:
    """Internal result for the slice-local static z march."""

    def __init__(self, *, A, theta_stack, relax_steps: int):
        self.A = A
        self.theta_stack = theta_stack
        self.relax_steps = int(relax_steps)


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

    def theta_relax(theta, intensity, outer):
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
            xp=xp,
        )

    controls = StaticRelaxControls(
        max_outer=request.solver.max_iterations,
        observer_stride=1,
    )

    theta_stack = xp.empty(
        (grid.Nz, grid.Nx, grid.Ny),
        dtype=bias.theta_2d.dtype,
    )
    A = A0.copy()
    relax_steps = 0

    for k in range(grid.Nz):
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

        slice_result = run_static_relax(
            theta_seed,
            A_slice_in,
            intensity_in,
            theta_relax=theta_relax,
            optics_update=propagate_slice,
            controls=controls,
        )

        theta_stack[k] = slice_result.theta
        theta_seed = slice_result.theta.copy()
        A = slice_result.A
        relax_steps += slice_result.outer_steps

    return _StaticZMarchResult(
        A=A,
        theta_stack=theta_stack,
        relax_steps=relax_steps,
    )
