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
    bias = build_bias(request.bias, grid)
    launch = build_launch(request.beams, grid, complex_dtype=np.complex128 if request.runtime.precision == "float64" else np.complex64)

    A0 = launch.A0.copy()
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

    coherent = launch.coherence == "coherent"

    if request.solver.workflow.strategy == "local_self_consistent":
        result = _run_static_relax_mode(
            request=request,
            grid=grid,
            bias=bias,
            launch=launch,
            kernel=kernel,
            wavelength_um=wavelength_um,
            n_ref=n_ref,
            coherent=coherent,
        )
        A = result.A
        theta = result.theta
        warnings = ()
        n_steps = result.outer_steps * grid.Nz

    else:
        A = A0.copy()
        theta = bias.theta_2d

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
    )


def _run_static_relax_mode(
    *,
    request: StaticRunRequest,
    grid,
    bias,
    launch,
    kernel,
    wavelength_um: float,
    n_ref: float,
    coherent: bool,
):
    """Self-consistency loop using migrated static_relax + Picard CN step."""

    xp = grid.xp
    theta0 = bias.theta_2d.copy()
    A0 = launch.A0.copy()

    intensity0 = weighted_theta_intensity(
        A0,
        launch.theta_weights,
        coherent=coherent,
        xp=xp,
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

    def optics_update(A_unused, theta, outer):
        A = A0.copy()
        I_mid = intensity0

        for _ in range(grid.Nz):
            A, _, _, I_mid = advance_slice_with_midintensity(
                A,
                theta,
                kernel=kernel,
                dz=grid.dz_um,
                wavelength=wavelength_um,
                n_ref=n_ref,
                ne=request.material.ne,
                no=request.material.no,
                coherent=coherent,
                theta_weights=launch.theta_weights,
                xp=xp,
            )

        return A, I_mid

    controls = StaticRelaxControls(
        max_outer=request.solver.max_iterations,
        observer_stride=1,
    )

    return run_static_relax(
        theta0,
        A0,
        intensity0,
        theta_relax=theta_relax,
        optics_update=optics_update,
        controls=controls,
    )
