"""Time-dependent LCProp workflow."""

from __future__ import annotations

from lcprop.core.requests import TimeDependentRunRequest, StaticRunRequest, StaticSolverOptions
from lcprop.core.results import TimeDependentRunResult
from lcprop.optics.launch import total_power
from lcprop.algorithms.td_zmarch import TDZMarchControls, run_td_zmarch
from lcprop.workflows.runtime import (
    build_runtime_components,
    make_td_optics_step,
    make_zcoupled_theta_step,
)


def run_timedependent(request: TimeDependentRunRequest) -> TimeDependentRunResult:
    """Run a time-dependent LC propagation workflow."""

    static_like_request = StaticRunRequest(
        grid=request.grid,
        material=request.material,
        bias=request.bias,
        beams=request.beams,
        solver=StaticSolverOptions(workflow=request.solver.workflow),
        output=request.output,
        runtime=request.runtime,
    )

    runtime = build_runtime_components(
        static_like_request,
        theta_dt=request.solver.dt,
        mobility=1.0,
    )

    A0 = runtime.launch.A0.copy()
    theta0 = runtime.bias.theta_stack.copy()

    power_initial = total_power(A0, runtime.grid)

    optics_step = make_td_optics_step(runtime)

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

    td = run_td_zmarch(
        theta0,
        A0,
        optics_step=optics_step,
        theta_step=theta_step,
        controls=controls,
    )

    power_final = total_power(td.A_last, runtime.grid)

    return TimeDependentRunResult(
        A_final=td.A_last,
        theta_final=td.theta,
        theta_bias=runtime.bias.theta_2d,
        power_initial=power_initial,
        power_final=power_final,
        grid_summary=runtime.grid.summary(),
        launch_summary=runtime.launch.summary(),
        bias_summary=runtime.bias.summary(),
        Nt=td.steps,
        method=request.solver.workflow.strategy,
        warnings=(),
    )
