"""LC-owned experiment request restoration for the LC application."""

from __future__ import annotations

from lcprop.adapters.launchplane import beam_stack_to_launchplane
from lcprop.lc.requests import (
    OutputOptions,
    RuntimeOptions,
    StaticRunRequest,
    StaticSolverOptions,
    StaticWorkflowOptions,
    TimeDependentRunRequest,
    TimeDependentSolverOptions,
)
from lcprop.lc.specs import BiasSpec, LCMaterial


LC_STATIC_WORKFLOW_ID = "static"
LC_TIMEDEPENDENT_WORKFLOW_ID = "timedependent"
LC_STATIC_EXPERIMENT = "Static propagation"
LC_TIMEDEPENDENT_EXPERIMENT = "Time-dependent propagation"


def workflow_id_for_lc_request(request) -> str:
    if isinstance(request, StaticRunRequest):
        return LC_STATIC_WORKFLOW_ID
    if isinstance(request, TimeDependentRunRequest):
        return LC_TIMEDEPENDENT_WORKFLOW_ID
    raise TypeError("unsupported LC GUI request type")


def experiment_name_for_workflow(workflow_id: str) -> str:
    if workflow_id == LC_STATIC_WORKFLOW_ID:
        return LC_STATIC_EXPERIMENT
    if workflow_id == LC_TIMEDEPENDENT_WORKFLOW_ID:
        return LC_TIMEDEPENDENT_EXPERIMENT
    raise ValueError(f"unsupported LC GUI workflow: {workflow_id!r}")


def validate_lc_gui_request_representable(request) -> None:
    """Reject valid headless settings for which the LC GUI has no controls."""

    workflow_id_for_lc_request(request)
    request.grid.validate()
    request.material.validate()
    request.bias.validate()
    request.beams.validate()
    if request.material.name != LCMaterial().name:
        raise ValueError("LC GUI cannot represent a non-default material name")
    default_bias = BiasSpec()
    if (
        request.bias.theta_min != default_bias.theta_min
        or request.bias.theta_max != default_bias.theta_max
        or request.bias.theta_center is not None
        or request.bias.b_override is not None
    ):
        raise ValueError("LC GUI cannot represent non-default hidden bias options")
    if request.initial_A is not None or request.initial_theta is not None:
        raise ValueError(
            "LC GUI experiment files cannot represent request-owned initial "
            "arrays; use checkpoint persistence for continuation"
        )
    if request.output != OutputOptions():
        raise ValueError("LC GUI cannot represent non-default output options")
    if isinstance(request, StaticRunRequest):
        if request.runtime != RuntimeOptions():
            raise ValueError("LC GUI cannot represent these runtime options")
        strategy = request.solver.workflow.strategy
        if strategy == "fixed_theta":
            workflow = StaticWorkflowOptions(
                strategy="fixed_theta",
                theta_solver="none",
                optics_solver="splitstep",
                coupling="frozen",
            )
        elif strategy == "local_self_consistent":
            workflow = StaticWorkflowOptions(
                strategy="local_self_consistent",
                theta_solver="picard_cn",
                optics_solver="splitstep",
                coupling="self_consistent",
            )
        else:
            raise ValueError(
                f"unsupported LC GUI static strategy: {strategy!r}"
            )
        represented_solver = StaticSolverOptions(
            workflow=workflow,
            max_iterations=request.solver.max_iterations,
            static_max_coupled_passes=request.solver.max_iterations,
        )
        if request.solver != represented_solver:
            raise ValueError("LC GUI cannot represent these static solver options")
    else:
        if request.runtime != RuntimeOptions(precision="float64"):
            raise ValueError("LC GUI cannot represent these TD runtime options")
        represented_solver = TimeDependentSolverOptions(
            Nt=request.solver.Nt,
            dt=request.solver.dt,
        )
        if request.solver != represented_solver:
            raise ValueError(
                "LC GUI cannot represent these time-dependent solver options"
            )
        if not float(request.solver.dt * 1e6).is_integer():
            raise ValueError("LC GUI TD dt must be an integer multiple of 1e-6")


def apply_lc_request(
    request,
    *,
    workflow_id: str,
    physics_panel,
    beam_panel,
    grid_panel,
    solver_panel,
    experiment_panel,
    beam_stack_definition=None,
) -> None:
    """Populate LC-owned controls from an already decoded request."""

    validate_lc_gui_request_representable(request)
    actual_workflow = workflow_id_for_lc_request(request)
    if workflow_id != actual_workflow:
        raise ValueError(
            f"LC request type disagrees with workflow_id {workflow_id!r}"
        )
    grid_panel.set_grid(request.grid)
    physics_panel.set_material(request.material)
    physics_panel.set_bias(request.bias)
    if isinstance(request, StaticRunRequest):
        solver_panel.set_solver(request.solver)
    else:
        solver_panel.set_td_solver(request.solver)
    beam_panel.set_aperture(
        request.grid.x_aperture_um,
        request.grid.y_aperture_um,
    )
    beam_panel.set_beam_stack_definition(
        beam_stack_to_launchplane(request.beams)
        if beam_stack_definition is None
        else beam_stack_definition
    )
    experiment_panel.set_current_experiment(
        experiment_name_for_workflow(workflow_id)
    )


__all__ = [
    "LC_STATIC_EXPERIMENT",
    "LC_STATIC_WORKFLOW_ID",
    "LC_TIMEDEPENDENT_EXPERIMENT",
    "LC_TIMEDEPENDENT_WORKFLOW_ID",
    "apply_lc_request",
    "experiment_name_for_workflow",
    "validate_lc_gui_request_representable",
    "workflow_id_for_lc_request",
]
