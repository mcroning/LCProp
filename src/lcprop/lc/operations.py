"""Explicit LC workflow compositions for shared execution."""

from dataclasses import replace

from lcprop.lc import LC_MATERIAL_ID
from lcprop.lc.products import (
    from_parameter_sweep_result,
    from_soliton_existence_result,
    from_soliton_result,
    from_static_result,
    from_timedependent_result,
)
from lcprop.runners.base import WorkflowOperation
from lcprop.lc.workflows import (
    continue_static,
    continue_timedependent,
    polish_soliton,
    run_parameter_sweep,
    run_soliton,
    run_soliton_existence,
    run_static,
    run_timedependent,
)


def run_soliton_with_optional_refinement(request, **kwargs):
    """Run the LC soliton workflow and its requested transverse refinement."""

    seed = run_soliton(request, **kwargs)
    if not request.refine_transverse or seed.status == "stopped":
        return seed

    polish_request = replace(
        request,
        theta_steps_per_outer=request.transverse_theta_steps_per_outer,
    )
    return polish_soliton(
        polish_request,
        seed,
        max_outer=request.transverse_max_outer,
        field_mix=request.transverse_field_mix,
        theta_mix=request.transverse_theta_mix,
        **kwargs,
    )

LC_STATIC_OPERATION = WorkflowOperation(
    material_id=LC_MATERIAL_ID,
    workflow_id="static",
    run=run_static,
    to_run_data=from_static_result,
)

LC_TIMEDEPENDENT_OPERATION = WorkflowOperation(
    material_id=LC_MATERIAL_ID,
    workflow_id="timedependent",
    run=run_timedependent,
    to_run_data=from_timedependent_result,
)

LC_SOLITON_OPERATION = WorkflowOperation(
    material_id=LC_MATERIAL_ID,
    workflow_id="soliton",
    run=run_soliton_with_optional_refinement,
    to_run_data=from_soliton_result,
)

LC_SOLITON_EXISTENCE_OPERATION = WorkflowOperation(
    material_id=LC_MATERIAL_ID,
    workflow_id="soliton_existence",
    run=run_soliton_existence,
    to_run_data=from_soliton_existence_result,
)

LC_PARAMETER_SWEEP_OPERATION = WorkflowOperation(
    material_id=LC_MATERIAL_ID,
    workflow_id="parameter_sweep",
    run=run_parameter_sweep,
    to_run_data=from_parameter_sweep_result,
)

# Compatibility-only continuation descriptors share the historical workflow
# IDs. They are executed explicitly rather than registered beside the primary
# operations, whose keys intentionally remain unique.
LC_CONTINUE_STATIC_OPERATION = WorkflowOperation(
    material_id=LC_MATERIAL_ID,
    workflow_id="static",
    run=continue_static,
    to_run_data=from_static_result,
)

LC_CONTINUE_TIMEDEPENDENT_OPERATION = WorkflowOperation(
    material_id=LC_MATERIAL_ID,
    workflow_id="timedependent",
    run=continue_timedependent,
    to_run_data=from_timedependent_result,
)


__all__ = [
    "LC_MATERIAL_ID",
    "LC_CONTINUE_STATIC_OPERATION",
    "LC_CONTINUE_TIMEDEPENDENT_OPERATION",
    "LC_PARAMETER_SWEEP_OPERATION",
    "LC_SOLITON_EXISTENCE_OPERATION",
    "LC_SOLITON_OPERATION",
    "LC_STATIC_OPERATION",
    "LC_TIMEDEPENDENT_OPERATION",
    "run_soliton_with_optional_refinement",
]
