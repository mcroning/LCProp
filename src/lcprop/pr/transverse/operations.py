"""Material-neutral execution composition for transverse PR Profile v1."""

from lcprop.pr.specs import PR_MATERIAL_ID
from lcprop.pr.transverse.products import (
    pr_transverse_result_to_run_data,
    pr_transverse_static_result_to_run_data,
)
from lcprop.pr.transverse.specs import PR_TRANSVERSE_TIMEDEPENDENT_WORKFLOW
from lcprop.pr.transverse.workflow import run_pr_transverse_timedependent
from lcprop.pr.transverse.static_workflow import (
    PR_TRANSVERSE_STATIC_WORKFLOW,
    run_pr_transverse_static,
)
from lcprop.runners.base import WorkflowOperation


PR_TRANSVERSE_TIMEDEPENDENT_OPERATION = WorkflowOperation(
    material_id=PR_MATERIAL_ID,
    workflow_id=PR_TRANSVERSE_TIMEDEPENDENT_WORKFLOW,
    run=run_pr_transverse_timedependent,
    to_run_data=pr_transverse_result_to_run_data,
)

PR_TRANSVERSE_STATIC_OPERATION = WorkflowOperation(
    material_id=PR_MATERIAL_ID,
    workflow_id=PR_TRANSVERSE_STATIC_WORKFLOW,
    run=run_pr_transverse_static,
    to_run_data=pr_transverse_static_result_to_run_data,
)


__all__ = [
    "PR_TRANSVERSE_STATIC_OPERATION",
    "PR_TRANSVERSE_TIMEDEPENDENT_OPERATION",
]
