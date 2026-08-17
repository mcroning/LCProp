"""Explicit PR workflow compositions for shared execution."""

from lcprop.pr.products import (
    pr_result_to_run_data,
    pr_static_result_to_run_data,
)
from lcprop.pr.specs import PR_MATERIAL_ID, PR_TIMEDEPENDENT_WORKFLOW
from lcprop.pr.static_workflow import (
    PR_STATIC_WORKFLOW,
    run_pr_static,
)
from lcprop.pr.workflow import run_pr_timedependent
from lcprop.runners.base import WorkflowOperation


PR_TIMEDEPENDENT_OPERATION = WorkflowOperation(
    material_id=PR_MATERIAL_ID,
    workflow_id=PR_TIMEDEPENDENT_WORKFLOW,
    run=run_pr_timedependent,
    to_run_data=pr_result_to_run_data,
)

PR_STATIC_OPERATION = WorkflowOperation(
    material_id=PR_MATERIAL_ID,
    workflow_id=PR_STATIC_WORKFLOW,
    run=run_pr_static,
    to_run_data=pr_static_result_to_run_data,
)


__all__ = [
    "PR_MATERIAL_ID",
    "PR_STATIC_OPERATION",
    "PR_TIMEDEPENDENT_OPERATION",
]
