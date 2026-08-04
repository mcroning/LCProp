"""Explicit PR workflow compositions for shared execution."""

from lcprop.pr.products import pr_result_to_run_data
from lcprop.pr.specs import PR_TIMEDEPENDENT_WORKFLOW
from lcprop.pr.workflow import run_pr_timedependent
from lcprop.runners.base import WorkflowOperation


PR_MATERIAL_ID = "pr"

PR_TIMEDEPENDENT_OPERATION = WorkflowOperation(
    material_id=PR_MATERIAL_ID,
    workflow_id=PR_TIMEDEPENDENT_WORKFLOW,
    run=run_pr_timedependent,
    to_run_data=pr_result_to_run_data,
)


__all__ = ["PR_MATERIAL_ID", "PR_TIMEDEPENDENT_OPERATION"]
