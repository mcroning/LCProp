"""Explicit LC workflow compositions for shared execution."""

from lcprop.lc import LC_MATERIAL_ID
from lcprop.products.data_model import (
    from_static_result,
    from_timedependent_result,
)
from lcprop.runners.base import WorkflowOperation
from lcprop.workflows import run_static, run_timedependent

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


__all__ = [
    "LC_MATERIAL_ID",
    "LC_STATIC_OPERATION",
    "LC_TIMEDEPENDENT_OPERATION",
]
