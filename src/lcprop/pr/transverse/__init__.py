"""Full-transverse photorefractive production API.

This package implements the frozen, unbiased isotropic Profile v1 without
changing the established reduced-x A7 workflow.
"""

from lcprop.pr.transverse.operations import PR_TRANSVERSE_TIMEDEPENDENT_OPERATION
from lcprop.pr.transverse.products import pr_transverse_result_to_run_data
from lcprop.pr.transverse.specs import (
    PR_FULL_TRANSVERSE_PROFILE_V1,
    PR_TRANSVERSE_EXPLICIT_EULER_REFERENCE,
    PR_TRANSVERSE_IMEX_EULER,
    PR_TRANSVERSE_INTEGRATORS,
    PR_TRANSVERSE_TIMEDEPENDENT_WORKFLOW,
    PRTransverseBoundaryProfile,
    PRTransverseDielectricProfile,
    PRTransverseProjectionProfile,
    PRTransverseRunRequest,
    PRTransverseRunResult,
    PRTransverseSolverOptions,
    PRTransverseTransportProfile,
)
from lcprop.pr.transverse.workflow import run_pr_transverse_timedependent

__all__ = [
    "PR_FULL_TRANSVERSE_PROFILE_V1",
    "PR_TRANSVERSE_EXPLICIT_EULER_REFERENCE",
    "PR_TRANSVERSE_IMEX_EULER",
    "PR_TRANSVERSE_INTEGRATORS",
    "PR_TRANSVERSE_TIMEDEPENDENT_OPERATION",
    "PR_TRANSVERSE_TIMEDEPENDENT_WORKFLOW",
    "PRTransverseBoundaryProfile",
    "PRTransverseDielectricProfile",
    "PRTransverseProjectionProfile",
    "PRTransverseRunRequest",
    "PRTransverseRunResult",
    "PRTransverseSolverOptions",
    "PRTransverseTransportProfile",
    "pr_transverse_result_to_run_data",
    "run_pr_transverse_timedependent",
]
