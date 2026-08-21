"""Full-transverse photorefractive production API.

This package implements the frozen, unbiased isotropic Profile v1 without
changing the established reduced-x A7 workflow.
"""

from lcprop.pr.transverse.operations import (
    PR_TRANSVERSE_STATIC_OPERATION,
    PR_TRANSVERSE_TIMEDEPENDENT_OPERATION,
)
from lcprop.pr.transverse.products import (
    pr_transverse_result_to_run_data,
    pr_transverse_static_result_to_run_data,
)
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
from lcprop.pr.transverse.static import (
    PRTransverseDiscreteStaticCorrectorOptions,
    PRTransverseDiscreteStaticMaterialResult,
    PRTransverseDiscreteStaticNewtonRecord,
    PRTransverseDiscreteStaticPlaneSummary,
    PRTransverseStaticMaterialResult,
    PRTransverseStaticMaterialSolverOptions,
    PRTransverseStaticNewtonRecord,
    PRTransverseStaticPlaneSummary,
    derivative_null_residual,
    project_production_resolved_modes,
    production_steady_jvp,
    production_steady_residual,
    solve_pr_transverse_discrete_static_intensity,
    solve_pr_transverse_static_intensity,
    static_equilibrium_residual,
)
from lcprop.pr.transverse.static_workflow import (
    PR_TRANSVERSE_STATIC_WORKFLOW,
    PRTransverseStaticCoupledRecord,
    PRTransverseStaticDiscreteIterationRecord,
    PRTransverseStaticMaterialIterationRecord,
    PRTransverseStaticRunRequest,
    PRTransverseStaticRunResult,
    PRTransverseStaticWorkflowOptions,
    run_pr_transverse_static,
)

__all__ = [
    "PR_FULL_TRANSVERSE_PROFILE_V1",
    "PR_TRANSVERSE_EXPLICIT_EULER_REFERENCE",
    "PR_TRANSVERSE_IMEX_EULER",
    "PR_TRANSVERSE_INTEGRATORS",
    "PR_TRANSVERSE_STATIC_OPERATION",
    "PR_TRANSVERSE_STATIC_WORKFLOW",
    "PR_TRANSVERSE_TIMEDEPENDENT_OPERATION",
    "PR_TRANSVERSE_TIMEDEPENDENT_WORKFLOW",
    "PRTransverseBoundaryProfile",
    "PRTransverseDielectricProfile",
    "PRTransverseProjectionProfile",
    "PRTransverseRunRequest",
    "PRTransverseRunResult",
    "PRTransverseSolverOptions",
    "PRTransverseDiscreteStaticCorrectorOptions",
    "PRTransverseDiscreteStaticMaterialResult",
    "PRTransverseDiscreteStaticNewtonRecord",
    "PRTransverseDiscreteStaticPlaneSummary",
    "PRTransverseStaticCoupledRecord",
    "PRTransverseStaticDiscreteIterationRecord",
    "PRTransverseStaticMaterialResult",
    "PRTransverseStaticMaterialIterationRecord",
    "PRTransverseStaticMaterialSolverOptions",
    "PRTransverseStaticNewtonRecord",
    "PRTransverseStaticPlaneSummary",
    "PRTransverseStaticRunRequest",
    "PRTransverseStaticRunResult",
    "PRTransverseStaticWorkflowOptions",
    "PRTransverseTransportProfile",
    "derivative_null_residual",
    "project_production_resolved_modes",
    "production_steady_jvp",
    "production_steady_residual",
    "pr_transverse_result_to_run_data",
    "pr_transverse_static_result_to_run_data",
    "run_pr_transverse_static",
    "run_pr_transverse_timedependent",
    "solve_pr_transverse_discrete_static_intensity",
    "solve_pr_transverse_static_intensity",
    "static_equilibrium_residual",
]
