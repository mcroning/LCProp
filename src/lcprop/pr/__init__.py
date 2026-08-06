"""Headless photorefractive material model and propagation workflow."""

from lcprop.pr.checkpoint import (
    PRTimeDependentCheckpoint,
    validate_pr_checkpoint,
    validate_pr_continuation,
)
from lcprop.pr.persistence import (
    PR_CHECKPOINT_SCHEMA_VERSION,
    load_pr_checkpoint,
    save_pr_checkpoint,
)
from lcprop.pr.specs import (
    PRMaterialSpec,
    PR_EULER_INTEGRATOR,
    PR_INTEGRATORS,
    PR_MATERIAL_ID,
    PRRunRequest,
    PRRunResult,
    PRSolverOptions,
    PR_SEMI_IMPLICIT_INTEGRATOR,
)
from lcprop.pr.static import (
    PRStaticResult,
    PRStaticSolverOptions,
    solve_pr_static_intensity,
    solve_pr_static_intensity_batched,
)
from lcprop.pr.static_workflow import (
    PRCoupledStaticIterationRecord,
    PRCoupledStaticSliceSummary,
    PRStaticRunRequest,
    PRStaticRunResult,
    PRStaticWorkflowOptions,
    PR_STATIC_WORKFLOW,
    run_pr_static,
)
from lcprop.pr.readiness import (
    PRImageAmplificationReadinessResult,
    PRImageAmplificationReadinessSpec,
    make_image_amplification_readiness_request,
    run_image_amplification_readiness,
)
from lcprop.pr.image_amplification import (
    PRImageAmplificationResult,
    PRImageAmplificationSpec,
    make_image_amplification_request,
    paper_absolute_signal_gain,
    paper_figure4_spec,
    run_image_amplification,
)
from lcprop.pr.geometry import crossing_beam_channels, paraxial_kernel_slope
from lcprop.pr.coupling import (
    FiniteGaussianCouplingSpec,
    PlaneWaveCouplingSpec,
    run_finite_gaussian_coupling,
    run_plane_wave_coupling,
)
from lcprop.pr.workflow import (
    advance_pr_slice_with_midpoint_source,
    continue_pr_timedependent,
    run_pr_timedependent,
)

__all__ = [
    "PRMaterialSpec",
    "PR_EULER_INTEGRATOR",
    "PR_INTEGRATORS",
    "PR_MATERIAL_ID",
    "PRRunRequest",
    "PRRunResult",
    "PRSolverOptions",
    "PR_SEMI_IMPLICIT_INTEGRATOR",
    "PRStaticResult",
    "PRStaticSolverOptions",
    "PRCoupledStaticIterationRecord",
    "PRCoupledStaticSliceSummary",
    "PRStaticRunRequest",
    "PRStaticRunResult",
    "PRStaticWorkflowOptions",
    "PR_STATIC_WORKFLOW",
    "PRImageAmplificationReadinessResult",
    "PRImageAmplificationReadinessSpec",
    "PRImageAmplificationResult",
    "PRImageAmplificationSpec",
    "PRTimeDependentCheckpoint",
    "PR_CHECKPOINT_SCHEMA_VERSION",
    "load_pr_checkpoint",
    "save_pr_checkpoint",
    "validate_pr_checkpoint",
    "validate_pr_continuation",
    "paraxial_kernel_slope",
    "crossing_beam_channels",
    "PlaneWaveCouplingSpec",
    "run_plane_wave_coupling",
    "FiniteGaussianCouplingSpec",
    "run_finite_gaussian_coupling",
    "make_image_amplification_readiness_request",
    "run_image_amplification_readiness",
    "make_image_amplification_request",
    "paper_absolute_signal_gain",
    "paper_figure4_spec",
    "run_image_amplification",
    "solve_pr_static_intensity",
    "solve_pr_static_intensity_batched",
    "advance_pr_slice_with_midpoint_source",
    "run_pr_static",
    "continue_pr_timedependent",
    "run_pr_timedependent",
]
