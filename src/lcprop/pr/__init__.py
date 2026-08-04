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
    PR_MATERIAL_ID,
    PRRunRequest,
    PRRunResult,
    PRSolverOptions,
)
from lcprop.pr.geometry import crossing_beam_channels, paraxial_kernel_slope
from lcprop.pr.coupling import (
    FiniteGaussianCouplingSpec,
    PlaneWaveCouplingSpec,
    run_finite_gaussian_coupling,
    run_plane_wave_coupling,
)
from lcprop.pr.workflow import continue_pr_timedependent, run_pr_timedependent

__all__ = [
    "PRMaterialSpec",
    "PR_MATERIAL_ID",
    "PRRunRequest",
    "PRRunResult",
    "PRSolverOptions",
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
    "continue_pr_timedependent",
    "run_pr_timedependent",
]
