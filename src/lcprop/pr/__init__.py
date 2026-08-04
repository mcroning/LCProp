"""Headless photorefractive material model and propagation workflow."""

from lcprop.pr.specs import (
    PRMaterialSpec,
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
from lcprop.pr.workflow import run_pr_timedependent

__all__ = [
    "PRMaterialSpec",
    "PRRunRequest",
    "PRRunResult",
    "PRSolverOptions",
    "paraxial_kernel_slope",
    "crossing_beam_channels",
    "PlaneWaveCouplingSpec",
    "run_plane_wave_coupling",
    "FiniteGaussianCouplingSpec",
    "run_finite_gaussian_coupling",
    "run_pr_timedependent",
]
