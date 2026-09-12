"""Deterministic coarse cost classification for PR GUI launches."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

from lcprop.core.grid import round_nz
from lcprop.pr.specs import PRRunRequest
from lcprop.pr.static_workflow import PRStaticRunRequest
from lcprop.pr.transverse.specs import (
    PR_MATERIAL_RESPONSE_LINEARIZED,
    PRTransverseRunRequest,
)
from lcprop.pr.transverse.static_workflow import PRTransverseStaticRunRequest


class LocalRunCostClass(str, Enum):
    """Coarse, deliberately non-temporal local-run cost classes."""

    NORMAL = "Normal"
    POTENTIALLY_EXPENSIVE = "Potentially Expensive"
    VERY_EXPENSIVE = "Very Expensive"


@dataclass(frozen=True)
class LocalRunCostAssessment:
    """Explainable result of the pre-launch local cost classifier."""

    classification: LocalRunCostClass
    model_label: str
    grid_shape: tuple[int, int, int]
    work_score: float
    rationale: str


def _grid_shape(request) -> tuple[int, int, int]:
    grid = request.grid
    return int(grid.Nx), int(grid.Ny), round_nz(grid.z_length_um, grid.dz_um)


def _hardware_precision_factor(request) -> float:
    backend = request.backend.backend
    # ``auto`` may resolve to NumPy on a local machine, so classify it with the
    # conservative CPU factor rather than assuming an available accelerator.
    backend_factor = {"numpy": 1.0, "auto": 1.0, "cupy": 0.25}.get(
        backend, 1.0
    )
    precision_factor = 1.0 if request.backend.precision == "float64" else 0.65
    return backend_factor * precision_factor


def classify_pr_run_cost(
    request,
    *,
    execution_target: str,
) -> LocalRunCostAssessment:
    """Classify a PR request without estimating wall-clock runtime.

    Scores count coarse upper-bound work factors already present in the
    immutable request.  The classes intentionally distinguish nonlinear
    Newton/Krylov work from the fixed-count Fourier linearized response.
    """

    shape = _grid_shape(request)
    nx, ny, nz = shape
    points = float(nx * ny * nz)
    factor = _hardware_precision_factor(request)

    if isinstance(request, PRTransverseStaticRunRequest):
        coupled = int(request.solver.max_coupled_iterations)
        if request.material_response.model == PR_MATERIAL_RESPONSE_LINEARIZED:
            model = "Full transverse PR transport — Linearized material response"
            score = points * coupled * 5.0 * factor
            potential_threshold = 2.0e10
            very_threshold = 2.0e11
            rationale = "fixed-count Fourier material response per coupled pass"
        else:
            material = request.solver.material_solver
            model = "Full transverse PR transport — Fully nonlinear"
            score = (
                points
                * coupled
                * int(material.max_newton_iterations)
                * math.sqrt(float(material.max_pcg_iterations))
                * factor
            )
            potential_threshold = 5.0e9
            very_threshold = 5.0e11
            rationale = "nonlinear coupled, per-plane Newton, and PCG work"
    elif isinstance(request, PRTransverseRunRequest):
        linearized = (
            request.material_response.model == PR_MATERIAL_RESPONSE_LINEARIZED
        )
        model = (
            "Full transverse PR transport — Linearized time dependent"
            if linearized
            else "Full transverse PR transport — Fully nonlinear time dependent"
        )
        score = points * int(request.solver.Nt) * (
            5.0 if linearized else 8.0
        ) * factor
        potential_threshold = 5.0e9
        very_threshold = 5.0e10
        rationale = (
            "exact plane-local Fourier material updates and optical z marches"
            if linearized
            else "full-transverse material steps and optical z marches"
        )
    elif isinstance(request, PRStaticRunRequest):
        if request.material_response.model == PR_MATERIAL_RESPONSE_LINEARIZED:
            model = "Reduced x-only PR transport — Linearized material response"
            score = (
                points
                * int(request.solver.max_coupled_passes)
                * 2.0
                * factor
            )
            rationale = "one-dimensional FFT material response per coupled pass"
        else:
            material_iterations = (
                request.solver.material_solver.max_iterations
                if request.solver.material_solver is not None
                else 40
            )
            model = "Reduced x-only PR transport — Fully nonlinear static"
            score = (
                points
                * int(request.solver.max_coupled_passes)
                * int(material_iterations)
                * factor
            )
            rationale = "reduced static material and coupled-pass work"
        potential_threshold = 2.0e10
        very_threshold = 2.0e11
    elif isinstance(request, PRRunRequest):
        model = "Reduced x-only PR transport — Fully nonlinear time dependent"
        score = points * int(request.solver.Nt) * factor
        potential_threshold = 2.0e10
        very_threshold = 2.0e11
        rationale = "reduced material steps and optical z marches"
    else:
        raise TypeError(f"unsupported PR run-cost request: {type(request).__name__}")

    if execution_target != "local":
        classification = LocalRunCostClass.NORMAL
        rationale = "remote execution bypasses the local-run cost guard"
    elif score >= very_threshold:
        classification = LocalRunCostClass.VERY_EXPENSIVE
    elif score >= potential_threshold:
        classification = LocalRunCostClass.POTENTIALLY_EXPENSIVE
    else:
        classification = LocalRunCostClass.NORMAL

    return LocalRunCostAssessment(
        classification=classification,
        model_label=model,
        grid_shape=shape,
        work_score=score,
        rationale=rationale,
    )


__all__ = [
    "LocalRunCostAssessment",
    "LocalRunCostClass",
    "classify_pr_run_cost",
]
