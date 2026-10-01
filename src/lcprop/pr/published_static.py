"""Published optical-first reduced Static: P(h), I, T, M(h), S(cell).

Material callbacks solve a fixed arriving intensity; they never repropagate
optics. No TD dispatch or time integrator participates in this workflow.
"""
from dataclasses import dataclass, field, asdict
from types import SimpleNamespace

from lcprop.core.backend import BackendSpec, get_backend
from lcprop.core.beams import BeamStack
from lcprop.core.context import GridSpec
from lcprop.optics.boundaries import TransverseBoundarySpec
from lcprop.pr.local_plane_workflow import _run_streaming_planes
from lcprop.pr.reduced_field_linear import PRReducedFieldLinearSpec, solve_pr_reduced_field_linear_intensity
from lcprop.pr.static import PRStaticSolverOptions, solve_pr_static_intensity_batched
from lcprop.pr.evolution import hopping_rhs, periodic_first_derivative_x
from lcprop.pr.specs import PRMaterialSpec
from lcprop.pr.transverse.specs import PRTransverseMaterialResponseSpec
from lcprop.pr.scattering import PRCanonicalScatteringSpec

PR_PUBLISHED_STATIC_WORKFLOW = "pr_static_published_optical_first_v1"
PR_PUBLISHED_STATIC_ARITHMETIC = "per_cell_full_linear_arriving_material_full_phase_scattering_v1"


@dataclass(frozen=True)
class PublishedStaticRequest:
    grid: GridSpec
    beams: BeamStack
    material: PRMaterialSpec = PRMaterialSpec()
    backend: BackendSpec = BackendSpec("numpy", "float64", False)
    material_response: PRTransverseMaterialResponseSpec = field(default_factory=lambda:
        PRTransverseMaterialResponseSpec(model="field_linear_local_intensity"))
    optical_boundary: TransverseBoundarySpec = TransverseBoundarySpec()
    scattering: PRCanonicalScatteringSpec | None = None
    launch_elements: tuple = ()
    initial_A: object | None = None
    residual_rms_tolerance: float = 1e-5
    residual_max_tolerance: float = 1e-4
    material_solver: PRStaticSolverOptions = PRStaticSolverOptions()


def material_residual(E, intensity, *, request, xp):
    dx = request.material.characteristic_wavenumber_per_um * request.grid.x_aperture_um/request.grid.Nx
    if request.material_response.model == "nonlinear":
        return hopping_rhs(E, intensity, applied_field=request.material.applied_field,
            background_intensity=request.material.background_intensity, dx_normalized=dx, xp=xp)
    from lcprop.pr.reduced_field_linear import reduced_field_linear_residual
    return reduced_field_linear_residual(E, intensity, spec=PRReducedFieldLinearSpec(
        request.material.applied_field, request.material.background_intensity, dx), xp=xp)


def solve_arriving_material(intensity, *, request):
    xp = get_backend(request.backend).xp
    dx = request.material.characteristic_wavenumber_per_um * request.grid.x_aperture_um/request.grid.Nx
    if request.material_response.model == "field_linear_local_intensity":
        return solve_pr_reduced_field_linear_intensity(intensity,
            spec=PRReducedFieldLinearSpec(request.material.applied_field,
                request.material.background_intensity, dx), backend=request.backend)
    if request.material_response.model != "nonlinear":
        raise ValueError("published Static supports Local-I or nonlinear reduced hopping only")
    result = solve_pr_static_intensity_batched(intensity,
        applied_field=request.material.applied_field,
        background_intensity=request.material.background_intensity, dx_normalized=dx,
        options=request.material_solver, xp=xp)
    if not result.converged:
        raise ValueError(f"material solve failed: {result.status}: {result.message}")
    carrier = 1 + periodic_first_derivative_x(result.E, dx_normalized=dx, xp=xp)
    if not bool(xp.all(xp.isfinite(carrier) & (carrier > 0)).item()):
        raise ValueError("material nonlinear carrier must be finite and positive")
    return SimpleNamespace(E=result.E, residual=material_residual(result.E, intensity,
        request=request, xp=xp), solver_evidence=dict(status=result.status,
        iterations=result.iterations, records=[asdict(r) for r in result.records]))


def run_published_static(request, *, retain_boundary_field=False,
                         cancellation_token=None, progress_callback=None,
                         observation_callback=None):
    from lcprop.pr.published_static_step import step_published_cell
    if not isinstance(request, PublishedStaticRequest):
        raise TypeError("requires explicit PublishedStaticRequest; no workflow migration")
    request.material_solver.validate()
    def step(A, **cell_kwargs):
        return step_published_cell(A, material_response=lambda I: solve_arriving_material(I, request=request),
            material_model=request.material_response.model, **cell_kwargs)
    return _run_streaming_planes(request, _request_type=PublishedStaticRequest,
        _models=("field_linear_local_intensity", "nonlinear"), _cell_step=step,
        _workflow=PR_PUBLISHED_STATIC_WORKFLOW, _arithmetic=PR_PUBLISHED_STATIC_ARITHMETIC,
        retain_boundary_field=retain_boundary_field, cancellation_token=cancellation_token,
        progress_callback=progress_callback, observation_callback=observation_callback)
