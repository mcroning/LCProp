"""Mapping and preflight for standalone PR GUI controls."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lcprop.adapters.launchplane import beam_stack_to_launchplane
from lcprop.core.backend import dtype_pair
from lcprop.core.grid import make_grid
from lcprop.pr.evolution import validate_timestep
from lcprop.pr.geometry import (
    PRBeamStackApertureReport,
    analyze_beam_stack_aperture,
)
from lcprop.pr.specs import PRRunRequest, PR_TIMEDEPENDENT_WORKFLOW
from lcprop.pr.static_workflow import (
    PRStaticRunRequest,
    PR_STATIC_WORKFLOW,
)
from lcprop.pr.transverse.specs import (
    PR_FULL_TRANSVERSE_PERIODIC_BIASED_CURRENT_V1,
    PR_FULL_TRANSVERSE_PROFILE_V1,
    PR_MATERIAL_RESPONSE_LINEARIZED,
    PR_MATERIAL_RESPONSE_NONLINEAR,
    PRTransverseBoundaryProfile,
    PRTransverseDielectricProfile,
    PRTransverseProjectionProfile,
    PRTransverseRunRequest,
    PRTransverseSolverOptions,
    PRTransverseTransportProfile,
    PR_TRANSVERSE_TIMEDEPENDENT_WORKFLOW,
)
from lcprop.pr.transverse.static_workflow import (
    PRTransverseStaticRunRequest,
    PRTransverseStaticWorkflowOptions,
    PR_TRANSVERSE_STATIC_WORKFLOW,
)


@dataclass(frozen=True)
class PRRequestPreflight:
    conservative_dt_limit: float
    aperture: PRBeamStackApertureReport

    @property
    def warnings(self) -> tuple[str, ...]:
        return self.aperture.warnings


@dataclass(frozen=True)
class PRStaticRequestPreflight:
    aperture: PRBeamStackApertureReport

    @property
    def warnings(self) -> tuple[str, ...]:
        return self.aperture.warnings


def _validate_pr_gui_common(request) -> PRBeamStackApertureReport:
    request.grid.validate()
    request.beams.validate()
    request.material.validate()
    request.solver.validate()
    request.backend.validate()
    wavelengths = tuple(
        float(channel.wavelength_um) for channel in request.beams.channels
    )
    if any(wavelength != wavelengths[0] for wavelength in wavelengths[1:]):
        raise ValueError("minimal PR workflow requires one shared wavelength")

    real_dtype, _ = dtype_pair(request.backend.precision)
    grid = make_grid(
        request.grid,
        xp=np,
        real_dtype=real_dtype,
    )
    return analyze_beam_stack_aperture(
        grid,
        request.beams,
        refractive_index=request.material.refractive_index,
        strict=False,
    )


def validate_pr_gui_request(request: PRRunRequest) -> PRRequestPreflight:
    """Validate one GUI request using PR-owned numerical rules."""

    if not isinstance(request, PRRunRequest):
        raise TypeError("request must be a PRRunRequest")
    aperture = _validate_pr_gui_common(request)
    real_dtype, _ = dtype_pair(request.backend.precision)
    grid = make_grid(
        request.grid,
        xp=np,
        real_dtype=real_dtype,
    )
    dt_limit = validate_timestep(
        request.solver.dt_normalized,
        grid,
        request.material,
        integrator=request.solver.integrator,
    )
    return PRRequestPreflight(
        conservative_dt_limit=dt_limit,
        aperture=aperture,
    )


def validate_pr_static_gui_request(
    request: PRStaticRunRequest,
) -> PRStaticRequestPreflight:
    """Validate one static GUI request without transient-time rules."""

    if not isinstance(request, PRStaticRunRequest):
        raise TypeError("request must be a PRStaticRunRequest")
    return PRStaticRequestPreflight(
        aperture=_validate_pr_gui_common(request),
    )


def validate_pr_transverse_static_gui_request(
    request: PRTransverseStaticRunRequest,
) -> PRStaticRequestPreflight:
    """Validate the GUI's canonical nonlinear 2D zero-flux request."""

    if not isinstance(request, PRTransverseStaticRunRequest):
        raise TypeError("request must be a PRTransverseStaticRunRequest")
    aperture = _validate_pr_gui_common(request)
    request.transport.validate()
    request.dielectric.validate()
    request.boundary.validate()
    request.projection.validate()
    request.material_response.validate_configuration(
        material_applied_field=request.material.applied_field,
        transport=request.transport,
        dielectric=request.dielectric,
        boundary=request.boundary,
        projection=request.projection,
    )
    if request.backend.backend not in ("numpy", "cupy"):
        raise ValueError(
            "2D zero-flux static workflow requires an explicit NumPy or "
            "CuPy backend"
        )
    if (
        request.material_response.model == PR_MATERIAL_RESPONSE_NONLINEAR
        and
        request.backend.backend == "numpy"
        and request.backend.precision != "float64"
    ):
        raise ValueError(
            "2D zero-flux static NumPy execution requires float64 precision"
        )
    return PRStaticRequestPreflight(aperture=aperture)


def validate_pr_transverse_gui_request(
    request: PRTransverseRunRequest,
) -> PRRequestPreflight:
    """Validate one canonical full-transverse material-time GUI request."""

    if not isinstance(request, PRTransverseRunRequest):
        raise TypeError("request must be a PRTransverseRunRequest")
    aperture = _validate_pr_gui_common(request)
    request.transport.validate()
    request.dielectric.validate()
    request.boundary.validate()
    request.projection.validate()
    if request.boundary.profile_id != PR_FULL_TRANSVERSE_PROFILE_V1:
        raise ValueError(
            "time-dependent full-transverse execution supports only the "
            "unbiased nonlinear Profile v1"
        )
    if request.backend.backend not in ("numpy", "cupy"):
        raise ValueError(
            "2D zero-flux time-dependent workflow requires an explicit "
            "NumPy or CuPy backend"
        )
    if float(request.material.applied_field) != 0.0:
        raise ValueError(
            "2D zero-flux Profile v1 requires zero normalized applied field"
        )
    return PRRequestPreflight(
        conservative_dt_limit=float("nan"),
        aperture=aperture,
    )


def validate_pr_gui_workflow_request(request):
    """Dispatch GUI preflight by the request's exact PR workflow type."""

    if isinstance(request, PRTransverseRunRequest):
        return validate_pr_transverse_gui_request(request)
    if isinstance(request, PRRunRequest):
        return validate_pr_gui_request(request)
    if isinstance(request, PRTransverseStaticRunRequest):
        return validate_pr_transverse_static_gui_request(request)
    if isinstance(request, PRStaticRunRequest):
        return validate_pr_static_gui_request(request)
    raise TypeError("unsupported PR GUI request type")


def build_pr_request(
    *,
    material_panel,
    beam_panel,
    grid_panel,
    evolution_panel,
) -> (
    PRRunRequest
    | PRStaticRunRequest
    | PRTransverseRunRequest
    | PRTransverseStaticRunRequest
):
    """Take one immutable, validated request snapshot from PR controls."""

    workflow_id = evolution_panel.workflow_id()
    launch = beam_panel.launch_configuration()
    common = {
        "grid": grid_panel.grid(),
        "beams": launch.beams,
        "launch_elements": launch.channel_elements,
        "material": material_panel.material(),
        "backend": evolution_panel.backend_spec(),
        "initial_A": None,
        "initial_E": None,
    }
    if workflow_id == PR_TIMEDEPENDENT_WORKFLOW:
        request = PRRunRequest(
            **common,
            solver=evolution_panel.solver(),
        )
    elif workflow_id == PR_STATIC_WORKFLOW:
        request = PRStaticRunRequest(
            **common,
            solver=evolution_panel.static_solver(),
        )
    elif workflow_id == PR_TRANSVERSE_STATIC_WORKFLOW:
        transverse_common = dict(common)
        transverse_common.pop("initial_E")
        material_response = evolution_panel.transverse_material_response()
        boundary_profile_id = (
            PR_FULL_TRANSVERSE_PERIODIC_BIASED_CURRENT_V1
            if material_response.model == PR_MATERIAL_RESPONSE_LINEARIZED
            else PR_FULL_TRANSVERSE_PROFILE_V1
        )
        request = PRTransverseStaticRunRequest(
            **transverse_common,
            transport=PRTransverseTransportProfile(),
            dielectric=PRTransverseDielectricProfile(),
            boundary=PRTransverseBoundaryProfile(
                profile_id=boundary_profile_id,
                applied_field_x=(
                    evolution_panel.transverse_applied_field.value()
                    if material_response.model == PR_MATERIAL_RESPONSE_LINEARIZED
                    else 0.0
                ),
            ),
            projection=PRTransverseProjectionProfile(),
            material_response=material_response,
            initial_psi=None,
            solver=evolution_panel.transverse_static_solver(),
        )
    elif workflow_id == PR_TRANSVERSE_TIMEDEPENDENT_WORKFLOW:
        transverse_common = dict(common)
        transverse_common.pop("initial_E")
        request = PRTransverseRunRequest(
            **transverse_common,
            initial_psi=None,
            solver=evolution_panel.transverse_solver(),
        )
    else:
        raise ValueError(f"unsupported PR GUI workflow: {workflow_id!r}")
    validate_pr_gui_workflow_request(request)
    return request


def validate_pr_gui_request_representable(request) -> None:
    """Reject headless PR settings that have no exact GUI representation."""

    validate_pr_gui_workflow_request(request)
    initial_material = (
        request.initial_psi
        if isinstance(
            request,
            (PRTransverseRunRequest, PRTransverseStaticRunRequest),
        )
        else request.initial_E
    )
    if request.initial_A is not None or initial_material is not None:
        raise ValueError(
            "PR GUI controls cannot represent request-owned initial arrays; "
            "load a PR checkpoint for continuation"
        )
    if request.backend.verbose:
        raise ValueError("PR GUI cannot represent backend verbose=True")
    if isinstance(request, PRTransverseRunRequest):
        represented = PRTransverseSolverOptions(
            Nt=request.solver.Nt,
            dt_normalized=request.solver.dt_normalized,
            optical_substeps=request.solver.optical_substeps,
            integrator=request.solver.integrator,
        )
        if request.solver != represented:
            raise ValueError(
                "PR GUI cannot represent these transverse TD solver options"
            )
        if request.scattering is not None:
            raise ValueError(
                "PR GUI cannot represent transverse TD scattering settings"
            )
    elif isinstance(request, PRTransverseStaticRunRequest):
        represented = PRTransverseStaticWorkflowOptions(
            max_coupled_iterations=request.solver.max_coupled_iterations,
            optical_substeps=request.solver.optical_substeps,
        )
        if request.solver != represented:
            raise ValueError(
                "PR GUI cannot represent these transverse static solver options"
            )
    elif isinstance(request, PRStaticRunRequest):
        represented = type(request.solver)(
            material_solver=None,
            max_coupled_passes=request.solver.max_coupled_passes,
            optical_substeps=request.solver.optical_substeps,
        )
        if request.solver != represented:
            raise ValueError(
                "PR GUI cannot represent these static solver options"
            )


def apply_pr_request(
    request: (
        PRRunRequest
        | PRStaticRunRequest
        | PRTransverseRunRequest
        | PRTransverseStaticRunRequest
    ),
    *,
    material_panel,
    beam_panel,
    grid_panel,
    evolution_panel,
    beam_stack_definition=None,
) -> None:
    """Populate every representable PR control from a saved request."""

    validate_pr_gui_request_representable(request)
    grid_panel.set_grid(request.grid)
    material_panel.set_material(request.material)
    if isinstance(request, PRTransverseRunRequest):
        evolution_panel.set_workflow_id(PR_TRANSVERSE_TIMEDEPENDENT_WORKFLOW)
        evolution_panel.set_transverse_solver(request.solver)
    elif isinstance(request, PRTransverseStaticRunRequest):
        evolution_panel.set_workflow_id(PR_TRANSVERSE_STATIC_WORKFLOW)
        evolution_panel.set_transverse_material_response(
            request.material_response,
            applied_field_x=request.boundary.applied_field_x,
        )
        evolution_panel.set_transverse_static_solver(request.solver)
    elif isinstance(request, PRStaticRunRequest):
        evolution_panel.set_workflow_id(PR_STATIC_WORKFLOW)
        evolution_panel.set_static_solver(request.solver)
    else:
        evolution_panel.set_workflow_id(PR_TIMEDEPENDENT_WORKFLOW)
        evolution_panel.set_solver(request.solver)
    evolution_panel.set_backend_spec(request.backend)
    beam_panel.set_aperture(
        request.grid.x_aperture_um,
        request.grid.y_aperture_um,
    )
    beam_panel.set_beam_stack_definition(
        beam_stack_to_launchplane(request.beams)
        if beam_stack_definition is None
        else beam_stack_definition
    )
    beam_panel.clear_launch_elements()


__all__ = [
    "PRRequestPreflight",
    "PRStaticRequestPreflight",
    "apply_pr_request",
    "build_pr_request",
    "validate_pr_gui_request",
    "validate_pr_gui_request_representable",
    "validate_pr_gui_workflow_request",
    "validate_pr_static_gui_request",
    "validate_pr_transverse_gui_request",
    "validate_pr_transverse_static_gui_request",
]
