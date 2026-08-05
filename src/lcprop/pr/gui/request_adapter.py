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
from lcprop.pr.specs import PRRunRequest


@dataclass(frozen=True)
class PRRequestPreflight:
    conservative_dt_limit: float
    aperture: PRBeamStackApertureReport

    @property
    def warnings(self) -> tuple[str, ...]:
        return self.aperture.warnings


def validate_pr_gui_request(request: PRRunRequest) -> PRRequestPreflight:
    """Validate one GUI request using PR-owned numerical rules."""

    if not isinstance(request, PRRunRequest):
        raise TypeError("request must be a PRRunRequest")
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
    dt_limit = validate_timestep(
        request.solver.dt_normalized,
        grid,
        request.material,
        integrator=request.solver.integrator,
    )
    aperture = analyze_beam_stack_aperture(
        grid,
        request.beams,
        refractive_index=request.material.refractive_index,
        strict=False,
    )
    return PRRequestPreflight(
        conservative_dt_limit=dt_limit,
        aperture=aperture,
    )


def build_pr_request(
    *,
    material_panel,
    beam_panel,
    grid_panel,
    evolution_panel,
) -> PRRunRequest:
    """Take one immutable, validated request snapshot from PR controls."""

    request = PRRunRequest(
        grid=grid_panel.grid(),
        beams=beam_panel.beams(),
        material=material_panel.material(),
        solver=evolution_panel.solver(),
        backend=evolution_panel.backend_spec(),
        initial_A=None,
        initial_E=None,
    )
    validate_pr_gui_request(request)
    return request


def apply_pr_request(
    request: PRRunRequest,
    *,
    material_panel,
    beam_panel,
    grid_panel,
    evolution_panel,
) -> None:
    """Populate every representable PR control from a saved request."""

    validate_pr_gui_request(request)
    if request.initial_A is not None or request.initial_E is not None:
        raise ValueError(
            "PR GUI controls cannot represent request-owned initial arrays; "
            "load a PR checkpoint for continuation"
        )
    grid_panel.set_grid(request.grid)
    material_panel.set_material(request.material)
    evolution_panel.set_solver(request.solver)
    evolution_panel.set_backend_spec(request.backend)
    beam_panel.set_aperture(
        request.grid.x_aperture_um,
        request.grid.y_aperture_um,
    )
    beam_panel.set_beam_stack_definition(
        beam_stack_to_launchplane(request.beams)
    )


__all__ = [
    "PRRequestPreflight",
    "apply_pr_request",
    "build_pr_request",
    "validate_pr_gui_request",
]
