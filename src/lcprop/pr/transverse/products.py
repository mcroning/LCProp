"""Shared presentation adapter for transverse PR workflow results."""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from lcprop.optics.splitstep import total_intensity
from lcprop.pr.transverse.projection import project_active_field
from lcprop.pr.transverse.specs import (
    PR_TRANSVERSE_TIMEDEPENDENT_WORKFLOW,
    PRTransverseProjectionProfile,
    PRTransverseRunResult,
)
from lcprop.pr.transverse.static_workflow import (
    PR_TRANSVERSE_STATIC_WORKFLOW,
    PRTransverseStaticRunResult,
)
from lcprop.pr.transverse.transport import state_from_potential
from lcprop.products.data_model import (
    CurveCollection,
    DiagnosticCollection,
    DiagnosticData,
    FieldCollection,
    Geometry,
    RunData,
    make_field,
)


def pr_transverse_result_to_run_data(result: PRTransverseRunResult) -> RunData:
    """Reconstruct Profile-v1 derived fields at the presentation boundary."""

    if not isinstance(result, PRTransverseRunResult):
        raise TypeError("result must be a PRTransverseRunResult")
    summary = result.grid_summary
    nx, ny, nz = (int(summary[key]) for key in ("Nx", "Ny", "Nz"))
    x = (np.arange(nx) - 0.5 * (nx - 1)) * float(summary["dx_um"])
    y = (np.arange(ny) - 0.5 * (ny - 1)) * float(summary["dy_um"])
    z = np.arange(nz) * float(summary["dz_um"])
    geometry = Geometry(x=x, y=y, z=z, units="um")
    psi = np.asarray(result.psi_final)
    if psi.shape != (nz, nx, ny):
        raise ValueError("psi_final does not match grid_summary")
    profile = result.resolved_profile
    state = state_from_potential(
        psi,
        dx_normalized=float(profile["dx_normalized"]),
        dy_normalized=float(profile["dy_normalized"]),
        h_y=float(profile["dielectric"]["h_y"]),
        applied_field_x=float(profile["boundary"]["applied_field_x"]),
    )
    projection = PRTransverseProjectionProfile(**profile["projection"])
    active = project_active_field(state.E_x, state.E_y, profile=projection)
    groups = tuple(result.launch_summary["coherence_groups"])
    input_intensity = total_intensity(
        result.A_initial, coherence_groups=groups, xp=np
    )
    output_intensity = total_intensity(
        result.A_final, coherence_groups=groups, xp=np
    )
    spatial_units = {"x": "um", "y": "um"}
    volume_units = {"z": "um", "x": "um", "y": "um"}
    fields = FieldCollection()
    for key, name, value, kind, cmap in (
        ("psi", "PR Electrostatic Potential", state.psi, "pr_potential", "coolwarm"),
        ("P", "Normalized Carrier Density", state.carrier_density, "pr_carrier", "viridis"),
        ("E_x", "Transverse Space-Charge Field E_x", state.E_x, "pr_space_charge", "coolwarm"),
        ("E_y", "Transverse Space-Charge Field E_y", state.E_y, "pr_space_charge", "coolwarm"),
        ("E_active", "Optically Active PR Field", active, "pr_space_charge", "coolwarm"),
    ):
        fields.add(
            key,
            make_field(
                key, name, value, ("z", "x", "y"), kind, volume_units,
                "longitudinal", quantity=key, value_unit="1", colormap=cmap,
            ),
        )
    fields.add("input_intensity", make_field(
        "input_intensity", "Input Plane Intensity", input_intensity,
        ("x", "y"), "intensity", spatial_units,
        quantity="normalized_intensity", value_unit="1/µm²",
    ))
    fields.add("output_intensity", make_field(
        "output_intensity", "Output Plane Intensity", output_intensity,
        ("x", "y"), "intensity", spatial_units,
        quantity="normalized_intensity", value_unit="1/µm²",
    ))
    diagnostics = DiagnosticCollection([
        ("summary", DiagnosticData("summary", "Summary", {
            "material": "photorefractive",
            "workflow": PR_TRANSVERSE_TIMEDEPENDENT_WORKFLOW,
            "status": result.status,
            "completed_material_steps": result.completed_steps,
            "requested_material_steps": result.requested_steps,
            "material_time_normalized": result.time_normalized,
            "physics_profile": deepcopy(result.resolved_profile),
            "backend": deepcopy(result.backend_summary),
        })),
        ("transverse_pr", DiagnosticData(
            "transverse_pr", "Transverse PR Diagnostics", deepcopy(result.diagnostics)
        )),
    ])
    return RunData(
        workflow=PR_TRANSVERSE_TIMEDEPENDENT_WORKFLOW,
        geometry=geometry,
        fields=fields,
        curves=CurveCollection(),
        diagnostics=diagnostics,
    )


def pr_transverse_static_result_to_run_data(
    result: PRTransverseStaticRunResult,
) -> RunData:
    """Reconstruct static Profile-v1 state and residual products."""

    if not isinstance(result, PRTransverseStaticRunResult):
        raise TypeError("result must be a PRTransverseStaticRunResult")
    summary = result.grid_summary
    nx, ny, nz = (int(summary[key]) for key in ("Nx", "Ny", "Nz"))
    x = (np.arange(nx) - 0.5 * (nx - 1)) * float(summary["dx_um"])
    y = (np.arange(ny) - 0.5 * (ny - 1)) * float(summary["dy_um"])
    z = np.arange(nz) * float(summary["dz_um"])
    geometry = Geometry(x=x, y=y, z=z, units="um")
    psi = np.asarray(result.psi_final)
    if psi.shape != (nz, nx, ny):
        raise ValueError("psi_final does not match grid_summary")
    profile = result.resolved_profile
    state = state_from_potential(
        psi,
        dx_normalized=float(profile["dx_normalized"]),
        dy_normalized=float(profile["dy_normalized"]),
        h_y=float(profile["dielectric"]["h_y"]),
        applied_field_x=float(profile["boundary"]["applied_field_x"]),
    )
    projection = PRTransverseProjectionProfile(**profile["projection"])
    active = project_active_field(state.E_x, state.E_y, profile=projection)
    groups = tuple(result.launch_summary["coherence_groups"])
    input_intensity = total_intensity(
        result.A_initial, coherence_groups=groups, xp=np
    )
    output_intensity = total_intensity(
        result.A_final, coherence_groups=groups, xp=np
    )
    spatial_units = {"x": "um", "y": "um"}
    volume_units = {"z": "um", "x": "um", "y": "um"}
    fields = FieldCollection()
    for key, name, value, kind, cmap in (
        ("psi", "PR Electrostatic Potential", state.psi, "pr_potential", "coolwarm"),
        ("P", "Normalized Carrier Density", state.carrier_density, "pr_carrier", "viridis"),
        ("E_x", "Transverse Space-Charge Field E_x", state.E_x, "pr_space_charge", "coolwarm"),
        ("E_y", "Transverse Space-Charge Field E_y", state.E_y, "pr_space_charge", "coolwarm"),
        ("E_active", "Optically Active PR Field", active, "pr_space_charge", "coolwarm"),
        (
            "transport_intensity",
            "Static PR Transport Intensity",
            result.source_intensity_stack,
            "intensity",
            "viridis",
        ),
        (
            "equilibrium_residual",
            "Authoritative Zero-Flux Static Residual",
            result.equilibrium_residual_stack,
            "residual",
            "coolwarm",
        ),
        (
            "td_rhs_residual",
            "Diagnostic Production TD RHS at Zero-Flux Static State",
            result.td_rhs_residual_stack,
            "residual",
            "coolwarm",
        ),
    ):
        fields.add(key, make_field(
            key, name, value, ("z", "x", "y"), kind, volume_units,
            "longitudinal", quantity=key, value_unit="1", colormap=cmap,
        ))
    fields.add("input_intensity", make_field(
        "input_intensity", "Input Plane Intensity", input_intensity,
        ("x", "y"), "intensity", spatial_units,
        quantity="normalized_intensity", value_unit="1/µm²",
    ))
    fields.add("output_intensity", make_field(
        "output_intensity", "Output Plane Intensity", output_intensity,
        ("x", "y"), "intensity", spatial_units,
        quantity="normalized_intensity", value_unit="1/µm²",
    ))
    diagnostics = DiagnosticCollection([
        ("summary", DiagnosticData("summary", "Summary", {
            "material": "photorefractive",
            "workflow": PR_TRANSVERSE_STATIC_WORKFLOW,
            "status": result.status,
            "converged": result.converged,
            "completed_coupled_iterations": result.completed_coupled_iterations,
            "physics_profile": deepcopy(result.resolved_profile),
            "backend": deepcopy(result.backend_summary),
            "timing": deepcopy(result.timing),
        })),
        ("transverse_pr", DiagnosticData(
            "transverse_pr", "Transverse PR Diagnostics", deepcopy(result.diagnostics)
        )),
        ("replay", DiagnosticData(
            "replay", "Independent Replay", deepcopy(result.replay_diagnostics)
        )),
    ])
    return RunData(
        workflow=PR_TRANSVERSE_STATIC_WORKFLOW,
        geometry=geometry,
        fields=fields,
        curves=CurveCollection(),
        diagnostics=diagnostics,
    )


__all__ = [
    "pr_transverse_result_to_run_data",
    "pr_transverse_static_result_to_run_data",
]
