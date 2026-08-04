"""Presentation products for photorefractive workflow results.

The concrete PR result remains the authoritative physical result.  This module
selects fields and diagnostics that can be represented faithfully by the
shared, material-neutral ``RunData`` presentation model.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from lcprop.core.backend import asnumpy
from lcprop.optics.splitstep import total_intensity
from lcprop.pr.specs import PRRunResult, PR_TIMEDEPENDENT_WORKFLOW
from lcprop.products.data_model import (
    CurveCollection,
    DiagnosticCollection,
    DiagnosticData,
    FieldCollection,
    Geometry,
    RunData,
    make_field,
)


def _copied_array(value: Any) -> np.ndarray:
    """Return a detached host copy suitable for presentation ownership."""

    return np.asarray(asnumpy(value)).copy()


def _geometry_from_pr_result(result: PRRunResult) -> Geometry:
    summary = result.grid_summary
    nx = int(summary["Nx"])
    ny = int(summary["Ny"])
    nz = int(summary["Nz"])
    dx_um = float(summary["dx_um"])
    dy_um = float(summary["dy_um"])
    dz_um = float(summary["dz_um"])

    if nx < 1 or ny < 1 or nz < 1:
        raise ValueError("PR grid dimensions must be positive")
    if dx_um <= 0.0 or dy_um <= 0.0 or dz_um <= 0.0:
        raise ValueError("PR grid spacings must be positive")

    x_um = (np.arange(nx) - 0.5 * (nx - 1)) * dx_um
    y_um = (np.arange(ny) - 0.5 * (ny - 1)) * dy_um
    z_um = np.arange(nz) * dz_um
    return Geometry(x=x_um, y=y_um, z=z_um, units="um")


def _validated_result_arrays(
    result: PRRunResult,
    geometry: Geometry,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    A_initial = _copied_array(result.A_initial)
    A_final = _copied_array(result.A_final)
    E_initial = _copied_array(result.E_initial)
    E_final = _copied_array(result.E_final)
    source_intensity = _copied_array(result.source_intensity_stack)

    nx = len(np.asarray(geometry.x))
    ny = len(np.asarray(geometry.y))
    nz = len(np.asarray(geometry.z))
    if A_initial.ndim != 3 or A_initial.shape[1:] != (nx, ny):
        raise ValueError(
            "PR A_initial must have shape (Nch, Nx, Ny) matching grid_summary"
        )
    if A_final.shape != A_initial.shape:
        raise ValueError("PR A_final shape must match A_initial")

    expected_volume_shape = (nz, nx, ny)
    for name, value in (
        ("E_initial", E_initial),
        ("E_final", E_final),
        ("source_intensity_stack", source_intensity),
    ):
        if value.shape != expected_volume_shape:
            raise ValueError(
                f"PR {name} must have shape (Nz, Nx, Ny) matching grid_summary"
            )

    return A_initial, A_final, E_initial, E_final, source_intensity


def _optical_intensity(A: np.ndarray, launch_summary: dict[str, Any]) -> np.ndarray:
    groups = launch_summary.get("coherence_groups")
    coherent = launch_summary.get("coherence") == "coherent"
    intensity = total_intensity(
        A,
        coherent=coherent,
        coherence_groups=None if groups is None else tuple(groups),
        xp=np,
    )
    return np.asarray(intensity).copy()


def pr_result_to_run_data(result: PRRunResult) -> RunData:
    """Convert a concrete PR workflow result into shared presentation data.

    ``E_initial`` and ``E_final`` remain normalized space-charge fields.  The
    source-intensity volume is the dimensionless PR-driving intensity used by
    the material equation, including configured dark and uniform background
    contributions.  No material-time history or benchmark-only diagnostics
    are inferred when they are absent from ``PRRunResult``.
    """

    if not isinstance(result, PRRunResult):
        raise TypeError("result must be a PRRunResult")

    geometry = _geometry_from_pr_result(result)
    (
        A_initial,
        A_final,
        E_initial,
        E_final,
        source_intensity,
    ) = _validated_result_arrays(result, geometry)
    launch_summary = deepcopy(result.launch_summary)
    input_intensity = _optical_intensity(A_initial, launch_summary)
    output_intensity = _optical_intensity(A_final, launch_summary)
    selected_z_index = E_final.shape[0] // 2

    spatial_units = {"x": "um", "y": "um"}
    volume_units = {"z": "um", "x": "um", "y": "um"}
    fields = FieldCollection([
        (
            "input_intensity",
            make_field(
                "input_intensity",
                "Input Plane Intensity",
                input_intensity,
                ("x", "y"),
                "intensity",
                spatial_units,
                quantity="normalized_intensity",
                value_unit="1/µm²",
            ),
        ),
        (
            "output_intensity",
            make_field(
                "output_intensity",
                "Output Plane Intensity",
                output_intensity,
                ("x", "y"),
                "intensity",
                spatial_units,
                quantity="normalized_intensity",
                value_unit="1/µm²",
            ),
        ),
        (
            "initial_E",
            make_field(
                "initial_E",
                "Initial PR Space-Charge Field",
                E_initial[selected_z_index].copy(),
                ("x", "y"),
                "pr_space_charge",
                spatial_units,
                quantity="normalized_space_charge_field",
                value_unit="1",
                colormap="coolwarm",
                source_volume_key="initial_E_stack",
            ),
        ),
        (
            "final_E",
            make_field(
                "final_E",
                "Final PR Space-Charge Field",
                E_final[selected_z_index].copy(),
                ("x", "y"),
                "pr_space_charge",
                spatial_units,
                quantity="normalized_space_charge_field",
                value_unit="1",
                colormap="coolwarm",
                source_volume_key="final_E_stack",
            ),
        ),
        (
            "initial_E_stack",
            make_field(
                "initial_E_stack",
                "Initial PR Space-Charge Field",
                E_initial,
                ("z", "x", "y"),
                "pr_space_charge",
                volume_units,
                "longitudinal",
                quantity="normalized_space_charge_field",
                value_unit="1",
                colormap="coolwarm",
            ),
        ),
        (
            "final_E_stack",
            make_field(
                "final_E_stack",
                "Final PR Space-Charge Field",
                E_final,
                ("z", "x", "y"),
                "pr_space_charge",
                volume_units,
                "longitudinal",
                quantity="normalized_space_charge_field",
                value_unit="1",
                colormap="coolwarm",
            ),
        ),
        (
            "pr_driving_intensity_stack",
            make_field(
                "pr_driving_intensity_stack",
                "Final PR-Driving Intensity",
                source_intensity,
                ("z", "x", "y"),
                "pr_driving_intensity",
                volume_units,
                "longitudinal",
                quantity="normalized_pr_driving_intensity",
                value_unit="1",
            ),
        ),
    ])

    diagnostics = DiagnosticCollection([
        (
            "summary",
            DiagnosticData(
                "summary",
                "Summary",
                {
                    "material": "photorefractive",
                    "normalized_field_integral_initial": float(
                        result.power_initial
                    ),
                    "normalized_field_integral_final": float(result.power_final),
                    "completed_material_steps": int(result.completed_steps),
                    "requested_material_steps": int(result.requested_steps),
                    "material_time_normalized": float(result.time_normalized),
                    "status": result.status,
                    "grid": deepcopy(result.grid_summary),
                    "launch": launch_summary,
                },
            ),
        ),
        (
            "pr_workflow",
            DiagnosticData(
                "pr_workflow",
                "PR Workflow Diagnostics",
                deepcopy(result.diagnostics),
            ),
        ),
    ])

    return RunData(
        workflow=PR_TIMEDEPENDENT_WORKFLOW,
        geometry=geometry,
        fields=fields,
        curves=CurveCollection(),
        diagnostics=diagnostics,
    )


__all__ = ["PR_TIMEDEPENDENT_WORKFLOW", "pr_result_to_run_data"]
