"""Shared presentation adapter for transverse PR workflow results."""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from lcprop.optics.farfield import direction_cosine_spectrum
from lcprop.optics.splitstep import total_intensity
from lcprop.pr.source import channel_peak_intensity_reference
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
from lcprop.pr.transverse.transport import PRTransverseState, state_from_potential
from lcprop.products.data_model import (
    CurveCollection,
    CurveData,
    DiagnosticCollection,
    DiagnosticData,
    FieldCollection,
    Geometry,
    RunData,
    make_field,
)


_FAR_FIELD_LOG_FLOOR_DB = -120.0
_STATE_RECONSTRUCTION_CHUNK_BYTES = 2 * 1024 * 1024
_STATE_RECONSTRUCTION_MIN_PLANE_CELLS = 512 * 512
_FAST_VOLUME_MESSAGE = (
    "Not retrieved in Fast mode; rerun with Full result retrieval to inspect "
    "this volume."
)


def _readonly_view(value) -> np.ndarray:
    view = np.asarray(value).view()
    view.flags.writeable = False
    return view


def _fast_optical_run_data(result, *, workflow: str, geometry: Geometry) -> RunData:
    """Present compact optical endpoints and far field for a Fast result."""

    groups = tuple(result.launch_summary["coherence_groups"])
    input_intensity = np.asarray(total_intensity(
        result.A_initial, coherence_groups=groups, xp=np
    ))
    output_intensity = np.asarray(total_intensity(
        result.A_final, coherence_groups=groups, xp=np
    ))
    profile = result.resolved_profile
    channels = profile["beam_request"]["channels"]
    spectrum = direction_cosine_spectrum(
        np.asarray(result.A_final),
        dx_um=float(result.grid_summary["dx_um"]),
        dy_um=float(result.grid_summary["dy_um"]),
        wavelength_um=float(channels[0]["wavelength_um"]),
        refractive_index=float(profile["material"]["refractive_index"]),
        coherence_groups=groups,
        xp=np,
    )
    fields = FieldCollection()
    for key, title, data in (
        ("input_intensity", "Input Plane Intensity", input_intensity),
        ("output_intensity", "Output Plane Intensity", output_intensity),
    ):
        fields.add(key, make_field(
            key, title, data, ("x", "y"), "intensity",
            {"x": "um", "y": "um"}, quantity="normalized_intensity",
            value_unit="1/µm²",
        ))
    fields.add("far_field_intensity", make_field(
        "far_field_intensity", "Output Far-Field Intensity",
        np.asarray(spectrum.intensity), ("s_x", "s_y"),
        "far_field_intensity", {"s_x": "1", "s_y": "1"},
        quantity="direction_cosine_power_density",
        value_unit="normalized power / direction-cosine²", colormap="magma",
        coordinates={"s_x": spectrum.s_x, "s_y": spectrum.s_y},
    ))
    summary = {
        "material": "photorefractive",
        "workflow": workflow,
        "status": result.status,
        "physics_profile": deepcopy(profile),
        "backend": deepcopy(result.backend_summary),
        "result_retention": deepcopy(result.retention_summary),
    }
    diagnostic_items = []
    if workflow == PR_TRANSVERSE_STATIC_WORKFLOW:
        summary.update({
            "converged": result.converged,
            "completed_coupled_iterations": result.completed_coupled_iterations,
            "timing": deepcopy(result.timing),
        })
    else:
        summary.update({
            "completed_material_steps": result.completed_steps,
            "requested_material_steps": result.requested_steps,
            "material_time_normalized": result.time_normalized,
        })
    diagnostic_items.extend([
        ("summary", DiagnosticData("summary", "Summary", summary)),
        ("transverse_pr", DiagnosticData(
            "transverse_pr", "Transverse PR Diagnostics", deepcopy(result.diagnostics)
        )),
    ])
    if workflow == PR_TRANSVERSE_STATIC_WORKFLOW:
        diagnostic_items.append(("replay", DiagnosticData(
            "replay", "Independent Replay", deepcopy(result.replay_diagnostics)
        )))
    diagnostics = DiagnosticCollection(diagnostic_items)
    return RunData(
        workflow=workflow, geometry=geometry, fields=fields,
        curves=CurveCollection(), diagnostics=diagnostics,
        longitudinal_enabled=False, longitudinal_message=_FAST_VOLUME_MESSAGE,
    )


def _state_from_potential_for_products(
    psi,
    *,
    dx_normalized: float,
    dy_normalized: float,
    h_y: float,
    applied_field_x: float,
) -> PRTransverseState:
    """Reconstruct a presentation state with bounded FFT temporaries.

    ``state_from_potential`` treats longitudinal planes independently, but a
    complete static result can contain tens of millions of transverse cells.
    Processing a small number of z planes at a time preserves the authoritative
    reconstruction operation while avoiding full-volume FFT work arrays.

    The direct path is retained for 2-D inputs, small volumes, and uniform
    blocks.  The latter preserves the transport helper's whole-volume uniform
    shortcut exactly when a mixed volume contains a locally uniform chunk.
    """

    potential = np.asarray(psi)

    def reconstruct(value) -> PRTransverseState:
        return state_from_potential(
            value,
            dx_normalized=dx_normalized,
            dy_normalized=dy_normalized,
            h_y=h_y,
            applied_field_x=applied_field_x,
            xp=np,
        )

    if potential.ndim != 3 or potential.dtype not in (
        np.dtype(np.float32),
        np.dtype(np.float64),
    ):
        return reconstruct(potential)

    plane_cells = potential.shape[-2] * potential.shape[-1]
    if plane_cells < _STATE_RECONSTRUCTION_MIN_PLANE_CELLS:
        return reconstruct(potential)
    plane_bytes = plane_cells * potential.itemsize
    chunk_planes = max(1, _STATE_RECONSTRUCTION_CHUNK_BYTES // plane_bytes)
    chunk_planes = min(potential.shape[0], chunk_planes)
    if chunk_planes >= potential.shape[0]:
        return reconstruct(potential)

    for start in range(0, potential.shape[0], chunk_planes):
        block = potential[start : start + chunk_planes]
        if np.all(block == block[..., :1, :1]):
            return reconstruct(potential)

    resolved = np.empty_like(potential)
    carrier = np.empty_like(potential)
    field_x = np.empty_like(potential)
    field_y = np.empty_like(potential)
    for start in range(0, potential.shape[0], chunk_planes):
        stop = min(start + chunk_planes, potential.shape[0])
        state = reconstruct(potential[start:stop])
        resolved[start:stop] = state.psi
        carrier[start:stop] = state.carrier_density
        field_x[start:stop] = state.E_x
        field_y[start:stop] = state.E_y
    return PRTransverseState(
        psi=resolved,
        carrier_density=carrier,
        E_x=field_x,
        E_y=field_y,
    )


def _carrier_exclusion_mask(
    s_x: np.ndarray,
    s_y: np.ndarray,
    *,
    beam_request,
    refractive_index: float,
) -> tuple[np.ndarray, tuple[dict[str, float], ...]]:
    """Return a documented legacy-style carrier-core exclusion mask."""

    keep = np.ones((s_x.size, s_y.size), dtype=bool)
    ds_x = abs(float(s_x[1] - s_x[0])) if s_x.size > 1 else 0.0
    ds_y = abs(float(s_y[1] - s_y[0])) if s_y.size > 1 else 0.0
    definitions = []
    for channel in beam_request["channels"]:
        wavelength = float(channel["wavelength_um"])
        k_medium = 2.0 * np.pi * float(refractive_index) / wavelength
        center_x = float(channel["tilt_x_rad_per_um"]) / k_medium
        center_y = float(channel["tilt_y_rad_per_um"]) / k_medium
        half_x = max(
            2.0 * ds_x,
            wavelength
            / (2.0 * float(refractive_index) * float(channel["waist_x_um"])),
        )
        half_y = max(
            2.0 * ds_y,
            wavelength
            / (2.0 * float(refractive_index) * float(channel["waist_y_um"])),
        )
        carrier_core = (
            np.abs(s_x[:, None] - center_x) <= half_x
        ) & (
            np.abs(s_y[None, :] - center_y) <= half_y
        )
        keep &= ~carrier_core
        definitions.append({
            "center_s_x": center_x,
            "center_s_y": center_y,
            "half_width_s_x": half_x,
            "half_width_s_y": half_y,
        })
    return keep, tuple(definitions)


def _static_presentation_products(
    *,
    result: PRTransverseStaticRunResult,
    geometry: Geometry,
    state,
    input_intensity: np.ndarray,
    output_intensity: np.ndarray,
):
    """Build compact PRProp3D-style products from canonical static state."""

    profile = result.resolved_profile
    material = profile["material"]
    beam_request = profile["beam_request"]
    channels = beam_request["channels"]
    wavelength = float(channels[0]["wavelength_um"])
    refractive_index = float(material["refractive_index"])
    dx_um = float(result.grid_summary["dx_um"])
    dy_um = float(result.grid_summary["dy_um"])
    spectrum = direction_cosine_spectrum(
        np.asarray(result.A_final),
        dx_um=dx_um,
        dy_um=dy_um,
        wavelength_um=wavelength,
        refractive_index=refractive_index,
        coherence_groups=tuple(result.launch_summary["coherence_groups"]),
        xp=np,
    )
    far_field = np.asarray(spectrum.intensity)
    far_peak = float(np.max(far_field))
    if not np.isfinite(far_peak) or far_peak <= 0.0:
        raise ValueError("output far-field intensity must have a finite positive peak")
    relative = np.maximum(far_field / far_peak, 10.0 ** (_FAR_FIELD_LOG_FLOOR_DB / 10.0))
    far_field_db = 10.0 * np.log10(relative)
    carrier_keep, carrier_regions = _carrier_exclusion_mask(
        np.asarray(spectrum.s_x),
        np.asarray(spectrum.s_y),
        beam_request=beam_request,
        refractive_index=refractive_index,
    )
    carrier_masked = np.where(carrier_keep, far_field, 0.0)

    x = np.asarray(geometry.x)
    y = np.asarray(geometry.y)
    z = np.asarray(geometry.z)
    ix = int(np.argmin(np.abs(x)))
    iy = int(np.argmin(np.abs(y)))
    iz = len(z) // 2
    peak_reference = channel_peak_intensity_reference(
        np.asarray(result.A_initial), xp=np
    )
    background = float(material["dark_intensity"]) + float(
        material["uniform_background_intensity"]
    )
    source = np.asarray(result.source_intensity_stack)
    optical_xz = (source[:, :, iy] - background) * peak_reference
    optical_yz = (source[:, ix, :] - background) * peak_reference

    field_units = {"x": "um", "y": "um", "z": "um"}
    angular_units = {"s_x": "1", "s_y": "1"}
    fields = (
        make_field(
            "optical_intensity_xz",
            f"Optical Intensity x-z at y={float(y[iy]):.6g} µm",
            optical_xz,
            ("z", "x"),
            "intensity",
            field_units,
            quantity="normalized_intensity",
            value_unit="1/µm²",
        ),
        make_field(
            "optical_intensity_yz",
            f"Optical Intensity y-z at x={float(x[ix]):.6g} µm",
            optical_yz,
            ("z", "y"),
            "intensity",
            field_units,
            quantity="normalized_intensity",
            value_unit="1/µm²",
        ),
        make_field(
            "E_x_xz",
            f"Material Electric Field E_x(x,z) at y={float(y[iy]):.6g} µm",
            _readonly_view(state.E_x[:, :, iy]),
            ("z", "x"),
            "pr_space_charge",
            field_units,
            quantity="normalized_space_charge_field",
            value_unit="1",
            colormap="coolwarm",
        ),
        make_field(
            "E_y_yz",
            f"Material Electric Field E_y(y,z) at x={float(x[ix]):.6g} µm",
            _readonly_view(state.E_y[:, ix, :]),
            ("z", "y"),
            "pr_space_charge",
            field_units,
            quantity="normalized_space_charge_field",
            value_unit="1",
            colormap="coolwarm",
        ),
        make_field(
            "far_field_intensity",
            "Output Far-Field Intensity",
            far_field,
            ("s_x", "s_y"),
            "far_field_intensity",
            angular_units,
            quantity="direction_cosine_power_density",
            value_unit="normalized power / direction-cosine²",
            colormap="magma",
            coordinates={"s_x": spectrum.s_x, "s_y": spectrum.s_y},
        ),
        make_field(
            "far_field_log_db",
            "Output Far Field (dB relative to peak)",
            far_field_db,
            ("s_x", "s_y"),
            "far_field_log",
            angular_units,
            quantity="relative_direction_cosine_power_density",
            value_unit="dB",
            colormap="magma",
            coordinates={"s_x": spectrum.s_x, "s_y": spectrum.s_y},
        ),
        make_field(
            "far_field_carrier_masked",
            "Carrier-Masked Far Field (diagnostic only)",
            carrier_masked,
            ("s_x", "s_y"),
            "far_field_intensity",
            angular_units,
            quantity="carrier_masked_direction_cosine_power_density",
            value_unit="normalized power / direction-cosine²",
            colormap="magma",
            coordinates={"s_x": spectrum.s_x, "s_y": spectrum.s_y},
        ),
    )
    selected_planes = (
        make_field(
            "selected_psi_plane",
            "Electrostatic Potential ψ (selected z plane)",
            _readonly_view(state.psi[iz]),
            ("x", "y"),
            "pr_potential",
            field_units,
            quantity="psi",
            value_unit="1",
            colormap="coolwarm",
            source_volume_key="psi",
        ),
        make_field(
            "selected_P_plane",
            "Normalized Carrier P (selected z plane)",
            _readonly_view(state.carrier_density[iz]),
            ("x", "y"),
            "pr_carrier",
            field_units,
            quantity="P",
            value_unit="1",
            source_volume_key="P",
        ),
        make_field(
            "selected_E_x_plane",
            "Material Electric Field E_x (selected z plane)",
            _readonly_view(state.E_x[iz]),
            ("x", "y"),
            "pr_space_charge",
            field_units,
            quantity="E_x",
            value_unit="1",
            colormap="coolwarm",
            source_volume_key="E_x",
        ),
        make_field(
            "selected_E_y_plane",
            "Material Electric Field E_y (selected z plane)",
            _readonly_view(state.E_y[iz]),
            ("x", "y"),
            "pr_space_charge",
            field_units,
            quantity="E_y",
            value_unit="1",
            colormap="coolwarm",
            source_volume_key="E_y",
        ),
    )
    curves = (
        CurveData(
            key="input_output_x_profile",
            display_name=f"Input/Output Intensity at y={float(y[iy]):.6g} µm",
            x=x,
            y=np.column_stack((input_intensity[:, iy], output_intensity[:, iy])),
            x_label="x",
            y_label="intensity",
            units={"x": "um", "intensity": "1/µm²"},
            series_labels=("Input", "Output"),
        ),
        CurveData(
            key="input_output_y_profile",
            display_name=f"Input/Output Intensity at x={float(x[ix]):.6g} µm",
            x=y,
            y=np.column_stack((input_intensity[ix, :], output_intensity[ix, :])),
            x_label="y",
            y_label="intensity",
            units={"y": "um", "intensity": "1/µm²"},
            series_labels=("Input", "Output"),
        ),
    )
    diagnostics = {
        "selected_x_index": ix,
        "selected_x_um": float(x[ix]),
        "selected_y_index": iy,
        "selected_y_um": float(y[iy]),
        "selected_z_index": iz,
        "selected_z_um": float(z[iz]),
        "far_field_coordinates": (
            "in-medium direction cosines s=(wavelength/n_ref)*f"
        ),
        "far_field_normalization": (
            "integral I_far ds_x ds_y equals normalized optical power"
        ),
        "far_field_log_floor_db": _FAR_FIELD_LOG_FLOOR_DB,
        "carrier_mask_role": "diagnostic_presentation_only",
        "carrier_mask_regions": carrier_regions,
        "carrier_mask_definition": (
            "union of per-channel rectangles centered at q/k_medium; "
            "half-width=max(two spectral bins, wavelength/(2*n_ref*waist))"
        ),
        "longitudinal_optical_observation": (
            "midpoint channel intensity reconstructed exactly from retained "
            "transport source minus uniform dark/background"
        ),
        "sparse_presentation_contract": True,
    }
    return fields, selected_planes, curves, diagnostics


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
    if result.retention_summary.get("policy", "full") == "fast":
        return _fast_optical_run_data(
            result, workflow=PR_TRANSVERSE_TIMEDEPENDENT_WORKFLOW,
            geometry=geometry,
        )
    psi = np.asarray(result.psi_final)
    if psi.shape != (nz, nx, ny):
        raise ValueError("psi_final does not match grid_summary")
    profile = result.resolved_profile
    state = _state_from_potential_for_products(
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
    z = (np.arange(nz) + 0.5) * float(summary["dz_um"])
    geometry = Geometry(x=x, y=y, z=z, units="um")
    if result.retention_summary.get("policy", "full") == "fast":
        return _fast_optical_run_data(
            result, workflow=PR_TRANSVERSE_STATIC_WORKFLOW,
            geometry=geometry,
        )
    psi = np.asarray(result.psi_final)
    if psi.shape != (nz, nx, ny):
        raise ValueError("psi_final does not match grid_summary")
    profile = result.resolved_profile
    state = _state_from_potential_for_products(
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
    (
        presentation_fields,
        selected_planes,
        presentation_curves,
        presentation_diagnostics,
    ) = _static_presentation_products(
        result=result,
        geometry=geometry,
        state=state,
        input_intensity=input_intensity,
        output_intensity=output_intensity,
    )
    for field in (*presentation_fields, *selected_planes):
        fields.add(field.key, field)
    curves = CurveCollection()
    for curve in presentation_curves:
        curves.add(curve.key, curve)
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
        ("presentation", DiagnosticData(
            "presentation",
            "PR Scientific Presentation",
            presentation_diagnostics,
        )),
    ])
    return RunData(
        workflow=PR_TRANSVERSE_STATIC_WORKFLOW,
        geometry=geometry,
        fields=fields,
        curves=curves,
        diagnostics=diagnostics,
    )


__all__ = [
    "pr_transverse_result_to_run_data",
    "pr_transverse_static_result_to_run_data",
]
