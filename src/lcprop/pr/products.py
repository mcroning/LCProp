"""Presentation products for photorefractive workflow results.

The concrete PR result remains the authoritative physical result.  This module
selects fields and diagnostics that can be represented faithfully by the
shared, material-neutral ``RunData`` presentation model.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from typing import Any

import numpy as np

from lcprop.core.backend import asnumpy
from lcprop.optics.farfield import direction_cosine_spectrum
from lcprop.optics.splitstep import total_intensity
from lcprop.pr.image_amplification import (
    PR_IMAGE_AMPLIFICATION_WORKFLOW,
    PRBeamPanelImageAmplificationRunRequest,
    PRImageAmplificationExperimentRequest,
    PRImageAmplificationResult,
    PRImageAmplificationRunRequest,
)
from lcprop.pr.specs import PRRunResult, PR_TIMEDEPENDENT_WORKFLOW
from lcprop.pr.static_workflow import (
    PRStaticRunResult,
    PR_STATIC_WORKFLOW,
)
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


_FAST_VOLUME_MESSAGE = (
    "Not retrieved in Fast mode; rerun with Full result retrieval to inspect "
    "this volume."
)


def _is_fast_result(result: Any) -> bool:
    return result.retention_summary.get("policy", "full") == "fast"


def _fast_optical_run_data(result: Any, *, workflow: str, geometry: Geometry) -> RunData:
    """Present retained optical endpoints without inventing missing volumes."""

    A_initial = _copied_array(result.A_initial)
    A_final = _copied_array(result.A_final)
    launch_summary = deepcopy(result.launch_summary)
    result_diagnostics = getattr(result, "diagnostics", {})
    backend_summary = getattr(
        result, "backend_summary", result_diagnostics.get("backend", {})
    )
    fields = FieldCollection()
    for key, title, value in (
        ("input_intensity", "Input Plane Intensity", A_initial),
        ("output_intensity", "Output Plane Intensity", A_final),
    ):
        fields.add(key, make_field(
            key,
            title,
            _optical_intensity(value, launch_summary),
            ("x", "y"),
            "intensity",
            {"x": "um", "y": "um"},
            quantity="normalized_intensity",
            value_unit="1/µm²",
        ))
    wavelengths = launch_summary.get("wavelengths_um", [])
    refractive_index = launch_summary.get("refractive_index")
    if wavelengths and refractive_index is not None:
        spectrum = direction_cosine_spectrum(
            A_final,
            dx_um=float(geometry.dx()),
            dy_um=float(geometry.dy()),
            wavelength_um=float(wavelengths[0]),
            refractive_index=float(refractive_index),
            coherence_groups=tuple(launch_summary["coherence_groups"]),
            xp=np,
        )
        fields.add("far_field_intensity", make_field(
            "far_field_intensity", "Output Far-Field Intensity",
            np.asarray(spectrum.intensity), ("s_x", "s_y"),
            "far_field_intensity", {"s_x": "1", "s_y": "1"},
            quantity="direction_cosine_power_density",
            value_unit="normalized power / direction-cosine²", colormap="magma",
            coordinates={"s_x": spectrum.s_x, "s_y": spectrum.s_y},
        ))
    if workflow == PR_STATIC_WORKFLOW:
        completed = int(result.completed_slices)
        power_drift = float(result.power_final - result.power_initial)
        summary = {
            "material": "photorefractive",
            "workflow": workflow,
            "status": result.status,
            "converged": bool(result.converged),
            "completed_slices": completed,
            "total_slices": int(result.grid_summary["Nz"]),
            "z_reached_um": completed * float(result.grid_summary["dz_um"]),
            "normalized_field_integral_initial": float(result.power_initial),
            "normalized_field_integral_final": float(result.power_final),
            "relative_power_drift": (
                power_drift / float(result.power_initial)
                if result.power_initial != 0.0 else 0.0
            ),
            "all_completed_slices_converged": all(
                value.converged for value in result.slice_summaries
            ),
            "grid": deepcopy(result.grid_summary),
            "launch": launch_summary,
            "backend": deepcopy(result.backend_summary),
            "resolved_tolerances": deepcopy(result.tolerance_provenance),
            "replay": deepcopy(result.replay_diagnostics),
            "result_retention": deepcopy(result.retention_summary),
        }
        diagnostics = DiagnosticCollection([
            ("summary", DiagnosticData("summary", "Summary", summary)),
            ("static_convergence", DiagnosticData(
                "static_convergence", "Static Convergence by Slice",
                {"rows": [asdict(value) for value in result.slice_summaries]},
            )),
            ("static_iteration_history", DiagnosticData(
                "static_iteration_history", "Static Coupled Iteration History",
                {"rows": [asdict(value) for value in result.iteration_records]},
            )),
        ])
    else:
        diagnostics = DiagnosticCollection([
            ("summary", DiagnosticData("summary", "Summary", {
                "material": "photorefractive",
                "normalized_field_integral_initial": float(result.power_initial),
                "normalized_field_integral_final": float(result.power_final),
                "completed_material_steps": int(result.completed_steps),
                "requested_material_steps": int(result.requested_steps),
                "material_time_normalized": float(result.time_normalized),
                "status": result.status,
                "grid": deepcopy(result.grid_summary),
                "launch": launch_summary,
                "result_retention": deepcopy(result.retention_summary),
            })),
            ("pr_workflow", DiagnosticData(
                "pr_workflow", "PR Workflow Diagnostics",
                deepcopy(result_diagnostics)
            )),
        ])
    return RunData(
        workflow=workflow,
        geometry=geometry,
        fields=fields,
        curves=CurveCollection(),
        diagnostics=diagnostics,
        longitudinal_enabled=False,
        longitudinal_message=_FAST_VOLUME_MESSAGE,
    )


def _copied_array(value: Any) -> np.ndarray:
    """Return a detached host copy suitable for presentation ownership."""

    return np.asarray(asnumpy(value)).copy()


def _readonly_shared_numpy_array(value: Any) -> np.ndarray:
    """Share a canonical host array through a read-only presentation view.

    Canonical PR workflows detach retained volumes at their result boundary.
    A view avoids duplicating those volumes while preventing GUI/presentation
    consumers from mutating the authoritative result.  Keep a defensive copy
    for non-NumPy inputs whose ownership is not established by the canonical
    workflow.
    """

    if not isinstance(value, np.ndarray):
        return _copied_array(value)
    return np.asarray(memoryview(value).toreadonly())


def _geometry_from_grid_summary(
    summary: dict[str, Any],
    *,
    completed_slices: int | None = None,
) -> Geometry:
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
    resolved_slices = nz if completed_slices is None else int(completed_slices)
    if not 0 <= resolved_slices <= nz:
        raise ValueError("completed PR slices must be between zero and Nz")
    z_um = np.arange(resolved_slices) * dz_um
    return Geometry(x=x_um, y=y_um, z=z_um, units="um")


def _geometry_from_pr_result(result: PRRunResult) -> Geometry:
    return _geometry_from_grid_summary(result.grid_summary)


def _image_amplification_default_display_extent(
    result: PRImageAmplificationResult,
    base: RunData,
) -> tuple[float, float, float, float] | None:
    """Return a clamped signal/screen view without cropping stored fields."""

    image_request = result.image_request
    try:
        if isinstance(
            image_request,
            (
                PRBeamPanelImageAmplificationRunRequest,
                PRImageAmplificationExperimentRequest,
            ),
        ):
            signal_index = int(image_request.signal_channel_index)
            screen = next(
                assignment.elements[0]
                for assignment in image_request.launch_configuration.channel_elements
                if assignment.channel_index == signal_index
            )
            screen_center_x = float(screen.placement.center_x_um)
            screen_center_y = float(screen.placement.center_y_um)
            screen_width = float(screen.placement.width_um)
            screen_height = float(screen.placement.height_um)
        elif isinstance(image_request, PRImageAmplificationRunRequest):
            signal_index = 1
            screen_size = float(image_request.launch.image_physical_size_um)
            screen_width = screen_size
            screen_height = screen_size
            screen_center_x = float(result.request.beams.channels[1].x0_um)
            screen_center_y = float(result.request.beams.channels[1].y0_um)
        else:
            return None
        signal = result.request.beams.channels[signal_index]
        center_x = float(signal.x0_um)
        center_y = float(signal.y0_um)
        waist_x = float(signal.waist_x_um)
        waist_y = float(signal.waist_y_um)
        x = np.asarray(base.geometry.x, dtype=float)
        y = np.asarray(base.geometry.y, dtype=float)
    except (AttributeError, IndexError, StopIteration, TypeError, ValueError):
        return None
    values = np.asarray([
        center_x,
        center_y,
        waist_x,
        waist_y,
        screen_center_x,
        screen_center_y,
        screen_width,
        screen_height,
    ])
    if (
        x.size < 2
        or y.size < 2
        or not np.all(np.isfinite(values))
        or waist_x <= 0.0
        or waist_y <= 0.0
        or screen_width <= 0.0
        or screen_height <= 0.0
    ):
        return None

    # BeamChannel waists are 1/e^2 intensity radii.  The established PR
    # aperture check uses a two-waist envelope, so presentation reuses it.
    x_min = min(center_x - 2.0 * waist_x, screen_center_x - 0.5 * screen_width)
    x_max = max(center_x + 2.0 * waist_x, screen_center_x + 0.5 * screen_width)
    y_min = min(center_y - 2.0 * waist_y, screen_center_y - 0.5 * screen_height)
    y_max = max(center_y + 2.0 * waist_y, screen_center_y + 0.5 * screen_height)
    x_margin = 0.15 * (x_max - x_min)
    y_margin = 0.15 * (y_max - y_min)
    extent = (
        max(float(x[0]), x_min - x_margin),
        min(float(x[-1]), x_max + x_margin),
        max(float(y[0]), y_min - y_margin),
        min(float(y[-1]), y_max + y_margin),
    )
    if extent[0] >= extent[1] or extent[2] >= extent[3]:
        return None
    return extent


def _validated_result_arrays(
    result: PRRunResult,
    geometry: Geometry,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    A_initial = _copied_array(result.A_initial)
    A_final = _copied_array(result.A_final)
    volume_adapter = (
        _readonly_shared_numpy_array
        if result.status == "cancelled"
        else _copied_array
    )
    E_initial = volume_adapter(result.E_initial)
    E_final = volume_adapter(result.E_final)
    source_intensity = volume_adapter(result.source_intensity_stack)

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
    if _is_fast_result(result):
        return _fast_optical_run_data(
            result, workflow=PR_TIMEDEPENDENT_WORKFLOW, geometry=geometry
        )
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
    launch_fallback = (
        result.status == "cancelled"
        and result.diagnostics.get("final_optical_observation")
        == "unpropagated_launch_fallback"
    )

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
                (
                    "Launch-Plane Intensity (Cancellation Fallback)"
                    if launch_fallback
                    else "Output Plane Intensity"
                ),
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
                (
                    "Launch-Plane PR-Driving Intensity "
                    "(Cancellation Fallback)"
                    if launch_fallback
                    else "Final PR-Driving Intensity"
                ),
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


def augment_pr_image_amplification_run_data(
    result: PRImageAmplificationResult,
    base: RunData,
) -> RunData:
    """Augment already-created ordinary PR products with image diagnostics."""

    if not isinstance(result, PRImageAmplificationResult):
        raise TypeError("result must be a PRImageAmplificationResult")
    if result.image_request is None:
        raise ValueError("image-amplification result lacks source provenance")
    if not isinstance(base, RunData):
        raise TypeError("base must be the selected operation's RunData")
    fields = FieldCollection(list(base.fields.items()))
    image_request = result.image_request
    source = np.asarray(
        image_request.source.grayscale, dtype=np.float64
    ).copy()
    source /= float(np.max(source))
    source_xy = source.T.copy()
    spatial_units = {"x": "um", "y": "um"}
    source_units = {"source_x": "pixel", "source_y": "pixel"}
    source_coordinates = {
        "source_x": np.arange(source_xy.shape[0], dtype=float),
        "source_y": np.arange(source_xy.shape[1], dtype=float),
    }
    default_display_extent = _image_amplification_default_display_extent(
        result,
        base,
    )
    for key, display_name, values, axes, kind, units, coordinates in (
        (
            "image_source",
            "Decoded Source Image",
            source_xy,
            ("source_x", "source_y"),
            "image_source",
            source_units,
            source_coordinates,
        ),
        (
            "image_transmission",
            "Simulation-Grid Image Transmission",
            result.image_transmission,
            ("x", "y"),
            "image_transmission",
            spatial_units,
            {},
        ),
        (
            "input_signal_intensity",
            "Image-Bearing Input Signal Intensity",
            np.abs(result.input_signal_field) ** 2,
            ("x", "y"),
            "intensity",
            spatial_units,
            {},
        ),
        (
            "output_signal_intensity",
            "Isolated Output Signal Intensity",
            np.abs(result.output_signal_field) ** 2,
            ("x", "y"),
            "intensity",
            spatial_units,
            {},
        ),
        (
            "amplified_image",
            "Back-Propagated Amplified Image",
            np.abs(result.backpropagated_signal_field) ** 2,
            ("x", "y"),
            "intensity",
            spatial_units,
            {},
        ),
        (
            "zero_response_image",
            "Zero-Response Reconstruction",
            np.abs(result.zero_response_backpropagated_signal_field) ** 2,
            ("x", "y"),
            "intensity",
            spatial_units,
            {},
        ),
    ):
        fields.add(
            key,
            make_field(
                key,
                display_name,
                np.asarray(values).copy(),
                axes,
                kind,
                units,
                quantity=key,
                value_unit="1",
                coordinates=coordinates,
                colormap="gray" if kind in ("image_source", "mask") else "viridis",
                default_display_extent=(
                    default_display_extent if axes == ("x", "y") else None
                ),
                initially_selected=key == "amplified_image",
            ),
        )

    if isinstance(
        image_request,
        (
            PRBeamPanelImageAmplificationRunRequest,
            PRImageAmplificationExperimentRequest,
        ),
    ):
        launch_configuration = image_request.launch_configuration
        signal = launch_configuration.beams.channels[
            image_request.signal_channel_index
        ]
        wavelength_um = float(signal.wavelength_um)
        coherence_group = launch_configuration.beams.coherence_groups[
            image_request.signal_channel_index
        ]
        incident_ratio = image_request.incident_signal_to_pump_power_ratio
        screen = next(
            assignment.elements[0]
            for assignment in launch_configuration.channel_elements
            if assignment.channel_index == image_request.signal_channel_index
        )
        preprocessing_policy = screen.preprocessing_policy
        pump_channel_index = image_request.pump_channel_index
        signal_channel_index = image_request.signal_channel_index
    else:
        launch = image_request.launch
        wavelength_um = float(launch.wavelength_um)
        coherence_group = launch.coherence_group
        incident_ratio = launch.incident_signal_to_pump_power_ratio
        preprocessing_policy = image_request.source.preprocessing_policy
        pump_channel_index = 0
        signal_channel_index = 1
    coherent_output = np.sum(np.asarray(result.run_result.A_final), axis=0)
    spectrum = direction_cosine_spectrum(
        coherent_output[None, ...],
        dx_um=float(result.run_result.grid_summary["dx_um"]),
        dy_um=float(result.run_result.grid_summary["dy_um"]),
        wavelength_um=wavelength_um,
        refractive_index=float(result.request.material.refractive_index),
        coherence_groups=(coherence_group,),
        xp=np,
    )
    far_field = np.asarray(spectrum.intensity)
    far_relative = np.maximum(far_field / float(np.max(far_field)), 1e-12)
    angular_coordinates = {
        "s_x": np.asarray(spectrum.s_x),
        "s_y": np.asarray(spectrum.s_y),
    }
    fields.add(
        "signal_carrier_mask",
        make_field(
            "signal_carrier_mask",
            "Signal-Carrier Fourier Mask",
            np.fft.fftshift(result.signal_carrier_mask).astype(float),
            ("s_x", "s_y"),
            "mask",
            {"s_x": "1", "s_y": "1"},
            quantity="signal_carrier_mask",
            value_unit="1",
            coordinates=angular_coordinates,
            colormap="gray",
        ),
    )
    for key, name, values, kind in (
        (
            "far_field_intensity",
            "Output Far-Field Intensity",
            far_field,
            "far_field_intensity",
        ),
        (
            "far_field_log_db",
            "Output Far Field (dB relative to peak)",
            10.0 * np.log10(far_relative),
            "far_field_log",
        ),
    ):
        fields.add(
            key,
            make_field(
                key,
                name,
                values,
                ("s_x", "s_y"),
                kind,
                {"s_x": "1", "s_y": "1"},
                quantity=key,
                value_unit="1",
                coordinates=angular_coordinates,
                colormap="magma",
            ),
        )

    diagnostics = DiagnosticCollection(list(base.diagnostics.items()))
    diagnostics.add(
        "image_amplification_metrics",
        DiagnosticData(
            "image_amplification_metrics",
            "Image Amplification Metrics",
            {
                "measured_signal_gain": result.measured_absolute_signal_gain,
                "analytic_signal_gain": result.analytic_absolute_signal_gain,
                "image_correlation": result.image_intensity_correlation,
                "normalized_image_rmse": result.normalized_image_rmse,
                "incident_signal_power_mW": result.incident_channel_powers_mW[1],
                "post_screen_signal_power_mW": (
                    result.post_element_channel_powers_mW[1]
                ),
                "signal_screen_throughput": result.signal_throughput_fraction,
                "measured_gain_reference_signal_power_mW": (
                    result.measured_gain_reference_signal_power_mW
                ),
                "output_isolated_signal_power_mW": (
                    result.output_isolated_signal_power_mW
                ),
                "total_power_entering_pr_medium_mW": (
                    result.post_element_total_power_mW
                ),
                "normalized_optical_power_drift": (
                    result.normalized_power_relative_drift
                ),
                "measured_gain_reference": (
                    "carrier-isolated post-screen field at z=0"
                ),
                "measured_gain_vs_z_available": False,
                "measured_gain_vs_z_reason": (
                    "base result stores input/output complex optical fields "
                    "but no per-z complex optical-field history"
                ),
            },
        ),
    )
    diagnostics.add(
        "image_amplification",
        DiagnosticData(
            "image_amplification",
            "Image Amplification",
            {
                "source_kind": image_request.source.source_kind,
                "source_asset_id": image_request.source.asset_id,
                "source_basename": image_request.source.basename,
                "source_sha256": image_request.source.sha256,
                "source_dimensions_pixels": [
                    image_request.source.width,
                    image_request.source.height,
                ],
                "source_decoded_mode": image_request.source.decoded_mode,
                "source_encoded_format": image_request.source.encoded_format,
                "preprocessing_policy": preprocessing_policy,
                "pump_channel_index": pump_channel_index,
                "signal_channel_index": signal_channel_index,
                "alpha_policy": "discarded_not_an_optical_mask",
                "simulation_grid": [result.request.grid.Nx, result.request.grid.Ny],
                "incident_pump_power_mW": result.incident_channel_powers_mW[0],
                "incident_signal_power_mW": result.incident_channel_powers_mW[1],
                "incident_total_power_mW": result.incident_total_power_mW,
                "post_element_pump_power_mW": (
                    result.post_element_channel_powers_mW[0]
                ),
                "post_element_signal_power_mW": (
                    result.post_element_channel_powers_mW[1]
                ),
                "power_entering_pr_medium_mW": result.post_element_total_power_mW,
                "signal_throughput_fraction": result.signal_throughput_fraction,
                "transparency_policy": result.transparency_policy,
                "derived_incident_signal_to_pump_power_ratio": (
                    incident_ratio
                ),
                "measured_absolute_signal_gain": result.measured_absolute_signal_gain,
                "analytic_absolute_signal_gain": result.analytic_absolute_signal_gain,
                "analytic_gamma_p_L": result.analytic_gamma_p_L,
                "image_intensity_correlation": result.image_intensity_correlation,
                "normalized_image_rmse": result.normalized_image_rmse,
                "normalized_power_relative_drift": (
                    result.normalized_power_relative_drift
                ),
            },
        ),
    )
    return RunData(
        workflow=PR_IMAGE_AMPLIFICATION_WORKFLOW,
        geometry=base.geometry,
        fields=fields,
        curves=base.curves,
        diagnostics=diagnostics,
        longitudinal_enabled=base.longitudinal_enabled,
        longitudinal_message=base.longitudinal_message,
    )


def pr_image_amplification_result_to_run_data(
    result: PRImageAmplificationResult,
) -> RunData:
    """Expose the historical reduced-TD image products in shared RunData."""

    return augment_pr_image_amplification_run_data(
        result,
        pr_result_to_run_data(result.run_result),
    )


def _validated_static_result_arrays(
    result: PRStaticRunResult,
    geometry: Geometry,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    A_initial = _copied_array(result.A_initial)
    A_final = _copied_array(result.A_final)
    E_initial = _readonly_shared_numpy_array(result.E_initial)
    E_final = _readonly_shared_numpy_array(result.E_final)
    source_intensity = _readonly_shared_numpy_array(
        result.source_intensity_stack
    )
    residual = _readonly_shared_numpy_array(result.residual_stack)

    nx = len(np.asarray(geometry.x))
    ny = len(np.asarray(geometry.y))
    completed = int(result.completed_slices)
    if A_initial.ndim != 3 or A_initial.shape[1:] != (nx, ny):
        raise ValueError(
            "PR static A_initial must have shape (Nch, Nx, Ny) "
            "matching grid_summary"
        )
    if A_final.shape != A_initial.shape:
        raise ValueError("PR static A_final shape must match A_initial")

    expected_volume_shape = (completed, nx, ny)
    for name, value in (
        ("E_initial", E_initial),
        ("E_final", E_final),
        ("source_intensity_stack", source_intensity),
        ("residual_stack", residual),
    ):
        if value.shape != expected_volume_shape:
            raise ValueError(
                f"PR static {name} must have shape "
                "(completed_slices, Nx, Ny)"
            )

    return (
        A_initial,
        A_final,
        E_initial,
        E_final,
        source_intensity,
        residual,
    )


def pr_static_result_to_run_data(result: PRStaticRunResult) -> RunData:
    """Convert a coupled-static PR result into shared presentation data."""

    if not isinstance(result, PRStaticRunResult):
        raise TypeError("result must be a PRStaticRunResult")

    completed = int(result.completed_slices)
    geometry = _geometry_from_grid_summary(
        result.grid_summary,
        completed_slices=completed,
    )
    if _is_fast_result(result):
        return _fast_optical_run_data(
            result, workflow=PR_STATIC_WORKFLOW, geometry=geometry
        )
    (
        A_initial,
        A_final,
        E_initial,
        E_final,
        source_intensity,
        residual,
    ) = _validated_static_result_arrays(result, geometry)
    launch_summary = deepcopy(result.launch_summary)
    input_intensity = _optical_intensity(A_initial, launch_summary)
    output_intensity = _optical_intensity(A_final, launch_summary)

    spatial_units = {"x": "um", "y": "um"}
    volume_units = {"z": "um", "x": "um", "y": "um"}
    field_items = [
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
                (
                    "Output Plane Intensity"
                    if result.status != "cancelled"
                    else "Intensity at Current z"
                ),
                output_intensity,
                ("x", "y"),
                "intensity",
                spatial_units,
                quantity="normalized_intensity",
                value_unit="1/µm²",
            ),
        ),
    ]
    if completed:
        selected_z_index = completed // 2
        field_items.extend([
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
                    "Static PR Space-Charge Field",
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
        ])
    field_items.extend([
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
                "Static PR Space-Charge Field",
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
                "Static PR-Driving Intensity",
                source_intensity,
                ("z", "x", "y"),
                "pr_driving_intensity",
                volume_units,
                "longitudinal",
                quantity="normalized_pr_driving_intensity",
                value_unit="1",
            ),
        ),
        (
            "pr_static_residual_stack",
            make_field(
                "pr_static_residual_stack",
                "Static PR Residual",
                residual,
                ("z", "x", "y"),
                "pr_static_residual",
                volume_units,
                "longitudinal",
                quantity="normalized_pr_static_residual",
                value_unit="1",
                colormap="coolwarm",
            ),
        ),
    ])

    curves = CurveCollection()
    if result.slice_summaries:
        z_um = np.asarray(
            [summary.z_um for summary in result.slice_summaries],
            dtype=float,
        )
        curve_specs = (
            (
                "static_final_residual_rms",
                "Final Residual RMS",
                "residual RMS",
                [summary.final_residual_rms for summary in result.slice_summaries],
                "log",
            ),
            (
                "static_final_residual_max",
                "Final Residual Maximum",
                "residual maximum",
                [summary.final_residual_max for summary in result.slice_summaries],
                "log",
            ),
            (
                "static_coupled_passes",
                "Coupled Passes",
                "coupled passes",
                [summary.coupled_passes for summary in result.slice_summaries],
                "linear",
            ),
            (
                "static_delta_E_max",
                "Final Material-State Change Maximum",
                "maximum |delta E|",
                [summary.final_delta_E_max for summary in result.slice_summaries],
                "log",
            ),
        )
        for key, display_name, y_label, values, y_scale in curve_specs:
            curves.add(
                key,
                CurveData(
                    key=key,
                    display_name=display_name,
                    x=z_um,
                    y=np.asarray(values, dtype=float),
                    x_label="z",
                    y_label=y_label,
                    units={"z": "um"},
                    y_scale=y_scale,
                ),
            )

    total_slices = int(result.grid_summary["Nz"])
    dz_um = float(result.grid_summary["dz_um"])
    power_drift = float(result.power_final - result.power_initial)
    relative_power_drift = (
        power_drift / float(result.power_initial)
        if result.power_initial != 0.0
        else 0.0
    )
    summary_values = {
        "material": "photorefractive",
        "workflow": PR_STATIC_WORKFLOW,
        "status": result.status,
        "converged": bool(result.converged),
        "completed_slices": completed,
        "total_slices": total_slices,
        "z_reached_um": float(completed * dz_um),
        "normalized_field_integral_initial": float(result.power_initial),
        "normalized_field_integral_final": float(result.power_final),
        "relative_power_drift": relative_power_drift,
        "all_completed_slices_converged": all(
            summary.converged for summary in result.slice_summaries
        ),
        "grid": deepcopy(result.grid_summary),
        "launch": launch_summary,
        "backend": deepcopy(result.backend_summary),
        "resolved_tolerances": deepcopy(result.tolerance_provenance),
        "replay": deepcopy(result.replay_diagnostics),
    }
    diagnostics = DiagnosticCollection([
        (
            "summary",
            DiagnosticData("summary", "Summary", summary_values),
        ),
        (
            "static_convergence",
            DiagnosticData(
                "static_convergence",
                "Static Convergence by Slice",
                {
                    "rows": [
                        asdict(summary) for summary in result.slice_summaries
                    ]
                },
            ),
        ),
        (
            "static_iteration_history",
            DiagnosticData(
                "static_iteration_history",
                "Static Coupled Iteration History",
                {
                    "rows": [
                        asdict(record) for record in result.iteration_records
                    ]
                },
            ),
        ),
    ])

    return RunData(
        workflow=PR_STATIC_WORKFLOW,
        geometry=geometry,
        fields=FieldCollection(field_items),
        curves=curves,
        diagnostics=diagnostics,
    )


__all__ = [
    "PR_IMAGE_AMPLIFICATION_WORKFLOW",
    "PR_STATIC_WORKFLOW",
    "PR_TIMEDEPENDENT_WORKFLOW",
    "pr_image_amplification_result_to_run_data",
    "pr_result_to_run_data",
    "pr_static_result_to_run_data",
]
