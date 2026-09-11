#!/usr/bin/env python3
"""Headless PR linearization and grating-resolution sweep.

The default path performs compact frozen-intensity material calculations. A
bounded production path is available only through ``--study production``;
this script never submits scheduler work or chooses a remote execution target.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
from time import perf_counter
from typing import Any, Sequence

import numpy as np

from lcprop.core.backend import BackendSpec
from lcprop.core.beams import BeamChannel, BeamStack
from lcprop.core.context import GridSpec
from lcprop.pr.carrier_power import carrier_power_diagnostic
from lcprop.pr.products import pr_static_result_to_run_data
from lcprop.pr.reduced_linearized import (
    PRReducedLinearizedSpec,
    solve_pr_reduced_linearized_intensity,
)
from lcprop.pr.specs import PRMaterialSpec
from lcprop.pr.static import PRStaticSolverOptions, solve_pr_static_intensity_batched
from lcprop.pr.static_workflow import (
    PRStaticRunRequest,
    PRStaticWorkflowOptions,
    run_pr_static,
)
from lcprop.pr.transverse.linearized_reference import (
    PRBiasedLinearizedReferenceSpec,
    solve_pr_biased_linearized_reference,
)
from lcprop.pr.transverse.products import pr_transverse_static_result_to_run_data
from lcprop.pr.transverse.specs import (
    PR_FULL_TRANSVERSE_PERIODIC_BIASED_CURRENT_V1,
    PR_MATERIAL_RESPONSE_LINEARIZED,
    PRTransverseBoundaryProfile,
    PRTransverseMaterialResponseSpec,
)
from lcprop.pr.transverse.static_workflow import (
    PRTransverseStaticRunRequest,
    PRTransverseStaticWorkflowOptions,
    run_pr_transverse_static,
)
from lcprop.pr.transverse.transport import state_from_potential


APERTURE_UM = 200.0
GRATING_MAGNITUDE_RAD_PER_UM = math.pi
GRATING_PERIOD_UM = 2.0
TOTAL_POWER_MW = 2.0
WAVELENGTH_UM = 0.633
REFRACTIVE_INDEX = 2.4
INTERACTION_LENGTH_UM = 1000.0
DARK_INTENSITY = 0.01
REFERENCE_INTENSITY = 1.0 + DARK_INTENSITY
DEFAULT_GRIDS = (256, 512, 1024)
DEFAULT_MODULATIONS = (0.5, 0.25, 0.125, 0.0625, 0.03125)
RESOLVED_POINTS_PER_PERIOD = 4.0
NYQUIST_POINTS_PER_PERIOD = 2.0


@dataclass(frozen=True)
class CarrierGeometry:
    """Exact symmetric two-carrier and launch-center geometry."""

    name: str
    beam_1_k_rad_per_um: tuple[float, float]
    beam_2_k_rad_per_um: tuple[float, float]
    delta_k_rad_per_um: tuple[float, float]
    grating_magnitude_rad_per_um: float
    grating_period_um: float
    beam_1_center_um: tuple[float, float]
    beam_2_center_um: tuple[float, float]


def carrier_geometry(name: str) -> CarrierGeometry:
    """Return the exact requested carriers and symmetric crossing centers."""

    medium_wavenumber = 2.0 * math.pi * REFRACTIVE_INDEX / WAVELENGTH_UM
    half_length = 0.5 * INTERACTION_LENGTH_UM
    if name == "x":
        first = (-0.5 * math.pi, 0.0)
        second = (0.5 * math.pi, 0.0)
    elif name == "45deg":
        component = math.pi / (2.0 * math.sqrt(2.0))
        first = (-component, component)
        second = (component, -component)
    else:
        raise ValueError("geometry must be 'x' or '45deg'")
    delta = (second[0] - first[0], second[1] - first[1])

    def launch_center(carrier: tuple[float, float]) -> tuple[float, float]:
        return (
            -carrier[0] * half_length / medium_wavenumber,
            -carrier[1] * half_length / medium_wavenumber,
        )

    return CarrierGeometry(
        name=name,
        beam_1_k_rad_per_um=first,
        beam_2_k_rad_per_um=second,
        delta_k_rad_per_um=delta,
        grating_magnitude_rad_per_um=math.hypot(*delta),
        grating_period_um=2.0 * math.pi / math.hypot(*delta),
        beam_1_center_um=launch_center(first),
        beam_2_center_um=launch_center(second),
    )


def powers_for_visibility(
    visibility: float, *, total_power_mW: float = TOTAL_POWER_MW
) -> tuple[float, float]:
    """Return unequal powers with fixed total and requested visibility."""

    modulation = float(visibility)
    total = float(total_power_mW)
    if not math.isfinite(modulation) or not 0.0 < modulation <= 1.0:
        raise ValueError("visibility must be finite and in (0, 1]")
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("total_power_mW must be finite and positive")
    imbalance = math.sqrt(max(0.0, 1.0 - modulation * modulation))
    return (0.5 * total * (1.0 + imbalance), 0.5 * total * (1.0 - imbalance))


def visibility_from_powers(first: float, second: float) -> float:
    total = float(first) + float(second)
    if total <= 0.0:
        raise ValueError("power sum must be positive")
    return 2.0 * math.sqrt(float(first) * float(second)) / total


def harmonic_resolution(points_per_period: float) -> str:
    """Classify discrete support using Nyquist and four-point gates."""

    value = float(points_per_period)
    if value < NYQUIST_POINTS_PER_PERIOD:
        return "underresolved"
    if value < RESOLVED_POINTS_PER_PERIOD:
        return "marginal"
    return "resolved"


def resolution_metadata(N: int, *, geometry: str | None = None) -> dict[str, Any]:
    """Return physical spacing and K/2K/3K sampling metadata."""

    size = int(N)
    if size < 1:
        raise ValueError("N must be positive")
    dx = APERTURE_UM / size
    fundamental = GRATING_PERIOD_UM / dx
    metadata: dict[str, Any] = {"N": size, "dx_um": dx}
    for harmonic in (1, 2, 3):
        points = fundamental / harmonic
        label = "K" if harmonic == 1 else f"{harmonic}K"
        metadata[f"points_per_{label}"] = points
        metadata[f"resolution_{label}"] = harmonic_resolution(points)
    if geometry is not None:
        model = carrier_geometry(geometry)
        fundamental_modes = tuple(
            component * APERTURE_UM / (2.0 * math.pi)
            for component in model.delta_k_rad_per_um
        )
        metadata["periodic_fundamental_mode_components"] = fundamental_modes
        metadata["periodic_fundamental_bin_offsets"] = tuple(
            abs(component - round(component)) for component in fundamental_modes
        )
        metadata["periodic_fundamental_commensurate"] = all(
            math.isclose(component, round(component), abs_tol=1e-12)
            for component in fundamental_modes
        )
    return metadata


def broad_beam_intensity(
    N: int,
    visibility: float,
    *,
    geometry: str,
    reference_intensity: float = REFERENCE_INTENSITY,
    collapse_x_control_y: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct ``I0*(1 + m*cos(K.r))`` on the physical aperture."""

    size = int(N)
    model = carrier_geometry(geometry)
    x = (np.arange(size, dtype=np.float64) - size / 2.0) * APERTURE_UM / size
    ny = 3 if collapse_x_control_y and geometry == "x" else size
    y = (np.arange(ny, dtype=np.float64) - ny / 2.0) * APERTURE_UM / ny
    phase = (
        model.delta_k_rad_per_um[0] * x[:, None]
        + model.delta_k_rad_per_um[1] * y[None, :]
    )
    modulation_pattern = np.cos(phase)
    modulation_pattern -= np.mean(modulation_pattern)
    intensity = float(reference_intensity) * (
        1.0 + float(visibility) * modulation_pattern
    )
    return intensity.astype(np.float64), x, y


def complex_harmonic(
    field: np.ndarray,
    *,
    x_um: np.ndarray,
    y_um: np.ndarray,
    wavevector_rad_per_um: tuple[float, float],
    harmonic: int,
    window: bool,
) -> complex:
    """Extract a compact complex harmonic by optional windowed demodulation."""

    values = np.asarray(field, dtype=np.float64)
    if values.shape != (x_um.size, y_um.size):
        raise ValueError("field shape must match x_um and y_um")
    phase = int(harmonic) * (
        float(wavevector_rad_per_um[0]) * x_um[:, None]
        + float(wavevector_rad_per_um[1]) * y_um[None, :]
    )
    if window:
        wx = np.hanning(x_um.size) if x_um.size > 3 else np.ones(x_um.size)
        wy = np.hanning(y_um.size) if y_um.size > 3 else np.ones(y_um.size)
        weights = wx[:, None] * wy[None, :]
    else:
        weights = np.ones(values.shape, dtype=np.float64)
    weighted_mean = float(np.sum(weights * values) / np.sum(weights))
    centered = values - weighted_mean
    return complex(
        np.sum(weights * centered * np.exp(-1j * phase)) / np.sum(weights)
    )


def observed_order(scales: Sequence[float], errors: Sequence[float]) -> float:
    """Return the least-squares log-log order for positive finite samples."""

    x = np.asarray(scales, dtype=np.float64)
    y = np.asarray(errors, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
    if np.count_nonzero(valid) < 2:
        return float("nan")
    return float(np.polyfit(np.log(x[valid]), np.log(y[valid]), 1)[0])


def _relative_l2(candidate: np.ndarray, reference: np.ndarray) -> float:
    difference = np.linalg.norm(np.asarray(candidate) - np.asarray(reference))
    scale = np.linalg.norm(np.asarray(reference))
    return float(difference / scale) if scale else float(difference)


def _rms(value: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.abs(np.asarray(value)) ** 2)))


def _harmonic_record(
    field: np.ndarray,
    *,
    x_um: np.ndarray,
    y_um: np.ndarray,
    geometry: CarrierGeometry,
    resolution: dict[str, Any],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    fundamental = None
    for harmonic in (1, 2, 3):
        label = "K" if harmonic == 1 else f"{harmonic}K"
        status = resolution[f"resolution_{label}"]
        coefficient = None
        if status != "underresolved":
            coefficient = complex_harmonic(
                field,
                x_um=x_um,
                y_um=y_um,
                wavevector_rad_per_um=geometry.delta_k_rad_per_um,
                harmonic=harmonic,
                window=geometry.name == "45deg",
            )
        magnitude = None if coefficient is None else abs(coefficient)
        if harmonic == 1:
            fundamental = magnitude
        output[f"harmonic_{harmonic}_magnitude"] = magnitude
        output[f"harmonic_{harmonic}_phase_rad"] = (
            None
            if coefficient is None
            else math.atan2(coefficient.imag, coefficient.real)
        )
        output[f"harmonic_{harmonic}_relative_to_K"] = (
            None
            if magnitude is None or fundamental in (None, 0.0)
            else magnitude / fundamental
        )
        output[f"harmonic_{harmonic}_resolution"] = status
    return output


def run_material_case(
    *,
    N: int,
    visibility: float,
    geometry_name: str,
    include_full_linearized: bool,
) -> list[dict[str, Any]]:
    """Run one compact frozen-intensity nonlinear/linearized comparison."""

    geometry = carrier_geometry(geometry_name)
    resolution = resolution_metadata(N, geometry=geometry_name)
    intensity, x_um, y_um = broad_beam_intensity(
        N,
        visibility,
        geometry=geometry_name,
        collapse_x_control_y=False,
    )
    dx_normalized = PRMaterialSpec().characteristic_wavenumber_per_um * (
        APERTURE_UM / int(N)
    )
    dy_normalized = dx_normalized if geometry_name == "45deg" else (
        PRMaterialSpec().characteristic_wavenumber_per_um
        * APERTURE_UM
        / y_um.size
    )
    nonlinear_started = perf_counter()
    nonlinear = solve_pr_static_intensity_batched(
        intensity,
        applied_field=0.0,
        background_intensity=DARK_INTENSITY,
        dx_normalized=dx_normalized,
        options=PRStaticSolverOptions(
            max_iterations=40,
            residual_rms_tolerance=1e-10,
            residual_max_tolerance=1e-9,
        ),
    )
    nonlinear_runtime = perf_counter() - nonlinear_started
    linearized_started = perf_counter()
    reduced_linearized = solve_pr_reduced_linearized_intensity(
        intensity,
        spec=PRReducedLinearizedSpec(
            reference_intensity=REFERENCE_INTENSITY,
            applied_field=0.0,
            background_intensity=DARK_INTENSITY,
            dx_normalized=dx_normalized,
        ),
    )
    reduced_linearized_runtime = perf_counter() - linearized_started
    difference = np.asarray(nonlinear.E) - np.asarray(reduced_linearized.E)
    common = {
        "study": "frozen_intensity_material",
        "fixture": "broad_beam",
        "geometry": geometry_name,
        "N": int(N),
        "m": float(visibility),
        "dx_um": resolution["dx_um"],
        "points_per_K": resolution["points_per_K"],
        "points_per_2K": resolution["points_per_2K"],
        "points_per_3K": resolution["points_per_3K"],
        "resolution_K": resolution["resolution_K"],
        "resolution_2K": resolution["resolution_2K"],
        "resolution_3K": resolution["resolution_3K"],
        "periodic_fundamental_mode_components": resolution[
            "periodic_fundamental_mode_components"
        ],
        "periodic_fundamental_bin_offsets": resolution[
            "periodic_fundamental_bin_offsets"
        ],
        "periodic_fundamental_commensurate": resolution[
            "periodic_fundamental_commensurate"
        ],
        "total_power_mW": TOTAL_POWER_MW,
        "reference_intensity": REFERENCE_INTENSITY,
        "launch_attenuation": "none",
        "material_rel_L2_vs_linearized": _relative_l2(
            nonlinear.E, reduced_linearized.E
        ),
        "material_abs_L2_vs_linearized": _rms(difference),
        "material_max_abs_vs_linearized": float(np.max(np.abs(difference))),
        "linear_response_rms": _rms(reduced_linearized.delta_E),
        "optical_rel_L2_vs_linearized": None,
        "output_intensity_rel_L2_vs_linearized": None,
        "pump_power_input_normalized": None,
        "pump_power_output_normalized": None,
        "pump_gain": None,
        "pump_delta_power_normalized": None,
        "signal_power_input_normalized": None,
        "signal_power_output_normalized": None,
        "signal_gain": None,
        "signal_delta_power_normalized": None,
        "carrier_gain_difference": None,
        "carrier_separation_quality": None,
        "carrier_power_balance_error": None,
        "power_drift": None,
    }
    rows = [
        {
            **common,
            "model": "reduced_nonlinear",
            "converged": nonlinear.converged,
            "runtime_s": nonlinear_runtime,
            **_harmonic_record(
                nonlinear.E,
                x_um=x_um,
                y_um=y_um,
                geometry=geometry,
                resolution=resolution,
            ),
        },
        {
            **common,
            "model": "reduced_linearized",
            "converged": True,
            "runtime_s": reduced_linearized_runtime,
            **_harmonic_record(
                reduced_linearized.E,
                x_um=x_um,
                y_um=y_um,
                geometry=geometry,
                resolution=resolution,
            ),
        },
    ]
    if include_full_linearized:
        full_started = perf_counter()
        full = solve_pr_biased_linearized_reference(
            intensity,
            spec=PRBiasedLinearizedReferenceSpec(
                reference_intensity=REFERENCE_INTENSITY,
                applied_field=0.0,
                dx_normalized=dx_normalized,
                dy_normalized=dy_normalized,
            ),
        )
        rows.append({
            **common,
            "model": "full_transverse_linearized",
            "converged": True,
            "runtime_s": perf_counter() - full_started,
            "material_rel_L2_vs_reduced_linearized": _relative_l2(
                full.delta_E_x, reduced_linearized.delta_E
            ),
            **_harmonic_record(
                full.delta_E_x,
                x_um=x_um,
                y_um=y_um,
                geometry=geometry,
                resolution=resolution,
            ),
        })
    return rows


def _production_beams(
    geometry: CarrierGeometry,
    visibility: float,
    *,
    waist_um: float,
) -> BeamStack:
    powers = powers_for_visibility(visibility)
    return BeamStack(
        channels=(
            BeamChannel(
                name="Pump",
                wavelength_um=WAVELENGTH_UM,
                power_mW=powers[0],
                waist_x_um=waist_um,
                waist_y_um=waist_um,
                x0_um=geometry.beam_1_center_um[0],
                y0_um=geometry.beam_1_center_um[1],
                tilt_x_rad_per_um=geometry.beam_1_k_rad_per_um[0],
                tilt_y_rad_per_um=geometry.beam_1_k_rad_per_um[1],
            ),
            BeamChannel(
                name="Signal",
                wavelength_um=WAVELENGTH_UM,
                power_mW=powers[1],
                waist_x_um=waist_um,
                waist_y_um=waist_um,
                x0_um=geometry.beam_2_center_um[0],
                y0_um=geometry.beam_2_center_um[1],
                tilt_x_rad_per_um=geometry.beam_2_k_rad_per_um[0],
                tilt_y_rad_per_um=geometry.beam_2_k_rad_per_um[1],
            ),
        ),
        coherence="coherent",
    )


def _production_request(
    *,
    N: int,
    visibility: float,
    geometry_name: str,
    waist_um: float,
    linearized: bool,
) -> PRStaticRunRequest:
    geometry = carrier_geometry(geometry_name)
    return PRStaticRunRequest(
        grid=GridSpec(
            Nx=int(N),
            Ny=int(N),
            x_aperture_um=APERTURE_UM,
            y_aperture_um=APERTURE_UM,
            dz_um=10.0,
            z_length_um=INTERACTION_LENGTH_UM,
        ),
        beams=_production_beams(geometry, visibility, waist_um=waist_um),
        material=PRMaterialSpec(
            dark_intensity=DARK_INTENSITY,
            uniform_background_intensity=0.0,
            applied_field=0.0,
            gain_length_product=3.0,
            refractive_index=REFRACTIVE_INDEX,
        ),
        solver=PRStaticWorkflowOptions(
            max_coupled_passes=20,
            optical_substeps=1,
        ),
        backend=BackendSpec("numpy", "float64", False),
        material_response=PRTransverseMaterialResponseSpec(
            model=(
                PR_MATERIAL_RESPONSE_LINEARIZED
                if linearized
                else "nonlinear"
            ),
            reference_intensity=REFERENCE_INTENSITY if linearized else None,
        ),
    )


def run_production_pair(
    *,
    N: int,
    visibility: float,
    geometry_name: str,
    waist_um: float,
    include_full_linearized: bool,
) -> list[dict[str, Any]]:
    """Run an explicitly requested anchor-scale optical comparison."""

    nonlinear_request = _production_request(
        N=N,
        visibility=visibility,
        geometry_name=geometry_name,
        waist_um=waist_um,
        linearized=False,
    )
    linearized_request = replace(
        nonlinear_request,
        material_response=PRTransverseMaterialResponseSpec(
            model=PR_MATERIAL_RESPONSE_LINEARIZED,
            reference_intensity=REFERENCE_INTENSITY,
        ),
    )
    started = perf_counter()
    nonlinear = run_pr_static(nonlinear_request)
    nonlinear_runtime = perf_counter() - started
    started = perf_counter()
    linearized = run_pr_static(linearized_request)
    linearized_runtime = perf_counter() - started
    nonlinear_carrier = pr_static_result_to_run_data(nonlinear).diagnostics[
        "carrier_power"
    ].values
    linearized_carrier = pr_static_result_to_run_data(linearized).diagnostics[
        "carrier_power"
    ].values
    material_difference = np.asarray(nonlinear.E_final) - np.asarray(
        linearized.E_final
    )
    optical_error = _relative_l2(nonlinear.A_final, linearized.A_final)
    nonlinear_intensity = np.abs(np.sum(nonlinear.A_final, axis=0)) ** 2
    linearized_intensity = np.abs(np.sum(linearized.A_final, axis=0)) ** 2
    intensity_error = _relative_l2(nonlinear_intensity, linearized_intensity)
    resolution = resolution_metadata(N, geometry=geometry_name)
    common = {
        "study": "production_static",
        "fixture": "broad_beam" if waist_um > 100.0 else "gaussian_20um",
        "geometry": geometry_name,
        "N": int(N),
        "m": float(visibility),
        "dx_um": resolution["dx_um"],
        "points_per_K": resolution["points_per_K"],
        "points_per_2K": resolution["points_per_2K"],
        "points_per_3K": resolution["points_per_3K"],
        "resolution_K": resolution["resolution_K"],
        "resolution_2K": resolution["resolution_2K"],
        "resolution_3K": resolution["resolution_3K"],
        "periodic_fundamental_mode_components": resolution[
            "periodic_fundamental_mode_components"
        ],
        "periodic_fundamental_bin_offsets": resolution[
            "periodic_fundamental_bin_offsets"
        ],
        "periodic_fundamental_commensurate": resolution[
            "periodic_fundamental_commensurate"
        ],
        "total_power_mW": sum(powers_for_visibility(visibility)),
        "reference_intensity": REFERENCE_INTENSITY,
        "launch_attenuation": "none",
        "material_rel_L2_vs_linearized": _relative_l2(
            nonlinear.E_final, linearized.E_final
        ),
        "material_abs_L2_vs_linearized": _rms(material_difference),
        "material_max_abs_vs_linearized": float(
            np.max(np.abs(material_difference))
        ),
        "linear_response_rms": _rms(linearized.E_final),
        "optical_rel_L2_vs_linearized": optical_error,
        "output_intensity_rel_L2_vs_linearized": intensity_error,
    }
    x_um = (
        np.arange(int(N), dtype=np.float64) - int(N) / 2.0
    ) * resolution["dx_um"]
    y_um = x_um.copy()
    geometry = carrier_geometry(geometry_name)

    def model_row(
        model: str,
        result,
        diagnostic,
        runtime: float,
        material_field: np.ndarray,
    ) -> dict[str, Any]:
        rows = diagnostic["rows"]
        return {
            **common,
            "model": model,
            "converged": bool(result.converged),
            "pump_power_input_normalized": rows[0]["input_power_normalized"],
            "pump_power_output_normalized": rows[0]["output_power_normalized"],
            "pump_gain": rows[0]["gain"],
            "pump_delta_power_normalized": rows[0]["delta_power_normalized"],
            "signal_power_input_normalized": rows[1]["input_power_normalized"],
            "signal_power_output_normalized": rows[1]["output_power_normalized"],
            "signal_gain": rows[1]["gain"],
            "signal_delta_power_normalized": rows[1]["delta_power_normalized"],
            "carrier_gain_difference": (
                None
                if rows[0]["gain"] is None or rows[1]["gain"] is None
                else rows[1]["gain"] - rows[0]["gain"]
            ),
            "carrier_separation_quality": diagnostic[
                "carrier_separation_quality"
            ],
            "carrier_power_balance_error": diagnostic[
                "carrier_power_balance_error"
            ],
            "power_drift": float(result.power_final - result.power_initial),
            "runtime_s": runtime,
            **_harmonic_record(
                material_field,
                x_um=x_um,
                y_um=y_um,
                geometry=geometry,
                resolution=resolution,
            ),
        }

    output = [
        model_row(
            "reduced_nonlinear",
            nonlinear,
            nonlinear_carrier,
            nonlinear_runtime,
            nonlinear.E_final[-1],
        ),
        model_row(
            "reduced_linearized",
            linearized,
            linearized_carrier,
            linearized_runtime,
            linearized.E_final[-1],
        ),
    ]
    if include_full_linearized:
        full_request = PRTransverseStaticRunRequest(
            grid=nonlinear_request.grid,
            beams=nonlinear_request.beams,
            material=nonlinear_request.material,
            boundary=PRTransverseBoundaryProfile(
                profile_id=PR_FULL_TRANSVERSE_PERIODIC_BIASED_CURRENT_V1,
                applied_field_x=0.0,
            ),
            solver=PRTransverseStaticWorkflowOptions(max_coupled_iterations=20),
            backend=nonlinear_request.backend,
            material_response=PRTransverseMaterialResponseSpec(
                model=PR_MATERIAL_RESPONSE_LINEARIZED,
                reference_intensity=REFERENCE_INTENSITY,
            ),
        )
        started = perf_counter()
        full = run_pr_transverse_static(full_request)
        full_runtime = perf_counter() - started
        full_carrier = pr_transverse_static_result_to_run_data(full).diagnostics[
            "carrier_power"
        ].values
        full_state = state_from_potential(
            full.psi_final,
            dx_normalized=(
                nonlinear_request.material.characteristic_wavenumber_per_um
                * resolution["dx_um"]
            ),
            dy_normalized=(
                nonlinear_request.material.characteristic_wavenumber_per_um
                * resolution["dx_um"]
            ),
            applied_field_x=0.0,
        )
        row = model_row(
            "full_transverse_linearized",
            full,
            full_carrier,
            full_runtime,
            full_state.E_x[-1],
        )
        row["material_rel_L2_vs_reduced_linearized"] = _relative_l2(
            full_state.E_x, linearized.E_final
        )
        output.append(row)
    return output


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Build compact asymptotic-order and resolution conclusions."""

    asymptotics = []
    keys = sorted({(row["geometry"], row["N"]) for row in rows})
    for geometry, size in keys:
        selected = [
            row
            for row in rows
            if row["study"] == "frozen_intensity_material"
            and row["geometry"] == geometry
            and row["N"] == size
            and row["model"] == "reduced_nonlinear"
            and row["material_abs_L2_vs_linearized"] is not None
        ]
        selected.sort(key=lambda row: row["m"])
        if len(selected) < 2:
            continue
        modulations = [row["m"] for row in selected]
        errors = [row["material_abs_L2_vs_linearized"] for row in selected]
        responses = [row["linear_response_rms"] for row in selected]
        asymptotics.append({
            "geometry": geometry,
            "N": size,
            "nonlinear_minus_linearized_order": observed_order(
                modulations, errors
            ),
            "linear_response_order": observed_order(modulations, responses),
            "error_over_m_squared": [
                error / (modulation * modulation)
                for error, modulation in zip(errors, modulations)
            ],
            "modulations_ascending": modulations,
        })
    return {
        "asymptotics": asymptotics,
        "resolution": [resolution_metadata(size) for size in DEFAULT_GRIDS],
        "grid_convergence_statement": (
            "N=256 is not adequate for nonlinear-harmonic conclusions; "
            "N=512 resolves K, leaves 2K marginal and 3K underresolved; "
            "therefore a 512-grid nonlinear carrier gain is not established "
            "as grid-converged without comparison to 1024 or finer."
        ),
    }


def write_outputs(
    output_dir: Path,
    *,
    rows: Sequence[dict[str, Any]],
    manifest: dict[str, Any],
    make_plots: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_rows = [_json_safe(row) for row in rows]
    fieldnames = sorted({key for row in safe_rows for key in row})
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(safe_rows)
    summary = _json_safe(summarize(rows))
    if make_plots:
        _write_plots(output_dir, rows)
    evidence_paths = [output_dir / "summary.csv", *sorted(output_dir.glob("*.png"))]
    resolved_manifest = {
        **_json_safe(manifest),
        "evidence_artifact_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in evidence_paths
        },
        "scientific_payload_sha256": hashlib.sha256(
            json.dumps(
                {"rows": safe_rows, "summary": summary},
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
    }
    payload = {
        "manifest": resolved_manifest,
        "summary": summary,
        "rows": safe_rows,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_plots(output_dir: Path, rows: Sequence[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nonlinear = [
        row
        for row in rows
        if row["model"] == "reduced_nonlinear"
        and row["material_abs_L2_vs_linearized"] is not None
    ]
    plot_specs = (
        (
            "material_error_vs_modulation.png",
            "material_abs_L2_vs_linearized",
            "Nonlinear - linearized material RMS",
            True,
        ),
        (
            "material_error_over_m2.png",
            "error_over_m2",
            "Material RMS / m²",
            True,
        ),
    )
    for filename, metric, ylabel, loglog in plot_specs:
        figure, axis = plt.subplots(figsize=(5.4, 3.6))
        for geometry in sorted({row["geometry"] for row in nonlinear}):
            for size in sorted({row["N"] for row in nonlinear}):
                selected = [
                    row
                    for row in nonlinear
                    if row["geometry"] == geometry and row["N"] == size
                ]
                selected.sort(key=lambda row: row["m"])
                if not selected:
                    continue
                y = [
                    row["material_abs_L2_vs_linearized"]
                    if metric != "error_over_m2"
                    else row["material_abs_L2_vs_linearized"] / row["m"] ** 2
                    for row in selected
                ]
                plot = axis.loglog if loglog else axis.plot
                plot(
                    [row["m"] for row in selected],
                    y,
                    marker="o",
                    label=f"{geometry}, N={size}",
                )
        axis.set(xlabel="Visibility m", ylabel=ylabel)
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(fontsize=7)
        figure.tight_layout()
        figure.savefig(output_dir / filename, dpi=150)
        plt.close(figure)

    material_nonlinear = [
        row
        for row in nonlinear
        if row["study"] == "frozen_intensity_material"
    ]
    if material_nonlinear:
        figure, axis = plt.subplots(figsize=(5.4, 3.6))
        for geometry in sorted({row["geometry"] for row in material_nonlinear}):
            selected_geometry = [
                row for row in material_nonlinear if row["geometry"] == geometry
            ]
            modulation = max(row["m"] for row in selected_geometry)
            selected = sorted(
                (row for row in selected_geometry if row["m"] == modulation),
                key=lambda row: row["N"],
            )
            for harmonic in (1, 2, 3):
                key = f"harmonic_{harmonic}_relative_to_K"
                points = [row for row in selected if row.get(key) is not None]
                if points:
                    axis.plot(
                        [row["N"] for row in points],
                        [row[key] for row in points],
                        marker="o",
                        label=f"{geometry}, {harmonic}K",
                    )
        axis.set(
            xlabel="Grid size N",
            ylabel="Harmonic magnitude / fundamental",
            yscale="log",
        )
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(fontsize=7)
        figure.tight_layout()
        figure.savefig(output_dir / "harmonics_vs_resolution.png", dpi=150)
        plt.close(figure)

        figure, axis = plt.subplots(figsize=(5.4, 3.6))
        for geometry in sorted({row["geometry"] for row in material_nonlinear}):
            for size in sorted({row["N"] for row in material_nonlinear}):
                selected = sorted(
                    (
                        row
                        for row in material_nonlinear
                        if row["geometry"] == geometry and row["N"] == size
                    ),
                    key=lambda row: row["m"],
                )
                axis.semilogx(
                    [row["m"] for row in selected],
                    [row["harmonic_1_phase_rad"] for row in selected],
                    marker="o",
                    label=f"{geometry}, N={size}",
                )
        axis.set(xlabel="Visibility m", ylabel="Fundamental phase (rad)")
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(fontsize=7)
        figure.tight_layout()
        figure.savefig(output_dir / "fundamental_phase_vs_modulation.png", dpi=150)
        plt.close(figure)

    carrier_rows = [
        row
        for row in rows
        if row.get("pump_gain") is not None and row.get("signal_gain") is not None
    ]
    if carrier_rows:
        figure, axis = plt.subplots(figsize=(5.4, 3.6))
        groups = sorted({
            (row["geometry"], row["model"], row["N"])
            for row in carrier_rows
        })
        for geometry, model, size in groups:
            selected = sorted(
                (
                    row
                    for row in carrier_rows
                    if row["geometry"] == geometry
                    and row["model"] == model
                    and row["N"] == size
                ),
                key=lambda row: row["m"],
            )
            for carrier in ("pump", "signal"):
                axis.semilogx(
                    [row["m"] for row in selected],
                    [row[f"{carrier}_gain"] for row in selected],
                    marker="o",
                    label=f"{geometry}, {model}, {carrier}",
                )
        axis.set(xlabel="Visibility m", ylabel="Carrier gain")
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(fontsize=6)
        figure.tight_layout()
        figure.savefig(output_dir / "carrier_gain_vs_modulation.png", dpi=150)
        plt.close(figure)

        if len({row["N"] for row in carrier_rows}) > 1:
            figure, axis = plt.subplots(figsize=(5.4, 3.6))
            groups = sorted({
                (row["geometry"], row["model"], row["m"])
                for row in carrier_rows
            })
            for geometry, model, modulation in groups:
                selected = sorted(
                    (
                        row
                        for row in carrier_rows
                        if row["geometry"] == geometry
                        and row["model"] == model
                        and row["m"] == modulation
                    ),
                    key=lambda row: row["N"],
                )
                for carrier in ("pump", "signal"):
                    axis.plot(
                        [row["N"] for row in selected],
                        [row[f"{carrier}_gain"] for row in selected],
                        marker="o",
                        label=f"{geometry}, {model}, {carrier}",
                    )
            axis.set(xlabel="Grid size N", ylabel="Carrier gain")
            axis.grid(True, alpha=0.25)
            axis.legend(fontsize=6)
            figure.tight_layout()
            figure.savefig(output_dir / "carrier_gain_vs_resolution.png", dpi=150)
            plt.close(figure)


def _parse_csv_numbers(value: str, *, cast=float) -> tuple[Any, ...]:
    return tuple(cast(item.strip()) for item in value.split(",") if item.strip())


def _git_text(*arguments: str) -> str | None:
    try:
        return subprocess.check_output(
            ("git", *arguments), text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study", choices=("material", "production"), default="material"
    )
    parser.add_argument("--grids", default="256,512,1024")
    parser.add_argument("--modulations", default="0.5,0.25,0.125,0.0625,0.03125")
    parser.add_argument("--geometries", default="x,45deg")
    parser.add_argument("--full-transverse-linearized", action="store_true")
    parser.add_argument("--fixture", choices=("broad", "gaussian"), default="broad")
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--expected-production-commit")
    parser.add_argument("--require-clean-production-source", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/pr_linearization_resolution_sweep"),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    production_source_commit = _git_text("rev-parse", "HEAD")
    production_source_status = _git_text("status", "--porcelain")
    if production_source_commit is None or production_source_status is None:
        raise RuntimeError("the production source must be an inspectable Git checkout")
    if (
        args.expected_production_commit is not None
        and production_source_commit != args.expected_production_commit
    ):
        raise RuntimeError(
            "production source commit does not match --expected-production-commit"
        )
    production_source_clean = production_source_status == ""
    if args.require_clean_production_source and not production_source_clean:
        raise RuntimeError("the production source checkout is not clean")
    grids = _parse_csv_numbers(args.grids, cast=int)
    modulations = _parse_csv_numbers(args.modulations)
    geometries = tuple(
        item.strip() for item in args.geometries.split(",") if item.strip()
    )
    rows: list[dict[str, Any]] = []
    for geometry_name in geometries:
        carrier_geometry(geometry_name)
        for size in grids:
            for modulation in modulations:
                powers = powers_for_visibility(modulation)
                if not math.isclose(sum(powers), TOTAL_POWER_MW, abs_tol=2e-15):
                    raise AssertionError("visibility construction changed total power")
                if args.study == "material":
                    rows.extend(run_material_case(
                        N=size,
                        visibility=modulation,
                        geometry_name=geometry_name,
                        include_full_linearized=args.full_transverse_linearized,
                    ))
                else:
                    rows.extend(run_production_pair(
                        N=size,
                        visibility=modulation,
                        geometry_name=geometry_name,
                        waist_um=500.0 if args.fixture == "broad" else 20.0,
                        include_full_linearized=args.full_transverse_linearized,
                    ))
    manifest = {
        "production_source_commit": production_source_commit,
        "production_source_clean": production_source_clean,
        "production_source_status_porcelain": production_source_status,
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "study": args.study,
        "grids": list(grids),
        "modulations": list(modulations),
        "geometries": [asdict(carrier_geometry(name)) for name in geometries],
        "fixed_total_power_mW": TOTAL_POWER_MW,
        "reference_experiment": {
            "aperture_um": [APERTURE_UM, APERTURE_UM],
            "interaction_length_um": INTERACTION_LENGTH_UM,
            "Nz": 100,
            "optical_dz_um": 10.0,
            "dark_intensity": DARK_INTENSITY,
            "uniform_background": 0.0,
            "normalized_applied_field": 0.0,
            "gain_length_product": 3.0,
            "refractive_index": REFRACTIVE_INDEX,
            "wavelength_um": WAVELENGTH_UM,
            "realistic_waist_um": 20.0,
            "max_coupled_iterations": 20,
            "optical_substeps": 1,
            "precision": "float64",
        },
        "pending_external_nonlinear_job_used": False,
        "scheduler_submission": "never",
    }
    write_outputs(args.output_dir, rows=rows, manifest=manifest, make_plots=args.plots)
    summary = summarize(rows)
    print(f"wrote {len(rows)} rows to {args.output_dir}")
    for record in summary["asymptotics"]:
        print(
            f"{record['geometry']} N={record['N']}: "
            f"nonlinear order={record['nonlinear_minus_linearized_order']:.6f}, "
            f"linear order={record['linear_response_order']:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
