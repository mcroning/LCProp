#!/usr/bin/env python3
"""Compare longitudinal step-size effects in the LCProp optical far field.

This is an isolated validation experiment.  It deliberately uses the public
``run_static`` workflow, selecting either its fixed-theta or local
self-consistent split-step path rather than calling the propagator directly.

The exact dark-bias profile is selected with ``theta_bc = 0`` and the value of
``b`` whose analytic center angle is pi/4.  All request parameters except
``GridSpec.dz_um`` are identical between runs.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter

import numpy as np

# Keep Matplotlib's runtime cache out of the repository when the user has not
# configured a writable cache location.
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "lcprop-matplotlib-cache"),
)

import matplotlib.pyplot as plt

from lcprop.core.backend import asnumpy
from lcprop.core.beams import BeamChannel, BeamStack
from lcprop.core.context import BiasSpec, GridSpec, LCMaterial
from lcprop.core.derived import resolved_b
from lcprop.core.grid import make_grid
from lcprop.core.requests import (
    OutputOptions,
    RuntimeOptions,
    StaticRunRequest,
    StaticSolverOptions,
    StaticWorkflowOptions,
)
from lcprop.lc.bias import b_from_theta0_zero_bc, theta0_from_b_zero_bc
from lcprop.optics.splitstep import total_intensity
from lcprop.products.diagnostics import rms_widths
from lcprop.workflows import run_static

from _experimental_scattering_noise import (
    CorrelatedPhaseNoise3D,
    grid_scale_spectral_fraction,
    normalized_spectrum,
)


DZ_VALUES_UM = (20.0, 5.0, 2.0)
REFERENCE_DZ_UM = 2.0
Z_LENGTH_UM = 3000.0
NX = 128
NY = 128
X_APERTURE_UM = 75.0
Y_APERTURE_UM = 100.0
WAIST_UM = 3.0
WAVELENGTH_UM = 0.633
POWER_MW = 1.0
LOG_DYNAMIC_RANGE_DECADES = 12.0
WORKFLOW_CHOICES = ("fixed_theta", "self_consistent")
NOISE_MODE_CHOICES = ("off", "independent")
BIAS_CASE_CHOICES = (
    "zero_bc_pi4_center",
    "zero_voltage_pi4_boundary",
)

MATERIAL = LCMaterial(
    ne=1.7,
    no=1.5,
    K=7.0e-12,
    delta_epsilon=13.0,
)
TARGET_THETA_CENTER_RAD = math.pi / 4.0
TARGET_B = b_from_theta0_zero_bc(TARGET_THETA_CENTER_RAD)


@dataclass
class ValidationRun:
    dz_um: float
    n_steps: int
    elapsed_s: float
    final_power_mW: float
    normalized_final_power: float
    near_rms_x_um: float
    far_rms_kx_rad_per_um: float
    outer_kx_power_fraction: float
    sampled_theta_center_rad: float
    unconverged_slices: int
    max_final_residual_rms: float
    max_final_residual_max: float
    noise_eps: float
    noise_sigma_um: float
    noise_seed: int
    representative_step_index: int
    representative_phase_mean_rad: float
    representative_phase_rms_rad: float
    raw_std_expected_rad: float
    raw_rms_measured_rad: float
    raw_rms_ratio: float
    raw_grid_scale_fraction: float
    phase_grid_scale_fraction: float
    screen_pointwise_intensity_max_error: float
    screen_relative_power_error: float
    representative_phase: np.ndarray
    representative_phase_spectrum: np.ndarray
    final_field: np.ndarray
    near_intensity: np.ndarray
    far_intensity_normalized: np.ndarray
    relative_l2_from_reference: float = math.nan


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for the .npz and .png outputs (default depends on workflow).",
    )
    parser.add_argument(
        "--workflow",
        choices=WORKFLOW_CHOICES,
        default="fixed_theta",
        help="Static workflow used for all three propagation runs.",
    )
    parser.add_argument(
        "--noise-mode",
        choices=NOISE_MODE_CHOICES,
        default="off",
        help="Optional PRProp-style phase layer applied once per accepted z step.",
    )
    parser.add_argument(
        "--noise-eps",
        type=float,
        default=0.002,
        help="Dimensionless PRProp scattering-noise strength.",
    )
    parser.add_argument(
        "--noise-sigma-um",
        type=float,
        default=0.4,
        help="Transverse Gaussian correlation sigma in micrometers.",
    )
    parser.add_argument(
        "--noise-seed",
        type=int,
        default=12345,
        help="Run seed used to derive deterministic per-step streams.",
    )
    parser.add_argument(
        "--bias-case",
        choices=BIAS_CASE_CHOICES,
        default="zero_bc_pi4_center",
        help=(
            "Prepared LC bias profile: the original zero-boundary analytic "
            "pi/4-center profile, or zero applied voltage with theta_bc=pi/4."
        ),
    )
    parser.set_defaults(repo_root=repo_root)
    return parser.parse_args()


def make_request(
    dz_um: float,
    workflow: str = "fixed_theta",
    bias_case: str = "zero_bc_pi4_center",
) -> StaticRunRequest:
    """Build the common trusted-workflow request, varying only dz."""

    if workflow == "fixed_theta":
        workflow_options = StaticWorkflowOptions(
            strategy="fixed_theta",
            theta_solver="none",
            optics_solver="splitstep",
            coupling="frozen",
        )
    elif workflow == "self_consistent":
        workflow_options = StaticWorkflowOptions(
            strategy="local_self_consistent",
            theta_solver="picard_cn",
            optics_solver="splitstep",
            coupling="self_consistent",
        )
    else:
        raise ValueError(f"unsupported workflow {workflow!r}")

    if bias_case == "zero_bc_pi4_center":
        bias = BiasSpec(
            theta_bc=0.0,
            b_override=TARGET_B,
        )
    elif bias_case == "zero_voltage_pi4_boundary":
        bias = BiasSpec(
            V_bias=0.0,
            theta_bc=math.pi / 4.0,
            b_override=None,
        )
    else:
        raise ValueError(f"unsupported bias case {bias_case!r}")

    return StaticRunRequest(
        grid=GridSpec(
            Nx=NX,
            Ny=NY,
            dz_um=dz_um,
            x_aperture_um=X_APERTURE_UM,
            y_aperture_um=Y_APERTURE_UM,
            z_length_um=Z_LENGTH_UM,
        ),
        material=MATERIAL,
        bias=bias,
        beams=BeamStack(
            channels=(
                BeamChannel(
                    wavelength_um=WAVELENGTH_UM,
                    power_mW=POWER_MW,
                    waist_x_um=WAIST_UM,
                    waist_y_um=WAIST_UM,
                    x0_um=0.0,
                    y0_um=0.0,
                    tilt_x_rad_per_um=0.0,
                    tilt_y_rad_per_um=0.0,
                ),
            ),
        ),
        solver=StaticSolverOptions(
            workflow=workflow_options,
            # Per-iteration records are not needed for this output-plane
            # diagnostic. Slice summaries and convergence metrics are retained.
            record_iteration_history=False,
        ),
        output=OutputOptions(save_slices=False, save_full=False),
        runtime=RuntimeOptions(precision="float64"),
    )


def array_module(array):
    """Return the NumPy-like module already used by an LCProp result array."""

    if type(array).__module__.split(".")[0] == "cupy":
        import cupy as cp  # type: ignore

        return cp
    return np


def centered_far_field(
    field,
    *,
    dx_um: float,
    dy_um: float,
    xp,
):
    """Return physical k axes and a unit-integral centered far-field density."""

    transformed = xp.fft.fftshift(
        xp.fft.fft2(xp.fft.ifftshift(field)),
    )
    raw_intensity = xp.abs(transformed) ** 2

    kx = 2.0 * math.pi * xp.fft.fftshift(
        xp.fft.fftfreq(field.shape[0], d=dx_um)
    )
    ky = 2.0 * math.pi * xp.fft.fftshift(
        xp.fft.fftfreq(field.shape[1], d=dy_um)
    )
    dkx = float(asnumpy(xp.abs(kx[1] - kx[0])))
    dky = float(asnumpy(xp.abs(ky[1] - ky[0])))
    spectral_power = xp.sum(raw_intensity) * dkx * dky
    normalized = raw_intensity / spectral_power
    return kx, ky, normalized, dkx, dky


def far_field_metrics(far_intensity, kx, *, dkx: float, dky: float, xp):
    """Return kx RMS width and power in the combined outermost 20%."""

    marginal_kx = xp.sum(far_intensity, axis=1) * dky
    mean_kx = xp.sum(kx * marginal_kx) * dkx
    variance_kx = xp.sum((kx - mean_kx) ** 2 * marginal_kx) * dkx

    # "Outer 20% of the kx bandwidth" is interpreted as the two edge bands
    # together: the outermost 10% at each end of the sampled kx interval.
    lower_edge = kx[0] - 0.5 * dkx
    upper_edge = kx[-1] + 0.5 * dkx
    edge_width = 0.1 * (upper_edge - lower_edge)
    outer_mask = (kx < lower_edge + edge_width) | (
        kx > upper_edge - edge_width
    )
    outer_fraction = xp.sum(marginal_kx[outer_mask]) * dkx

    return (
        float(asnumpy(xp.sqrt(xp.maximum(variance_kx, 0.0)))),
        float(asnumpy(outer_fraction)),
    )


def execute_case(
    dz_um: float,
    workflow: str = "fixed_theta",
    *,
    noise_mode: str = "off",
    noise_eps: float = 0.002,
    noise_sigma_um: float = 0.4,
    noise_seed: int = 12345,
    bias_case: str = "zero_bc_pi4_center",
) -> tuple[ValidationRun, np.ndarray, np.ndarray]:
    request = make_request(dz_um, workflow, bias_case)
    runtime_grid = make_grid(
        request.grid,
        xp=np,
        real_dtype=np.float64,
    )
    phase_noise = CorrelatedPhaseNoise3D(
        nx=runtime_grid.Nx,
        ny=runtime_grid.Ny,
        dx_um=runtime_grid.dx_um,
        dy_um=runtime_grid.dy_um,
        z_length_um=request.grid.z_length_um,
        dz_um=request.grid.dz_um,
        eps=noise_eps,
        sigma_um=noise_sigma_um,
        seed=noise_seed,
        mode=noise_mode,
    )
    representative_step_index = phase_noise.n_steps // 2
    representative_raw = phase_noise.raw_layer(representative_step_index)
    representative_phase = phase_noise.phase_layer(
        representative_step_index
    )
    representative_phase_spectrum = normalized_spectrum(
        representative_phase
    )

    raw_rms = float(np.sqrt(np.mean(representative_raw**2)))
    raw_ratio = (
        1.0
        if phase_noise.raw_std == 0.0
        else raw_rms / phase_noise.raw_std
    )
    probe = np.exp(
        1j
        * np.linspace(
            -0.75,
            0.75,
            runtime_grid.Nx * runtime_grid.Ny,
        ).reshape(1, runtime_grid.Nx, runtime_grid.Ny)
    )
    probe_intensity_before = np.abs(probe) ** 2
    probe_power_before = float(np.sum(probe_intensity_before))
    phase_noise(probe, representative_step_index)
    probe_intensity_after = np.abs(probe) ** 2
    pointwise_error = float(
        np.max(np.abs(probe_intensity_after - probe_intensity_before))
    )
    relative_power_error = abs(
        float(np.sum(probe_intensity_after)) - probe_power_before
    ) / probe_power_before

    # Cheap deterministic/helper invariants, evaluated without consuming any
    # mutable RNG state because each layer owns a derived child stream.
    np.testing.assert_array_equal(
        representative_phase,
        phase_noise.phase_layer(representative_step_index),
    )
    zero_noise = replace(phase_noise, eps=0.0, mode="independent")
    if np.any(zero_noise.phase_layer(representative_step_index)):
        raise AssertionError("eps=0 must generate an exactly zero phase layer")
    if noise_mode == "independent" and noise_eps > 0.0:
        if np.array_equal(
            representative_phase,
            phase_noise.phase_layer(
                (representative_step_index + 1) % phase_noise.n_steps
            ),
        ):
            raise AssertionError("different step indices produced one phase layer")
        if np.array_equal(
            representative_phase,
            replace(phase_noise, seed=noise_seed + 1).phase_layer(
                representative_step_index
            ),
        ):
            raise AssertionError("different seeds produced one phase layer")
        if not (0.95 <= raw_ratio <= 1.05):
            raise AssertionError(
                f"measured raw RMS ratio {raw_ratio:g} is not close to one"
            )
        if not (
            grid_scale_spectral_fraction(representative_phase)
            < grid_scale_spectral_fraction(representative_raw)
        ):
            raise AssertionError(
                "Gaussian filtering did not suppress grid-scale spectral power"
            )
    if pointwise_error > 10.0 * np.finfo(np.float64).eps:
        raise AssertionError("phase screen changed pointwise intensity")
    if relative_power_error > 10.0 * np.finfo(np.float64).eps:
        raise AssertionError("phase screen changed total probe power")

    started = perf_counter()
    result = run_static(
        request,
        _phase_screen_fn=(
            phase_noise if noise_mode == "independent" else None
        ),
    )
    elapsed_s = perf_counter() - started

    xp = array_module(result.A_final)
    if xp is not np:
        runtime_grid = make_grid(
            request.grid,
            xp=xp,
            real_dtype=np.float64,
        )
    final_field_backend = result.A_final[0]
    near_intensity_backend = total_intensity(
        result.A_final,
        coherence_groups=request.beams.coherence_groups,
        xp=xp,
    )
    near_rms_x_um, _ = rms_widths(near_intensity_backend, runtime_grid)

    kx_backend, ky_backend, far_backend, dkx, dky = centered_far_field(
        final_field_backend,
        dx_um=runtime_grid.dx_um,
        dy_um=runtime_grid.dy_um,
        xp=xp,
    )
    far_rms, outer_fraction = far_field_metrics(
        far_backend,
        kx_backend,
        dkx=dkx,
        dky=dky,
        xp=xp,
    )

    theta_bias = np.asarray(asnumpy(result.theta_bias))
    center_pair = theta_bias[NX // 2 - 1 : NX // 2 + 1, :]
    unconverged_slices = sum(
        not summary.converged for summary in result.slice_summaries
    )
    run = ValidationRun(
        dz_um=dz_um,
        n_steps=int(result.total_slices),
        elapsed_s=elapsed_s,
        final_power_mW=float(result.physical_power_final_mW),
        normalized_final_power=float(result.power_final),
        near_rms_x_um=near_rms_x_um,
        far_rms_kx_rad_per_um=far_rms,
        outer_kx_power_fraction=outer_fraction,
        sampled_theta_center_rad=float(np.mean(center_pair)),
        unconverged_slices=unconverged_slices,
        max_final_residual_rms=(
            math.nan
            if result.max_final_residual_rms is None
            else float(result.max_final_residual_rms)
        ),
        max_final_residual_max=(
            math.nan
            if result.max_final_residual_max is None
            else float(result.max_final_residual_max)
        ),
        noise_eps=noise_eps if noise_mode == "independent" else 0.0,
        noise_sigma_um=noise_sigma_um,
        noise_seed=noise_seed,
        representative_step_index=representative_step_index,
        representative_phase_mean_rad=float(np.mean(representative_phase)),
        representative_phase_rms_rad=float(
            np.sqrt(np.mean(representative_phase**2))
        ),
        raw_std_expected_rad=phase_noise.raw_std,
        raw_rms_measured_rad=raw_rms,
        raw_rms_ratio=raw_ratio,
        raw_grid_scale_fraction=grid_scale_spectral_fraction(
            representative_raw
        ),
        phase_grid_scale_fraction=grid_scale_spectral_fraction(
            representative_phase
        ),
        screen_pointwise_intensity_max_error=pointwise_error,
        screen_relative_power_error=relative_power_error,
        representative_phase=representative_phase.copy(),
        representative_phase_spectrum=representative_phase_spectrum.copy(),
        final_field=np.asarray(asnumpy(final_field_backend)).copy(),
        near_intensity=np.asarray(asnumpy(near_intensity_backend)).copy(),
        far_intensity_normalized=np.asarray(asnumpy(far_backend)).copy(),
    )
    kx = np.asarray(asnumpy(kx_backend)).copy()
    ky = np.asarray(asnumpy(ky_backend)).copy()

    # The workflow result retains longitudinal diagnostic/checkpoint arrays.
    # This validation only needs the output plane, so release each full result
    # before starting the next dz case.
    del result
    gc.collect()
    return run, kx, ky


def add_reference_differences(runs: list[ValidationRun]) -> None:
    reference = next(run for run in runs if run.dz_um == REFERENCE_DZ_UM)
    reference_norm = np.linalg.norm(reference.far_intensity_normalized)
    for run in runs:
        run.relative_l2_from_reference = float(
            np.linalg.norm(
                run.far_intensity_normalized
                - reference.far_intensity_normalized
            )
            / reference_norm
        )


def save_numerical_results(
    path: Path,
    runs: list[ValidationRun],
    x_um: np.ndarray,
    y_um: np.ndarray,
    kx_rad_per_um: np.ndarray,
    ky_rad_per_um: np.ndarray,
    far_log10: np.ndarray,
    *,
    workflow: str,
    noise_mode: str,
    bias_case: str,
) -> None:
    central_ky = int(np.argmin(np.abs(ky_rad_per_um)))
    metadata = {
        "workflow": workflow,
        "noise_mode": noise_mode,
        "bias_case": bias_case,
        "noise_operation": "A *= exp(1j * noise_xy)",
        "noise_layer_relation": "independent per longitudinal step",
        "backend": "numpy",
        "fft_extent": "full computational field and full Nyquist spectrum",
        "outer_spectrum_definition": (
            "combined outermost 10 percent at each kx edge, integrated over ky"
        ),
        "nx": NX,
        "ny": NY,
        "x_aperture_um": X_APERTURE_UM,
        "y_aperture_um": Y_APERTURE_UM,
        "z_length_um": Z_LENGTH_UM,
        "wavelength_um": WAVELENGTH_UM,
        "beam_power_mW": POWER_MW,
        "beam_waist_um": WAIST_UM,
    }
    np.savez_compressed(
        path,
        dz_um=np.asarray([run.dz_um for run in runs]),
        n_steps=np.asarray([run.n_steps for run in runs], dtype=np.int64),
        elapsed_s=np.asarray([run.elapsed_s for run in runs]),
        final_power_mW=np.asarray([run.final_power_mW for run in runs]),
        normalized_final_power=np.asarray(
            [run.normalized_final_power for run in runs]
        ),
        near_rms_x_um=np.asarray([run.near_rms_x_um for run in runs]),
        far_rms_kx_rad_per_um=np.asarray(
            [run.far_rms_kx_rad_per_um for run in runs]
        ),
        outer_kx_power_fraction=np.asarray(
            [run.outer_kx_power_fraction for run in runs]
        ),
        relative_l2_from_dz_2=np.asarray(
            [run.relative_l2_from_reference for run in runs]
        ),
        sampled_theta_center_rad=np.asarray(
            [run.sampled_theta_center_rad for run in runs]
        ),
        unconverged_slices=np.asarray(
            [run.unconverged_slices for run in runs],
            dtype=np.int64,
        ),
        max_final_residual_rms=np.asarray(
            [run.max_final_residual_rms for run in runs]
        ),
        max_final_residual_max=np.asarray(
            [run.max_final_residual_max for run in runs]
        ),
        noise_eps=np.asarray([run.noise_eps for run in runs]),
        noise_sigma_um=np.asarray(
            [run.noise_sigma_um for run in runs]
        ),
        noise_seed=np.asarray(
            [run.noise_seed for run in runs],
            dtype=np.int64,
        ),
        representative_step_index=np.asarray(
            [run.representative_step_index for run in runs],
            dtype=np.int64,
        ),
        representative_phase_mean_rad=np.asarray(
            [run.representative_phase_mean_rad for run in runs]
        ),
        representative_phase_rms_rad=np.asarray(
            [run.representative_phase_rms_rad for run in runs]
        ),
        raw_std_expected_rad=np.asarray(
            [run.raw_std_expected_rad for run in runs]
        ),
        raw_rms_measured_rad=np.asarray(
            [run.raw_rms_measured_rad for run in runs]
        ),
        raw_rms_ratio=np.asarray([run.raw_rms_ratio for run in runs]),
        raw_grid_scale_fraction=np.asarray(
            [run.raw_grid_scale_fraction for run in runs]
        ),
        phase_grid_scale_fraction=np.asarray(
            [run.phase_grid_scale_fraction for run in runs]
        ),
        screen_pointwise_intensity_max_error=np.asarray(
            [run.screen_pointwise_intensity_max_error for run in runs]
        ),
        screen_relative_power_error=np.asarray(
            [run.screen_relative_power_error for run in runs]
        ),
        target_theta_center_rad=np.asarray(TARGET_THETA_CENTER_RAD),
        bias_voltage_V=np.asarray(
            make_request(DZ_VALUES_UM[0], workflow, bias_case).bias.V_bias
        ),
        theta_bc_rad=np.asarray(
            make_request(DZ_VALUES_UM[0], workflow, bias_case).bias.theta_bc
        ),
        b_resolved=np.asarray(
            resolved_b(
                MATERIAL,
                make_request(DZ_VALUES_UM[0], workflow, bias_case).bias,
            )
        ),
        b_override=np.asarray(
            math.nan
            if make_request(
                DZ_VALUES_UM[0], workflow, bias_case
            ).bias.b_override
            is None
            else make_request(
                DZ_VALUES_UM[0], workflow, bias_case
            ).bias.b_override
        ),
        x_um=x_um,
        y_um=y_um,
        kx_rad_per_um=kx_rad_per_um,
        ky_rad_per_um=ky_rad_per_um,
        final_field=np.stack([run.final_field for run in runs]),
        near_field_intensity=np.stack([run.near_intensity for run in runs]),
        far_field_intensity_normalized=np.stack(
            [run.far_intensity_normalized for run in runs]
        ),
        central_far_field_cut=np.stack(
            [
                run.far_intensity_normalized[:, central_ky]
                for run in runs
            ]
        ),
        far_field_log10=far_log10,
        representative_phase_screen=np.stack(
            [run.representative_phase for run in runs]
        ),
        representative_phase_spectrum=np.stack(
            [run.representative_phase_spectrum for run in runs]
        ),
        run_metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )


def save_comparison_figure(
    path: Path,
    runs: list[ValidationRun],
    x_um: np.ndarray,
    y_um: np.ndarray,
    kx_rad_per_um: np.ndarray,
    ky_rad_per_um: np.ndarray,
    far_log10: np.ndarray,
    log_vmin: float,
    log_vmax: float,
    workflow: str,
    noise_mode: str,
    noise_eps: float,
    noise_sigma_um: float,
    noise_seed: int,
    bias_case: str,
) -> None:
    figure = plt.figure(figsize=(14.5, 8.3), constrained_layout=True)
    layout = figure.add_gridspec(2, 6, height_ratios=(0.82, 1.0))
    near_axis = figure.add_subplot(layout[0, 0:3])
    far_cut_axis = figure.add_subplot(layout[0, 3:6])

    central_y = int(np.argmin(np.abs(y_um)))
    central_ky = int(np.argmin(np.abs(ky_rad_per_um)))
    colors = ("tab:red", "tab:green", "tab:blue")
    for run, color in zip(runs, colors, strict=True):
        label = rf"$dz={run.dz_um:g}\,\mu$m"
        near_axis.plot(
            x_um,
            run.near_intensity[:, central_y],
            color=color,
            label=label,
        )
        far_cut_axis.semilogy(
            kx_rad_per_um,
            np.maximum(
                run.far_intensity_normalized[:, central_ky],
                10.0**log_vmin,
            ),
            color=color,
            label=label,
        )

    near_axis.set(
        title="Final near-field central x cut",
        xlabel=r"$x$ ($\mu$m)",
        ylabel=r"Intensity ($\mu$m$^{-2}$)",
    )
    far_cut_axis.set(
        title="Final normalized far-field central kx cut",
        xlabel=r"$k_x$ (rad/$\mu$m)",
        ylabel="Spectral power density",
    )
    near_axis.grid(alpha=0.25)
    far_cut_axis.grid(alpha=0.25)
    near_axis.legend()
    far_cut_axis.legend()

    image_axes = [
        figure.add_subplot(layout[1, 0:2]),
        figure.add_subplot(layout[1, 2:4]),
        figure.add_subplot(layout[1, 4:6]),
    ]
    image = None
    for index, (run, axis) in enumerate(
        zip(runs, image_axes, strict=True)
    ):
        image = axis.imshow(
            far_log10[index].T,
            origin="lower",
            extent=(
                kx_rad_per_um[0],
                kx_rad_per_um[-1],
                ky_rad_per_um[0],
                ky_rad_per_um[-1],
            ),
            vmin=log_vmin,
            vmax=log_vmax,
            cmap="magma",
            aspect="equal",
            interpolation="nearest",
        )
        axis.set(
            title=rf"$dz={run.dz_um:g}\,\mu$m",
            xlabel=r"$k_x$ (rad/$\mu$m)",
            ylabel=r"$k_y$ (rad/$\mu$m)",
        )

    assert image is not None
    colorbar = figure.colorbar(image, ax=image_axes, shrink=0.92, pad=0.02)
    colorbar.set_label(r"$\log_{10}$ normalized spectral power density")
    workflow_title = (
        "fixed-director propagation"
        if workflow == "fixed_theta"
        else "static self-consistent propagation"
    )
    figure.suptitle(
        f"LCProp 3 mm {workflow_title}: longitudinal step-size comparison"
        + (
            ""
            if bias_case == "zero_bc_pi4_center"
            else "\nV=0, theta_bc=π/4"
        )
        + (
            ""
            if noise_mode == "off"
            else (
                f"\nPRProp-style independent phase layers: "
                f"eps={noise_eps:g}, sigma={noise_sigma_um:g} µm, "
                f"seed={noise_seed}"
            )
        )
    )
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_phase_figure(
    path: Path,
    runs: list[ValidationRun],
    x_um: np.ndarray,
    y_um: np.ndarray,
    kx_rad_per_um: np.ndarray,
    ky_rad_per_um: np.ndarray,
) -> None:
    """Save the representative dz=2 phase layer and its full spectrum."""

    run = next(item for item in runs if item.dz_um == REFERENCE_DZ_UM)
    spectrum = run.representative_phase_spectrum
    positive = spectrum[spectrum > 0.0]
    spectrum_floor = (
        float(np.max(spectrum)) * 1e-12
        if positive.size
        else np.finfo(float).tiny
    )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(11.5, 4.7),
        constrained_layout=True,
    )
    phase_image = axes[0].imshow(
        run.representative_phase.T,
        origin="lower",
        extent=(x_um[0], x_um[-1], y_um[0], y_um[-1]),
        cmap="coolwarm",
        aspect="equal",
    )
    axes[0].set(
        title=(
            f"Representative phase, dz={run.dz_um:g} µm, "
            f"step={run.representative_step_index}"
        ),
        xlabel="x (µm)",
        ylabel="y (µm)",
    )
    figure.colorbar(phase_image, ax=axes[0], label="phase (rad)")

    spectrum_image = axes[1].imshow(
        np.log10(np.maximum(spectrum, spectrum_floor)).T,
        origin="lower",
        extent=(
            kx_rad_per_um[0],
            kx_rad_per_um[-1],
            ky_rad_per_um[0],
            ky_rad_per_um[-1],
        ),
        cmap="magma",
        aspect="equal",
    )
    axes[1].set(
        title="Full centered phase-screen spectrum",
        xlabel="kx (rad/µm)",
        ylabel="ky (rad/µm)",
    )
    figure.colorbar(
        spectrum_image,
        ax=axes[1],
        label="log10 normalized spectral power",
    )
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def print_table(runs: list[ValidationRun]) -> None:
    header = (
        "dz_um  n_steps  eps      sigma_um  seed   phase_mean_rad  "
        "phase_rms_rad  final_power_mW  near_rms_x_um  "
        "far_rms_kx_rad_per_um  outer20_fraction  rel_L2_from_dz2"
    )
    print(header)
    for run in runs:
        print(
            f"{run.dz_um:5.1f}  "
            f"{run.n_steps:7d}  "
            f"{run.noise_eps:7.4f}  "
            f"{run.noise_sigma_um:8.3f}  "
            f"{run.noise_seed:5d}  "
            f"{run.representative_phase_mean_rad:14.7e}  "
            f"{run.representative_phase_rms_rad:13.7e}  "
            f"{run.final_power_mW:14.9f}  "
            f"{run.near_rms_x_um:13.7f}  "
            f"{run.far_rms_kx_rad_per_um:21.9f}  "
            f"{run.outer_kx_power_fraction:16.9e}  "
            f"{run.relative_l2_from_reference:15.9e}"
        )


def main() -> None:
    args = parse_args()
    if args.noise_eps < 0.0:
        raise ValueError("--noise-eps must be nonnegative")
    if args.noise_sigma_um < 0.0:
        raise ValueError("--noise-sigma-um must be nonnegative")
    if args.noise_mode == "independent" and args.workflow != "self_consistent":
        raise ValueError(
            "experimental phase-screen insertion currently targets only "
            "--workflow self_consistent"
        )

    output_stem = "validation_01_longitudinal_step_far_field"
    if args.workflow == "self_consistent":
        output_stem += "_static_self_consistent"
    if args.bias_case == "zero_voltage_pi4_boundary":
        output_stem += "_V_0_theta_bc_pi4"
    if args.noise_mode == "independent":
        eps_slug = f"{args.noise_eps:g}".replace(".", "p")
        sigma_slug = f"{args.noise_sigma_um:g}".replace(".", "p")
        output_stem += (
            f"_noise_independent_eps_{eps_slug}"
            f"_sigma_{sigma_slug}_seed_{args.noise_seed}"
        )
    output_dir = (
        args.repo_root / "outputs" / output_stem
        if args.output_dir is None
        else args.output_dir
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / f"{output_stem}.npz"
    png_path = output_dir / f"{output_stem}.png"
    phase_png_path = output_dir / f"{output_stem}_representative_phase.png"

    runs: list[ValidationRun] = []
    kx_rad_per_um = None
    ky_rad_per_um = None
    print(
        "configuration: "
        f"Nx={NX}, Ny={NY}, apertures=({X_APERTURE_UM:g}, "
        f"{Y_APERTURE_UM:g}) um, z={Z_LENGTH_UM:g} um, "
        f"waist={WAIST_UM:g} um, power={POWER_MW:g} mW"
    )
    print(f"workflow: {args.workflow}")
    print(
        "noise: "
        f"mode={args.noise_mode}, eps={args.noise_eps:g}, "
        f"sigma_um={args.noise_sigma_um:g}, seed={args.noise_seed}"
    )
    bias_request = make_request(
        DZ_VALUES_UM[0],
        args.workflow,
        args.bias_case,
    )
    bias_b = resolved_b(MATERIAL, bias_request.bias)
    if args.bias_case == "zero_bc_pi4_center":
        print(
            "director: theta_bc=0, "
            f"analytic theta(0)="
            f"{theta0_from_b_zero_bc(TARGET_B):.12f} rad, "
            f"b_override={TARGET_B:.12f}"
        )
    else:
        print(
            "director: "
            f"V_bias={bias_request.bias.V_bias:g} V, "
            f"theta_bc={bias_request.bias.theta_bc:.12f} rad, "
            f"resolved_b={bias_b:.12f}, uniform prepared theta=pi/4"
        )

    for dz_um in DZ_VALUES_UM:
        print(f"running dz={dz_um:g} um ...", flush=True)
        run, case_kx, case_ky = execute_case(
            dz_um,
            args.workflow,
            noise_mode=args.noise_mode,
            noise_eps=args.noise_eps,
            noise_sigma_um=args.noise_sigma_um,
            noise_seed=args.noise_seed,
            bias_case=args.bias_case,
        )
        if kx_rad_per_um is None:
            kx_rad_per_um = case_kx
            ky_rad_per_um = case_ky
        else:
            np.testing.assert_allclose(case_kx, kx_rad_per_um, rtol=0, atol=0)
            np.testing.assert_allclose(case_ky, ky_rad_per_um, rtol=0, atol=0)
        runs.append(run)
        print(f"completed dz={dz_um:g} um in {run.elapsed_s:.3f} s")

    assert kx_rad_per_um is not None
    assert ky_rad_per_um is not None
    add_reference_differences(runs)

    reference_grid = make_grid(
        make_request(
            DZ_VALUES_UM[0],
            args.workflow,
            args.bias_case,
        ).grid
    )
    x_um = np.asarray(asnumpy(reference_grid.x_um))
    y_um = np.asarray(asnumpy(reference_grid.y_um))

    global_peak = max(
        float(np.max(run.far_intensity_normalized)) for run in runs
    )
    log_vmax = math.log10(global_peak)
    log_vmin = log_vmax - LOG_DYNAMIC_RANGE_DECADES
    log_floor = 10.0**log_vmin
    far_log10 = np.stack(
        [
            np.log10(np.maximum(run.far_intensity_normalized, log_floor))
            for run in runs
        ]
    )

    save_numerical_results(
        npz_path,
        runs,
        x_um,
        y_um,
        kx_rad_per_um,
        ky_rad_per_um,
        far_log10,
        workflow=args.workflow,
        noise_mode=args.noise_mode,
        bias_case=args.bias_case,
    )
    save_comparison_figure(
        png_path,
        runs,
        x_um,
        y_um,
        kx_rad_per_um,
        ky_rad_per_um,
        far_log10,
        log_vmin,
        log_vmax,
        args.workflow,
        args.noise_mode,
        args.noise_eps,
        args.noise_sigma_um,
        args.noise_seed,
        args.bias_case,
    )
    if args.noise_mode == "independent":
        save_phase_figure(
            phase_png_path,
            runs,
            x_um,
            y_um,
            kx_rad_per_um,
            ky_rad_per_um,
        )

    print()
    print_table(runs)
    print(
        f"sampled_theta_center_rad = "
        f"{runs[0].sampled_theta_center_rad:.12f}"
    )
    if args.workflow == "self_consistent":
        print("self-consistent convergence:")
        for run in runs:
            print(
                f"  dz={run.dz_um:g} um: "
                f"unconverged_slices={run.unconverged_slices}/{run.n_steps}, "
                f"max_residual_rms={run.max_final_residual_rms:.9e}, "
                f"max_residual_max={run.max_final_residual_max:.9e}"
            )
    if args.noise_mode == "independent":
        print("phase-layer checks:")
        for run in runs:
            print(
                f"  dz={run.dz_um:g} um: "
                f"raw_std_expected={run.raw_std_expected_rad:.9e}, "
                f"raw_rms_measured={run.raw_rms_measured_rad:.9e}, "
                f"ratio={run.raw_rms_ratio:.6f}, "
                f"grid_scale_fraction="
                f"{run.raw_grid_scale_fraction:.6f}->"
                f"{run.phase_grid_scale_fraction:.6f}, "
                f"pointwise_I_error="
                f"{run.screen_pointwise_intensity_max_error:.3e}, "
                f"relative_power_error="
                f"{run.screen_relative_power_error:.3e}"
            )
    print(f"npz_path = {npz_path}")
    print(f"png_path = {png_path}")
    if args.noise_mode == "independent":
        print(f"phase_png_path = {phase_png_path}")


if __name__ == "__main__":
    main()
