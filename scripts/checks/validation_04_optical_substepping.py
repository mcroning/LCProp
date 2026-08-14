#!/usr/bin/env python3
"""Controlled optical substepping through one frozen self-consistent LC stack.

The nominal LC/director interval remains 20 um.  A trusted Nsub=1
self-consistent run first generates the accepted theta distribution for every
nominal interval.  The optical launch is then replayed through that identical
theta stack with the existing split-step implementation, using a diffraction
kernel for dz/Nsub and the existing fractional nonlinear phase update.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from dataclasses import dataclass, replace
from pathlib import Path
import tempfile
from time import perf_counter

import numpy as np

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "lcprop-matplotlib-cache"),
)

import matplotlib.pyplot as plt
from scipy.signal import find_peaks

import validation_01_longitudinal_step_far_field as validation
from lcprop.core.backend import asnumpy
from lcprop.core.grid import make_grid
from lcprop.lc import run_static
from lcprop.lc.propagation import advance_slice, neff_from_theta
from lcprop.optics.launch import (
    build_launch,
    normalized_power,
    reconstructed_physical_powers_mW,
)
from lcprop.optics.splitstep import (
    linear_kernel,
    total_intensity,
)
from lcprop.products.diagnostics import rms_widths


NOMINAL_DZ_UM = 20.0
NSUB_VALUES = (1, 2, 4, 10)
LOW_ORDERS = (1, 2, 3, 4)
THETA_IN_RAD = 0.0
LOG_DYNAMIC_RANGE_DECADES = 12.0
ANNULAR_HALF_WIDTH_BINS = 1.0


@dataclass
class ReplayRun:
    nsub: int
    dz_sub_um: float
    effective_screen_spacing_um: float
    nominal_intervals: int
    total_steps: int
    elapsed_s: float
    final_power_mW: float
    normalized_final_power: float
    near_rms_x_um: float
    far_rms_kx_rad_per_um: float
    final_field: np.ndarray
    near_intensity: np.ndarray
    far_intensity_normalized: np.ndarray
    baseline_relative_l2: float
    interval_phase_max_error_rad: float
    interval_phase_factor_max_error: float
    total_phase_max_error_rad: float


def radial_profile(
    intensity: np.ndarray,
    kx: np.ndarray,
    ky: np.ndarray,
    *,
    bin_spacing: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    radius = np.sqrt(kx[:, None] ** 2 + ky[None, :] ** 2)
    edges = np.arange(
        0.0,
        float(np.max(radius)) + 2.0 * bin_spacing,
        bin_spacing,
    )
    weighted, _ = np.histogram(
        radius.ravel(),
        bins=edges,
        weights=intensity.ravel(),
    )
    counts, _ = np.histogram(radius.ravel(), bins=edges)
    profile = np.divide(
        weighted,
        counts,
        out=np.zeros_like(weighted),
        where=counts > 0,
    )
    return 0.5 * (edges[:-1] + edges[1:]), profile, counts


def predicted_rings(
    *,
    nsub: int,
    n_ref: float,
    wavelength_um: float,
) -> list[dict[str, float | int | bool]]:
    dz_screen_um = NOMINAL_DZ_UM / nsub
    rows: list[dict[str, float | int | bool]] = []
    for order in LOW_ORDERS:
        term = (
            n_ref * order / (wavelength_um * dz_screen_um)
            - (order / dz_screen_um) ** 2
        )
        if term <= 0.0:
            continue
        f_jy = math.sqrt(term)
        f_jx = math.cos(THETA_IN_RAD) * f_jy
        rows.append(
            {
                "nominal_dz_um": NOMINAL_DZ_UM,
                "Nsub": nsub,
                "dz_sub_um": dz_screen_um,
                "effective_screen_spacing_um": dz_screen_um,
                "artifact_order_j": order,
                "predicted_f_jx_cycles_per_um": f_jx,
                "predicted_f_jy_cycles_per_um": f_jy,
                "predicted_k_jx_rad_per_um": 2.0 * math.pi * f_jx,
                "predicted_k_jy_rad_per_um": 2.0 * math.pi * f_jy,
            }
        )
    return rows


def annular_mean(
    intensity: np.ndarray,
    radius: np.ndarray,
    lower: float,
    upper: float,
) -> tuple[float, int]:
    mask = (radius > lower) & (radius < upper)
    count = int(np.count_nonzero(mask))
    if count == 0:
        return math.nan, 0
    return float(np.mean(intensity[mask])), count


def add_ring_measurements(
    rows: list[dict[str, float | int | bool]],
    intensity: np.ndarray,
    kx: np.ndarray,
    ky: np.ndarray,
    radial_k: np.ndarray,
    radial_mean: np.ndarray,
    *,
    radial_bin_spacing: float,
    annular_half_width: float,
) -> None:
    radius = np.sqrt(kx[:, None] ** 2 + ky[None, :] ** 2)
    peak_indices, _ = find_peaks(radial_mean)
    if peak_indices.size == 0:
        peak_indices = np.asarray([int(np.argmax(radial_mean))])
    measured_peak_radii = radial_k[peak_indices]
    radial_support = float(np.max(radius))
    kx_limit = float(np.max(np.abs(kx)))
    ky_limit = float(np.max(np.abs(ky)))

    for row in rows:
        predicted = float(row["predicted_k_jy_rad_per_um"])
        if predicted <= radial_support:
            peak_index = int(
                np.argmin(np.abs(measured_peak_radii - predicted))
            )
            measured = float(measured_peak_radii[peak_index])
            error = measured - predicted
        else:
            measured = math.nan
            error = math.nan

        ring_mean, ring_count = annular_mean(
            intensity,
            radius,
            predicted - annular_half_width,
            predicted + annular_half_width,
        )
        inner_mean, inner_count = annular_mean(
            intensity,
            radius,
            predicted - 3.0 * annular_half_width,
            predicted - 2.0 * annular_half_width,
        )
        outer_mean, outer_count = annular_mean(
            intensity,
            radius,
            predicted + 2.0 * annular_half_width,
            predicted + 3.0 * annular_half_width,
        )
        neighbor_mean = 0.5 * (inner_mean + outer_mean)
        contrast = (
            ring_mean / neighbor_mean
            if np.isfinite(neighbor_mean) and neighbor_mean > 0.0
            else math.nan
        )
        row.update(
            {
                "inside_x_nyquist": predicted <= kx_limit,
                "inside_y_nyquist": predicted <= ky_limit,
                "inside_radial_corner_support": predicted <= radial_support,
                "measured_nearest_radial_peak_rad_per_um": measured,
                "peak_error_rad_per_um": error,
                "peak_error_fft_bins": error / radial_bin_spacing,
                "peak_uncertainty_rad_per_um": radial_bin_spacing,
                "annular_half_width_rad_per_um": annular_half_width,
                "ring_mean": ring_mean,
                "inner_mean": inner_mean,
                "outer_mean": outer_mean,
                "annular_contrast": contrast,
                "ring_pixel_count": ring_count,
                "inner_pixel_count": inner_count,
                "outer_pixel_count": outer_count,
            }
        )


def replay_frozen_theta(
    *,
    nsub: int,
    request,
    grid,
    launch,
    theta_stack: np.ndarray,
    baseline_final_field: np.ndarray,
    wavelength_um: float,
    n_ref: float,
) -> ReplayRun:
    dz_sub_um = NOMINAL_DZ_UM / nsub
    kernel_sub = linear_kernel(
        grid.fxy2_um,
        dz=dz_sub_um,
        wavelength=wavelength_um,
        n_ref=n_ref,
        xp=np,
    )
    A = np.asarray(launch.A0).copy()
    k0 = 2.0 * math.pi / wavelength_um
    interval_phase_max_error = 0.0
    interval_factor_max_error = 0.0
    total_full_phase = np.zeros(theta_stack.shape[1:], dtype=np.float64)
    total_sub_phase = np.zeros_like(total_full_phase)

    started = perf_counter()
    for theta in theta_stack:
        dn = (
            neff_from_theta(
                theta,
                ne=request.material.ne,
                no=request.material.no,
                xp=np,
            )
            - n_ref
        )
        phase_full = k0 * NOMINAL_DZ_UM * dn
        phase_sub = k0 * dz_sub_um * dn
        accumulated_sub_phase = nsub * phase_sub
        interval_phase_max_error = max(
            interval_phase_max_error,
            float(np.max(np.abs(accumulated_sub_phase - phase_full))),
        )
        interval_factor_max_error = max(
            interval_factor_max_error,
            float(
                np.max(
                    np.abs(
                        np.exp(1j * phase_sub) ** nsub
                        - np.exp(1j * phase_full)
                    )
                )
            ),
        )
        total_full_phase += phase_full
        total_sub_phase += accumulated_sub_phase
        advance_slice(
            A,
            theta,
            kernel=kernel_sub,
            dz=NOMINAL_DZ_UM,
            wavelength=wavelength_um,
            n_ref=n_ref,
            ne=request.material.ne,
            no=request.material.no,
            Nsub=nsub,
            xp=np,
        )
    elapsed_s = perf_counter() - started

    final_field = np.asarray(A[0]).copy()
    near_intensity = np.asarray(
        total_intensity(
            A,
            coherence_groups=request.beams.coherence_groups,
            xp=np,
        )
    )
    near_rms_x_um, _ = rms_widths(near_intensity, grid)
    kx, ky, far_intensity, dkx, dky = validation.centered_far_field(
        final_field,
        dx_um=grid.dx_um,
        dy_um=grid.dy_um,
        xp=np,
    )
    far_rms, _ = validation.far_field_metrics(
        far_intensity,
        kx,
        dkx=dkx,
        dky=dky,
        xp=np,
    )
    baseline_norm = float(np.linalg.norm(baseline_final_field))
    return ReplayRun(
        nsub=nsub,
        dz_sub_um=dz_sub_um,
        effective_screen_spacing_um=dz_sub_um,
        nominal_intervals=theta_stack.shape[0],
        total_steps=theta_stack.shape[0] * nsub,
        elapsed_s=elapsed_s,
        final_power_mW=float(
            reconstructed_physical_powers_mW(A, grid, launch).sum()
        ),
        normalized_final_power=float(normalized_power(A, grid)),
        near_rms_x_um=float(near_rms_x_um),
        far_rms_kx_rad_per_um=float(far_rms),
        final_field=final_field,
        near_intensity=near_intensity.copy(),
        far_intensity_normalized=np.asarray(far_intensity).copy(),
        baseline_relative_l2=float(
            np.linalg.norm(final_field - baseline_final_field)
            / baseline_norm
        ),
        interval_phase_max_error_rad=interval_phase_max_error,
        interval_phase_factor_max_error=interval_factor_max_error,
        total_phase_max_error_rad=float(
            np.max(np.abs(total_sub_phase - total_full_phase))
        ),
    )


def line_style(order: int) -> str:
    return {1: "-", 2: "--", 3: ":", 4: "-."}[order]


def draw_ring_overlays(
    axis,
    rows: list[dict[str, float | int | bool]],
    kx: np.ndarray,
    ky: np.ndarray,
) -> None:
    phi = np.linspace(0.0, 2.0 * math.pi, 1000)
    for row in rows:
        order = int(row["artifact_order_j"])
        k_jx = float(row["predicted_k_jx_rad_per_um"])
        k_jy = float(row["predicted_k_jy_rad_per_um"])
        curve_x = k_jx * np.cos(phi)
        curve_y = k_jy * np.sin(phi)
        valid = (
            (curve_x >= kx[0])
            & (curve_x <= kx[-1])
            & (curve_y >= ky[0])
            & (curve_y <= ky[-1])
        )
        if not np.any(valid):
            continue
        axis.plot(
            np.where(valid, curve_x, np.nan),
            np.where(valid, curve_y, np.nan),
            color="white",
            linestyle=line_style(order),
            linewidth=0.75,
            alpha=0.72,
        )
        label_index = np.flatnonzero(valid)[len(np.flatnonzero(valid)) // 5]
        axis.text(
            curve_x[label_index],
            curve_y[label_index],
            f"j={order}",
            color="white",
            fontsize=6,
            alpha=0.8,
            ha="center",
            va="center",
        )


def save_far_figure(
    path: Path,
    runs: list[ReplayRun],
    predictions,
    kx: np.ndarray,
    ky: np.ndarray,
) -> None:
    global_peak = max(
        float(np.max(run.far_intensity_normalized)) for run in runs
    )
    vmax = math.log10(global_peak)
    vmin = vmax - LOG_DYNAMIC_RANGE_DECADES
    floor = 10.0**vmin
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12.8, 10.4),
        constrained_layout=True,
    )
    image = None
    for run, rows, axis in zip(
        runs, predictions, axes.flat, strict=True
    ):
        image = axis.imshow(
            np.log10(np.maximum(run.far_intensity_normalized, floor)).T,
            origin="lower",
            extent=(kx[0], kx[-1], ky[0], ky[-1]),
            vmin=vmin,
            vmax=vmax,
            cmap="magma",
            interpolation="nearest",
            aspect="equal",
        )
        draw_ring_overlays(axis, rows, kx, ky)
        axis.set(
            title=(
                f"Nsub={run.nsub}, effective spacing="
                f"{run.effective_screen_spacing_um:g} µm"
            ),
            xlabel="kx (rad/µm)",
            ylabel="ky (rad/µm)",
        )
    assert image is not None
    colorbar = figure.colorbar(image, ax=axes, shrink=0.88, pad=0.02)
    colorbar.set_label("log10 normalized spectral power density")
    figure.suptitle(
        "Frozen self-consistent theta: fractional optical substepping",
        y=1.02,
    )
    figure.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(figure)


def save_radial_figure(
    path: Path,
    runs,
    predictions,
    radial_data,
    kx: np.ndarray,
    ky: np.ndarray,
) -> None:
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12.8, 8.8),
        constrained_layout=True,
        sharey=True,
    )
    for run, rows, (radial_k, profile, counts), axis in zip(
        runs, predictions, radial_data, axes.flat, strict=True
    ):
        positive = profile[profile > 0.0]
        floor = float(np.max(positive)) * 1e-14
        axis.semilogy(
            radial_k,
            np.maximum(profile, floor),
            color="black",
            linewidth=1.2,
        )
        for row in rows:
            predicted = float(row["predicted_k_jy_rad_per_um"])
            if predicted > radial_k[-1]:
                continue
            order = int(row["artifact_order_j"])
            axis.axvline(
                predicted,
                color="tab:red",
                linestyle=line_style(order),
                linewidth=0.8,
                alpha=0.72,
            )
            axis.text(
                predicted,
                0.92 - 0.08 * (order - 1),
                f"j={order}",
                transform=axis.get_xaxis_transform(),
                rotation=90,
                color="tab:red",
                fontsize=6,
                va="top",
                ha="right",
            )
        axis.set(
            title=f"Nsub={run.nsub}, spacing={run.dz_sub_um:g} µm",
            xlabel="radial k (rad/µm)",
            ylabel="Azimuthal mean spectral density",
            xlim=(0.0, radial_k[-1]),
        )
        axis.grid(alpha=0.2)
        # Radii above min(|kx|_max, |ky|_max) have only partial azimuthal
        # coverage in the rectangular FFT window.
        axis.axvspan(
            min(float(np.max(np.abs(kx))), float(np.max(np.abs(ky)))),
            radial_k[-1],
            color="0.7",
            alpha=0.08,
            linewidth=0.0,
        )
    figure.suptitle(
        "Radial far-field profiles (gray region has partial angular coverage)"
    )
    figure.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(figure)


def save_cut_figure(
    path: Path,
    runs,
    predictions,
    kx: np.ndarray,
    ky: np.ndarray,
) -> None:
    figure, axes = plt.subplots(
        4,
        2,
        figsize=(13.2, 14.0),
        constrained_layout=True,
    )
    ix0 = int(np.argmin(np.abs(kx)))
    iy0 = int(np.argmin(np.abs(ky)))
    for row_index, (run, rows) in enumerate(
        zip(runs, predictions, strict=True)
    ):
        x_cut = run.far_intensity_normalized[:, iy0]
        y_cut = run.far_intensity_normalized[ix0, :]
        x_axis, y_axis = axes[row_index]
        x_axis.semilogy(
            kx,
            np.maximum(x_cut, float(np.max(x_cut)) * 1e-14),
            color="black",
        )
        y_axis.semilogy(
            ky,
            np.maximum(y_cut, float(np.max(y_cut)) * 1e-14),
            color="black",
        )
        for row in rows:
            order = int(row["artifact_order_j"])
            for sign in (-1.0, 1.0):
                x_position = (
                    sign * float(row["predicted_k_jx_rad_per_um"])
                )
                y_position = (
                    sign * float(row["predicted_k_jy_rad_per_um"])
                )
                if kx[0] <= x_position <= kx[-1]:
                    x_axis.axvline(
                        x_position,
                        color="tab:red",
                        linestyle=line_style(order),
                        linewidth=0.75,
                        alpha=0.7,
                    )
                if ky[0] <= y_position <= ky[-1]:
                    y_axis.axvline(
                        y_position,
                        color="tab:red",
                        linestyle=line_style(order),
                        linewidth=0.75,
                        alpha=0.7,
                    )
        x_axis.set(
            title=f"Nsub={run.nsub}: I(kx, ky=0)",
            xlabel="kx (rad/µm)",
            ylabel="Spectral density",
        )
        y_axis.set(
            title=f"Nsub={run.nsub}: I(kx=0, ky)",
            xlabel="ky (rad/µm)",
            ylabel="Spectral density",
        )
        x_axis.grid(alpha=0.2)
        y_axis.grid(alpha=0.2)
    figure.suptitle("Central far-field cuts with predicted intersections")
    figure.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = (
        repo_root
        / "outputs"
        / "validation_04_optical_substepping"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    far_png = output_dir / "validation_04_far_fields.png"
    radial_png = output_dir / "validation_04_radial_profiles.png"
    cuts_png = output_dir / "validation_04_xy_cuts.png"
    csv_path = output_dir / "validation_04_ring_metrics.csv"
    npz_path = output_dir / "validation_04_optical_substepping_data.npz"

    request = validation.make_request(
        NOMINAL_DZ_UM,
        "self_consistent",
        "zero_voltage_pi4_boundary",
    )
    request = replace(
        request,
        runtime=replace(
            request.runtime,
            optical_substeps_enabled=False,
        ),
    )
    grid = make_grid(request.grid, xp=np, real_dtype=np.float64)
    launch = build_launch(
        request.beams,
        grid,
        complex_dtype=np.complex128,
    )
    wavelength_um = float(request.beams.channels[0].wavelength_um)
    n_ref = float(request.material.no)
    print(f"wavelength_um = {wavelength_um:.12f}")
    print(f"prediction_n_ref = {n_ref:.12f}")
    print("prediction_n_source = run_static n_ref = request.material.no")
    print(
        "substep_mode = accepted Nsub=1 theta stack frozen; "
        "advance_slice applies dz/Nsub nonlinear phase before each "
        "matching dz/Nsub diffraction hop"
    )

    print("running trusted self-consistent Nsub=1 baseline ...", flush=True)
    baseline_started = perf_counter()
    baseline_result = run_static(request)
    baseline_elapsed_s = perf_counter() - baseline_started
    theta_stack = np.asarray(asnumpy(baseline_result.theta_final)).copy()
    baseline_final_field = np.asarray(
        asnumpy(baseline_result.A_final[0])
    ).copy()
    if theta_stack.shape != (grid.Nz, grid.Nx, grid.Ny):
        raise AssertionError("expected one accepted theta plane per nominal dz")
    theta_sha256 = hashlib.sha256(
        np.ascontiguousarray(theta_stack).view(np.uint8)
    ).hexdigest()
    unconverged_slices = sum(
        not summary.converged
        for summary in baseline_result.slice_summaries
    )
    print(
        f"baseline completed in {baseline_elapsed_s:.3f} s; "
        f"theta_sha256={theta_sha256}"
    )

    runs: list[ReplayRun] = []
    kx = ky = None
    for nsub in NSUB_VALUES:
        print(f"replaying frozen theta with Nsub={nsub} ...", flush=True)
        run = replay_frozen_theta(
            nsub=nsub,
            request=request,
            grid=grid,
            launch=launch,
            theta_stack=theta_stack,
            baseline_final_field=baseline_final_field,
            wavelength_um=wavelength_um,
            n_ref=n_ref,
        )
        runs.append(run)
        case_kx, case_ky, _, _, _ = validation.centered_far_field(
            run.final_field,
            dx_um=grid.dx_um,
            dy_um=grid.dy_um,
            xp=np,
        )
        if kx is None:
            kx = np.asarray(case_kx)
            ky = np.asarray(case_ky)
        else:
            np.testing.assert_array_equal(kx, case_kx)
            np.testing.assert_array_equal(ky, case_ky)
        print(f"completed Nsub={nsub} in {run.elapsed_s:.3f} s")

    assert kx is not None and ky is not None
    dkx = float(kx[1] - kx[0])
    dky = float(ky[1] - ky[0])
    radial_bin_spacing = min(dkx, dky)
    annular_half_width = ANNULAR_HALF_WIDTH_BINS * radial_bin_spacing
    if not (np.all(np.diff(kx) > 0.0) and np.all(np.diff(ky) > 0.0)):
        raise AssertionError("FFT axes must be strictly increasing")
    for run in runs:
        if run.final_field.shape != (kx.size, ky.size):
            raise AssertionError("array axis 0/1 must map to kx/ky")
        integral = (
            float(np.sum(run.far_intensity_normalized)) * dkx * dky
        )
        if not np.isclose(integral, 1.0, rtol=1e-12, atol=1e-12):
            raise AssertionError("far field is not unit-integral")

    # Replaying the accepted theta stack with one substep must exactly follow
    # the accepted optical path in the trusted self-consistent baseline.
    if runs[0].baseline_relative_l2 > 1e-13:
        raise AssertionError("Nsub=1 did not reproduce the trusted baseline")

    predictions = []
    radial_data = []
    all_rows = []
    for run in runs:
        rows = predicted_rings(
            nsub=run.nsub,
            n_ref=n_ref,
            wavelength_um=wavelength_um,
        )
        radial = radial_profile(
            run.far_intensity_normalized,
            kx,
            ky,
            bin_spacing=radial_bin_spacing,
        )
        add_ring_measurements(
            rows,
            run.far_intensity_normalized,
            kx,
            ky,
            radial[0],
            radial[1],
            radial_bin_spacing=radial_bin_spacing,
            annular_half_width=annular_half_width,
        )
        for row in rows:
            row.update(
                {
                    "nominal_intervals": run.nominal_intervals,
                    "total_steps": run.total_steps,
                    "final_power_mW": run.final_power_mW,
                    "normalized_final_power": run.normalized_final_power,
                    "near_rms_x_um": run.near_rms_x_um,
                    "far_rms_kx_rad_per_um": run.far_rms_kx_rad_per_um,
                    "baseline_relative_l2": run.baseline_relative_l2,
                    "interval_phase_max_error_rad": (
                        run.interval_phase_max_error_rad
                    ),
                    "interval_phase_factor_max_error": (
                        run.interval_phase_factor_max_error
                    ),
                    "total_phase_max_error_rad": (
                        run.total_phase_max_error_rad
                    ),
                    "theta_max_abs_difference": 0.0,
                    "theta_sha256": theta_sha256,
                }
            )
        predictions.append(rows)
        radial_data.append(radial)
        all_rows.extend(rows)

    save_far_figure(far_png, runs, predictions, kx, ky)
    save_radial_figure(
        radial_png,
        runs,
        predictions,
        radial_data,
        kx,
        ky,
    )
    save_cut_figure(cuts_png, runs, predictions, kx, ky)

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)

    metadata = {
        "workflow": "static local_self_consistent baseline plus frozen-theta optical replay",
        "bias_case": "V_app=0, theta_bc=pi/4",
        "noise_eps": 0.0,
        "nominal_dz_um": NOMINAL_DZ_UM,
        "z_length_um": float(request.grid.z_length_um),
        "wavelength_um": wavelength_um,
        "prediction_n_ref": n_ref,
        "prediction_n_source": "run_static n_ref = request.material.no",
        "nonlinear_phase_application": (
            "fractional phase k0*(neff(theta)-n_ref)*dz/Nsub applied "
            "at every optical substep; theta is not recomputed within or "
            "between replay cases"
        ),
        "theta_identical_across_replays": True,
        "theta_sha256": theta_sha256,
        "baseline_elapsed_s": baseline_elapsed_s,
        "baseline_unconverged_slices": unconverged_slices,
        "baseline_max_final_residual_rms": (
            baseline_result.max_final_residual_rms
        ),
        "baseline_max_final_residual_max": (
            baseline_result.max_final_residual_max
        ),
        "kx_first_last_rad_per_um": [float(kx[0]), float(kx[-1])],
        "ky_first_last_rad_per_um": [float(ky[0]), float(ky[-1])],
        "kx_abs_nyquist_rad_per_um": float(np.max(np.abs(kx))),
        "ky_abs_nyquist_rad_per_um": float(np.max(np.abs(ky))),
        "dkx_rad_per_um": dkx,
        "dky_rad_per_um": dky,
        "radial_bin_spacing_rad_per_um": radial_bin_spacing,
        "annular_half_width_rad_per_um": annular_half_width,
        "fft_axis_orientation": (
            "array axis 0 -> kx; array axis 1 -> ky; transpose only for imshow"
        ),
        "fft_extent": "full 128x128 computational field and discrete Nyquist spectrum",
        "frequency_conversion": "k = 2*pi*f",
        "common_log_dynamic_range_decades": LOG_DYNAMIC_RANGE_DECADES,
    }
    np.savez_compressed(
        npz_path,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        Nsub=np.asarray([run.nsub for run in runs], dtype=np.int64),
        nominal_dz_um=np.asarray(NOMINAL_DZ_UM),
        dz_sub_um=np.asarray([run.dz_sub_um for run in runs]),
        effective_screen_spacing_um=np.asarray(
            [run.effective_screen_spacing_um for run in runs]
        ),
        nominal_intervals=np.asarray(
            [run.nominal_intervals for run in runs], dtype=np.int64
        ),
        total_steps=np.asarray(
            [run.total_steps for run in runs], dtype=np.int64
        ),
        final_power_mW=np.asarray([run.final_power_mW for run in runs]),
        normalized_final_power=np.asarray(
            [run.normalized_final_power for run in runs]
        ),
        near_rms_x_um=np.asarray([run.near_rms_x_um for run in runs]),
        far_rms_kx_rad_per_um=np.asarray(
            [run.far_rms_kx_rad_per_um for run in runs]
        ),
        replay_elapsed_s=np.asarray([run.elapsed_s for run in runs]),
        baseline_relative_l2=np.asarray(
            [run.baseline_relative_l2 for run in runs]
        ),
        interval_phase_max_error_rad=np.asarray(
            [run.interval_phase_max_error_rad for run in runs]
        ),
        interval_phase_factor_max_error=np.asarray(
            [run.interval_phase_factor_max_error for run in runs]
        ),
        total_phase_max_error_rad=np.asarray(
            [run.total_phase_max_error_rad for run in runs]
        ),
        theta_max_abs_difference=np.zeros(len(runs)),
        frozen_theta_stack=theta_stack,
        kx_rad_per_um=kx,
        ky_rad_per_um=ky,
        final_field=np.stack([run.final_field for run in runs]),
        near_field_intensity=np.stack(
            [run.near_intensity for run in runs]
        ),
        far_field_intensity_normalized=np.stack(
            [run.far_intensity_normalized for run in runs]
        ),
        radial_k_rad_per_um=np.stack([item[0] for item in radial_data]),
        radial_profile=np.stack([item[1] for item in radial_data]),
        radial_sample_counts=np.stack([item[2] for item in radial_data]),
        artifact_Nsub=np.asarray(
            [row["Nsub"] for row in all_rows], dtype=np.int64
        ),
        artifact_order_j=np.asarray(
            [row["artifact_order_j"] for row in all_rows],
            dtype=np.int64,
        ),
        predicted_k_rad_per_um=np.asarray(
            [row["predicted_k_jy_rad_per_um"] for row in all_rows]
        ),
        measured_peak_rad_per_um=np.asarray(
            [
                row["measured_nearest_radial_peak_rad_per_um"]
                for row in all_rows
            ]
        ),
        peak_error_rad_per_um=np.asarray(
            [row["peak_error_rad_per_um"] for row in all_rows]
        ),
        peak_error_fft_bins=np.asarray(
            [row["peak_error_fft_bins"] for row in all_rows]
        ),
        annular_contrast=np.asarray(
            [row["annular_contrast"] for row in all_rows]
        ),
    )

    print()
    print(
        "nom_dz  Nsub  dz_sub  steps  power_mW  near_rms_x  far_rms_kx  "
        "j  k_pred  k_peak  delta_k  bins  contrast"
    )
    for row in all_rows:
        print(
            f"{float(row['nominal_dz_um']):6.1f}  "
            f"{int(row['Nsub']):4d}  "
            f"{float(row['dz_sub_um']):6.1f}  "
            f"{int(row['total_steps']):5d}  "
            f"{float(row['final_power_mW']):8.6f}  "
            f"{float(row['near_rms_x_um']):10.6f}  "
            f"{float(row['far_rms_kx_rad_per_um']):10.6f}  "
            f"{int(row['artifact_order_j']):1d}  "
            f"{float(row['predicted_k_jy_rad_per_um']):7.4f}  "
            f"{float(row['measured_nearest_radial_peak_rad_per_um']):7.4f}  "
            f"{float(row['peak_error_rad_per_um']):+8.4f}  "
            f"{float(row['peak_error_fft_bins']):+6.2f}  "
            f"{float(row['annular_contrast']):9.4f}"
        )
    print()
    print(f"theta_identical_across_replays = True")
    print(f"theta_max_abs_difference = 0.0")
    print(f"theta_sha256 = {theta_sha256}")
    print(
        f"Nsub1_baseline_relative_l2 = "
        f"{runs[0].baseline_relative_l2:.16e}"
    )
    print(
        "maximum_interval_phase_error_rad = "
        f"{max(run.interval_phase_max_error_rad for run in runs):.16e}"
    )
    print(
        "maximum_interval_phase_factor_error = "
        f"{max(run.interval_phase_factor_max_error for run in runs):.16e}"
    )
    print(
        "maximum_total_phase_error_rad = "
        f"{max(run.total_phase_max_error_rad for run in runs):.16e}"
    )
    print(f"baseline_unconverged_slices = {unconverged_slices}")
    print(
        f"kx_first_last_rad_per_um = {kx[0]:.12f}, {kx[-1]:.12f}"
    )
    print(
        f"ky_first_last_rad_per_um = {ky[0]:.12f}, {ky[-1]:.12f}"
    )
    print(f"dkx_rad_per_um = {dkx:.12f}")
    print(f"dky_rad_per_um = {dky:.12f}")
    print(
        f"annular_half_width_rad_per_um = {annular_half_width:.12f}"
    )
    print(f"far_field_figure = {far_png}")
    print(f"radial_figure = {radial_png}")
    print(f"cuts_figure = {cuts_png}")
    print(f"csv_path = {csv_path}")
    print(f"npz_path = {npz_path}")


if __name__ == "__main__":
    main()
