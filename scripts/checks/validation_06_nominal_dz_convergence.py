#!/usr/bin/env python3
"""Nominal LC/director dz convergence with production optical substeps."""

from __future__ import annotations

import csv
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

import validation_01_longitudinal_step_far_field as validation
from lcprop.core.backend import asnumpy
from lcprop.core.grid import make_grid
from lcprop.optics.splitstep import total_intensity
from lcprop.workflows import run_static


DZ_VALUES_UM = (20.0, 10.0, 5.0, 2.0)
REFERENCE_DZ_UM = 2.0
SECONDARY_REFERENCE_DZ_UM = 5.0
COMMON_THETA_SPACING_UM = 20.0
OPTICAL_MAX_SUBSTEPS = 32
LOG_DYNAMIC_RANGE_DECADES = 12.0


@dataclass
class RunData:
    dz_um: float
    elapsed_s: float
    nominal_slices: int
    accepted_optical_substeps: int
    executed_optical_substeps: int
    final_power_mW: float
    optical_plan: dict
    final_field: np.ndarray
    near_intensity: np.ndarray
    final_theta_last: np.ndarray
    theta_common_planes: np.ndarray
    far_intensity: np.ndarray
    z_um: np.ndarray
    histories: dict[str, np.ndarray]
    converged: np.ndarray
    residual_rms: np.ndarray
    residual_max: np.ndarray
    relaxation_iterations: np.ndarray
    optical_passes: np.ndarray
    convergence: dict
    final_near: dict
    final_far: dict


def scalar_moments(
    intensity: np.ndarray,
    x_um: np.ndarray,
    y_um: np.ndarray,
    *,
    dx_um: float,
    dy_um: float,
) -> dict[str, float]:
    area = dx_um * dy_um
    power = float(np.sum(intensity) * area)
    cx = float(np.sum(x_um[:, None] * intensity) * area / power)
    cy = float(np.sum(y_um[None, :] * intensity) * area / power)
    sx = math.sqrt(
        max(
            float(
                np.sum((x_um[:, None] - cx) ** 2 * intensity)
                * area
                / power
            ),
            0.0,
        )
    )
    sy = math.sqrt(
        max(
            float(
                np.sum((y_um[None, :] - cy) ** 2 * intensity)
                * area
                / power
            ),
            0.0,
        )
    )
    return {
        "power": power,
        "centroid_x_um": cx,
        "centroid_y_um": cy,
        "rms_x_um": sx,
        "rms_y_um": sy,
        "peak_intensity": float(np.max(intensity)),
    }


def spectral_metrics(
    intensity: np.ndarray,
    kx: np.ndarray,
    ky: np.ndarray,
) -> dict[str, float]:
    dkx = float(kx[1] - kx[0])
    dky = float(ky[1] - ky[0])
    area = dkx * dky
    cx = float(np.sum(kx[:, None] * intensity) * area)
    cy = float(np.sum(ky[None, :] * intensity) * area)
    sx = math.sqrt(
        max(
            float(
                np.sum((kx[:, None] - cx) ** 2 * intensity) * area
            ),
            0.0,
        )
    )
    sy = math.sqrt(
        max(
            float(
                np.sum((ky[None, :] - cy) ** 2 * intensity) * area
            ),
            0.0,
        )
    )
    x_lower = kx[0] - 0.5 * dkx
    x_upper = kx[-1] + 0.5 * dkx
    y_lower = ky[0] - 0.5 * dky
    y_upper = ky[-1] + 0.5 * dky
    x_edge = 0.1 * (x_upper - x_lower)
    y_edge = 0.1 * (y_upper - y_lower)
    outer_x = (kx < x_lower + x_edge) | (kx > x_upper - x_edge)
    outer_y = (ky < y_lower + y_edge) | (ky > y_upper - y_edge)
    outer = outer_x[:, None] | outer_y[None, :]
    return {
        "centroid_kx_rad_per_um": cx,
        "centroid_ky_rad_per_um": cy,
        "rms_kx_rad_per_um": sx,
        "rms_ky_rad_per_um": sy,
        "outer_spectrum_power_fraction": float(
            np.sum(intensity[outer]) * area
        ),
    }


def unconverged_ranges(
    converged: np.ndarray,
    *,
    dz_um: float,
) -> tuple[list[tuple[float, float]], int]:
    bad = np.flatnonzero(~converged)
    if bad.size == 0:
        return [], 0
    groups = np.split(bad, np.flatnonzero(np.diff(bad) > 1) + 1)
    ranges = [
        (float(group[0] * dz_um), float((group[-1] + 1) * dz_um))
        for group in groups
    ]
    return ranges, max(len(group) for group in groups)


def convergence_metrics(result, dz_um: float) -> tuple[dict, ...]:
    summaries = result.slice_summaries
    converged = np.asarray([item.converged for item in summaries], dtype=bool)
    residual_rms = np.asarray(
        [item.final_residual_rms for item in summaries], dtype=float
    )
    residual_max = np.asarray(
        [item.final_residual_max for item in summaries], dtype=float
    )
    iterations = np.asarray(
        [item.relaxation_iterations for item in summaries], dtype=np.int64
    )
    optical_passes = np.asarray(
        [item.optical_passes for item in summaries], dtype=np.int64
    )
    ranges, longest = unconverged_ranges(converged, dz_um=dz_um)
    count = int(np.count_nonzero(converged))
    total = converged.size
    metrics = {
        "converged_slices": count,
        "converged_fraction": count / total,
        "unconverged_slices": total - count,
        "unconverged_fraction": (total - count) / total,
        "longest_unconverged_sequence": longest,
        "unconverged_z_ranges_um": ranges,
        "max_residual_rms": float(np.max(residual_rms)),
        "median_residual_rms": float(np.median(residual_rms)),
        "p95_residual_rms": float(np.percentile(residual_rms, 95.0)),
        "max_residual_max": float(np.max(residual_max)),
        "max_local_iteration_count": int(np.max(iterations)),
        "median_local_iteration_count": float(np.median(iterations)),
    }
    return (
        metrics,
        converged,
        residual_rms,
        residual_max,
        iterations,
        optical_passes,
    )


def execute_case(dz_um: float, common_theta_z: np.ndarray) -> RunData:
    request = validation.make_request(
        dz_um,
        "self_consistent",
        "zero_voltage_pi4_boundary",
    )
    request = replace(
        request,
        runtime=replace(
            request.runtime,
            optical_substeps_enabled=True,
            optical_dn_max_est=0.02,
            optical_max_phase_per_substep_rad=0.30,
            optical_max_substeps=OPTICAL_MAX_SUBSTEPS,
        ),
    )
    grid = make_grid(request.grid, xp=np, real_dtype=np.float64)
    started = perf_counter()
    result = run_static(request)
    elapsed_s = perf_counter() - started

    final_field = np.asarray(asnumpy(result.A_final[0])).copy()
    near_intensity = np.asarray(
        asnumpy(
            total_intensity(
                result.A_final,
                coherence_groups=request.beams.coherence_groups,
                xp=grid.xp,
            )
        )
    ).copy()
    kx, ky, far_intensity, _, _ = validation.centered_far_field(
        final_field,
        dx_um=grid.dx_um,
        dy_um=grid.dy_um,
        xp=np,
    )
    kx = np.asarray(kx)
    ky = np.asarray(ky)
    far_intensity = np.asarray(far_intensity)

    intensity_stack = np.asarray(asnumpy(result.intensity_stack))
    theta_stack = np.asarray(asnumpy(result.theta_final))
    z_um = np.asarray(
        [item.z_um for item in result.slice_summaries], dtype=float
    )
    histories = {
        "centroid_x_um": np.empty(grid.Nz),
        "centroid_y_um": np.empty(grid.Nz),
        "rms_x_um": np.empty(grid.Nz),
        "rms_y_um": np.empty(grid.Nz),
        "peak_intensity": np.empty(grid.Nz),
        "optical_power": np.empty(grid.Nz),
        "theta_max_rad": np.max(theta_stack, axis=(1, 2)),
    }
    x_um = np.asarray(grid.x_um)
    y_um = np.asarray(grid.y_um)
    for index, intensity in enumerate(intensity_stack):
        moments = scalar_moments(
            intensity,
            x_um,
            y_um,
            dx_um=grid.dx_um,
            dy_um=grid.dy_um,
        )
        for name in (
            "centroid_x_um",
            "centroid_y_um",
            "rms_x_um",
            "rms_y_um",
            "peak_intensity",
        ):
            histories[name][index] = moments[name]
        histories["optical_power"][index] = moments["power"]

    common_indices = np.rint(common_theta_z / dz_um).astype(np.int64)
    np.testing.assert_allclose(z_um[common_indices], common_theta_z)
    theta_common_planes = theta_stack[common_indices].copy()
    final_theta_last = theta_stack[-1].copy()

    (
        convergence,
        converged,
        residual_rms,
        residual_max,
        iterations,
        optical_passes,
    ) = convergence_metrics(result, dz_um)
    optical_plan = dict(result.provenance)
    accepted_optical_substeps = grid.Nz * int(optical_plan["optical_Nsub"])
    executed_optical_substeps = int(
        np.sum(1 + optical_passes) * int(optical_plan["optical_Nsub"])
    )
    final_near = scalar_moments(
        near_intensity,
        x_um,
        y_um,
        dx_um=grid.dx_um,
        dy_um=grid.dy_um,
    )
    final_far = spectral_metrics(far_intensity, kx, ky)

    return RunData(
        dz_um=dz_um,
        elapsed_s=elapsed_s,
        nominal_slices=grid.Nz,
        accepted_optical_substeps=accepted_optical_substeps,
        executed_optical_substeps=executed_optical_substeps,
        final_power_mW=float(result.physical_power_final_mW),
        optical_plan=optical_plan,
        final_field=final_field,
        near_intensity=near_intensity,
        final_theta_last=final_theta_last,
        theta_common_planes=theta_common_planes,
        far_intensity=far_intensity,
        z_um=z_um,
        histories=histories,
        converged=converged,
        residual_rms=residual_rms,
        residual_max=residual_max,
        relaxation_iterations=iterations,
        optical_passes=optical_passes,
        convergence=convergence,
        final_near=final_near,
        final_far=final_far,
    )


def phase_aligned_field_error(test: np.ndarray, reference: np.ndarray) -> float:
    phase_offset = float(np.angle(np.sum(np.conj(reference) * test)))
    aligned = test * np.exp(-1j * phase_offset)
    return float(
        np.linalg.norm(aligned - reference) / np.linalg.norm(reference)
    )


def compare_final(test: RunData, reference: RunData) -> dict[str, float]:
    near_ref_norm = np.linalg.norm(reference.near_intensity)
    far_ref_norm = np.linalg.norm(reference.far_intensity)
    theta_test = test.theta_common_planes[-1]
    theta_ref = reference.theta_common_planes[-1]
    theta_delta = theta_test - theta_ref
    common_stack_delta = test.theta_common_planes - reference.theta_common_planes
    result = {
        "field_phase_aligned_relative_l2": phase_aligned_field_error(
            test.final_field, reference.final_field
        ),
        "near_relative_l2": float(
            np.linalg.norm(test.near_intensity - reference.near_intensity)
            / near_ref_norm
        ),
        "near_peak_difference": (
            test.final_near["peak_intensity"]
            - reference.final_near["peak_intensity"]
        ),
        "near_peak_relative_difference": (
            test.final_near["peak_intensity"]
            / reference.final_near["peak_intensity"]
            - 1.0
        ),
        "near_rms_x_difference_um": (
            test.final_near["rms_x_um"] - reference.final_near["rms_x_um"]
        ),
        "near_rms_y_difference_um": (
            test.final_near["rms_y_um"] - reference.final_near["rms_y_um"]
        ),
        "near_centroid_x_difference_um": (
            test.final_near["centroid_x_um"]
            - reference.final_near["centroid_x_um"]
        ),
        "near_centroid_y_difference_um": (
            test.final_near["centroid_y_um"]
            - reference.final_near["centroid_y_um"]
        ),
        "theta_rms_difference_rad": float(
            np.sqrt(np.mean(theta_delta * theta_delta))
        ),
        "theta_max_abs_difference_rad": float(np.max(np.abs(theta_delta))),
        "theta_max_difference_rad": float(
            np.max(theta_test) - np.max(theta_ref)
        ),
        "theta_relative_l2": float(
            np.linalg.norm(theta_delta) / np.linalg.norm(theta_ref)
        ),
        "theta_common_stack_rms_difference_rad": float(
            np.sqrt(np.mean(common_stack_delta * common_stack_delta))
        ),
        "theta_common_stack_max_abs_difference_rad": float(
            np.max(np.abs(common_stack_delta))
        ),
        "far_relative_l2": float(
            np.linalg.norm(test.far_intensity - reference.far_intensity)
            / far_ref_norm
        ),
    }
    for name in (
        "rms_kx_rad_per_um",
        "rms_ky_rad_per_um",
        "outer_spectrum_power_fraction",
        "centroid_kx_rad_per_um",
        "centroid_ky_rad_per_um",
    ):
        result[f"far_{name}_difference"] = (
            test.final_far[name] - reference.final_far[name]
        )
    return result


def interpolate_histories(
    run: RunData,
    common_z_um: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        name: np.interp(common_z_um, run.z_um, values)
        for name, values in run.histories.items()
    }


def compare_histories(test, reference) -> dict[str, float]:
    result = {}
    for name in reference:
        difference = test[name] - reference[name]
        result[f"{name}_max_abs_difference"] = float(
            np.max(np.abs(difference))
        )
        result[f"{name}_rms_difference"] = float(
            np.sqrt(np.mean(difference * difference))
        )
    return result


def acceptance(
    run: RunData,
    comparison: dict[str, float],
    reference: RunData,
) -> dict[str, bool]:
    centroid_difference = max(
        abs(comparison["near_centroid_x_difference_um"]),
        abs(comparison["near_centroid_y_difference_um"]),
    )
    width_relative_difference = max(
        abs(comparison["near_rms_x_difference_um"])
        / reference.final_near["rms_x_um"],
        abs(comparison["near_rms_y_difference_um"])
        / reference.final_near["rms_y_um"],
    )
    values = {
        "near": comparison["near_relative_l2"],
        "far": comparison["far_relative_l2"],
        "theta": comparison["theta_rms_difference_rad"],
        "centroid": centroid_difference,
        "width": width_relative_difference,
        "unconverged": run.convergence["unconverged_fraction"],
    }
    strict_limits = {
        "near": 0.01,
        "far": 0.02,
        "theta": 1e-3,
        "centroid": 0.1,
        "width": 0.01,
        "unconverged": 0.01,
    }
    moderate_limits = {
        "near": 0.05,
        "far": 0.05,
        "theta": 5e-3,
        "centroid": 0.25,
        "width": 0.05,
        "unconverged": 0.05,
    }
    result: dict[str, bool] = {}
    for level, limits in (
        ("strict", strict_limits),
        ("moderate", moderate_limits),
    ):
        for name, limit in limits.items():
            result[f"{level}_{name}"] = values[name] <= limit
        result[f"{level}_overall"] = all(
            result[f"{level}_{name}"] for name in limits
        )
    result.update({f"value_{name}": value for name, value in values.items()})
    return result


def save_convergence_figure(path: Path, runs: list[RunData]) -> None:
    figure, axes = plt.subplots(
        3, 1, figsize=(11.5, 9.0), constrained_layout=True, sharex=True
    )
    for run in runs:
        label = f"dz={run.dz_um:g} µm"
        axes[0].semilogy(run.z_um, run.residual_rms, label=label)
        axes[1].step(
            run.z_um,
            run.converged.astype(float),
            where="post",
            label=label,
        )
        axes[2].plot(run.z_um, run.relaxation_iterations, label=label)
    axes[0].set(ylabel="Residual RMS", title="Nominal-slice convergence")
    axes[1].set(ylabel="Converged flag", ylim=(-0.05, 1.05))
    axes[2].set(ylabel="Relaxation iterations", xlabel="z (µm)")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(ncol=2, fontsize=8)
    figure.savefig(path, dpi=190)
    plt.close(figure)


def save_director_figure(path: Path, runs: list[RunData]) -> None:
    figure, axes = plt.subplots(
        3, 1, figsize=(11.5, 9.0), constrained_layout=True, sharex=True
    )
    for run in runs:
        label = f"dz={run.dz_um:g} µm"
        axes[0].plot(run.z_um, run.histories["theta_max_rad"], label=label)
        axes[1].semilogy(
            run.z_um, run.histories["peak_intensity"], label=label
        )
        axes[2].plot(run.z_um, run.histories["optical_power"], label=label)
    axes[0].set(ylabel="theta max (rad)", title="Director and intensity histories")
    axes[1].set(ylabel="Peak intensity")
    axes[2].set(ylabel="Normalized power", xlabel="z (µm)")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(ncol=2, fontsize=8)
    figure.savefig(path, dpi=190)
    plt.close(figure)


def save_beam_figure(path: Path, runs: list[RunData]) -> None:
    figure, axes = plt.subplots(
        2, 1, figsize=(11.5, 7.0), constrained_layout=True, sharex=True
    )
    for run in runs:
        base = f"dz={run.dz_um:g}"
        axes[0].plot(run.z_um, run.histories["rms_x_um"], label=base + " x")
        axes[0].plot(
            run.z_um,
            run.histories["rms_y_um"],
            linestyle="--",
            label=base + " y",
        )
        axes[1].plot(
            run.z_um, run.histories["centroid_x_um"], label=base + " x"
        )
        axes[1].plot(
            run.z_um,
            run.histories["centroid_y_um"],
            linestyle="--",
            label=base + " y",
        )
    axes[0].set(ylabel="RMS width (µm)", title="Beam moments")
    axes[1].set(ylabel="Centroid (µm)", xlabel="z (µm)")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(ncol=4, fontsize=7)
    figure.savefig(path, dpi=190)
    plt.close(figure)


def save_field_figure(
    path: Path,
    runs: list[RunData],
    x_um: np.ndarray,
    y_um: np.ndarray,
    kx: np.ndarray,
    ky: np.ndarray,
) -> None:
    figure = plt.figure(figsize=(14.0, 16.0), constrained_layout=True)
    grid_spec = figure.add_gridspec(4, 2)
    axes = [figure.add_subplot(grid_spec[index, column]) for index in range(2) for column in range(2)]
    near_x_axis, near_y_axis, far_x_axis, far_y_axis = axes
    ix = int(np.argmin(np.abs(x_um)))
    iy = int(np.argmin(np.abs(y_um)))
    ikx = int(np.argmin(np.abs(kx)))
    iky = int(np.argmin(np.abs(ky)))
    for run in runs:
        label = f"dz={run.dz_um:g} µm"
        near_x_axis.plot(x_um, run.near_intensity[:, iy], label=label)
        near_y_axis.plot(y_um, run.near_intensity[ix, :], label=label)
        far_x_axis.semilogy(
            kx,
            np.maximum(
                run.far_intensity[:, iky],
                float(np.max(run.far_intensity[:, iky])) * 1e-14,
            ),
            label=label,
        )
        far_y_axis.semilogy(
            ky,
            np.maximum(
                run.far_intensity[ikx, :],
                float(np.max(run.far_intensity[ikx, :])) * 1e-14,
            ),
            label=label,
        )
    near_x_axis.set(title="Final near-field central x cut", xlabel="x (µm)", ylabel="Intensity")
    near_y_axis.set(title="Final near-field central y cut", xlabel="y (µm)", ylabel="Intensity")
    far_x_axis.set(title="Final far-field central kx cut", xlabel="kx (rad/µm)", ylabel="Spectral density")
    far_y_axis.set(title="Final far-field central ky cut", xlabel="ky (rad/µm)", ylabel="Spectral density")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)

    global_peak = max(float(np.max(run.far_intensity)) for run in runs)
    vmax = math.log10(global_peak)
    vmin = vmax - LOG_DYNAMIC_RANGE_DECADES
    floor = 10.0**vmin
    image = None
    for index, run in enumerate(runs):
        axis = figure.add_subplot(grid_spec[2 + index // 2, index % 2])
        image = axis.imshow(
            np.log10(np.maximum(run.far_intensity, floor)).T,
            origin="lower",
            extent=(kx[0], kx[-1], ky[0], ky[-1]),
            vmin=vmin,
            vmax=vmax,
            cmap="magma",
            interpolation="nearest",
            aspect="equal",
        )
        axis.set(
            title=f"dz={run.dz_um:g} µm",
            xlabel="kx (rad/µm)",
            ylabel="ky (rad/µm)",
        )
    assert image is not None
    colorbar = figure.colorbar(image, ax=figure.axes, shrink=0.55, pad=0.015)
    colorbar.set_label("log10 normalized spectral power density")
    figure.suptitle("Nominal-dz final-field convergence", y=1.01)
    figure.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(figure)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = repo_root / "outputs" / "validation_06_nominal_dz_convergence"
    output_dir.mkdir(parents=True, exist_ok=True)
    convergence_png = output_dir / "validation_06_convergence.png"
    director_png = output_dir / "validation_06_director_intensity.png"
    beam_png = output_dir / "validation_06_beam_moments.png"
    fields_png = output_dir / "validation_06_final_fields.png"
    npz_path = output_dir / "validation_06_convergence_data.npz"
    summary_csv = output_dir / "validation_06_final_comparisons.csv"
    history_csv = output_dir / "validation_06_history_comparisons.csv"
    convergence_csv = output_dir / "validation_06_slice_convergence.csv"
    acceptance_csv = output_dir / "validation_06_acceptance.csv"
    metadata_json_path = output_dir / "validation_06_metadata.json"

    common_theta_z = np.arange(
        0.0, 3000.0, COMMON_THETA_SPACING_UM, dtype=float
    )
    runs = []
    for dz_um in DZ_VALUES_UM:
        print(f"running nominal dz={dz_um:g} um ...", flush=True)
        run = execute_case(dz_um, common_theta_z)
        runs.append(run)
        print(
            f"completed dz={dz_um:g} um in {run.elapsed_s:.3f} s; "
            f"plan={run.optical_plan}",
            flush=True,
        )

    reference = next(run for run in runs if run.dz_um == REFERENCE_DZ_UM)
    secondary = next(
        run for run in runs if run.dz_um == SECONDARY_REFERENCE_DZ_UM
    )
    final_vs_2 = {run.dz_um: compare_final(run, reference) for run in runs}
    final_vs_5 = {run.dz_um: compare_final(run, secondary) for run in runs}

    common_z_max = min(float(run.z_um[-1]) for run in runs)
    common_z = reference.z_um[reference.z_um <= common_z_max]
    interpolated = {
        run.dz_um: interpolate_histories(run, common_z) for run in runs
    }
    history_vs_2 = {
        run.dz_um: compare_histories(
            interpolated[run.dz_um], interpolated[REFERENCE_DZ_UM]
        )
        for run in runs
    }
    history_vs_5 = {
        run.dz_um: compare_histories(
            interpolated[run.dz_um],
            interpolated[SECONDARY_REFERENCE_DZ_UM],
        )
        for run in runs
    }
    acceptances = {
        run.dz_um: acceptance(run, final_vs_2[run.dz_um], reference)
        for run in runs
    }

    request = validation.make_request(
        REFERENCE_DZ_UM,
        "self_consistent",
        "zero_voltage_pi4_boundary",
    )
    grid = make_grid(request.grid, xp=np, real_dtype=np.float64)
    x_um = np.asarray(grid.x_um)
    y_um = np.asarray(grid.y_um)
    kx, ky, _, _, _ = validation.centered_far_field(
        reference.final_field,
        dx_um=grid.dx_um,
        dy_um=grid.dy_um,
        xp=np,
    )
    kx = np.asarray(kx)
    ky = np.asarray(ky)
    save_convergence_figure(convergence_png, runs)
    save_director_figure(director_png, runs)
    save_beam_figure(beam_png, runs)
    save_field_figure(fields_png, runs, x_um, y_um, kx, ky)

    summary_rows = []
    for run in runs:
        row = {
            "dz_nominal_um": run.dz_um,
            "reference_dz_um": REFERENCE_DZ_UM,
            **run.optical_plan,
            "elapsed_s": run.elapsed_s,
            "time_per_nominal_slice_s": run.elapsed_s / run.nominal_slices,
            "speedup_vs_dz2": reference.elapsed_s / run.elapsed_s,
            "nominal_slices": run.nominal_slices,
            "accepted_optical_substeps": run.accepted_optical_substeps,
            "executed_optical_substeps": run.executed_optical_substeps,
            "final_power_mW": run.final_power_mW,
            **{f"final_near_{key}": value for key, value in run.final_near.items()},
            **{f"final_far_{key}": value for key, value in run.final_far.items()},
            **run.convergence,
            **{f"vs_dz2_{key}": value for key, value in final_vs_2[run.dz_um].items()},
            **{f"vs_dz5_{key}": value for key, value in final_vs_5[run.dz_um].items()},
        }
        row["unconverged_z_ranges_um"] = json.dumps(
            row["unconverged_z_ranges_um"]
        )
        summary_rows.append(row)
    write_csv(summary_csv, summary_rows)

    history_rows = []
    for run in runs:
        history_rows.append(
            {
                "dz_nominal_um": run.dz_um,
                "reference_dz_um": REFERENCE_DZ_UM,
                **history_vs_2[run.dz_um],
                **{
                    f"vs_dz5_{key}": value
                    for key, value in history_vs_5[run.dz_um].items()
                },
            }
        )
    write_csv(history_csv, history_rows)

    convergence_rows = []
    for run in runs:
        for index in range(run.nominal_slices):
            convergence_rows.append(
                {
                    "dz_nominal_um": run.dz_um,
                    "z_um": run.z_um[index],
                    "converged": bool(run.converged[index]),
                    "residual_rms": run.residual_rms[index],
                    "residual_max": run.residual_max[index],
                    "relaxation_iterations": int(
                        run.relaxation_iterations[index]
                    ),
                    "optical_passes": int(run.optical_passes[index]),
                }
            )
    write_csv(convergence_csv, convergence_rows)

    acceptance_rows = []
    for run in runs:
        acceptance_rows.append(
            {"dz_nominal_um": run.dz_um, **acceptances[run.dz_um]}
        )
    write_csv(acceptance_csv, acceptance_rows)

    arrays = {
        "common_scalar_z_um": common_z,
        "common_theta_z_um": common_theta_z,
        "kx_rad_per_um": kx,
        "ky_rad_per_um": ky,
    }
    for run in runs:
        suffix = f"dz{int(run.dz_um)}"
        arrays[f"final_field_{suffix}"] = run.final_field
        arrays[f"final_near_intensity_{suffix}"] = run.near_intensity
        arrays[f"final_theta_last_{suffix}"] = run.final_theta_last
        arrays[f"theta_common_planes_{suffix}"] = run.theta_common_planes
        arrays[f"far_intensity_normalized_{suffix}"] = run.far_intensity
        arrays[f"z_um_{suffix}"] = run.z_um
        arrays[f"converged_{suffix}"] = run.converged
        arrays[f"residual_rms_{suffix}"] = run.residual_rms
        arrays[f"residual_max_{suffix}"] = run.residual_max
        arrays[f"relaxation_iterations_{suffix}"] = run.relaxation_iterations
        arrays[f"optical_passes_{suffix}"] = run.optical_passes
        for name, values in run.histories.items():
            arrays[f"history_{name}_{suffix}"] = values
        for name, values in interpolated[run.dz_um].items():
            arrays[f"common_history_{name}_{suffix}"] = values
    np.savez_compressed(npz_path, **arrays)

    metadata = {
        "physical_parameters": {
            "z_length_um": 3000.0,
            "waist_um": 3.0,
            "wavelength_um": 0.633,
            "V_app": 0.0,
            "theta_bc_rad": math.pi / 4.0,
            "noise_eps": 0.0,
            "power_mW": 1.0,
            "Nx": 128,
            "Ny": 128,
            "x_aperture_um": 75.0,
            "y_aperture_um": 100.0,
            "material_ne": 1.7,
            "material_no": 1.5,
            "material_K_N": 7.0e-12,
            "material_delta_epsilon": 13.0,
        },
        "numerical_parameters": {
            "workflow_strategy": request.solver.workflow.strategy,
            "theta_solver": request.solver.workflow.theta_solver,
            "optics_solver": request.solver.workflow.optics_solver,
            "coupling": request.solver.workflow.coupling,
            "precision": request.runtime.precision,
            "static_residual_rms_tol": request.solver.static_residual_rms_tol,
            "static_residual_max_tol": request.solver.static_residual_max_tol,
            "static_delta_theta_rms_tol": (
                request.solver.resolved_delta_theta_rms_tol
            ),
            "static_delta_theta_max_tol": (
                request.solver.resolved_delta_theta_max_tol
            ),
            "static_max_relax_iterations": (
                request.solver.static_max_relax_iterations
            ),
            "static_max_coupled_passes": (
                request.solver.resolved_static_max_coupled_passes
            ),
            "record_iteration_history": (
                request.solver.record_iteration_history
            ),
            "nominal_dz_values_um": list(DZ_VALUES_UM),
        },
        "optical_policy": {
            "enabled": True,
            "dn_max_est": 0.02,
            "max_phase_per_substep_rad": 0.30,
            "max_substeps": OPTICAL_MAX_SUBSTEPS,
        },
        "reference": (
            "dz=2 um is the finest nominal step in this study, not a "
            "mathematically converged solution"
        ),
        "common_scalar_z_range_um": [float(common_z[0]), float(common_z[-1])],
        "common_theta_plane_spacing_um": COMMON_THETA_SPACING_UM,
        "fft_axis_orientation": "array axis 0 -> kx; array axis 1 -> ky",
        "far_field_normalization": "unit integral over dkx*dky",
        "outer_spectrum_definition": (
            "union of the outermost 10 percent at each edge of kx or ky"
        ),
        "runs": {
            str(run.dz_um): {
                "optical_plan": run.optical_plan,
                "elapsed_s": run.elapsed_s,
                "convergence": run.convergence,
            }
            for run in runs
        },
        "total_wall_time_s": sum(run.elapsed_s for run in runs),
    }
    metadata_json_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )

    print()
    print(
        "total measured workflow wall time = "
        f"{sum(run.elapsed_s for run in runs):.3f} s"
    )
    print()
    print("dz  Nsub  dz_sub  phi_est  cap  slices  elapsed_s  speedup")
    for run in runs:
        print(
            f"{run.dz_um:2.0f}  {int(run.optical_plan['optical_Nsub']):5d}  "
            f"{float(run.optical_plan['optical_dz_sub_um']):6.3f}  "
            f"{float(run.optical_plan['optical_phi_est_rad']):7.4f}  "
            f"{str(bool(run.optical_plan['optical_substep_cap_reached'])):>5}  "
            f"{run.nominal_slices:6d}  {run.elapsed_s:9.3f}  "
            f"{reference.elapsed_s / run.elapsed_s:7.3f}"
        )
    print()
    print("dz  field_L2  near_L2  far_L2  theta_RMS  theta_max_abs  unc_frac")
    for run in runs:
        comp = final_vs_2[run.dz_um]
        print(
            f"{run.dz_um:2.0f}  "
            f"{comp['field_phase_aligned_relative_l2']:8.5f}  "
            f"{comp['near_relative_l2']:7.5f}  "
            f"{comp['far_relative_l2']:7.5f}  "
            f"{comp['theta_rms_difference_rad']:9.3e}  "
            f"{comp['theta_max_abs_difference_rad']:13.3e}  "
            f"{run.convergence['unconverged_fraction']:8.4f}"
        )
    print()
    print("dz  strict  moderate  failed_strict  failed_moderate")
    for run in runs:
        item = acceptances[run.dz_um]
        strict_failed = [
            name.removeprefix("strict_")
            for name, passed in item.items()
            if name.startswith("strict_") and name != "strict_overall" and not passed
        ]
        moderate_failed = [
            name.removeprefix("moderate_")
            for name, passed in item.items()
            if name.startswith("moderate_") and name != "moderate_overall" and not passed
        ]
        print(
            f"{run.dz_um:2.0f}  {str(item['strict_overall']):>6}  "
            f"{str(item['moderate_overall']):>8}  "
            f"{','.join(strict_failed) or '-':>13}  "
            f"{','.join(moderate_failed) or '-'}"
        )
    print(f"convergence_figure = {convergence_png}")
    print(f"director_figure = {director_png}")
    print(f"beam_figure = {beam_png}")
    print(f"fields_figure = {fields_png}")
    print(f"npz_path = {npz_path}")
    print(f"summary_csv = {summary_csv}")
    print(f"history_csv = {history_csv}")
    print(f"convergence_csv = {convergence_csv}")
    print(f"acceptance_csv = {acceptance_csv}")
    print(f"metadata_json = {metadata_json_path}")


if __name__ == "__main__":
    main()
