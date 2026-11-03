#!/usr/bin/env python3
"""Overlay PRProp longitudinal-step artifact predictions on LCProp far fields."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "lcprop-matplotlib-cache"),
)

import matplotlib.pyplot as plt
from scipy.signal import find_peaks

import validation_01_longitudinal_step_far_field as validation
from lcprop.core.backend import asnumpy
from lcprop.lc.optical_response import compute_neff
from lcprop.core.grid import make_grid


DZ_VALUES_UM = (20.0, 30.0, 40.0)
WORKFLOW = "self_consistent"
BIAS_CASE = "zero_voltage_pi4_boundary"
THETA_IN_RAD = 0.0
LOG_DYNAMIC_RANGE_DECADES = 12.0
RADIAL_BIN_SPACING_RAD_PER_UM = 2.0 * math.pi / 100.0


def line_style(order: int) -> tuple[str, float]:
    styles = ("-", "--", ":", "-.")
    if order <= 3:
        return styles[order - 1], 0.82
    return styles[(order - 1) % len(styles)], max(0.18, 0.72 - 0.008 * order)


def artifact_predictions(
    *,
    dz_um: float,
    n: float,
    wavelength_um: float,
    theta_in_rad: float,
    kx_limit: float,
    ky_limit: float,
) -> list[dict[str, float | int | bool]]:
    rows = []
    maximum_order = int(math.ceil(n * dz_um / wavelength_um))
    for order in range(1, maximum_order + 1):
        radial_term = (
            n * order / (wavelength_um * dz_um)
            - (order / dz_um) ** 2
        )
        if radial_term <= 0.0:
            continue
        f_jy = math.sqrt(radial_term)
        f_jx = math.cos(theta_in_rad) * f_jy
        k_jx = 2.0 * math.pi * f_jx
        k_jy = 2.0 * math.pi * f_jy
        inside_x = k_jx <= kx_limit
        inside_y = k_jy <= ky_limit
        if inside_x or inside_y:
            rows.append(
                {
                    "dz_um": float(dz_um),
                    "artifact_order_j": order,
                    "predicted_f_jx_cycles_per_um": f_jx,
                    "predicted_f_jy_cycles_per_um": f_jy,
                    "predicted_k_jx_rad_per_um": k_jx,
                    "predicted_k_jy_rad_per_um": k_jy,
                    "inside_x_nyquist": inside_x,
                    "inside_y_nyquist": inside_y,
                }
            )
    return rows


def radial_profile(
    intensity: np.ndarray,
    kx: np.ndarray,
    ky: np.ndarray,
    *,
    bin_spacing: float,
) -> tuple[np.ndarray, np.ndarray]:
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
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, profile


def measure_nearest_peaks(
    rows: list[dict[str, float | int | bool]],
    radius: np.ndarray,
    profile: np.ndarray,
    *,
    uncertainty: float,
) -> None:
    peak_indices, _ = find_peaks(profile)
    if peak_indices.size == 0:
        peak_indices = np.asarray([int(np.argmax(profile))])
    peak_radii = radius[peak_indices]
    for row in rows:
        predicted = float(row["predicted_k_jy_rad_per_um"])
        nearest = int(np.argmin(np.abs(peak_radii - predicted)))
        measured = float(peak_radii[nearest])
        delta = measured - predicted
        row["measured_nearest_radial_peak_rad_per_um"] = measured
        row["delta_k_rad_per_um"] = delta
        row["delta_k_fft_bins"] = delta / uncertainty
        row["estimated_uncertainty_rad_per_um"] = uncertainty


def draw_prediction_curves(axis, rows, kx_limit: float, ky_limit: float) -> None:
    phi = np.linspace(0.0, 2.0 * math.pi, 1000)
    for row_index, row in enumerate(rows):
        order = int(row["artifact_order_j"])
        k_jx = float(row["predicted_k_jx_rad_per_um"])
        k_jy = float(row["predicted_k_jy_rad_per_um"])
        style, alpha = line_style(order)
        x_curve = k_jx * np.cos(phi)
        y_curve = k_jy * np.sin(phi)
        axis.plot(
            x_curve,
            y_curve,
            linestyle=style,
            color="white",
            linewidth=0.65,
            alpha=alpha,
        )

        # Choose a visible point and vary the label angle to reduce collisions.
        valid = (np.abs(x_curve) <= kx_limit) & (np.abs(y_curve) <= ky_limit)
        valid_indices = np.flatnonzero(valid)
        if valid_indices.size:
            fraction = (0.17 + 0.19 * (row_index % 4)) % 1.0
            label_index = valid_indices[
                min(int(fraction * valid_indices.size), valid_indices.size - 1)
            ]
            axis.text(
                x_curve[label_index],
                y_curve[label_index],
                str(order),
                color="white",
                fontsize=4.5,
                alpha=max(alpha, 0.32),
                ha="center",
                va="center",
            )


def save_far_field_figure(
    path: Path,
    runs,
    predictions,
    kx: np.ndarray,
    ky: np.ndarray,
) -> None:
    global_peak = max(
        float(np.max(run.far_intensity_normalized)) for run in runs
    )
    log_vmax = math.log10(global_peak)
    log_vmin = log_vmax - LOG_DYNAMIC_RANGE_DECADES
    floor = 10.0**log_vmin
    kx_limit = float(np.max(np.abs(kx)))
    ky_limit = float(np.max(np.abs(ky)))

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(16.2, 6.2),
        constrained_layout=True,
    )
    image = None
    for run, rows, axis in zip(runs, predictions, axes, strict=True):
        image = axis.imshow(
            np.log10(np.maximum(run.far_intensity_normalized, floor)).T,
            origin="lower",
            extent=(kx[0], kx[-1], ky[0], ky[-1]),
            vmin=log_vmin,
            vmax=log_vmax,
            cmap="magma",
            interpolation="nearest",
            aspect="equal",
        )
        draw_prediction_curves(axis, rows, kx_limit, ky_limit)
        axis.set(
            xlabel="kx (rad/µm)",
            ylabel="ky (rad/µm)",
        )
        axis.set_title(f"dz={run.dz_um:g} µm", pad=8)
        radius_pairs = [
            f"{int(row['artifact_order_j'])}:{float(row['predicted_k_jx_rad_per_um']):.2f}"
            for row in rows
        ]
        radii = "\n".join(
            ", ".join(radius_pairs[start : start + 7])
            for start in range(0, len(radius_pairs), 7)
        )
        axis.text(
            0.5,
            -0.19,
            "j:k [rad/µm]\n" + radii,
            transform=axis.transAxes,
            color="black",
            fontsize=4.5,
            alpha=0.9,
            va="top",
            ha="center",
            clip_on=False,
        )

    assert image is not None
    colorbar = figure.colorbar(image, ax=axes, shrink=0.9, pad=0.02)
    colorbar.set_label("log10 normalized spectral power density")
    figure.suptitle(
        "PRProp artifact predictions — static self-consistent, "
        "V=0, theta_bc=π/4, eps=0",
        y=1.02,
    )
    figure.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(figure)


def save_radial_figure(path: Path, runs, predictions, radial_data) -> None:
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(16.2, 4.7),
        constrained_layout=True,
        sharey=True,
    )
    for run, rows, (radius, profile), axis in zip(
        runs,
        predictions,
        radial_data,
        axes,
        strict=True,
    ):
        positive = profile[profile > 0.0]
        floor = float(np.max(profile)) * 1e-14 if positive.size else 1e-300
        axis.semilogy(radius, np.maximum(profile, floor), color="black")
        for row_index, row in enumerate(rows):
            order = int(row["artifact_order_j"])
            predicted = float(row["predicted_k_jy_rad_per_um"])
            style, alpha = line_style(order)
            axis.axvline(
                predicted,
                linestyle=style,
                color="tab:red",
                linewidth=0.7,
                alpha=alpha,
            )
            y_fraction = 0.92 - 0.06 * (row_index % 5)
            axis.text(
                predicted,
                y_fraction,
                str(order),
                transform=axis.get_xaxis_transform(),
                color="tab:red",
                fontsize=5,
                alpha=max(alpha, 0.35),
                rotation=90,
                va="top",
                ha="right",
            )
        axis.set(
            title=f"dz={run.dz_um:g} µm",
            xlabel="radial k (rad/µm)",
            ylabel="Azimuthal mean spectral density",
            xlim=(0.0, max(float(row["predicted_k_jy_rad_per_um"]) for row in rows) + 0.2),
        )
        axis.grid(alpha=0.2)
    figure.suptitle(
        "Azimuthally averaged far fields with predicted artifact locations"
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
        3,
        2,
        figsize=(13.5, 11.0),
        constrained_layout=True,
    )
    central_kx = int(np.argmin(np.abs(kx)))
    central_ky = int(np.argmin(np.abs(ky)))

    for row_index, (run, rows) in enumerate(
        zip(runs, predictions, strict=True)
    ):
        x_cut = run.far_intensity_normalized[:, central_ky]
        y_cut = run.far_intensity_normalized[central_kx, :]
        x_floor = float(np.max(x_cut)) * 1e-14
        y_floor = float(np.max(y_cut)) * 1e-14
        x_axis, y_axis = axes[row_index]
        x_axis.semilogy(kx, np.maximum(x_cut, x_floor), color="black")
        y_axis.semilogy(ky, np.maximum(y_cut, y_floor), color="black")
        for row in rows:
            order = int(row["artifact_order_j"])
            style, alpha = line_style(order)
            for sign in (-1.0, 1.0):
                x_axis.axvline(
                    sign * float(row["predicted_k_jx_rad_per_um"]),
                    linestyle=style,
                    color="tab:red",
                    linewidth=0.65,
                    alpha=alpha,
                )
                y_axis.axvline(
                    sign * float(row["predicted_k_jy_rad_per_um"]),
                    linestyle=style,
                    color="tab:red",
                    linewidth=0.65,
                    alpha=alpha,
                )
        x_axis.set(
            title=f"dz={run.dz_um:g} µm: I(kx, ky=0)",
            xlabel="kx (rad/µm)",
            ylabel="Spectral density",
        )
        y_axis.set(
            title=f"dz={run.dz_um:g} µm: I(kx=0, ky)",
            xlabel="ky (rad/µm)",
            ylabel="Spectral density",
        )
        x_axis.grid(alpha=0.2)
        y_axis.grid(alpha=0.2)
    figure.suptitle("Central far-field cuts with predicted ± artifact intersections")
    figure.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = (
        repo_root
        / "outputs"
        / "validation_03_prprop_artifact_rings_dz20_30_40"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    far_png = output_dir / "validation_03_far_fields_with_predicted_rings.png"
    radial_png = output_dir / "validation_03_radial_profiles.png"
    cuts_png = output_dir / "validation_03_xy_cuts.png"
    csv_path = output_dir / "validation_03_predicted_and_measured_rings.csv"
    npz_path = output_dir / "validation_03_artifact_ring_data.npz"

    reference_request = validation.make_request(
        DZ_VALUES_UM[0],
        WORKFLOW,
        BIAS_CASE,
    )
    wavelength_um = float(reference_request.beams.channels[0].wavelength_um)
    n_prediction = float(reference_request.material.no)
    neff_pi4 = float(
        compute_neff(
            math.pi / 4.0,
            ne=reference_request.material.ne,
            no=reference_request.material.no,
        )
    )
    print(f"wavelength_um = {wavelength_um:.12f}")
    print(f"prediction_n = {n_prediction:.12f}")
    print("prediction_n_source = run_static n_ref = request.material.no")
    print(f"neff_pi_over_4_for_comparison = {neff_pi4:.12f}")

    runs = []
    kx = None
    ky = None
    for dz_um in DZ_VALUES_UM:
        print(f"running dz={dz_um:g} um ...", flush=True)
        run, case_kx, case_ky = validation.execute_case(
            dz_um,
            WORKFLOW,
            noise_mode="off",
            noise_eps=0.0,
            noise_sigma_um=0.4,
            noise_seed=0,
            bias_case=BIAS_CASE,
        )
        runs.append(run)
        if kx is None:
            kx = case_kx
            ky = case_ky
        else:
            np.testing.assert_array_equal(kx, case_kx)
            np.testing.assert_array_equal(ky, case_ky)
        print(f"completed dz={dz_um:g} um in {run.elapsed_s:.3f} s")

    assert kx is not None
    assert ky is not None
    kx_limit = float(np.max(np.abs(kx)))
    ky_limit = float(np.max(np.abs(ky)))
    dkx = float(abs(kx[1] - kx[0]))
    dky = float(abs(ky[1] - ky[0]))
    radial_spacing = min(dkx, dky)
    if not np.isclose(radial_spacing, RADIAL_BIN_SPACING_RAD_PER_UM):
        raise AssertionError("unexpected radial FFT-bin spacing")
    if not (np.all(np.diff(kx) > 0.0) and np.all(np.diff(ky) > 0.0)):
        raise AssertionError("centered FFT axes must be strictly increasing")
    for run in runs:
        if run.final_field.shape != (kx.size, ky.size):
            raise AssertionError("field axis 0/1 must map to kx/ky")
        spectral_integral = (
            float(np.sum(run.far_intensity_normalized)) * dkx * dky
        )
        if not np.isclose(spectral_integral, 1.0, rtol=1e-12, atol=1e-12):
            raise AssertionError("far field must have unit integrated power")

    predictions = []
    radial_data = []
    all_rows = []
    for run in runs:
        rows = artifact_predictions(
            dz_um=run.dz_um,
            n=n_prediction,
            wavelength_um=wavelength_um,
            theta_in_rad=THETA_IN_RAD,
            kx_limit=kx_limit,
            ky_limit=ky_limit,
        )
        radius, profile = radial_profile(
            run.far_intensity_normalized,
            kx,
            ky,
            bin_spacing=radial_spacing,
        )
        measure_nearest_peaks(
            rows,
            radius,
            profile,
            uncertainty=radial_spacing,
        )
        predictions.append(rows)
        radial_data.append((radius, profile))
        all_rows.extend(rows)

    save_far_field_figure(far_png, runs, predictions, kx, ky)
    save_radial_figure(radial_png, runs, predictions, radial_data)
    save_cut_figure(cuts_png, runs, predictions, kx, ky)

    fieldnames = list(all_rows[0])
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    metadata = {
        "wavelength_um": wavelength_um,
        "prediction_n": n_prediction,
        "prediction_n_source": "run_static n_ref = request.material.no",
        "neff_pi_over_4_for_comparison": neff_pi4,
        "theta_in_rad": THETA_IN_RAD,
        "kx_nyquist_limit_rad_per_um": kx_limit,
        "ky_nyquist_limit_rad_per_um": ky_limit,
        "kx_first_last_rad_per_um": [float(kx[0]), float(kx[-1])],
        "ky_first_last_rad_per_um": [float(ky[0]), float(ky[-1])],
        "dkx_rad_per_um": dkx,
        "dky_rad_per_um": dky,
        "radial_peak_uncertainty_rad_per_um": radial_spacing,
        "fft_axis_orientation": "axis 0 -> kx; axis 1 -> ky; transpose only for imshow",
        "frequency_conversion": "k = 2*pi*f; f cycles/um; k rad/um",
        "fft_extent": "full 128x128 computational field and discrete Nyquist spectrum",
    }
    np.savez_compressed(
        npz_path,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        dz_um=np.asarray(DZ_VALUES_UM),
        kx_rad_per_um=kx,
        ky_rad_per_um=ky,
        radial_k_rad_per_um=np.stack([item[0] for item in radial_data]),
        radial_profile=np.stack([item[1] for item in radial_data]),
        n_steps=np.asarray([run.n_steps for run in runs], dtype=np.int64),
        elapsed_s=np.asarray([run.elapsed_s for run in runs]),
        final_power_mW=np.asarray([run.final_power_mW for run in runs]),
        normalized_final_power=np.asarray(
            [run.normalized_final_power for run in runs]
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
        final_field=np.stack([run.final_field for run in runs]),
        far_field_intensity_normalized=np.stack(
            [run.far_intensity_normalized for run in runs]
        ),
        artifact_dz_um=np.asarray([row["dz_um"] for row in all_rows]),
        artifact_order_j=np.asarray(
            [row["artifact_order_j"] for row in all_rows],
            dtype=np.int64,
        ),
        predicted_f_jx_cycles_per_um=np.asarray(
            [row["predicted_f_jx_cycles_per_um"] for row in all_rows]
        ),
        predicted_f_jy_cycles_per_um=np.asarray(
            [row["predicted_f_jy_cycles_per_um"] for row in all_rows]
        ),
        predicted_k_jx_rad_per_um=np.asarray(
            [row["predicted_k_jx_rad_per_um"] for row in all_rows]
        ),
        predicted_k_jy_rad_per_um=np.asarray(
            [row["predicted_k_jy_rad_per_um"] for row in all_rows]
        ),
        inside_x_nyquist=np.asarray(
            [row["inside_x_nyquist"] for row in all_rows],
            dtype=bool,
        ),
        inside_y_nyquist=np.asarray(
            [row["inside_y_nyquist"] for row in all_rows],
            dtype=bool,
        ),
        measured_nearest_radial_peak_rad_per_um=np.asarray(
            [
                row["measured_nearest_radial_peak_rad_per_um"]
                for row in all_rows
            ]
        ),
        delta_k_rad_per_um=np.asarray(
            [row["delta_k_rad_per_um"] for row in all_rows]
        ),
        delta_k_fft_bins=np.asarray(
            [row["delta_k_fft_bins"] for row in all_rows]
        ),
        estimated_uncertainty_rad_per_um=np.asarray(
            [row["estimated_uncertainty_rad_per_um"] for row in all_rows]
        ),
    )

    print()
    print(
        "dz_um  j  f_jx_cyc_um  f_jy_cyc_um  k_jx_rad_um  k_jy_rad_um  "
        "inside_x  inside_y  measured_k  delta_k  delta_bins"
    )
    for row in all_rows:
        print(
            f"{float(row['dz_um']):5.1f}  "
            f"{int(row['artifact_order_j']):2d}  "
            f"{float(row['predicted_f_jx_cycles_per_um']):11.8f}  "
            f"{float(row['predicted_f_jy_cycles_per_um']):11.8f}  "
            f"{float(row['predicted_k_jx_rad_per_um']):11.8f}  "
            f"{float(row['predicted_k_jy_rad_per_um']):11.8f}  "
            f"{str(bool(row['inside_x_nyquist'])):>8}  "
            f"{str(bool(row['inside_y_nyquist'])):>8}  "
            f"{float(row['measured_nearest_radial_peak_rad_per_um']):10.7f}  "
            f"{float(row['delta_k_rad_per_um']):+10.7f}  "
            f"{float(row['delta_k_fft_bins']):+10.4f}"
        )
    print(f"kx_limit_rad_per_um = {kx_limit:.12f}")
    print(f"ky_limit_rad_per_um = {ky_limit:.12f}")
    print(f"kx_first_last_rad_per_um = {kx[0]:.12f}, {kx[-1]:.12f}")
    print(f"ky_first_last_rad_per_um = {ky[0]:.12f}, {ky[-1]:.12f}")
    print(f"dkx_rad_per_um = {dkx:.12f}")
    print(f"dky_rad_per_um = {dky:.12f}")
    print(f"radial_peak_uncertainty_rad_per_um = {radial_spacing:.12f}")
    print()
    print("dz_um  n_steps  final_power_mW  unconverged_slices")
    for run in runs:
        print(
            f"{run.dz_um:5.1f}  {run.n_steps:7d}  "
            f"{run.final_power_mW:14.9f}  {run.unconverged_slices:18d}"
        )
    print(f"far_field_figure = {far_png}")
    print(f"radial_figure = {radial_png}")
    print(f"cuts_figure = {cuts_png}")
    print(f"csv_path = {csv_path}")
    print(f"npz_path = {npz_path}")


if __name__ == "__main__":
    main()
